import csv
import datetime
from datetime import timedelta
from typing import List

import pytz

from prosimos.control_flow_manager import (
    BPMN,
    CustomDatetimeAndSeconds,
)
from prosimos.execution_info import EnabledEvent, TaskEvent, Trace
from prosimos.file_manager import FileManager
from prosimos.prioritisation import CasePrioritisation
from prosimos.simulation_properties_parser import parse_datetime
from prosimos.simulation_queues_ds import (
    DiffResourceQueue,
    EventQueue,
)
from prosimos.simulation_setup import SimDiffSetup
from prosimos.simulation_stats_calculator import LogInfo


class SimResource:
    def __init__(self):
        self.switching_time = 0
        self.allocated_tasks = 0
        self.worked_time = 0
        self.available_time = 0
        self.last_released = 0


class SimBPMEnv:
    def __init__(self, sim_setup: SimDiffSetup, stat_fwriter, log_fwriter):
        self.sim_setup = sim_setup
        self.sim_resources = dict()
        self.stat_fwriter = stat_fwriter
        self.log_writer = FileManager(10000, log_fwriter, self.sim_setup.case_attributes.get_columns_generated())
        self.log_info = LogInfo(sim_setup)
        self.executed_events = 0
        self.time_update_process_state = 0

        r_first_available = dict()
        for r_id in sim_setup.resources_map:
            self.sim_resources[r_id] = SimResource()
            r_first_available[r_id] = self.sim_setup.next_resting_time(r_id, self.sim_setup.start_datetime)

        self.resource_queue = DiffResourceQueue(self.sim_setup.task_resource, r_first_available)
        self.events_queue = EventQueue()
        self.all_process_states = dict()  # store all process states with a case_id as a key

        self.case_prioritisation = CasePrioritisation(
            self.sim_setup.total_num_cases,
            self.sim_setup.case_attributes,
            self.sim_setup.prioritisation_rules,
        )

    def calc_priority_and_append_to_queue(self, enabled_event: EnabledEvent, is_arrival_event: bool):
        if enabled_event.is_inter_event:
            # append with the highest priority
            highest_priority = 0
            self.append_enabled_event_to_queue(enabled_event, is_arrival_event, highest_priority)
            return

        case_priority = self.calc_priority_for_task_or_batch(enabled_event)

        self.append_enabled_event_to_queue(enabled_event, is_arrival_event, case_priority)

    def calc_priority_for_task_or_batch(self, enabled_event):
        """
        Calculate case priority by following one of two path:
        1) no batching  - use current case's priority
        2) batching     - find case id with the highest priority
        """
        if enabled_event.batch_info_exec is not None:
            # batched task
            multiple_cases_dict = enabled_event.batch_info_exec.case_ids
            multiple_cases_arr = [(k, v) for k, v in multiple_cases_dict.items()]
            case_priority = self.case_prioritisation.calculate_max_priority(multiple_cases_arr)
        else:
            case_priority = self.case_prioritisation.get_priority_by_case_id(enabled_event.p_case)

        return case_priority

    def append_enabled_event_to_queue(self, enabled_event: EnabledEvent, is_arrival_event: bool, case_priority):
        "Append as either an arrival event or enabled intermediate/end event"
        if is_arrival_event:
            self.events_queue.append_arrival_event(enabled_event, case_priority)
        else:
            self.events_queue.append_enabled_event(enabled_event, case_priority)

    def generate_all_arrival_events(self):
        sim_setup = self.sim_setup
        arrival_time = 0
        for p_case in range(0, sim_setup.total_num_cases):
            enabled_datetime = self._update_initial_event_info(self.sim_setup, p_case, arrival_time)
            arrival_time += sim_setup.next_arrival_time(enabled_datetime)

    def generate_fixed_arrival_events(self, starting_times):
        p_case = 0
        for arrival_time in starting_times:
            self._update_initial_event_info(self.sim_setup, p_case, arrival_time)
            p_case += 1

    def _update_initial_event_info(self, sim_setup, p_case, arrival_time):
        for e_id in sim_setup.bpmn_graph.last_datetime:
            sim_setup.bpmn_graph.last_datetime[e_id][p_case] = None
        p_state = sim_setup.initial_state()
        enabled_datetime = self.simulation_datetime_from(arrival_time)
        enabled_time = CustomDatetimeAndSeconds(arrival_time, enabled_datetime)
        enabled_tasks, _ = sim_setup.update_process_state(
            p_case, sim_setup.bpmn_graph.starting_event, p_state, enabled_time
        )
        self.all_process_states[p_case] = p_state
        self.log_info.trace_list.append(Trace(p_case, enabled_datetime))
        for task in enabled_tasks:
            task_id = task.task_id
            self.calc_priority_and_append_to_queue(
                EnabledEvent(
                    p_case,
                    p_state,
                    task_id,
                    arrival_time,
                    enabled_datetime,
                    task.batch_info_exec,
                    task.duration_sec,
                    task.is_event,
                ),
                True,
            )
        return enabled_datetime

    def execute_enabled_event(self, c_event: EnabledEvent):
        self.executed_events += 1

        event_element_info = self.sim_setup.bpmn_graph.element_info[c_event.task_id]

        if event_element_info.type == BPMN.TASK and c_event.batch_info_exec is not None:
            # execute batched task
            executed_tasks = self.execute_task_batch(c_event)

            for task in executed_tasks:
                completed_at, completed_datetime, p_case = task
                p_state = self.all_process_states[p_case]
                enabled_time = CustomDatetimeAndSeconds(completed_at, completed_datetime)
                enabled_tasks, visited_at = self.sim_setup.update_process_state(
                    p_case,
                    c_event.task_id,
                    self.all_process_states[p_case],
                    enabled_time,
                )

                for next_task in enabled_tasks:
                    self.calc_priority_and_append_to_queue(
                        EnabledEvent(
                            p_case,
                            p_state,
                            next_task.task_id,
                            visited_at[next_task.task_id].seconds_from_start,
                            visited_at[next_task.task_id].datetime,
                            next_task.batch_info_exec,
                            next_task.duration_sec,
                            next_task.is_event,
                        ),
                        False,
                    )
        else:
            if event_element_info.type == BPMN.TASK:
                # execute not batched task
                completed_at, completed_datetime = self.execute_task(c_event)
            else:
                completed_at, completed_datetime = self.execute_event(c_event)

            # Updating the process state. Retrieving/enqueuing enabled tasks, it also schedules the corresponding event
            # s_t = datetime.datetime.now()
            enabled_time = CustomDatetimeAndSeconds(completed_at, completed_datetime)
            enabled_tasks, visited_at = self.sim_setup.update_process_state(
                c_event.p_case, c_event.task_id, c_event.p_state, enabled_time
            )
            # self.time_update_process_state += (datetime.datetime.now() - s_t).total_seconds()

            for next_task in enabled_tasks:
                self.calc_priority_and_append_to_queue(
                    EnabledEvent(
                        c_event.p_case,
                        c_event.p_state,
                        next_task.task_id,
                        visited_at[next_task.task_id].seconds_from_start,
                        visited_at[next_task.task_id].datetime,
                        next_task.batch_info_exec,
                        next_task.duration_sec,
                        next_task.is_event,
                    ),
                    False,
                )

    def pop_and_allocate_resource(self, task_id: str, num_allocated_tasks: int):
        r_id, r_avail_at = self.resource_queue.pop_resource_for(task_id)
        self.sim_resources[r_id].allocated_tasks += num_allocated_tasks
        return r_id, r_avail_at

    def execute_task(self, c_event: EnabledEvent):
        r_id, r_avail_at = self.pop_and_allocate_resource(c_event.task_id, 1)

        r_avail_at = max(c_event.enabled_at, r_avail_at)
        avail_datetime = self._datetime_from(r_avail_at)
        is_working, _ = self.sim_setup.get_resource_calendar(r_id).is_working_datetime(avail_datetime)
        if not is_working:
            r_avail_at = r_avail_at + self.sim_setup.next_resting_time(r_id, avail_datetime)

        full_evt = TaskEvent(
            c_event.p_case,
            c_event.task_id,
            r_id,
            r_avail_at,
            c_event.enabled_at,
            c_event.enabled_datetime,
            self,
        )

        self.log_info.add_event_info(c_event.p_case, full_evt, self.sim_setup.resources_map[r_id].cost_per_hour)

        r_next_available = full_evt.completed_at

        if self.sim_resources[r_id].switching_time > 0:
            r_next_available += self.sim_setup.next_resting_time(r_id, full_evt.completed_datetime)

        self.resource_queue.update_resource_availability(r_id, r_next_available)
        self.sim_resources[r_id].worked_time += full_evt.ideal_duration

        self.log_writer.add_csv_row(self.get_csv_row_data(full_evt))

        completed_at = full_evt.completed_at
        completed_datetime = full_evt.completed_datetime

        return completed_at, completed_datetime

    def get_csv_row_data(self, full_event: TaskEvent):
        """
        Return array of values for one line of the csv file based on full_event information.
        In case we have defined case attributes setup, we will have additional columns besides the basic ones.
        """

        resource_name = (
            self.sim_setup.resources_map[full_event.resource_id].resource_name
            if (hasattr(full_event, "resource_id"))
            else "No assigned resource"
        )

        row_basic_info = verify_miliseconds(
            [
                full_event.p_case,
                self.sim_setup.bpmn_graph.element_info[full_event.task_id].name,
                full_event.enabled_datetime,
                full_event.started_datetime,
                full_event.completed_datetime,
                resource_name,
            ]
        )

        extended_with_case_atrr = self.case_prioritisation.get_case_attr_values(full_event.p_case)

        return [*row_basic_info, *extended_with_case_atrr]

    def append_any_enabled_batch_tasks(self, current_event: EnabledEvent) -> List[EnabledEvent]:
        enabled_datetime = CustomDatetimeAndSeconds(current_event.enabled_at, current_event.enabled_datetime)
        enabled_batch_task_ids = self.sim_setup.is_any_batch_enabled(enabled_datetime)

        if enabled_batch_task_ids is not None:
            for batch_task_id, batch_info in enabled_batch_task_ids.items():
                start_time_from_rule = batch_info.start_time_from_rule

                # TODO: cover with additional test cases
                # when start_time_from_rule > current_event.enabled_datetime

                if start_time_from_rule < current_event.enabled_datetime:
                    # get needed value in seconds according to the
                    # already existing pair of seconds and datetime
                    timedelta_sec = (current_event.enabled_datetime - start_time_from_rule).total_seconds()
                    enabled_at = current_event.enabled_at - timedelta_sec
                    enabled_datetime = start_time_from_rule
                else:
                    enabled_at = current_event.enabled_at
                    enabled_datetime = current_event.enabled_datetime

                c_event = EnabledEvent(
                    current_event.p_case,
                    current_event.p_state,
                    batch_task_id,
                    enabled_at,
                    enabled_datetime,
                    batch_info,
                )
                self.calc_priority_and_append_to_queue(c_event, False)

    def execute_if_any_unexecuted_batch(self, last_task_enabled_time: CustomDatetimeAndSeconds):
        for case_id, enabled_datetime in self.sim_setup.is_any_unexecuted_batch(last_task_enabled_time):
            if not enabled_datetime:
                return

            enabled_batch_task_ids = self.sim_setup.is_any_batch_enabled(enabled_datetime)

            if not len(enabled_batch_task_ids):
                # no rules were satisfied
                # check whether there are some invalid rules
                invalid_batches = self.sim_setup.get_invalid_batches_if_any(last_task_enabled_time)
                if invalid_batches is not None:
                    for key, item in invalid_batches.items():
                        if key not in enabled_batch_task_ids:
                            enabled_batch_task_ids[key] = item

            if enabled_batch_task_ids is not None:
                for batch_task_id, batch_info in enabled_batch_task_ids.items():
                    c_event = EnabledEvent(
                        case_id,
                        self.all_process_states[case_id],
                        batch_task_id,
                        self.simulation_at_from_datetime(batch_info.start_time_from_rule),
                        batch_info.start_time_from_rule,
                        batch_info,
                    )
                    self.calc_priority_and_append_to_queue(c_event, False)

    def _get_chunk(self, batch_spec, curr_index, all_case_ids):
        """Return only the part of the all_case_ids that will be executed as a batch"""
        acc_tasks_in_batch = 0
        for i in range(0, curr_index):
            acc_tasks_in_batch = acc_tasks_in_batch + batch_spec[i]
        num_tasks_in_batch = batch_spec[curr_index]
        return all_case_ids[acc_tasks_in_batch : acc_tasks_in_batch + num_tasks_in_batch]

    def execute_task_batch(self, c_event: EnabledEvent):
        all_tasks_waiting = len(c_event.batch_info_exec.case_ids)

        if all_tasks_waiting == 0:
            print("WARNING: Number of tasks in the enabled batch is 0.")

        all_case_ids = list(c_event.batch_info_exec.case_ids.items())
        ordered_case_ids = self.case_prioritisation.get_ordered_case_ids_by_priority(all_case_ids)
        batch_spec = c_event.batch_info_exec.batch_spec
        chunks = [self._get_chunk(batch_spec, i, ordered_case_ids) for i in range(0, len(batch_spec))]

        if c_event.batch_info_exec.is_sequential():
            return self.execute_seq_task_batch(c_event, chunks)
        elif c_event.batch_info_exec.is_parallel():
            return self.execute_parallel_task_batch(c_event, chunks)
        else:
            print(f"WARNING: {c_event.batch_info_exec.task_batch_info.type} not supported")

    def execute_seq_task_batch(self, c_event: EnabledEvent, chunks):
        start_time_from_rule_seconds = (
            c_event.batch_info_exec.start_time_from_rule - self.sim_setup.start_datetime
        ).total_seconds()

        for batch_item in chunks:
            num_tasks_in_batch = len(batch_item)

            r_id, r_avail_at = self.pop_and_allocate_resource(c_event.task_id, num_tasks_in_batch)

            completed_at = 0

            for case_id, enabled_time in batch_item:
                p_case = case_id
                task_id = c_event.task_id
                enabled_at = enabled_time.seconds_from_start
                enabled_datetime = enabled_time.datetime
                enabled_batch = c_event.enabled_at

                r_avail_at = max(
                    enabled_at,
                    r_avail_at,
                    enabled_batch,
                    completed_at,
                    start_time_from_rule_seconds,
                )
                avail_datetime = self._datetime_from(r_avail_at)
                is_working, _ = self.sim_setup.get_resource_calendar(r_id).is_working_datetime(avail_datetime)
                if not is_working:
                    r_avail_at = r_avail_at + self.sim_setup.next_resting_time(r_id, avail_datetime)

                full_evt = TaskEvent(
                    p_case,
                    task_id,
                    r_id,
                    r_avail_at,
                    enabled_at,
                    enabled_datetime,
                    self,
                    num_tasks_in_batch,
                )

                self.sim_resources[r_id].worked_time += full_evt.ideal_duration
                (
                    completed_at,
                    completed_datetime,
                ) = self._update_logs_and_resource_availability(full_evt, r_id)

                yield completed_at, completed_datetime, p_case

    def execute_parallel_task_batch(self, c_event: EnabledEvent, chunks):
        task_id = c_event.task_id

        start_time_from_rule_datetime = c_event.batch_info_exec.start_time_from_rule
        if start_time_from_rule_datetime == None:
            start_time_from_rule_seconds = 0
            enabled_batch = c_event.enabled_at
        else:
            # edge case: start_time_from_rule overwrite the enabled time from the execution
            # happens when we entered the day (e.g., Monday) during the time
            # waiting for the task execution in the queue
            start_time_from_rule_seconds = (
                c_event.batch_info_exec.start_time_from_rule - self.sim_setup.start_datetime
            ).total_seconds()
            enabled_batch = 0

        for batch_item in chunks:
            num_tasks_in_batch = len(batch_item)

            r_id, r_avail_at = self.pop_and_allocate_resource(c_event.task_id, num_tasks_in_batch)

            r_avail_at = max(r_avail_at, enabled_batch, start_time_from_rule_seconds)
            avail_datetime = self._datetime_from(r_avail_at)
            is_working, _ = self.sim_setup.get_resource_calendar(r_id).is_working_datetime(avail_datetime)
            if not is_working:
                r_avail_at = r_avail_at + self.sim_setup.next_resting_time(r_id, avail_datetime)

            for case_id, enabled_time in batch_item:
                p_case = case_id
                enabled_at = enabled_time.seconds_from_start
                enabled_datetime = enabled_time.datetime

                full_evt = TaskEvent(
                    p_case,
                    task_id,
                    r_id,
                    r_avail_at,
                    enabled_at,
                    enabled_datetime,
                    self,
                    num_tasks_in_batch,
                )

                (
                    completed_at,
                    completed_datetime,
                ) = self._update_logs_and_resource_availability(full_evt, r_id)

                yield completed_at, completed_datetime, p_case

            # since the tasks are executed in parallel
            # we add their duration only once cause they were happening at the same time
            self.sim_resources[r_id].worked_time += full_evt.ideal_duration

    def _update_logs_and_resource_availability(self, full_evt: TaskEvent, r_id):
        self.log_info.add_event_info(full_evt.p_case, full_evt, self.sim_setup.resources_map[r_id].cost_per_hour)

        r_next_available = full_evt.completed_at

        if self.sim_resources[r_id].switching_time > 0:
            r_next_available += self.sim_setup.next_resting_time(r_id, full_evt.completed_datetime)

        self.resource_queue.update_resource_availability(r_id, r_next_available)

        self.log_writer.add_csv_row(self.get_csv_row_data(full_evt))

        completed_at = full_evt.completed_at
        completed_datetime = full_evt.completed_datetime

        return completed_at, completed_datetime

    def execute_event(self, c_event):
        # Handle event types separately (they don't need assigned resource)
        event_duration_seconds = None
        event_element = self.sim_setup.bpmn_graph.element_info[c_event.task_id]
        [event_duration_seconds] = self.sim_setup.bpmn_graph.event_duration(event_element.id)

        completed_at = c_event.enabled_at + event_duration_seconds
        completed_datetime = c_event.enabled_datetime + timedelta(seconds=event_duration_seconds)

        full_evt = TaskEvent.create_event_entity(c_event, completed_at, completed_datetime)

        self.log_info.add_event_info(c_event.p_case, full_evt, 0)

        if self.sim_setup.is_event_added_to_log:
            self.log_writer.add_csv_row(self.get_csv_row_data(full_evt))

        return completed_at, completed_datetime

    def _datetime_from(self, in_seconds):
        return self.simulation_datetime_from(in_seconds) if in_seconds is not None else None

    def simulation_datetime_from(self, simpy_time):
        return self.sim_setup.start_datetime + timedelta(seconds=simpy_time)

    def simulation_at_from_datetime(self, datetime):
        td = datetime - self.sim_setup.start_datetime
        return td.seconds

    def get_utilization_for(self, resource_id):
        if self.sim_resources[resource_id].available_time == 0:
            return -1
        return self.sim_resources[resource_id].worked_time / self.sim_resources[resource_id].available_time

    def _find_worked_times(self, event_info, completed_events):
        i = len(completed_events) - 1
        resource_calendar = self.sim_setup.get_resource_calendar(event_info.resource_id)
        current_end = event_info.completed_at
        duration = 0
        while i >= 0:
            prev_event = completed_events[i]
            if event_info.started_at >= prev_event.completed_at:
                break
            else:
                if prev_event.completed_at < current_end:
                    duration += resource_calendar.find_working_time(prev_event.completed_at, current_end)
                if event_info.started_at < prev_event.started_at:
                    current_end = prev_event.started_at
                else:
                    return duration
            i -= 1
        return duration + resource_calendar.find_working_time(event_info.started_at, current_end)


def execute_full_process(bpm_env: SimBPMEnv, fixed_starting_times=None):
    # Initialize event queue with the arrival times of all the cases to simulate,
    # i.e., all the initial events are enqueued and sorted by their arrival times
    # s_t = datetime.datetime.now()
    if fixed_starting_times is None:
        bpm_env.generate_all_arrival_events()
    else:
        bpm_env.generate_fixed_arrival_events(fixed_starting_times)

    # print("Generation of all cases: %s" %
    #       str(datetime.timedelta(seconds=(datetime.datetime.now() - s_t).total_seconds())))
    current_event = bpm_env.events_queue.pop_next_event()
    while current_event is not None:
        bpm_env.execute_enabled_event(current_event)

        # find the next event to be executed
        # double-check whether there are elements that need to be executed before the start of the event
        # add founded elements to the queue, if any
        intermediate_event = bpm_env.events_queue.peek()
        if intermediate_event is not None:
            bpm_env.append_any_enabled_batch_tasks(intermediate_event)

        current_event = bpm_env.events_queue.pop_next_event()
        if current_event is not None:
            # save the datetime of the last executed task in the flow
            last_event_datetime = CustomDatetimeAndSeconds(current_event.enabled_at, current_event.enabled_datetime)
        else:
            # we reached the point where all tasks enabled for the execution were executed
            # add to the events_queue batched tasks if any
            bpm_env.execute_if_any_unexecuted_batch(last_event_datetime)

            # verifying whether we still have (batched) tasks to be executed in the future
            current_event = bpm_env.events_queue.pop_next_event()


def run_simulation(
    bpmn_path,
    json_path,
    total_cases,
    stat_out_path=None,
    log_out_path=None,
    starting_at=None,
    is_event_added_to_log=False,
    fixed_arrival_times=None,
):
    diffsim_info = SimDiffSetup(bpmn_path, json_path, is_event_added_to_log, total_cases)

    if not diffsim_info:
        return None

    starting_at_datetime = (
        parse_datetime(starting_at, True) if starting_at else pytz.utc.localize(datetime.datetime.now())
    )
    diffsim_info.set_starting_datetime(starting_at_datetime)

    # if not stat_out_path and not log_out_path:
    #     stat_out_path = os.path.join(os.path.dirname(__file__), Path("%s.csv" % diffsim_info.process_name))
    if stat_out_path is None and log_out_path is None:
        return run_simpy_simulation(diffsim_info, None, None, fixed_arrival_times)
    elif stat_out_path:
        with open(stat_out_path, mode="w", newline="", encoding="utf-8") as stat_csv_file:
            if log_out_path:
                with open(log_out_path, mode="w", newline="", encoding="utf-8") as log_csv_file:
                    return run_simpy_simulation(
                        diffsim_info,
                        csv.writer(
                            stat_csv_file,
                            delimiter=",",
                            quotechar='"',
                            quoting=csv.QUOTE_MINIMAL,
                        ),
                        csv.writer(
                            log_csv_file,
                            delimiter=",",
                            quotechar='"',
                            quoting=csv.QUOTE_MINIMAL,
                        ),
                        fixed_arrival_times,
                    )
            else:
                return run_simpy_simulation(
                    diffsim_info,
                    csv.writer(
                        stat_csv_file,
                        delimiter=",",
                        quotechar='"',
                        quoting=csv.QUOTE_MINIMAL,
                    ),
                    None,
                    fixed_arrival_times,
                )
    else:
        with open(log_out_path, mode="w", newline="", encoding="utf-8") as log_csv_file:
            return run_simpy_simulation(
                diffsim_info,
                None,
                csv.writer(
                    log_csv_file,
                    delimiter=",",
                    quotechar='"',
                    quoting=csv.QUOTE_MINIMAL,
                ),
                fixed_arrival_times,
            )


def run_simpy_simulation(diffsim_info, stat_fwriter, log_fwriter, fixed_starting_times=None):
    bpm_env = SimBPMEnv(diffsim_info, stat_fwriter, log_fwriter)
    execute_full_process(bpm_env, fixed_starting_times)
    if fixed_starting_times is not None:
        return bpm_env
    if log_fwriter is None and stat_fwriter is None:
        return bpm_env.log_info.compute_process_kpi(bpm_env), bpm_env.log_info
    if log_fwriter:
        bpm_env.log_writer.force_write()
    if stat_fwriter:
        bpm_env.log_info.save_joint_statistics(bpm_env)

    return None


def verify_miliseconds(array):
    """
    In case of datetime.microsecond = 0, standard converter does not print microseconds
    So we force the conversion, so that the datetime format is the same for every datetime in the final file
    Indexes correspond to the next values:
        2 - enabled_datetime
        3 - start_datetime
        4 - end_datetime
    """
    for i in range(2, 5):
        if array[i].microsecond == 0:
            array[i] = _get_string_from_datetime(array[i])

    return array


def _get_string_from_datetime(datetime):
    datetime_without_colon = datetime.strftime("%Y-%m-%d %H:%M:%S.%f%z")
    return "{0}:{1}".format(datetime_without_colon[:-2], datetime_without_colon[-2:])
