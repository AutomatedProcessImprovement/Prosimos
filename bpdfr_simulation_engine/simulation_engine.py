import csv
import os
from pathlib import Path

import pytz
import datetime
from datetime import timedelta
from bpdfr_simulation_engine.control_flow_manager import BPMN, EVENT_TYPE, CustomDatetimeAndSeconds, EnabledTask

from bpdfr_simulation_engine.file_manager import FileManager
from bpdfr_simulation_engine.execution_info import Trace, TaskEvent, EnabledEvent
from bpdfr_simulation_engine.simulation_queues_ds import PriorityQueue, DiffResourceQueue, EventQueue
from bpdfr_simulation_engine.simulation_setup import SimDiffSetup
from bpdfr_simulation_engine.simulation_stats_calculator import LogInfo


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
        self.log_writer = FileManager(10000, log_fwriter)
        self.log_info = LogInfo(sim_setup)
        self.executed_events = 0
        self.time_update_process_state = 0

        r_first_available = dict()
        for r_id in sim_setup.resources_map:
            self.sim_resources[r_id] = SimResource()
            r_first_available[r_id] = self.sim_setup.next_resting_time(r_id, self.sim_setup.start_datetime)

        self.resource_queue = DiffResourceQueue(self.sim_setup.task_resource, r_first_available)
        self.events_queue = EventQueue()
        self.all_process_states = dict()

    def generate_all_arrival_events(self, total_cases):
        sim_setup = self.sim_setup
        arrival_time = 0
        # prev = 0
        for p_case in range(0, total_cases):
            p_state = sim_setup.initial_state()
            enabled_datetime = self.simulation_datetime_from(arrival_time)
            enabled_time = CustomDatetimeAndSeconds(arrival_time, enabled_datetime)
            enabled_tasks: list[EnabledTask] = sim_setup.update_process_state(p_case, sim_setup.bpmn_graph.starting_event, p_state, enabled_time)
            self.all_process_states[p_case] = p_state
            self.log_info.trace_list.append(Trace(p_case, enabled_datetime))
            for task in enabled_tasks:
                task_id = task.task_id
                self.events_queue.append_arrival_event(EnabledEvent(p_case, p_state, task_id, arrival_time,
                                                                    enabled_datetime))
            arrival_time += sim_setup.next_arrival_time(enabled_datetime)

    def execute_enabled_event(self, c_event: EnabledEvent):
        self.executed_events += 1

        event_element_info = self.sim_setup.bpmn_graph.element_info[c_event.task_id]
        # self.is_any_batch_enabled(c_event.enabled_at)

        if event_element_info.type == BPMN.TASK and c_event.batch_info_exec is not None:
            # execute batched task
            executed_tasks = self.execute_task_batch(c_event)

            for task in executed_tasks:
                completed_at, completed_datetime, p_case = task
                p_state = self.all_process_states[p_case]
                enabled_time = CustomDatetimeAndSeconds(completed_at, completed_datetime)
                enabled_tasks: list[EnabledTask] = self.sim_setup.update_process_state(
                    p_case, c_event.task_id, self.all_process_states[p_case], 
                    enabled_time)

                for next_task in enabled_tasks:
                    self.events_queue.append_enabled_event(
                        EnabledEvent(p_case, p_state, next_task.task_id, completed_at,
                                    completed_datetime, next_task.batch_info_exec))
        else:
            if event_element_info.type == BPMN.TASK:
                # execute not batched task
                completed_at, completed_datetime = \
                    self.execute_task(c_event)
            else:
                completed_at, completed_datetime = \
                    self.execute_event(c_event)

            # Updating the process state. Retrieving/enqueuing enabled tasks, it also schedules the corresponding event
            # s_t = datetime.datetime.now()
            enabled_time = CustomDatetimeAndSeconds(completed_at, completed_datetime)
            enabled_tasks: list[EnabledTask] = self.sim_setup.update_process_state(c_event.p_case, c_event.task_id, c_event.p_state, enabled_time)
            # self.time_update_process_state += (datetime.datetime.now() - s_t).total_seconds()

            for next_task in enabled_tasks:
                self.events_queue.append_enabled_event(
                    EnabledEvent(c_event.p_case, c_event.p_state, next_task.task_id, completed_at,
                                completed_datetime, next_task.batch_info_exec))

    def execute_task(self, c_event: EnabledEvent):
        r_id, r_avail_at = self.resource_queue.pop_resource_for(c_event.task_id)
        self.sim_resources[r_id].allocated_tasks += 1

        r_avail_at = max(c_event.enabled_at, r_avail_at)
        avail_datetime = self._datetime_from(r_avail_at)
        is_working, _ = self.sim_setup.get_resource_calendar(r_id).is_working_datetime(avail_datetime)
        if not is_working:
            r_avail_at = r_avail_at + self.sim_setup.next_resting_time(r_id, avail_datetime)

        full_evt = TaskEvent(c_event.p_case, c_event.task_id, r_id, r_avail_at, c_event.enabled_at,
                            c_event.enabled_datetime, self)

        self.log_info.add_event_info(c_event.p_case, full_evt, self.sim_setup.resources_map[r_id].cost_per_hour)

        r_next_available = full_evt.completed_at

        if self.sim_resources[r_id].switching_time > 0:
            r_next_available += self.sim_setup.next_resting_time(r_id, full_evt.completed_datetime)

        self.resource_queue.update_resource_availability(r_id, r_next_available)
        self.sim_resources[r_id].worked_time += full_evt.ideal_duration
        
        self.log_writer.add_csv_row([c_event.p_case,
                                    self.sim_setup.bpmn_graph.element_info[c_event.task_id].name,
                                    full_evt.enabled_datetime,
                                    full_evt.started_datetime,
                                    full_evt.completed_datetime,
                                    self.sim_setup.resources_map[full_evt.resource_id].resource_name])

        completed_at = full_evt.completed_at
        completed_datetime = full_evt.completed_datetime

        return completed_at, completed_datetime

    
    def is_any_batch_enabled(self, current_event: EnabledEvent):
        # TODO: implement and call before every execution of the task
        enabled_datetime = CustomDatetimeAndSeconds(current_event.enabled_at, current_event.enabled_datetime)
        enabled_batch_task_ids = self.sim_setup.is_any_batch_enabled(enabled_datetime)
        if enabled_batch_task_ids != None:
            for (batch_task_id, batch_info) in enabled_batch_task_ids.items():
                c_event = EnabledEvent(
                    current_event.p_case,
                    current_event.p_state,
                    batch_task_id,
                    current_event.enabled_at,
                    current_event.enabled_datetime,
                    batch_info
                )

                self.execute_enabled_event(c_event)


    def is_any_unexecuted_batch(self):
        return self.sim_setup.is_any_unexecuted_batch()


    def _get_chunk(self, batch_spec, curr_index, all_case_ids):
        """ Return only the part of the all_case_ids that will be executed as a batch """
        acc_tasks_in_batch = 0
        for i in range(0, curr_index):
            acc_tasks_in_batch = acc_tasks_in_batch + batch_spec[i]
        num_tasks_in_batch = batch_spec[curr_index]
        return all_case_ids[ acc_tasks_in_batch:acc_tasks_in_batch+num_tasks_in_batch ]
    
    def execute_task_batch(self, c_event: EnabledEvent):
        all_tasks_waiting = len(c_event.batch_info_exec.case_ids)

        if all_tasks_waiting == 0:
            print("WARNING: Number of tasks in the enabled batch is 0.")

        all_case_ids = list(c_event.batch_info_exec.case_ids.items())

        batch_spec = c_event.batch_info_exec.batch_spec
        chunks = [self._get_chunk(batch_spec, i, all_case_ids) for i in range(0, len(batch_spec))]

        if c_event.batch_info_exec.is_sequential():
           return self.execute_seq_task_batch(c_event, chunks)
        elif c_event.batch_info_exec.is_parallel():
            return self.execute_parallel_task_batch(c_event, chunks)
        else:
            print(f"WARNING: {c_event.batch_info_exec.task_batch_info.type} not supported")

    def execute_seq_task_batch(self, c_event: EnabledEvent, chunks):
        start_time_from_rule_seconds = (c_event.batch_info_exec.start_time_from_rule - self.sim_setup.start_datetime).total_seconds()

        for batch_item in chunks:
            num_tasks_in_batch = len(batch_item)
            
            r_id, r_avail_at = self.resource_queue.pop_resource_for(c_event.task_id)
            self.sim_resources[r_id].allocated_tasks += num_tasks_in_batch
            
            completed_at = 0

            for (case_id, enabled_time) in batch_item:
                p_case = case_id
                task_id = c_event.task_id
                enabled_at = enabled_time.seconds_from_start
                enabled_datetime = enabled_time.datetime
                enabled_batch = c_event.enabled_at

                r_avail_at = max(enabled_at, r_avail_at, enabled_batch, completed_at, start_time_from_rule_seconds)
                avail_datetime = self._datetime_from(r_avail_at)
                is_working, _ = self.sim_setup.get_resource_calendar(r_id).is_working_datetime(avail_datetime)
                if not is_working:
                    r_avail_at = r_avail_at + self.sim_setup.next_resting_time(r_id, avail_datetime)

                full_evt = TaskEvent(p_case, task_id, r_id, r_avail_at, enabled_at,
                                    enabled_datetime, self, num_tasks_in_batch)

                self.log_info.add_event_info(p_case, full_evt, self.sim_setup.resources_map[r_id].cost_per_hour)

                r_next_available = full_evt.completed_at

                if self.sim_resources[r_id].switching_time > 0:
                    r_next_available += self.sim_setup.next_resting_time(r_id, full_evt.completed_datetime)

                self.resource_queue.update_resource_availability(r_id, r_next_available)
                self.sim_resources[r_id].worked_time += full_evt.ideal_duration
                
                self.log_writer.add_csv_row([p_case,
                                            self.sim_setup.bpmn_graph.element_info[task_id].name,
                                            full_evt.enabled_datetime,
                                            full_evt.started_datetime,
                                            full_evt.completed_datetime,
                                            self.sim_setup.resources_map[full_evt.resource_id].resource_name])

                completed_at = full_evt.completed_at
                completed_datetime = full_evt.completed_datetime

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
            start_time_from_rule_seconds = \
                (c_event.batch_info_exec.start_time_from_rule - self.sim_setup.start_datetime).total_seconds()
            enabled_batch = 0

        for batch_item in chunks:
            num_tasks_in_batch = len(batch_item)

            r_id, r_avail_at = self.resource_queue.pop_resource_for(task_id)
            self.sim_resources[r_id].allocated_tasks += num_tasks_in_batch

            r_avail_at = max(r_avail_at, enabled_batch, start_time_from_rule_seconds)
            avail_datetime = self._datetime_from(r_avail_at)
            is_working, _ = self.sim_setup.get_resource_calendar(r_id).is_working_datetime(avail_datetime)
            if not is_working:
                r_avail_at = r_avail_at + self.sim_setup.next_resting_time(r_id, avail_datetime)
            
            for (case_id, enabled_time) in batch_item:
                p_case = case_id
                enabled_at = enabled_time.seconds_from_start
                enabled_datetime = enabled_time.datetime

                full_evt = TaskEvent(p_case, task_id, r_id, r_avail_at, enabled_at,
                                    enabled_datetime, self, num_tasks_in_batch)

                self.log_info.add_event_info(p_case, full_evt, self.sim_setup.resources_map[r_id].cost_per_hour)

                r_next_available = full_evt.completed_at

                if self.sim_resources[r_id].switching_time > 0:
                    r_next_available += self.sim_setup.next_resting_time(r_id, full_evt.completed_datetime)

                self.resource_queue.update_resource_availability(r_id, r_next_available)
                self.sim_resources[r_id].worked_time += full_evt.ideal_duration
                
                self.log_writer.add_csv_row([p_case,
                                            self.sim_setup.bpmn_graph.element_info[task_id].name,
                                            full_evt.enabled_datetime,
                                            full_evt.started_datetime,
                                            full_evt.completed_datetime,
                                            self.sim_setup.resources_map[full_evt.resource_id].resource_name])

                completed_at = full_evt.completed_at
                completed_datetime = full_evt.completed_datetime

                yield completed_at, completed_datetime, p_case


    def execute_event(self, c_event):
        # Handle event types separately (they don't need assigned resource)
        event_duration_seconds = None
        event_element = self.sim_setup.bpmn_graph.element_info[c_event.task_id]
        if (event_element.event_type == EVENT_TYPE.TIMER):
            # parse timer name
            event_duration_seconds = self.sim_setup.bpmn_graph.parse_timer_duration(event_element, c_event.enabled_datetime)
        else:
            # all other type should have defined probabilities
            event_duration_seconds = self.sim_setup.bpmn_graph.event_duration(event_element.id)

        completed_at = c_event.enabled_at + event_duration_seconds
        completed_datetime = c_event.enabled_datetime + timedelta(seconds=event_duration_seconds)

        full_evt = TaskEvent.create_event_entity(c_event, completed_at, completed_datetime)

        self.log_info.add_event_info(c_event.p_case, full_evt, 0)

        if (self.sim_setup.is_event_added_to_log):
            self.log_writer.add_csv_row([c_event.p_case,
                            self.sim_setup.bpmn_graph.element_info[c_event.task_id].name,
                            full_evt.enabled_datetime,
                            full_evt.started_datetime,
                            full_evt.completed_datetime,
                            "No assigned resource"])

        return completed_at, completed_datetime

    def _datetime_from(self, in_seconds):
        return self.simulation_datetime_from(in_seconds) if in_seconds is not None else None

    def simulation_datetime_from(self, simpy_time):
        return self.sim_setup.start_datetime + timedelta(seconds=simpy_time)

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


def execute_full_process(bpm_env: SimBPMEnv, total_cases):
    # Initialize event queue with the arrival times of all the cases to simulate,
    # i.e., all the initial events are enqueued and sorted by their arrival times
    # s_t = datetime.datetime.now()
    bpm_env.generate_all_arrival_events(total_cases)
    # print("Generation of all cases: %s" %
    #       str(datetime.timedelta(seconds=(datetime.datetime.now() - s_t).total_seconds())))
    current_event = bpm_env.events_queue.pop_next_event()
    while current_event is not None:
        bpm_env.is_any_batch_enabled(current_event)
        bpm_env.execute_enabled_event(current_event)
        current_event = bpm_env.events_queue.pop_next_event()

    if bpm_env.is_any_unexecuted_batch():
        print("not executed")
    # TODO: validate that we don't have any tasks left in the batch


def run_simulation(bpmn_path, json_path, total_cases, stat_out_path=None, log_out_path=None, starting_at=None, is_event_added_to_log=False):
    diffsim_info = SimDiffSetup(bpmn_path, json_path, is_event_added_to_log)

    if not diffsim_info:
        return None

    diffsim_info.set_starting_datetime(starting_at if starting_at else pytz.utc.localize(datetime.datetime.now()))

    if not stat_out_path and not log_out_path:
        stat_out_path = os.path.join(os.path.dirname(__file__), Path("%s.csv" % diffsim_info.process_name))
    if stat_out_path:
        with open(stat_out_path, mode='w', newline='', encoding='utf-8') as stat_csv_file:
            if log_out_path:
                with open(log_out_path, mode='w', newline='', encoding='utf-8') as log_csv_file:
                    run_simpy_simulation(diffsim_info, total_cases,
                                         csv.writer(stat_csv_file, delimiter=',', quotechar='"',
                                                    quoting=csv.QUOTE_MINIMAL),
                                         csv.writer(log_csv_file, delimiter=',', quotechar='"',
                                                    quoting=csv.QUOTE_MINIMAL))
            else:
                run_simpy_simulation(diffsim_info, total_cases,
                                     csv.writer(stat_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL),
                                     None)
    else:
        with open(log_out_path, mode='w', newline='', encoding='utf-8') as log_csv_file:
            run_simpy_simulation(diffsim_info, total_cases,
                                 None,
                                 csv.writer(log_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL))


def run_simpy_simulation(diffsim_info, total_cases, stat_fwriter, log_fwriter):
    bpm_env = SimBPMEnv(diffsim_info, stat_fwriter, log_fwriter)
    add_simulation_event_log_header(log_fwriter)
    execute_full_process(bpm_env, total_cases)
    if log_fwriter:
        bpm_env.log_writer.force_write()
    if stat_fwriter:
        bpm_env.log_info.save_joint_statistics(bpm_env)


def add_simulation_event_log_header(log_fwriter):
    if log_fwriter:
        log_fwriter.writerow([
            'case_id', 'activity', 'enable_time', 'start_time', 'end_time', 'resource', ])
