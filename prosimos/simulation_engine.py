import csv
import datetime
import os
import numpy as np
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
from prosimos.warning_logger import warning_logger


class SimResource:
    def __init__(self):
        self.switching_time = 0
        self.allocated_tasks = 0
        self.worked_time = 0
        self.available_time = 0
        self.last_released = 0


class SimBPMEnv:
    def __init__(self, sim_setup: SimDiffSetup, stat_fwriter, log_fwriter, process_state=None, simulation_horizon=None):
        self.sim_setup = sim_setup
        self.sim_resources = dict()
        self.stat_fwriter = stat_fwriter
        self.additional_columns = self.sim_setup.all_attributes.get_all_columns_generated()
        self.log_writer = FileManager(10000, log_fwriter, self.additional_columns)
        self.log_info = LogInfo(sim_setup)
        self.executed_events = 0
        self.time_update_process_state = 0
        self.all_process_states = dict()
        self.simulation_horizon = simulation_horizon

        self.case_prioritisation = CasePrioritisation(
            self.sim_setup.total_num_cases,
            self.sim_setup.case_attributes,
            self.sim_setup.prioritisation_rules,
        )

        all_attributes = {
            "global": self.sim_setup.all_attributes.global_attribute_initial_values,
            **self.case_prioritisation.all_case_attributes
        }

        self.sim_setup.bpmn_graph.all_attributes = all_attributes

        if process_state:
            self.initialize_from_process_state(process_state)
        else:
            r_first_available = dict()
            for r_id in sim_setup.resources_map:
                self.sim_resources[r_id] = SimResource()
                r_first_available[r_id] = self.sim_setup.next_resting_time(r_id, self.sim_setup.start_datetime)

            self.resource_queue = DiffResourceQueue(self.sim_setup.task_resource, r_first_available)
            self.events_queue = EventQueue()
            self.generate_all_arrival_events()

    def filter_event_log(self):
        filtered_traces = {}
        for case_id, trace in self.log_info.trace_list.items():
            # Get the start time of the case (first event's start time)
            if trace.event_list:
                case_start_time = trace.event_list[0].started_datetime
            else:
                continue  # Skip cases with no events

            if case_start_time <= self.sim_setup.simulation_horizon:
                # Include this case
                filtered_traces[case_id] = trace
            else:
                # Exclude this case
                pass  # Events of cases started after the horizon are discarded

        # Replace the trace_list with the filtered traces
        self.log_info.trace_list = filtered_traces

    def initialize_from_process_state(self, process_state):
        # Initialize resource mapping (resource name to ID) from simulation setup
        resource_name_to_id = self.sim_setup.resource_name_to_id

        # Initialize resources based on the process state
        r_first_available = {}
        for resource_name, end_time in process_state.get('resource_last_end_times', {}).items():
            if isinstance(end_time, str):
                end_time = parse_datetime(end_time)
            resource_id = resource_name_to_id.get(resource_name)
            if resource_id and resource_id in self.sim_setup.resources_map:
                # Initialize resource
                sim_resource = SimResource()
                sim_resource.available_time = (end_time - self.sim_setup.start_datetime).total_seconds()
                self.sim_resources[resource_id] = sim_resource
                r_first_available[resource_id] = sim_resource.available_time
            else:
                # Resource not in resource pool; treat as external resource
                pass  # Do not add to sim_resources or r_first_available

        # For resources not mentioned in process_state, set their available time to start datetime or next resting time
        for r_id in self.sim_setup.resources_map:
            if r_id not in r_first_available:
                self.sim_resources[r_id] = SimResource()
                r_first_available[r_id] = self.sim_setup.next_resting_time(r_id, self.sim_setup.start_datetime)

        # Initialize resource queue
        self.resource_queue = DiffResourceQueue(self.sim_setup.task_resource, r_first_available)

        # Initialize event queue
        self.events_queue = EventQueue()

        # Initialize process states and execute ongoing activities
        for case_id_str, case_data in process_state.get('cases', {}).items():
            case_id = int(case_id_str)
            # Create ProcessState
            p_state = self.sim_setup.initial_state()
            # Set tokens based on control_flow_state
            tokens = {}
            flows = case_data.get('control_flow_state', {}).get('flows', [])
            for flow_id in flows:
                tokens[flow_id] = 1  # Assuming 1 token per flow?
            p_state.set_tokens(tokens)
            print(f"Set tokens for case {case_id}: {tokens}")

            self.all_process_states[case_id] = p_state
            # Initialize log trace for the case
            self.log_info.trace_list[case_id] = Trace(case_id, self.sim_setup.start_datetime)

            # Execute ongoing activities
            for activity in case_data.get('ongoing_activities', []):
                task_name = activity['name']
                task_id = self.sim_setup.bpmn_graph.get_task_id_by_name(task_name)
                resource_name = activity.get('resource')
                start_time = activity['start_time']
                if isinstance(start_time, str):
                    start_time = parse_datetime(start_time)
                remaining_duration = activity.get('remaining_duration')
                if remaining_duration is None and 'remaining_time' in activity:
                    remaining_duration = float(activity['remaining_time'])

                # Map resource name to ID
                resource_id = resource_name_to_id.get(resource_name)
                if resource_id is None:
                    # Resource not found in simulation parameters
                    print(
                        f"Resource '{resource_name}' not found in simulation parameters. Treating as external resource.")
                    resource_in_pool = False
                    resource_id = resource_name
                else:
                    resource_in_pool = True
                    # Check if resource can perform the task
                    if task_id not in self.sim_setup.task_resource or resource_id not in self.sim_setup.task_resource[
                        task_id]:
                        print(
                            f"Resource '{resource_id}' cannot perform task '{task_id}'. Assigning a capable resource.")
                        possible_resources = self.sim_setup.task_resource.get(task_id, {})
                        if possible_resources:
                            resource_id = np.random.choice(list(possible_resources.keys()))
                            resource_in_pool = resource_id in self.sim_setup.resources_map
                        else:
                            print(f"No resources available for task '{task_id}'.")
                            continue

                if remaining_duration is None:
                    # Compute remaining_duration based on total duration and time elapsed
                    possible_resources = self.sim_setup.task_resource.get(task_id, {})
                    if possible_resources:
                        # Use the duration distribution from any available resource for the task
                        any_resource_id = next(iter(possible_resources))
                        total_duration = self.sim_setup.task_resource[task_id][any_resource_id].generate_sample(1)[0]
                        time_elapsed = (self.sim_setup.start_datetime - start_time).total_seconds()
                        remaining_duration = total_duration - time_elapsed
                        if remaining_duration < 0:
                            remaining_duration = 0
                        print(
                            f"WARNING: Using duration distribution from resource '{any_resource_id}' for task '{task_id}'.")
                    else:
                        print(
                            f"Cannot compute remaining_duration for task '{task_id}'. No duration distribution available.")
                        continue
                    time_elapsed = (self.sim_setup.start_datetime - start_time).total_seconds()
                    remaining_duration = total_duration - time_elapsed
                    if remaining_duration < 0:
                        remaining_duration = 0

                enabled_at = (start_time - self.sim_setup.start_datetime).total_seconds()
                if enabled_at < 0:
                    print(f"Adjusted enabled_at for case {case_id} from {enabled_at} to 0.")
                    enabled_at = 0  # Adjust if enabled_at is negative

                # Create EnabledEvent for the ongoing activity
                enabled_event = EnabledEvent(
                    case_id,
                    p_state,
                    task_id,
                    enabled_at,
                    start_time,
                    duration_sec=remaining_duration,
                    assigned_resource_id=resource_id
                )

                # Add the enabled event to the events queue
                self.calc_priority_and_append_to_queue(enabled_event, is_arrival_event=False)


                print(
                    f"Scheduled ongoing activity '{task_name}' (ID: '{task_id}') for case {case_id} at simulation time {enabled_at} with remaining duration {remaining_duration}."
                )

                # Set tokens for the ongoing activity
                task = self.sim_setup.bpmn_graph.element_info[task_id]

                # Retrieve the incoming flow IDs
                incoming_flows = task.incoming_flows
                tokens = {flow_id: 1 for flow_id in incoming_flows}
                p_state.set_tokens(tokens)
                print(f"Set tokens for case {case_id}: {tokens}")

            # Schedule enabled activities (those that are ready to start)
            for activity in case_data.get('enabled_activities', []):
                task_name = activity['name']
                task_id = self.sim_setup.bpmn_graph.get_task_id_by_name(task_name)
                enabled_time = activity['enabled_time']
                if isinstance(enabled_time, str):
                    enabled_time = parse_datetime(enabled_time)
                enabled_at = (enabled_time - self.sim_setup.start_datetime).total_seconds()

                # Create and schedule EnabledEvent
                enabled_event = EnabledEvent(
                    case_id,
                    p_state,
                    task_id,
                    enabled_at,
                    enabled_time
                )
                self.calc_priority_and_append_to_queue(enabled_event, is_arrival_event=False)
                # print(f"Scheduling enabled activity '{task_name}' for case {case_id} enabled at {enabled_time}.")

            # if case_id == '15':
            print(f"Initialized case {case_id} with process state and scheduled ongoing and enabled activities.")

        print("Events scheduled in the event queue after initialization:", self.events_queue.enabled_events)
        # Generate remaining arrival events
        self.generate_remaining_arrival_events(process_state)


    def generate_remaining_arrival_events(self, process_state):
        existing_case_ids = set(int(cid) for cid in process_state['cases'].keys())
        total_existing_cases = len(existing_case_ids)
        total_cases_needed = self.sim_setup.total_num_cases

        if total_existing_cases >= total_cases_needed:
            return

        last_arrival_time = process_state['last_case_arrival']
        current_time = last_arrival_time
        case_id = max(existing_case_ids) + 1

        while total_existing_cases < total_cases_needed:
            # Generate inter-arrival time
            inter_arrival_seconds = self.sim_setup.next_arrival_time(current_time)
            next_arrival_time = current_time + timedelta(seconds=inter_arrival_seconds)

            # Schedule arrival event
            enabled_at = (next_arrival_time - self.sim_setup.start_datetime).total_seconds()
            arrival_event = EnabledEvent(
                p_case=case_id,
                p_state=self.sim_setup.initial_state(),
                task_id=self.sim_setup.bpmn_graph.starting_event,
                enabled_at=enabled_at,
                enabled_datetime=next_arrival_time
            )
            self.calc_priority_and_append_to_queue(arrival_event, is_arrival_event=True)

            # Update for next iteration
            current_time = next_arrival_time
            case_id += 1
            total_existing_cases += 1

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

    def execute_enabled_event(self, c_event: EnabledEvent, resource_in_pool=True):
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
                    p_state,
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
                        ),
                        False,
                    )
        else:
            if event_element_info.type == BPMN.TASK:
                # Execute non-batched task
                completed_at, completed_datetime = self.execute_task(c_event, resource_in_pool)
            else:
                completed_at, completed_datetime = self.execute_event(c_event)

            # Update the process state and retrieve enabled tasks
            enabled_time = CustomDatetimeAndSeconds(completed_at, completed_datetime)
            enabled_tasks, visited_at = self.sim_setup.update_process_state(
                c_event.p_case, c_event.task_id, c_event.p_state, enabled_time
            )

            if not enabled_tasks:
                pass
            else:
                for next_task in enabled_tasks:
                    self.calc_priority_and_append_to_queue(
                        EnabledEvent(
                            c_event.p_case,
                            c_event.p_state,
                            next_task.task_id,
                            visited_at[next_task.task_id].seconds_from_start,
                            visited_at[next_task.task_id].datetime,
                        ),
                        False,
                    )
        print(f"Process state after executing event for case {c_event.p_case}: {c_event.p_state.tokens}")

    def pop_and_allocate_resource(self, task_id: str, num_allocated_tasks: int):
        r_id, r_avail_at = self.resource_queue.pop_resource_for(task_id)
        self.sim_resources[r_id].allocated_tasks += num_allocated_tasks
        return r_id, r_avail_at

    def execute_task(self, c_event: EnabledEvent, resource_in_pool=True):
        task_id = c_event.task_id
        p_case = c_event.p_case

        # Use assigned resource if provided
        if c_event.assigned_resource_id:
            resource_id = c_event.assigned_resource_id
            # Assume resource is available at the task's enabled time
            resource_available_at = c_event.enabled_at
        else:
            # Allocate a resource using pop_and_allocate_resource
            resource_id, resource_available_at = self.pop_and_allocate_resource(task_id, num_allocated_tasks=1)
            self.sim_resources[resource_id].allocated_tasks += 1

        # Determine the duration
        if c_event.duration_sec is not None:
            duration = c_event.duration_sec
        else:
            # Calculate duration using the task's duration distribution
            duration = self.sim_setup.ideal_task_duration(task_id, resource_id, num_tasks_in_batch=0)

        # The task cannot start before both the event's enabled_at and the resource's availability
        started_at = max(c_event.enabled_at, resource_available_at)
        started_datetime = self.simulation_datetime_from(started_at)

        if resource_in_pool:
            # Resource is in the pool, calculate real duration considering the resource's calendar
            real_duration = self.sim_setup.real_task_duration(duration, resource_id, started_datetime)
        else:
            # Resource not in pool, use the duration as is
            real_duration = duration

        completed_at = started_at + real_duration
        completed_datetime = self.simulation_datetime_from(completed_at)

        # Create TaskEvent
        full_evt = TaskEvent(
            p_case=p_case,
            task_id=task_id,
            resource_id=resource_id,
            started_at=started_at,
            started_datetime=started_datetime,
            enabled_at=c_event.enabled_at,
            enabled_datetime=c_event.enabled_datetime,
            real_duration=real_duration,
            ideal_duration=duration,
            bpm_env=self
        )

        # Update logs and resource availability
        self._update_logs_and_resource_availability(full_evt, resource_id, resource_in_pool)

        return completed_at, completed_datetime

    def update_attributes(self, current_event):
        event_attributes = self.sim_setup.all_attributes.event_attributes.attributes
        global_event_attributes = self.sim_setup.all_attributes.global_event_attributes.attributes

        all_attribute_values = {
            **self.sim_setup.bpmn_graph.all_attributes["global"],
            **self.sim_setup.bpmn_graph.all_attributes[current_event.p_case]
        }

        new_global_attr_values = self._extract_attributes_for_event(current_event.task_id, global_event_attributes, all_attribute_values)
        new_event_attr_values = self._extract_attributes_for_event(current_event.task_id, event_attributes, all_attribute_values)

        self.sim_setup.bpmn_graph.all_attributes["global"].update(new_global_attr_values)
        self.sim_setup.bpmn_graph.all_attributes[current_event.p_case].update(new_event_attr_values)

    def _extract_attributes_for_event(self, task_id, source_attributes, all_attribute_values):
        new_attributes = {}

        if task_id in source_attributes:
            task_attributes = source_attributes[task_id]
            for key, value in task_attributes.items():
                new_attributes[key] = value.get_next_value(all_attribute_values)

        return new_attributes

    def get_csv_row_data(self, full_event: TaskEvent):
        """
        Return array of values for one line of the csv file based on full_event information.
        Includes filtering based on the simulation horizon.
        """

        # Check if the case started after the simulation horizon
        case_id = full_event.p_case

        # Get the start time of the case
        trace = self.log_info.trace_list.get(case_id)
        if trace and trace.event_list:
            case_start_time = trace.event_list[0].started_datetime
        else:
            # If no events yet, use the start time of the current event
            case_start_time = full_event.started_datetime

        if case_start_time > self.sim_setup.simulation_horizon:
            # Skip this event, as the case started after the horizon
            return None

        # Proceed to generate the row data as before

        # Check if the resource_id is in resources_map
        if hasattr(full_event, "resource_id"):
            if full_event.resource_id in self.sim_setup.resources_map:
                resource_name = self.sim_setup.resources_map[full_event.resource_id].resource_name
            else:
                # Resource is external;
                resource_name = full_event.resource_id
        else:
            resource_name = "No assigned resource"

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

        all_attrs = self.sim_setup.bpmn_graph.get_all_attributes(full_event.p_case)
        values = ["" if all_attrs.get(col) is None else all_attrs.get(col) for col in self.additional_columns]

        return [*row_basic_info, *values]

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

    def _update_logs_and_resource_availability(self, full_evt: TaskEvent, r_id, resource_in_pool=True):
        # Log the task execution
        if resource_in_pool and r_id in self.sim_setup.resources_map:
            resource_cost = self.sim_setup.resources_map[r_id].cost_per_hour
        else:
            resource_cost = 0

        self.log_info.add_event_info(full_evt.p_case, full_evt, resource_cost)

        if resource_in_pool and r_id in self.sim_setup.resources_map:
            # Update resource availability
            r_next_available = full_evt.completed_at

            if self.sim_resources[r_id].switching_time > 0:
                r_next_available += self.sim_setup.next_resting_time(
                    r_id, self.simulation_datetime_from(r_next_available)
                )

            self.resource_queue.update_resource_availability(r_id, r_next_available)

            # Update resource's worked time
            self.sim_resources[r_id].worked_time += full_evt.real_duration

        # Get the CSV row data
        row_data = self.get_csv_row_data(full_evt)
        if row_data:
            # Write event to log file
            self.log_writer.add_csv_row(row_data)
        else:
            # Event is not included due to filtering
            pass

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
            # Get the CSV row data
            row_data = self.get_csv_row_data(full_evt)
            if row_data:
                # Write event to log file
                self.log_writer.add_csv_row(row_data)
            else:
                # Event is not included due to filtering
                pass

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
    # Only generate arrival events if the events_queue is empty
    if bpm_env.events_queue.is_empty():
        if fixed_starting_times is None:
            bpm_env.generate_all_arrival_events()
        else:
            bpm_env.generate_fixed_arrival_events(fixed_starting_times)

    # print("Generation of all cases: %s" %
    #       str(datetime.timedelta(seconds=(datetime.datetime.now() - s_t).total_seconds())))
    current_event = bpm_env.events_queue.pop_next_event()
    executed_cases = set()

    while current_event is not None:
        if current_event.p_case not in executed_cases:
            executed_cases.add(current_event.p_case)
            global_case_attributes = bpm_env.sim_setup.all_attributes.global_case_attributes.attributes
            new_attributes = {attr.name: attr.get_next_value() for attr in global_case_attributes}
            bpm_env.sim_setup.bpmn_graph.all_attributes["global"].update(new_attributes)

        print(f"Processing event at simulation time: {current_event}")
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
    process_state=None,
    simulation_horizon=None
):
    diffsim_info = SimDiffSetup(bpmn_path, json_path, is_event_added_to_log, total_cases, process_state=process_state, simulation_horizon=simulation_horizon)

    if not diffsim_info:
        return None

    if process_state:
        if starting_at:
            print("WARNING: 'starting_at' parameter is ignored when 'process_state' is provided.")
    else:
        # When no process_state, set start_datetime based on starting_at or current time
        starting_at_datetime = (
            parse_datetime(starting_at, True) if starting_at else pytz.utc.localize(datetime.datetime.now())
        )
        diffsim_info.set_starting_datetime(starting_at_datetime)

    if stat_out_path is None and log_out_path is None:
        return run_simpy_simulation(diffsim_info, None, None, process_state=process_state, simulation_horizon=simulation_horizon)

    csv_writer_config = {
        'delimiter': ',',
        'quotechar': '"',
        'quoting': csv.QUOTE_MINIMAL
    }

    stat_csv_file = open(stat_out_path, mode="w", newline="", encoding="utf-8") if stat_out_path else None
    log_csv_file = open(log_out_path, mode="w", newline="", encoding="utf-8") if log_out_path else None

    try:
        stat_writer = csv.writer(stat_csv_file, **csv_writer_config) if stat_csv_file else None
        log_writer = csv.writer(log_csv_file, **csv_writer_config) if log_csv_file else None

        result = run_simpy_simulation(diffsim_info, stat_writer, log_writer, fixed_starting_times=fixed_arrival_times, process_state=process_state, simulation_horizon=simulation_horizon)
        print("run_simulation: result =", result)
    finally:
        if stat_csv_file:
            stat_csv_file.close()
        if log_csv_file:
            log_csv_file.close()

    warning_file_name = "simulation_warnings.txt"
    if stat_out_path:
        warning_file_path = os.path.join(os.path.dirname(stat_out_path), warning_file_name)
    elif log_out_path:
        warning_file_path = os.path.join(os.path.dirname(log_out_path), warning_file_name)
    else:
        warning_file_path = warning_file_name

    with open(warning_file_path, "w") as warning_file:
        for warning in warning_logger.get_all_warnings():
            warning_file.write(f"{warning}\n")

    return result


def run_simpy_simulation(diffsim_info, stat_fwriter, log_fwriter, fixed_starting_times=None, process_state=None, simulation_horizon=None):
    bpm_env = SimBPMEnv(diffsim_info, stat_fwriter, log_fwriter, process_state=process_state, simulation_horizon=simulation_horizon)
    execute_full_process(bpm_env, fixed_starting_times)

    bpm_env.filter_event_log()
    if fixed_starting_times is not None:
        return bpm_env
    if log_fwriter is None and stat_fwriter is None:
        return bpm_env.log_info.compute_process_kpi(bpm_env), bpm_env.log_info
    if log_fwriter:
        bpm_env.log_writer.force_write()
    if stat_fwriter:
        bpm_env.log_info.save_joint_statistics(bpm_env)

    warning_logger.add_warnings(bpm_env.sim_setup.bpmn_graph.simulation_execution_stats.find_issues())

    print("run_simpy_simulation: bpm_env =", bpm_env)
    return bpm_env


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
