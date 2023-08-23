import datetime
import ntpath
from datetime import timedelta
from typing import Optional

import pytz
from pix_framework.discovery.resource_calendar_and_performance.crisp.resource_calendar import RCalendar
from pix_framework.statistics.distribution import DurationDistribution

from prosimos.batch_processing import BatchConfigPerTask
from prosimos.control_flow_manager import BPMN, ElementInfo, ProcessState
from prosimos.exceptions import InvalidSimScenarioException
from prosimos.histogram_distribution import HistogramDistribution
from prosimos.simulation_properties_parser import parse_json_sim_parameters, parse_simulation_model


class SimDiffSetup:
    def __init__(self, bpmn_path, json_path, is_event_added_to_log, total_cases):
        self.process_name = ntpath.basename(bpmn_path).split(".")[0]
        self.start_datetime = datetime.datetime.now(pytz.utc)

        (
            self.resources_map,
            self.calendars_map,
            self.element_probability,
            self.task_resource,
            self.arrival_calendar,
            self.event_distibution,
            self.batch_processing,
            self.case_attributes,
            self.prioritisation_rules,
            self.model_type,
        ) = parse_json_sim_parameters(json_path)

        self.bpmn_graph = parse_simulation_model(bpmn_path)
        self.bpmn_graph.set_additional_fields_from_json(
            self.element_probability, self.task_resource, self.event_distibution, self.batch_processing
        )
        if not self.arrival_calendar:
            self.arrival_calendar = self.find_arrival_calendar()

        self.is_event_added_to_log = is_event_added_to_log
        self.total_num_cases = total_cases  # how many process cases should be simulated

    def verify_simulation_input(self):
        for e_id in self.bpmn_graph.element_info:
            e_info: ElementInfo = self.bpmn_graph[e_id]
            if e_info.type == BPMN.TASK and e_info.id not in self.task_resource:
                print("WARNING: No resource assigned to task %s" % e_info.name)
            if e_info.type in [BPMN.INCLUSIVE_GATEWAY, BPMN.EXCLUSIVE_GATEWAY] and e_info.is_split():
                if e_info.id not in self.element_probability:
                    print("WARNING: No probability assigned to gateway %s" % e_info.name)

    def name_from_id(self, resource_id):
        return self.resources_map[resource_id].resource_name

    def get_resource_calendar(self, resource_id):
        if resource_id in self.resources_map:
            return self.calendars_map[self.resources_map[resource_id].calendar_id]
        return None

    def next_resting_time(self, resource_id, starting_from):
        if resource_id in self.resources_map:
            return self.calendars_map[self.resources_map[resource_id].calendar_id].next_available_time(starting_from)
        return 0

    def next_arrival_time(self, starting_from):
        # duration: float = 0.0

        # decide how to calculate value based on whether it is function distribution or histogram one
        if isinstance(self.element_probability["arrivalTime"], DurationDistribution):
            [duration] = self.element_probability["arrivalTime"].generate_sample(1)
        elif isinstance(self.element_probability["arrivalTime"], HistogramDistribution):
            duration = self.element_probability["arrivalTime"].generate_value()
        else:
            raise InvalidSimScenarioException("Not supported arrival distribution")

        return duration + self.arrival_calendar.next_available_time(starting_from + timedelta(seconds=duration))

    def initial_state(self):
        return ProcessState(self.bpmn_graph)

    def is_enabled(self, e_id, p_state):
        return self.bpmn_graph.is_enabled(e_id, p_state)

    def update_process_state(self, p_case, e_id, p_state, completed_time_prev_event):
        return self.bpmn_graph.update_process_state(p_case, e_id, p_state, completed_time_prev_event)

    def is_any_batch_enabled(self, started_datetime):
        return self.bpmn_graph.is_any_batch_enabled(started_datetime)

    def get_invalid_batches_if_any(self, current_point_of_time):
        return self.bpmn_graph.get_invalid_batches_if_any(current_point_of_time)

    def is_any_unexecuted_batch(self, last_task_enabled_time):
        return self.bpmn_graph.is_any_unexecuted_batch(last_task_enabled_time)

    def find_arrival_calendar(self):
        # TODO: make sure this 0 as p_case does not break anything
        enabled_tasks = self.update_process_state(0, self.bpmn_graph.starting_event, self.initial_state(), None)
        starter_resources = set()
        arrival_calendar = RCalendar("arrival_calendar")
        for task_id in enabled_tasks:
            for r_id in self.task_resource[task_id]:
                if r_id in starter_resources:
                    continue
                arrival_calendar.combine_calendar(self.calendars_map[self.resources_map[r_id].calendar_id])
                starter_resources.add(r_id)
        return arrival_calendar

    def ideal_task_duration(self, task_id, resource_id, num_tasks_in_batch):
        # calculate duration based on defined distribution for the resource allocation
        [duration] = self.task_resource[task_id][resource_id].generate_sample(1)

        if num_tasks_in_batch == 0:
            # task executed NOT in batch
            return duration
        else:
            # task executed as a part of the batch
            curr_batch_info: Optional[BatchConfigPerTask] = self.batch_processing.get(task_id, None)
            if curr_batch_info is None:
                print(f"WARNING: Could not find info about batch_processing for task {task_id}")

            return curr_batch_info.calculate_ideal_duration(duration, num_tasks_in_batch)

    def real_task_duration(self, task_duration, resource_id, enabled_at, worked_intervals=None):
        if self.model_type == "FUZZY":
            return self.calendars_map[self.resources_map[resource_id].calendar_id].find_idle_time(
                enabled_at, task_duration, worked_intervals
            )
        else:
            return self.calendars_map[self.resources_map[resource_id].calendar_id].find_idle_time(
                enabled_at, task_duration
            )

    def set_starting_datetime(self, new_datetime):
        (
            is_inside_arrival_calendar,
            _,
        ) = self.arrival_calendar.is_working_datetime(new_datetime)
        if is_inside_arrival_calendar:
            self.start_datetime = new_datetime
        else:
            # the start datetime provided in the simulation scenario
            # is outside of the arrival calendar range
            # so we need to get the next earliest datetime
            # next_in_sec = self.arrival_calendar.next_available_time(new_datetime)
            next_in_sec = self.next_arrival_time(new_datetime)
            self.start_datetime = new_datetime + timedelta(seconds=next_in_sec)
