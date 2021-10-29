import pytz
import datetime
from datetime import timedelta
import ntpath

from bpdfr_simulation_engine.control_flow_manager import ProcessState
from bpdfr_simulation_engine.probability_distributions import generate_number_from, random_uniform
from bpdfr_simulation_engine.resource_calendar import RCalendar
from bpdfr_simulation_engine.simulation_properties_parser import parse_simulation_model, parse_json_sim_parameters


class SimulationStep:
    def __init__(self, trace_info, task_id, ideal_duration, p_state, enabled_at, enabled_by=None):
        self.trace_info = trace_info
        self.task_id = task_id
        self.ideal_duration = ideal_duration
        self.p_state = p_state
        self.enabled_at = enabled_at
        self.enabled_by = enabled_by
        self.started_at = None
        self.completed_at = None
        self.performed_by_resource = None


class SimDiffSetup:
    def __init__(self, bpmn_path, json_path, with_enabled_state=False, with_csv_state_column=False):
        self.with_enabled_state = with_enabled_state
        self.with_csv_state_column = with_csv_state_column
        self.process_name = ntpath.basename(bpmn_path).split(".")[0]
        self.start_datetime = datetime.datetime.now(pytz.utc)

        self.resources_map, self.calendars_map, self.element_probability, self.task_resource, self.arrival_calendar \
            = parse_json_sim_parameters(json_path)

        self.bpmn_graph = parse_simulation_model(bpmn_path)
        self.bpmn_graph.set_element_probabilities(self.element_probability, self.task_resource)
        if not self.arrival_calendar:
            self.arrival_calendar = self.find_arrival_calendar()

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
        val = generate_number_from(self.element_probability['arrivalTime']['distribution_name'],
                                   self.element_probability['arrivalTime']['distribution_params'])
        return val + self.arrival_calendar.next_available_time(starting_from + timedelta(seconds=val))

    def initial_state(self):
        return ProcessState(self.bpmn_graph)

    def is_enabled(self, e_id, p_state):
        return self.bpmn_graph.is_enabled(e_id, p_state)

    def update_process_state(self, e_id, p_state):
        return self.bpmn_graph.update_process_state(e_id, p_state)

    def find_arrival_calendar(self):
        enabled_tasks = self.update_process_state(self.bpmn_graph.starting_event, self.initial_state())
        starter_resources = set()
        arrival_calendar = RCalendar("arrival_calendar")
        for task_id in enabled_tasks:
            for r_id in self.task_resource[task_id]:
                if r_id in starter_resources:
                    continue
                arrival_calendar.combine_calendar(self.calendars_map[self.resources_map[r_id].calendar_id])
                starter_resources.add(r_id)
        return arrival_calendar

    def ideal_task_duration(self, task_id, resource_id):
        val = generate_number_from(self.task_resource[task_id][resource_id]['distribution_name'],
                                   self.task_resource[task_id][resource_id]['distribution_params'])
        return val

    def real_task_duration(self, task_duration, resource_id, enabled_at):
        return self.calendars_map[self.resources_map[resource_id].calendar_id].find_idle_time(enabled_at, task_duration)

    def set_starting_satetime(self, new_datetime):
        self.start_datetime = new_datetime
