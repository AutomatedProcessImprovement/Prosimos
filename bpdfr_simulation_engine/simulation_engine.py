import csv
import os
from pathlib import Path

import pytz
import simpy
import datetime
from datetime import timedelta
from collections import deque

from bpdfr_simulation_engine.execution_info import Trace, TaskEvent
from bpdfr_simulation_engine.priority_queue import PriorityQueue
from bpdfr_simulation_engine.simulation_setup import SimulationStep, SimDiffSetup
from bpdfr_simulation_engine.simulation_stats_calculator import LogInfo, print_event_state


class SimResource:
    def __init__(self, simpy_resource):
        self.simpy_resource = simpy_resource
        self.is_available = True
        self.worked_time = 0
        self.available_time = 0
        self.completed_events = list()
        self.events_queue = deque()
        self.last_released = 0


class SimBPMEnv:
    def __init__(self, env, sim_setup, stat_fwriter, log_fwriter):
        self.simpy_env = env
        self.sim_setup = sim_setup
        self.sim_resources = dict()
        self.completed_events = list()
        self.real_duration = dict()
        self.stat_fwriter = stat_fwriter
        self.log_fwriter = log_fwriter
        self.log_info = LogInfo(sim_setup)

        self.resource_total_allocated_tasks = dict()
        for r_id in sim_setup.resources_map:
            self.resource_total_allocated_tasks[r_id] = 0
            self.sim_resources[r_id] = SimResource(simpy.Resource(env, sim_setup.resources_map[r_id].resource_amount))
            self.real_duration[r_id] = 0
        self._is_event_completed = list()
        self._pending_events = list()

    def enqueue_event(self, trace_info, task_id, p_state, enabled_by=None, available_resources=PriorityQueue()):
        s_step = SimulationStep(trace_info, task_id, None, p_state, self.simpy_env.now, enabled_by)
        self._pending_events.append(s_step)
        self._is_event_completed.append(False)
        if self.log_fwriter and self.sim_setup.with_enabled_state:
            self.log_fwriter.writerow([trace_info.p_case,
                                       self.sim_setup.bpmn_graph.element_info[task_id].name,
                                       None, 'enabled', self.simulation_datetime_from(s_step.enabled_at)])
        for r_id in self.sim_setup.task_resource[task_id]:
            self.sim_resources[r_id].events_queue.append(len(self._pending_events) - 1)
            if self.sim_resources[r_id].is_available:
                available_resources.add_task(r_id, self.sim_resources[r_id].last_released)
        return s_step

    def pop_event_for(self, resource_id):
        if resource_id not in self.sim_resources or not self.sim_resources[resource_id].is_available:
            return None
        while len(self.sim_resources[resource_id].events_queue) > 0:
            event_index = self.sim_resources[resource_id].events_queue.popleft()
            if not self._is_event_completed[event_index]:
                self._is_event_completed[event_index] = True
                return self._pending_events[event_index]
        return None

    def start_task(self, e_step, resource_id, sim_setup):
        e_step.started_at = self.simpy_env.now
        e_step.performed_by_resource = sim_setup.name_from_id(resource_id)
        started_at = self.simulation_datetime_from(e_step.started_at)
        if self.log_fwriter and self.sim_setup.with_enabled_state:
            self.log_fwriter.writerow([e_step.trace_info.p_case,
                                       self.sim_setup.bpmn_graph.element_info[e_step.task_id].name,
                                       resource_id, 'started', started_at])
        e_step.ideal_duration = sim_setup.ideal_task_duration(e_step.task_id, resource_id)
        real_duration = self.sim_setup.real_task_duration(e_step.ideal_duration, resource_id, started_at)
        event_index = e_step.trace_info.start_event(e_step.task_id,
                                                    self.sim_setup.bpmn_graph.element_info[e_step.task_id].name,
                                                    started_at, resource_id,
                                                    self.simulation_datetime_from(e_step.enabled_at),
                                                    e_step.enabled_by)
        self.real_duration[resource_id] += e_step.ideal_duration
        return event_index, real_duration

    def complete_task(self, e_step, event_index, real_duration):
        e_step.completed_at = self.simpy_env.now
        e_step.trace_info.complete_event(event_index, self.simulation_datetime_from(self.simpy_env.now),
                                         real_duration - e_step.ideal_duration)
        self.register_event(e_step.trace_info.event_list[event_index])
        self.log_fwriter.writerow(self._extract_event_data_from_step(e_step))
        return self.simpy_env.now

    def _extract_event_data_from_step(self, e_step: SimulationStep):
        case_id = e_step.trace_info.p_case
        activity = self.sim_setup.bpmn_graph.element_info[e_step.task_id].name
        resource = e_step.performed_by_resource
        completed_at = self._datetime_from(e_step.completed_at)

        if self.sim_setup.with_csv_state_column:
            return [case_id, activity, resource, 'completed', completed_at]
        else:
            started_at = self._datetime_from(e_step.started_at)
            if self.sim_setup.with_enabled_state:
                return [case_id, activity, self._datetime_from(e_step.enabled_at), started_at, completed_at, resource]
            else:
                return [case_id, activity, started_at, completed_at, resource]

    def _datetime_from(self, in_seconds):
        return self.simulation_datetime_from(in_seconds) if in_seconds is not None else None

    def current_simulation_date(self):
        return self.sim_setup.start_datetime + timedelta(seconds=self.simpy_env.now)

    def simulation_datetime_from(self, simpy_time):
        return self.sim_setup.start_datetime + timedelta(seconds=simpy_time)

    def get_utilization_for(self, resource_id):
        if resource_id not in self.sim_resources:
            return None
        if self.sim_resources[resource_id].available_time == 0:
            if self.sim_resources[resource_id].worked_time != 0:  # This conditional is for testing only (remove later)
                return -2
            return -1
        return self.sim_resources[resource_id].worked_time / self.sim_resources[resource_id].available_time

    def simulation_started_at(self):
        if len(self.completed_events) > 0:
            return self.completed_events[0].enabled_at
        return None

    def simulation_completed_at(self):
        if len(self.completed_events) > 0:
            return self.completed_events[len(self.completed_events) - 1].completed_at
        return None

    def request_resource(self, r_id):
        request = self.sim_resources[r_id].simpy_resource.request()
        self.sim_resources[r_id].is_available = False
        return request

    def release_resource(self, r_id, request):
        self.sim_resources[r_id].simpy_resource.release(request)
        self.sim_resources[r_id].is_available = True
        self.resource_total_allocated_tasks[r_id] += 1

    def register_event(self, event_info: TaskEvent):
        resource_id = event_info.resource_id
        worked_time = self._find_worked_times(event_info, self.sim_resources[resource_id].completed_events)
        self.sim_resources[resource_id].worked_time += worked_time
        self.completed_events.append(event_info)
        self.sim_resources[event_info.resource_id].completed_events.append(event_info)
        self.log_info.register_completed_task(event_info, self.sim_setup.resources_map[event_info.resource_id])

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
                    # event_info.print_event_info()
                    duration += resource_calendar.find_working_time(prev_event.completed_at, current_end)
                if event_info.started_at < prev_event.started_at:
                    current_end = prev_event.started_at
                else:
                    return duration
            i -= 1
        # event_info.print_event_info()
        return duration + resource_calendar.find_working_time(event_info.started_at, current_end)


def schedule_resource(bpm_env, resource_id):
    if resource_id in bpm_env.sim_resources and bpm_env.sim_resources[resource_id].is_available:
        simpy_env = bpm_env.simpy_env
        sim_setup = bpm_env.sim_setup

        to_rest = sim_setup.next_resting_time(resource_id, bpm_env.current_simulation_date())
        if to_rest > 0:
            request = bpm_env.request_resource(resource_id)
            yield simpy_env.timeout(to_rest)
            # print(simpy_env.now)
            bpm_env.release_resource(resource_id, request)
            bpm_env.sim_resources[resource_id].last_released = simpy_env.now

        to_execute = bpm_env.pop_event_for(resource_id)
        if to_execute is not None:
            simpy_env.process(execute_task_case(bpm_env, to_execute, resource_id))


def execute_task_case(bpm_env, e_step, r_id):
    simpy_env, sim_setup = bpm_env.simpy_env, bpm_env.sim_setup
    # Locking the resource performing the task
    request = bpm_env.request_resource(r_id)
    # Computing real duration (+idle time) and creating (starting) the new event
    event_index, real_duration = bpm_env.start_task(e_step, r_id, sim_setup)
    # This conditional is for testing purposes .. remove later
    if real_duration < e_step.ideal_duration:
        bpm_env.start_task(e_step, r_id, sim_setup)
    # print_event_state('Started', e_step, bpm_env, r_id)

    # Waiting for the event (task) to be completed
    # print(real_duration)
    # if to_rest > 0:
    #     sim_setup.next_resting_time(resource_id, bpm_env.current_simulation_date())
    #     print(resource_id)
    #     print(str(datetime.timedelta(seconds=to_rest)))
    #     print(bpm_env.current_simulation_date())
    #     print(bpm_env.current_simulation_date().weekday())
    #     print('-------------')
    yield simpy_env.timeout(real_duration)
    # Completing the event after execution
    end_time = bpm_env.complete_task(e_step, event_index, real_duration)
    # Unlocking the resource that performed the task
    bpm_env.release_resource(r_id, request)
    # print_event_state('Completed', e_step, bpm_env, r_id)
    # Updating process case state, and retrieving the tasks enabled after completion of the current event (task)
    enabled_tasks = sim_setup.update_process_state(e_step.task_id, e_step.p_state)
    available_resources = PriorityQueue()
    available_resources.add_task(r_id, bpm_env.sim_resources[r_id].last_released)
    # Enqueuing the new enabled tasks (for execution), and retrieving the resource candidates
    for next_task in enabled_tasks:
        bpm_env.enqueue_event(e_step.trace_info, next_task, e_step.p_state, event_index, available_resources)
        # print_event_state('Enabled', e_step, bpm_env, r_id)
    # Select among the resource candidates, who will perform each of the enabled tasks (FIFO allocation)
    while not available_resources.is_empty():
        simpy_env.process(schedule_resource(bpm_env, available_resources.pop_task()))


def execute_full_process(bpm_env, total_cases):
    simpy_env = bpm_env.simpy_env
    sim_setup = bpm_env.sim_setup
    current_case = 0
    while True:
        # Triggering the start event from the initial state
        p_state = sim_setup.initial_state()
        enabled_tasks = sim_setup.update_process_state(sim_setup.bpmn_graph.starting_event, p_state)
        # Starting a new process case ...
        # Executing all the tasks enabled by the starting event
        trace_info = Trace(current_case, bpm_env.current_simulation_date())
        bpm_env.log_info.trace_list.append(trace_info)
        for task_id in enabled_tasks:
            e_step = bpm_env.enqueue_event(trace_info, task_id, p_state)
            # print_event_state('Enabled', e_step, bpm_env, 'New-Case')
        # print('Started:' + str(bpm_env.current_simulation_date()))
        for r_id in sim_setup.resources_map:
            simpy_env.process(schedule_resource(bpm_env, r_id))
        # Verifying if there are more cases to execute
        current_case += 1
        # bpm_env.log_info.post_process_completed_trace()
        if current_case >= total_cases:
            break
        # Waiting for the next case to start according to the arrival time distribution
        # simpy_env.process(create_new_process_case(bpm_env))
        next_arrival = sim_setup.next_arrival_time(bpm_env.current_simulation_date())

        yield simpy_env.timeout(next_arrival)
        # print(current_case)


def run_simulation(bpmn_path, json_path, total_cases, stat_out_path=None, log_out_path=None, starting_at=None,
                   with_enabled_state=False, with_csv_state_column=False):
    diffsim_info = SimDiffSetup(bpmn_path, json_path, with_enabled_state, with_csv_state_column)

    if not diffsim_info:
        return None

    diffsim_info.set_starting_satetime(starting_at if starting_at else pytz.utc.localize(datetime.datetime.now()))

    if not stat_out_path:
        stat_out_path = os.path.join(os.path.dirname(__file__), Path("%s.csv" % diffsim_info.process_name))
    with open(stat_out_path, mode='w', newline='') as stat_csv_file:
        if log_out_path:
            with open(log_out_path, mode='w', newline='') as log_csv_file:
                run_simpy_simulation(diffsim_info, total_cases,
                                     csv.writer(stat_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL),
                                     csv.writer(log_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL))
        else:
            run_simpy_simulation(diffsim_info, total_cases,
                                 csv.writer(stat_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL))


def run_simpy_simulation(diffsim_info, total_cases, stat_fwriter, log_fwriter=None):
    env = simpy.Environment()
    bpm_env = SimBPMEnv(env, diffsim_info, stat_fwriter, log_fwriter)
    add_simulation_event_log_header(log_fwriter, diffsim_info.with_enabled_state, diffsim_info.with_csv_state_column)

    env.process(execute_full_process(bpm_env, total_cases))
    env.run()

    # print("Simulation Compeleted in: %s" % str(
    #     datetime.timedelta(seconds=(datetime.datetime.now() - started_at).total_seconds())))
    # print('---------------------------------------------------------')
    bpm_env.log_info.save_joint_statistics(bpm_env)


def add_simulation_event_log_header(log_fwriter, with_enabled_state, with_csv_state_column):
    if log_fwriter:
        if with_csv_state_column:
            log_fwriter.writerow(['CaseID', 'Activity', 'org:resource', 'lifecycle:transition', 'time:timestamp'])
        else:
            if with_enabled_state:
                log_fwriter.writerow([
                    'CaseID', 'Activity', 'EnableTimestamp', 'StartTimestamp', 'EndTimestamp', 'Resource', ])
            else:
                log_fwriter.writerow(['CaseID', 'Activity', 'StartTimestamp', 'EndTimestamp', 'Resource'])