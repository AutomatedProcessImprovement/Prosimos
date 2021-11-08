import csv
import os
from enum import Enum
from pathlib import Path

import pytz
import simpy
import datetime
from datetime import timedelta
from collections import deque

from bpdfr_simulation_engine.file_manager import FileManager
from bpdfr_simulation_engine.execution_info import Trace, TaskEvent
from bpdfr_simulation_engine.priority_queue import PriorityQueue
from bpdfr_simulation_engine.simulation_setup import SimulationStep, SimDiffSetup
from bpdfr_simulation_engine.simulation_stats_calculator import LogInfo, print_event_state


class ResourceState(Enum):
    AVAILABLE = 0
    ALLOCATED = 1
    RESERVED = 2


class SimResource:
    def __init__(self, simpy_resource):
        self.simpy_resource = simpy_resource
        self.state = ResourceState.AVAILABLE
        self.allocated_to = None
        self.worked_time = 0
        self.available_time = 0
        self.completed_events = list()
        self.events_queue = deque()
        self.last_released = 0


class DiffSimQueue:
    # Two tasks share a resource queue iff the share all the resources. If the two tasks share only a set of resources,
    # then they will point to different resource queues. Therefore, a resource may be repeated in many queues.
    def __init__(self, task_resource_map, r_initial_availability):
        self._resource_queues = list()  # List of (shared) resource queues, i.e., many tasks may share a resource queue
        self._resource_queue_map = dict()  # Map relating the indexes of the queues where a resource r_id is contained
        self._task_queue_map = dict()  # Map with the index of the resource queue that can perform a task r_id

        self._init_simulation_queues(task_resource_map, r_initial_availability)

    def pop_resource_for(self, task_id):
        return self._resource_queues[self._task_queue_map[task_id]].pop_min()

    def upddate_resource_availability(self, resource_id, released_at):
        for q_index in self._resource_queue_map[resource_id]:
            self._resource_queues[q_index].insert(resource_id, released_at)

    def _init_simulation_queues(self, task_resource_map, r_initial_availability):
        joint_sets = list()
        taken = set()
        # Grouping the tasks that share all the resources
        for task_id_1 in task_resource_map:
            if task_id_1 not in taken:
                joint_tasks = set()
                for task_id_2 in task_resource_map:
                    is_joint = True
                    for r_id in task_resource_map[task_id_2]:
                        if r_id not in task_resource_map[task_id_1]:
                            is_joint = False
                            break
                    if is_joint:
                        taken.add(task_id_2)
                        joint_tasks.add(task_id_2)
                joint_sets.append(joint_tasks)

        index = 0
        for j_set in joint_sets:
            resource_queue = PriorityQueue()
            for task_id in j_set:
                self._task_queue_map[task_id] = index
                if resource_queue.is_empty():
                    for r_id in task_resource_map[task_id]:
                        if r_id not in self._resource_queue_map:
                            self._resource_queue_map[r_id] = list()
                        resource_queue.insert(r_id, r_initial_availability[r_id])
                        self._resource_queue_map[r_id].append(index)
            self._resource_queues.append(resource_queue)
            index += 1


class SimBPMEnv:
    def __init__(self, env, sim_setup: SimDiffSetup, stat_fwriter, log_fwriter):
        self.simpy_env = env
        self.sim_setup = sim_setup
        self.sim_resources = dict()
        self.completed_events = list()
        self.real_duration = dict()
        self.stat_fwriter = stat_fwriter
        self.log_fwriter = log_fwriter
        self.log_writer = FileManager(1000, self.log_fwriter)
        self.log_info = LogInfo(sim_setup)

        self.resource_total_allocated_tasks = dict()

        r_first_available = dict()
        for r_id in sim_setup.resources_map:
            self.resource_total_allocated_tasks[r_id] = 0
            self.sim_resources[r_id] = SimResource(simpy.Resource(env, sim_setup.resources_map[r_id].resource_amount))
            self.real_duration[r_id] = 0
            r_first_available[r_id] = self.sim_setup.next_resting_time(r_id, self.current_simulation_date())

        self.sim_queue = DiffSimQueue(self.sim_setup.task_resource, r_first_available)

    def current_simulation_date(self):
        return self.sim_setup.start_datetime + timedelta(seconds=self.simpy_env.now)

    def enqueue_event(self, trace_info, task_id, p_state, enabled_by=None, prev_step: SimulationStep = None):
        resource_id, resource_available_at = self.sim_queue.pop_resource_for(task_id)

        started_at = max(resource_available_at, prev_step.completed_at) if prev_step else max(resource_available_at,
                                                                                              self.simpy_env.now)

        self.resource_total_allocated_tasks[resource_id] += 1

        e_step = SimulationStep(trace_info, task_id, self.sim_setup.ideal_task_duration(task_id, resource_id), p_state,
                                self.simpy_env.now, self.simulation_datetime_from(self.simpy_env.now), enabled_by)
        e_step.started_at = started_at
        e_step.started_datetime = self.simulation_datetime_from(started_at)

        e_step.resource_id = resource_id

        e_step.real_duration = self.sim_setup.real_task_duration(e_step.ideal_duration, resource_id,
                                                                 e_step.started_datetime)
        self.log_writer.add_csv_row([trace_info.p_case, self.sim_setup.bpmn_graph.element_info[task_id].name,
                                     None, 'enabled', e_step.enabled_datetime])

        e_step.completed_at = started_at + e_step.real_duration
        e_step.completed_datetime = self.simulation_datetime_from(e_step.completed_at)
        n_t = e_step.completed_at + self.sim_setup.next_resting_time(resource_id, e_step.completed_datetime)
        self.sim_queue.upddate_resource_availability(resource_id, n_t)
        return e_step

    def start_task(self, e_step):
        current = self.simpy_env.now
        if e_step.started_at != current:
            print('hola started')
        self.log_writer.add_csv_row([e_step.trace_info.p_case,
                                     self.sim_setup.bpmn_graph.element_info[e_step.task_id].name,
                                     e_step.resource_id, 'started', e_step.started_datetime])
        event_index = e_step.trace_info.start_event(e_step.task_id,
                                                    self.sim_setup.bpmn_graph.element_info[e_step.task_id].name,
                                                    e_step.started_datetime, e_step.resource_id,
                                                    e_step.enabled_datetime,
                                                    e_step.enabled_by)
        self.real_duration[e_step.resource_id] += e_step.ideal_duration
        return event_index

    def complete_task(self, e_step, event_index):
        current = self.simpy_env.now
        if e_step.completed_at != current:
            print('hola completed')
        e_step.trace_info.complete_event(event_index, e_step.completed_datetime,
                                         e_step.real_duration - e_step.ideal_duration)
        self.register_event(e_step.trace_info.event_list[event_index])
        self.log_writer.add_csv_row(self._extract_event_data_from_step(e_step))

        return self.simpy_env.now

    def _extract_event_data_from_step(self, e_step: SimulationStep):
        case_id = e_step.trace_info.p_case
        activity = self.sim_setup.bpmn_graph.element_info[e_step.task_id].name
        resource = e_step.resource_id
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

    def request_resource(self, r_id, task_id):
        request = self.sim_resources[r_id].simpy_resource.request()
        self.sim_resources[r_id].allocated_to = task_id
        return request

    def release_allocated_resource(self, r_id, task_id, request):
        self.sim_resources[r_id].simpy_resource.release(request)
        self.sim_resources[r_id].allocated_to = None
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


# def schedule_resource(bpm_env: SimBPMEnv, task_id):
#     # if resource_id in bpm_env.sim_resources and bpm_env.sim_resources[resource_id].is_available:
#     simpy_env = bpm_env.simpy_env
#     sim_setup = bpm_env.sim_setup
#
#     r_id, request = bpm_env.allocate_available_resource(task_id)
#     if r_id is not None:
#         to_rest = sim_setup.next_resting_time(r_id, bpm_env.current_simulation_date())
#         if to_rest > 0:
#             yield simpy_env.timeout(to_rest)
#             bpm_env.release_allocated_resource(r_id, request)
#             simpy_env.process(execute_task_case(bpm_env, task_id, r_id))
#     else:
#         # This case occurrs if the resource is available, but there are not tasks available that he/she can execute
#         # Add the resource to the queue of all the possible tasks that he/she can execute
#         return 1


def schedule_enqueued_event(bpm_env: SimBPMEnv, e_step: SimulationStep):
    # print_event_state('Enabled', e_step, bpm_env, e_step.resource_id, e_step.enabled_at)
    simpy_env, sim_setup = bpm_env.simpy_env, bpm_env.sim_setup
    # Enabled events must wait until the allocated resource is available to perform it.
    if simpy_env.now < e_step.started_at:
        yield simpy_env.timeout(e_step.started_at - simpy_env.now)
    # Starting the activity, this will add an event in the event log with state started
    event_index = bpm_env.start_task(e_step)
    # Waiting for the event to complete the execution, i.e., including the idle times of the allocated resource
    # print_event_state('Started', e_step, bpm_env, e_step.resource_id, e_step.started_at)
    yield simpy_env.timeout(e_step.real_duration)
    # Registering the completion of an activity
    # print_event_state('Ended  ', e_step, bpm_env, e_step.resource_id, e_step.completed_at)
    bpm_env.complete_task(e_step, event_index)
    # Updating the process state. Retrieving/enqueuing enabled tasks, it also schedules the corresponding event
    enabled_tasks = sim_setup.update_process_state(e_step.task_id, e_step.p_state)
    for next_task in enabled_tasks:
        next_step = bpm_env.enqueue_event(e_step.trace_info, next_task, e_step.p_state, event_index, e_step)
        simpy_env.process(schedule_enqueued_event(bpm_env, next_step))


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
            simpy_env.process(schedule_enqueued_event(bpm_env,
                                                      bpm_env.enqueue_event(trace_info, task_id, p_state)))
        # print_event_state('Enabled', e_step, bpm_env, 'New-Case')
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
    with open(stat_out_path, mode='w', newline='', encoding='utf-8') as stat_csv_file:
        if log_out_path:
            with open(log_out_path, mode='w', newline='', encoding='utf-8') as log_csv_file:
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
