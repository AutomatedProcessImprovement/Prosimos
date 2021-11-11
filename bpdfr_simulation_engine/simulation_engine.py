import csv
import os
from pathlib import Path

import pytz
import simpy
import datetime
from datetime import timedelta

from bpdfr_simulation_engine.control_flow_manager import ProcessState
from bpdfr_simulation_engine.file_manager import FileManager
from bpdfr_simulation_engine.execution_info import Trace, TaskEvent
from bpdfr_simulation_engine.priority_queue import PriorityQueue
from bpdfr_simulation_engine.simulation_setup import SimDiffSetup
from bpdfr_simulation_engine.simulation_stats_calculator import LogInfo, print_event_state


class SimResource:
    def __init__(self):
        self.allocated_tasks = 0
        self.worked_time = 0
        self.available_time = 0
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
        self.stat_fwriter = stat_fwriter
        self.log_writer = FileManager(10000, log_fwriter)
        self.log_info = LogInfo(sim_setup)

        r_first_available = dict()
        for r_id in sim_setup.resources_map:
            self.sim_resources[r_id] = SimResource()
            r_first_available[r_id] = self.sim_setup.next_resting_time(r_id, self.current_simulation_date())

        self.sim_queue = DiffSimQueue(self.sim_setup.task_resource, r_first_available)

    def current_simulation_date(self):
        return self.sim_setup.start_datetime + timedelta(seconds=self.simpy_env.now)

    def create_new_process_case(self):
        p_case = len(self.log_info.trace_list)
        trace_info = Trace(p_case, self.current_simulation_date())
        self.log_info.trace_list.append(trace_info)
        return p_case

    def enqueue_event(self, p_case: int, task_id: str, enabled_by: TaskEvent = None):
        resource_id, r_available_at = self.sim_queue.pop_resource_for(task_id)
        self.sim_resources[resource_id].allocated_tasks += 1
        new_event = TaskEvent(p_case, task_id, resource_id, r_available_at, enabled_by, self)
        self.log_info.add_event_info(p_case, new_event, self.sim_setup.resources_map[resource_id].cost_per_hour)
        self.log_writer.add_csv_row([p_case, self.sim_setup.bpmn_graph.element_info[task_id].name,
                                     None, 'enabled', new_event.enabled_datetime])
        self.sim_queue.upddate_resource_availability(resource_id,
                                                     new_event.completed_at + self.sim_setup.next_resting_time(
                                                         resource_id,
                                                         new_event.completed_datetime))
        self.sim_resources[resource_id].worked_time += new_event.real_duration
        return new_event

    def extract_event_data(self, p_case: int, event_info: TaskEvent):
        activity = self.sim_setup.bpmn_graph.element_info[event_info.task_id].name
        resource = event_info.resource_id
        completed_at = event_info.completed_datetime

        if self.sim_setup.with_csv_state_column:
            return [p_case, activity, resource, 'completed', completed_at]
        else:
            started_at = self._datetime_from(event_info.started_at)
            if self.sim_setup.with_enabled_state:
                return [p_case, activity, event_info.enabled_datetime, started_at, completed_at, resource]
            else:
                return [p_case, activity, started_at, completed_at, resource]

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


def schedule_enqueued_event(bpm_env: SimBPMEnv, current_event: TaskEvent, p_case: int, p_state: ProcessState):
    # print_event_state('Enabled', e_step, bpm_env, e_step.resource_id, e_step.enabled_at)
    simpy_env, sim_setup = bpm_env.simpy_env, bpm_env.sim_setup
    # Enabled events must wait until the allocated resource is available to perform it.
    if simpy_env.now < current_event.started_at:
        yield simpy_env.timeout(current_event.started_at - simpy_env.now)
    # At this point the activity is started, which will add an event in the event log with state started
    # _test_current = bpm_env.simpy_env.now
    # if current_event.started_at != _test_current:
    #     print('hola started')
    bpm_env.log_writer.add_csv_row([p_case, bpm_env.sim_setup.bpmn_graph.element_info[current_event.task_id].name,
                                    current_event.resource_id, 'started', current_event.started_datetime])

    # Waiting for the event to complete the execution, i.e., including the idle times of the allocated resource
    # print_event_state('Started', e_step, bpm_env, e_step.resource_id, e_step.started_at)
    yield simpy_env.timeout(current_event.real_duration)
    # Registering the completion of an activity
    # print_event_state('Ended  ', e_step, bpm_env, e_step.resource_id, e_step.completed_at)
    # _test_current = bpm_env.simpy_env.now
    # if current_event.completed_at != _test_current:
    #     print('hola completed')
    bpm_env.log_writer.add_csv_row(bpm_env.extract_event_data(p_case, current_event))

    # Updating the process state. Retrieving/enqueuing enabled tasks, it also schedules the corresponding event
    enabled_tasks = sim_setup.update_process_state(current_event.task_id, p_state)
    for next_task in enabled_tasks:
        next_step = bpm_env.enqueue_event(p_case, next_task, current_event)
        simpy_env.process(schedule_enqueued_event(bpm_env, next_step, p_case, p_state))


def execute_full_process(bpm_env: SimBPMEnv, total_cases):
    simpy_env = bpm_env.simpy_env
    sim_setup = bpm_env.sim_setup
    p_case = -1
    while p_case < total_cases - 1:
        # Triggering the start event from the initial state
        p_state = sim_setup.initial_state()
        enabled_tasks = sim_setup.update_process_state(sim_setup.bpmn_graph.starting_event, p_state)
        # Starting a new process case ...
        # Executing all the tasks enabled by the starting event
        p_case = bpm_env.create_new_process_case()
        for task_id in enabled_tasks:
            simpy_env.process(
                schedule_enqueued_event(bpm_env, bpm_env.enqueue_event(p_case, task_id, None), p_case, p_state)
            )

        # Waiting for the next case to start according to the arrival time distribution
        next_arrival = sim_setup.next_arrival_time(bpm_env.current_simulation_date())
        yield simpy_env.timeout(next_arrival)


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
