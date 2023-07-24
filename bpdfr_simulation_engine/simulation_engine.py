import csv
import os
from pathlib import Path

import pytz
import datetime
from datetime import timedelta

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

        self.test_resource_events = dict()

        r_first_available = dict()

        for r_id in sim_setup.resources_map:
            self.sim_resources[r_id] = SimResource()
            r_first_available[r_id] = self.sim_setup.next_resting_time(r_id, self.sim_setup.start_datetime)
            self.test_resource_events[r_id] = list()

        self.resource_queue = DiffResourceQueue(self.sim_setup.task_resource, r_first_available)
        self.events_queue = EventQueue()

    def generate_all_arrival_events(self, total_cases):
        sim_setup = self.sim_setup
        arrival_time = 0
        # prev = 0
        for p_case in range(0, total_cases):
            p_state = sim_setup.initial_state()
            enabled_tasks = sim_setup.update_process_state(sim_setup.bpmn_graph.starting_event, p_state)
            enabled_datetime = self.simulation_datetime_from(arrival_time)
            self.log_info.trace_list.append(Trace(p_case, enabled_datetime))
            for task_id in enabled_tasks:
                self.events_queue.append_arrival_event(EnabledEvent(p_case, p_state, task_id, arrival_time,
                                                                    enabled_datetime))
            arrival_time += sim_setup.next_arrival_time(enabled_datetime)

    def generate_fixed_arrival_events(self, starting_times):
        sim_setup = self.sim_setup

        p_case = 0
        for arrival_time in starting_times:
            p_state = sim_setup.initial_state()
            enabled_tasks = sim_setup.update_process_state(sim_setup.bpmn_graph.starting_event, p_state)
            enabled_datetime = self.simulation_datetime_from(arrival_time)
            self.log_info.trace_list.append(Trace(p_case, enabled_datetime))
            for task_id in enabled_tasks:
                self.events_queue.append_arrival_event(EnabledEvent(p_case, p_state, task_id, arrival_time,
                                                                    enabled_datetime))
            p_case += 1

    def execute_enabled_event(self, c_event: EnabledEvent):
        self.executed_events += 1
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

        self.resource_queue.upddate_resource_availability(r_id, r_next_available)
        self.sim_resources[r_id].worked_time += full_evt.ideal_duration

        if full_evt.enabled_datetime > full_evt.started_datetime:
            print("hola")
        self.log_writer.add_csv_row([c_event.p_case,
                                     self.sim_setup.bpmn_graph.element_info[c_event.task_id].name,
                                     full_evt.enabled_datetime,
                                     full_evt.started_datetime,
                                     full_evt.completed_datetime,
                                     self.sim_setup.resources_map[full_evt.resource_id].resource_name])

        # Updating the process state. Retrieving/enqueuing enabled tasks, it also schedules the corresponding event
        # s_t = datetime.datetime.now()
        enabled_tasks = self.sim_setup.update_process_state(c_event.task_id, c_event.p_state)
        # self.time_update_process_state += (datetime.datetime.now() - s_t).total_seconds()

        for next_task in enabled_tasks:
            self.events_queue.append_enabled_event(
                EnabledEvent(c_event.p_case, c_event.p_state, next_task, full_evt.completed_at,
                             full_evt.completed_datetime))

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


def execute_full_process(bpm_env: SimBPMEnv, total_cases, fixed_starting_times=None):
    # Initialize event queue with the arrival times of all the cases to simulate,
    # i.e., all the initial events are enqueued and sorted by their arrival times
    # s_t = datetime.datetime.now()
    if fixed_starting_times is None:
        bpm_env.generate_all_arrival_events(total_cases)
    else:
        bpm_env.generate_fixed_arrival_events(fixed_starting_times)
    # print("Generation of all cases: %s" %
    #       str(datetime.timedelta(seconds=(datetime.datetime.now() - s_t).total_seconds())))
    current_event = bpm_env.events_queue.pop_next_event()
    while current_event is not None:
        bpm_env.execute_enabled_event(current_event)
        current_event = bpm_env.events_queue.pop_next_event()


def run_simulation(bpmn_path, json_path, tot_cases, is_fuzzy, stat_out_path=None, log_out_path=None, starting_at=None,
                   fixed_arrival_times=None):
    diffsim_info = SimDiffSetup(bpmn_path, json_path, is_fuzzy)

    if not diffsim_info:
        return None

    diffsim_info.set_starting_satetime(starting_at if starting_at else pytz.utc.localize(datetime.datetime.now()))

    # if not stat_out_path and not log_out_path:
    #     stat_out_path = os.path.join(os.path.dirname(__file__), Path("%s.csv" % diffsim_info.process_name))
    if stat_out_path is None and log_out_path is None:
        return run_simpy_simulation(diffsim_info, tot_cases, None, None, fixed_arrival_times)
    elif stat_out_path:
        with open(stat_out_path, mode='w', newline='', encoding='utf-8') as stat_csv_file:
            if log_out_path:
                with open(log_out_path, mode='w', newline='', encoding='utf-8') as log_csv_file:
                    return run_simpy_simulation(diffsim_info, tot_cases,
                                                csv.writer(stat_csv_file, delimiter=',', quotechar='"',
                                                           quoting=csv.QUOTE_MINIMAL),
                                                csv.writer(log_csv_file, delimiter=',', quotechar='"',
                                                           quoting=csv.QUOTE_MINIMAL), fixed_arrival_times)
            else:
                return run_simpy_simulation(diffsim_info, tot_cases,
                                            csv.writer(stat_csv_file, delimiter=',', quotechar='"',
                                                       quoting=csv.QUOTE_MINIMAL), None, fixed_arrival_times)
    else:
        with open(log_out_path, mode='w', newline='', encoding='utf-8') as log_csv_file:
            return run_simpy_simulation(diffsim_info, tot_cases,
                                        None, csv.writer(log_csv_file, delimiter=',', quotechar='"',
                                                         quoting=csv.QUOTE_MINIMAL), fixed_arrival_times)


def run_simpy_simulation(diffsim_info, total_cases, stat_fwriter, log_fwriter, fixed_starting_times=None):
    bpm_env = SimBPMEnv(diffsim_info, stat_fwriter, log_fwriter)
    add_simulation_event_log_header(log_fwriter)
    execute_full_process(bpm_env, total_cases, fixed_starting_times)
    if fixed_starting_times is not None:
        return bpm_env
    if log_fwriter is None and stat_fwriter is None:
        return bpm_env.log_info.compute_process_kpi(bpm_env)
    if log_fwriter:
        bpm_env.log_writer.force_write()
    if stat_fwriter:
        bpm_env.log_info.save_joint_statistics(bpm_env)
    return None


def add_simulation_event_log_header(log_fwriter):
    if log_fwriter:
        log_fwriter.writerow([
            'case_id', 'activity', 'enable_time', 'start_time', 'end_time', 'resource', ])
