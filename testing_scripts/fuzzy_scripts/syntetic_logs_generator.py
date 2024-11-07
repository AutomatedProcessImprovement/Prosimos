import csv
from collections import deque
from datetime import datetime, timedelta
import random
from enum import Enum

import pandas as pd
import numpy as np
from prosimos.control_flow_manager import BPMNGraph, ProcessState

from bpdfr_discovery.log_parser import event_list_from_csv
from prosimos.execution_info import TaskEvent
from prosimos.file_manager import FileManager
from prosimos.probability_distributions import Choice, best_fit_distribution, generate_number_from
from prosimos.simulation_properties_parser import parse_simulation_model
from prosimos.simulation_queues_ds import PriorityQueue
from prosimos.weekday_helper import CustomDatetimeAndSeconds

from testing_scripts.fuzzy_scripts.fuzzy_test_files import FileType, get_file_path_fuzzy
from testing_scripts.fuzzy_scripts.old_fuzzy_structs import ProcInfo
from testing_scripts.multitasking_scripts.multitasking_files import get_file_path_multi


class IType(Enum):
    FULL_TIME = 1
    MORNING = 2
    AFTERNOON = 3
    ALTERNATE = 4
    ALWAYS = 5
    MWF_24 = 6


class WorkingCalendar:
    # Monday:0, ..., Sunday: 6
    def __init__(self, c_type: IType, last_worked_date: datetime):
        self.c_type = c_type
        self.last_worked_day = last_worked_date.replace(hour=00, minute=00, second=00, microsecond=00)
        self.i_length = 86400 if [IType.ALWAYS, IType.ALTERNATE] else 14000

    def next_available(self, from_date: datetime):
        if self.c_type is IType.ALWAYS:
            return 0
        c_time = from_date.time()
        week_day = from_date.weekday()

        full_day = 86400
        current_second = (from_date.hour * 3600) + (from_date.minute * 60) + from_date.second
        to_00 = full_day - current_second
        from_00_to_08 = 28800
        from_00_to_13 = 46800

        if self.c_type is not IType.ALTERNATE:
            if self.c_type is IType.FULL_TIME:
                if week_day == 4 and c_time.hour >= 17:  # If Friday
                    return to_00 + (2 * full_day) + from_00_to_08
                if week_day == 5:  # If Saturday
                    return to_00 + full_day + from_00_to_08
                if week_day == 6:  # If Sunday
                    return to_00 + from_00_to_08
                if (8 <= c_time.hour < 12) or (13 <= c_time.hour < 17):
                    return 0
                if c_time.hour < 8:
                    return from_00_to_08 - current_second
                if 12 <= c_time.hour < 13:
                    return from_00_to_13 - current_second
                if c_time.hour >= 17:
                    return to_00 + from_00_to_08
            if self.c_type is IType.MORNING:
                if week_day == 4 and c_time.hour >= 12:  # If Friday
                    return to_00 + (2 * full_day) + from_00_to_08
                if week_day == 5:  # If Saturday
                    return to_00 + full_day + from_00_to_08
                if week_day == 6:  # If Sunday
                    return to_00 + from_00_to_08
                if 8 <= c_time.hour < 12:
                    return 0
                if c_time.hour < 8:
                    return from_00_to_08 - current_second
                if c_time.hour >= 12:
                    return to_00 + from_00_to_08
            if self.c_type is IType.AFTERNOON:
                if week_day == 4 and c_time.hour >= 17:  # If Friday
                    return to_00 + (2 * full_day) + from_00_to_13
                if week_day == 5:  # If Saturday
                    return to_00 + full_day + from_00_to_08
                if week_day == 6:  # If Sunday
                    return to_00 + from_00_to_08
                if 13 <= c_time.hour < 17:
                    return 0
                if c_time.hour < 13:
                    return from_00_to_13 - current_second
                if c_time.hour >= 17:
                    return to_00 + from_00_to_13
        else:
            if from_date.date() == self.last_worked_day.date():
                return 0
            if from_date.date() > self.last_worked_day.date():
                while self.last_worked_day.date() < from_date.date():
                    self.last_worked_day += timedelta(days=2)
                return (
                        self.last_worked_day - from_date).total_seconds() if from_date.date() < self.last_worked_day.date() else 0
            if from_date.date() < self.last_worked_day.date():
                return (self.last_worked_day - from_date).total_seconds()
        return None

    def to_interval_end(self, from_date: datetime):
        c_time = from_date.time()
        full_day = 86400.0
        current_second = (from_date.hour * 3600) + (from_date.minute * 60) + from_date.second
        from_00_to_12 = 43200.0
        from_00_to_17 = 61200.0
        if self.c_type in [IType.FULL_TIME, IType.MORNING]:
            if 8 <= c_time.hour < 12:
                return from_00_to_12 - current_second
        if self.c_type in [IType.FULL_TIME, IType.AFTERNOON]:
            if 13 <= c_time.hour < 17:
                return from_00_to_17 - current_second
        if self.c_type is IType.ALTERNATE:
            if self.last_worked_day.date() == from_date.date():
                return full_day - current_second
        return 0.0

    def adjust_duration(self, from_date: datetime, duration: float):
        # from_date must be a date in which the resource is available
        unavailable = self.next_available(from_date)
        if unavailable > 0:
            from_date += timedelta(seconds=unavailable)
        if self.c_type is IType.ALWAYS:
            return duration

        if duration <= self.to_interval_end(from_date):
            return duration + unavailable
        elif duration <= self.i_length:
            return duration + self.next_available(from_date + timedelta(seconds=duration)) + unavailable
        else:
            real_duration = 0
            c_date = from_date
            resting_time = 0

            while duration > 0:
                to_add = min(self.to_interval_end(c_date), duration)
                real_duration += (to_add + resting_time)
                duration -= to_add
                resting_time = self.next_available(c_date + timedelta(seconds=to_add))
                c_date += timedelta(seconds=(to_add + resting_time))
            return real_duration + unavailable

    def available_minutes(self):
        minutes = []
        from_m = 0
        if self.c_type is IType.FULL_TIME:
            self._update_minutes(minutes, 8, 12)
            self._update_minutes(minutes, 13, 17)
            from_m = 8 * 60
        if self.c_type is IType.ALWAYS:
            self._update_minutes(minutes, 0, 24)
        return from_m, minutes

    @staticmethod
    def _update_minutes(minutes, from_m, to_m):
        for i in range(from_m * 60, to_m * 60):
            minutes.append(i)


def generate_syntetic_multitasking_log(proc_name, is_fuzzy, total_cases):
    generate_syntetic_log(proc_name, is_fuzzy, total_cases)


def generate_syntetic_log(proc_name, is_fuzzy, total_cases):
    bpmn_graph, p_info, start_datetime, arrival_dates, task_res_distributions = generate_base_sim_model(proc_name,
                                                                                                        total_cases,
                                                                                                        is_fuzzy)
    for even_res_workload in [True, False]:
        task_res_allocations = get_task_resource_workload(p_info, even_res_workload)
        traces = dict()
        for i in range(0, total_cases):
            traces[i] = generate_trace(i, bpmn_graph, task_res_distributions, task_res_allocations)

        _assign_resource_24_hours(proc_name, p_info, traces, arrival_dates, start_datetime, even_res_workload, is_fuzzy)

        _assign_resource_8_hours(p_info, proc_name, traces, arrival_dates, start_datetime, even_res_workload, is_fuzzy)

        _assign_resource_8_4_4_hours(p_info, proc_name, traces, arrival_dates, start_datetime, even_res_workload,
                                     is_fuzzy)

        _assign_resource_8_4_4_24_hours(p_info, proc_name, traces, arrival_dates, start_datetime, even_res_workload,
                                        is_fuzzy)


def generate_syntetic_log_vacation(proc_name, is_fuzzy, total_cases, num_res, single_resource, arrival_rate):
    bpmn_graph, p_info, start_datetime, arrival_dates, task_res_distributions = generate_base_sim_model(proc_name,
                                                                                                        total_cases,
                                                                                                        is_fuzzy,
                                                                                                        arrival_rate)
    t_r_distributions = dict()
    if single_resource:
        for t_id in task_res_distributions:
            t_r_distributions[t_id] = dict()
            r_id = random.choice(list(task_res_distributions[t_id].keys()))
            t_r_distributions[t_id]["R1"] = task_res_distributions[t_id][r_id]
            # if num_res > 1:
            #     t_r_distributions[t_id]["R2"] = task_res_distributions[t_id][r_id]

        traces = dict()
        candidates = ["R1", "R2"]
        for i in range(0, total_cases):
            # traces[i] = generate_trace(i, bpmn_graph, t_r_distributions, dict(), random.choice(candidates))
            traces[i] = generate_trace(i, bpmn_graph, t_r_distributions, dict(), "R1")

        r_calendars = dict()
        r_calendars["R1"] = WorkingCalendar(IType.FULL_TIME, start_datetime)
        # r_calendars["R2"] = WorkingCalendar(IType.FULL_TIME, start_datetime)

        _assign_times_to_tasks(proc_name, traces, arrival_dates, 2, True, r_calendars, is_fuzzy)

        _save_event_log(
            out_log_path=get_file_path(is_fuzzy=False, proc_name=proc_name, file_type=FileType.ORIGINAL_CSV_LOG,
                                       calendar_type=5, even=True),
            traces=traces,
            is_fuzzy=True)
    else:
        task_res_allocations = get_task_resource_workload(p_info, True)
        traces = dict()
        for i in range(0, total_cases):
            traces[i] = generate_trace(i, bpmn_graph, task_res_distributions, task_res_allocations)

        _assign_resource_8_hours(p_info, proc_name, traces, arrival_dates, start_datetime, True, is_fuzzy)



def get_file_path(is_fuzzy, proc_name: str, file_type: FileType, granule=15, angle=0, file_index=0, calendar_type=1,
                  even=True):
    if is_fuzzy:
        return get_file_path_fuzzy(proc_name, file_type, granule, angle, file_index, calendar_type, even)
    return get_file_path_multi(proc_name, file_type, granule, angle, file_index, calendar_type, even)


def generate_base_sim_model(proc_name, total_cases, is_fuzzy, mean_x_day=108):
    bpmn_graph = parse_simulation_model(get_file_path(is_fuzzy, proc_name=proc_name, file_type=FileType.BPMN))

    gen_trace = event_list_from_csv(
        get_file_path(is_fuzzy, proc_name=proc_name, file_type=FileType.GENERATOR_LOG, even=True))

    p_info = ProcInfo(gen_trace, bpmn_graph, 60, False)
    bpmn_graph.set_element_probabilities(
        parse_gateway_probabilities(bpmn_graph.compute_branching_probability(p_info.flow_arcs_frequency)), None)

    start_datetime = pd.to_datetime('2023-04-17 08:00:00.000000+00:00')
    arrival_dates = get_arrival_times(total_cases, start_datetime, mean_x_day,
                                      WorkingCalendar(IType.FULL_TIME, start_datetime), is_fuzzy)
    task_res_distributions = get_task_res_distsibution(p_info)

    return bpmn_graph, p_info, start_datetime, arrival_dates, task_res_distributions


def generate_vacation_sim_model(proc_name, total_cases, is_fuzzy, mean_x_day):
    bpmn_graph = parse_simulation_model(get_file_path(is_fuzzy, proc_name=proc_name, file_type=FileType.BPMN))

    gen_trace = event_list_from_csv(
        get_file_path(is_fuzzy, proc_name=proc_name, file_type=FileType.GENERATOR_LOG, even=True))

    p_info = ProcInfo(gen_trace, bpmn_graph, 60, False)
    bpmn_graph.set_element_probabilities(
        parse_gateway_probabilities(bpmn_graph.compute_branching_probability(p_info.flow_arcs_frequency)), None)

    start_datetime = pd.to_datetime('2023-04-17 08:00:00.000000+00:00')
    arrival_dates = get_arrival_times(total_cases, start_datetime, mean_x_day,
                                      WorkingCalendar(IType.FULL_TIME, start_datetime), is_fuzzy)
    task_res_distributions = get_task_res_distsibution(p_info)

    return bpmn_graph, p_info, start_datetime, arrival_dates, task_res_distributions


def _assign_resource_24_hours(proc_name, p_info: ProcInfo, traces: dict, arrival_times, start_datetime, is_even,
                              is_fuzzy):
    if is_fuzzy:
        _assign_resource_24_hours_fuzzy(proc_name, p_info, traces, arrival_times, start_datetime, is_even)
    else:
        _assign_resource_24_hours_multi(proc_name, p_info, traces, arrival_times, start_datetime, is_even)


def _assign_resource_24_hours_multi(proc_name, p_info: ProcInfo, traces: dict, arrival_times, start_datetime, is_even):
    print("Generating 24 Hours Calendars. Even Resource Workload %s" % str(is_even))

    ev_queue = PriorityQueue()
    for t_id in traces:
        traces[t_id][0].task_name = 0
        ev_queue.insert(traces[t_id][0], arrival_times[t_id])

    while not ev_queue.is_empty():
        c_event, enabled_at = ev_queue.pop_min()
        c_event.enaled_at = enabled_at
        c_event.started_at = enabled_at
        c_event.completed_at = c_event.started_at + timedelta(seconds=c_event.processing_time)

        if c_event.task_name < len(traces[c_event.p_case]) - 1:
            next_ev = traces[c_event.p_case][c_event.task_name + 1]
            next_ev.task_name = c_event.task_name + 1
            ev_queue.insert(next_ev, c_event.completed_at)

    _save_event_log(
        out_log_path=get_file_path(is_fuzzy=False, proc_name=proc_name, file_type=FileType.ORIGINAL_CSV_LOG,
                                   even=is_even),
        traces=traces,
        is_fuzzy=False)


def _assign_resource_24_hours_fuzzy(proc_name, p_info: ProcInfo, traces: dict, arrival_times, start_datetime, is_even):
    print("Generating 24 Hours Calendars. Even Resource Workload %s" % str(is_even))
    r_next_available = dict()
    for r_id in p_info.resource_tasks:
        r_next_available[r_id] = start_datetime

    ev_queue = PriorityQueue()
    for t_id in traces:
        traces[t_id][0].task_name = 0
        ev_queue.insert(traces[t_id][0], arrival_times[t_id])

    while not ev_queue.is_empty():
        c_event, enabled_at = ev_queue.pop_min()
        c_event.enaled_at = enabled_at
        r_available = max(r_next_available[c_event.resource_id], enabled_at)
        c_event.started_at = r_available
        c_event.completed_at = r_available + timedelta(seconds=c_event.processing_time)

        r_next_available[c_event.resource_id] = max(c_event.completed_at, r_next_available[c_event.resource_id])
        if c_event.task_name < len(traces[c_event.p_case]) - 1:
            next_ev = traces[c_event.p_case][c_event.task_name + 1]
            next_ev.task_name = c_event.task_name + 1
            ev_queue.insert(next_ev, c_event.completed_at)

    _save_event_log(
        out_log_path=get_file_path(is_fuzzy=True, proc_name=proc_name, file_type=FileType.ORIGINAL_CSV_LOG,
                                   even=is_even),
        traces=traces,
        is_fuzzy=True)


def _assign_resource_8_hours(p_info: ProcInfo, proc_name, traces, arrival_dates, start_datetime, is_even, is_fuzzy):
    print("Generating 8 Hours Calendars, 5 days x week. Even Resource Workload %s" % str(is_even))
    r_calendars = dict()
    for r_id in p_info.resource_tasks:
        r_calendars[r_id] = WorkingCalendar(IType.FULL_TIME, start_datetime)

    _assign_times_to_tasks(proc_name, traces, arrival_dates, 2, is_even, r_calendars, is_fuzzy)


def _assign_resource_8_4_4_hours(p_info: ProcInfo, proc_name, traces, arrival_dates, start_datetime, is_even, is_fuzzy):
    print("Generating 8 Hours, Morning, and Afternoon Calendars. Even Resource Workload %s" % str(is_even))
    r_calendars = dict()
    index = 0
    for r_id in p_info.resource_tasks:
        if index % 2 == 0:
            r_calendars[r_id] = WorkingCalendar(IType.FULL_TIME, start_datetime)
        if index % 2 == 1:
            r_calendars[r_id] = WorkingCalendar(IType.AFTERNOON, start_datetime)
        index += 1

    _assign_times_to_tasks(proc_name, traces, arrival_dates, 3, is_even, r_calendars, is_fuzzy)


def _assign_resource_8_4_4_24_hours(p_info: ProcInfo, proc_name, traces, arrival_dates, start_datetime, is_even,
                                    is_fuzzy):
    print(
        "Generating 8 Hours and Alternate Calendars 1 day work/ 2 days rest. Even Resource Workload %s" % str(is_even))
    r_calendars = dict()
    index = 0
    for r_id in p_info.resource_tasks:
        if index % 3 == 0:
            r_calendars[r_id] = WorkingCalendar(IType.ALTERNATE, start_datetime)
        if index % 3 == 1:
            r_calendars[r_id] = WorkingCalendar(IType.FULL_TIME, start_datetime)
        if index % 3 == 2:
            r_calendars[r_id] = WorkingCalendar(IType.AFTERNOON, start_datetime)
        index += 1

    _assign_times_to_tasks(proc_name, traces, arrival_dates, 5, is_even, r_calendars, is_fuzzy)


def _assign_times_to_tasks(proc_name, traces, arrival_dates, calendar_type, is_even, r_calendars, is_fuzzy):
    if is_fuzzy:
        _assign_times_to_tasks_fuzzy(proc_name, traces, arrival_dates, calendar_type, is_even, r_calendars)
    else:
        _assign_times_to_tasks_multi(proc_name, traces, arrival_dates, calendar_type, is_even, r_calendars)


def _assign_times_to_tasks_multi(proc_name, traces, arrival_dates, calendar_type, is_even, r_calendars):
    ev_queue = PriorityQueue()

    for t_id in traces:
        traces[t_id][0].task_name = 0
        ev_queue.insert(traces[t_id][0], arrival_dates[t_id])

    while not ev_queue.is_empty():
        c_event, enabled_at = ev_queue.pop_min()
        c_event.enaled_at = enabled_at

        c_event.started_at = enabled_at + timedelta(seconds=r_calendars[c_event.resource_id].next_available(enabled_at))
        real_time = r_calendars[c_event.resource_id].adjust_duration(c_event.started_at, c_event.processing_time)
        c_event.completed_at = c_event.started_at + timedelta(seconds=real_time)

        if c_event.task_name < len(traces[c_event.p_case]) - 1:
            next_ev = traces[c_event.p_case][c_event.task_name + 1]
            next_ev.task_name = c_event.task_name + 1
            ev_queue.insert(next_ev, c_event.completed_at)

    _save_event_log(
        out_log_path=get_file_path(is_fuzzy=False, proc_name=proc_name, file_type=FileType.ORIGINAL_CSV_LOG,
                                   calendar_type=calendar_type, even=is_even),
        traces=traces,
        is_fuzzy=False)


def _assign_times_to_tasks_fuzzy(proc_name, traces, arrival_dates, calendar_type, is_even, r_calendars):
    ev_queue = PriorityQueue()
    r_last_avail = dict()
    for r_id in r_calendars:
        r_last_avail[r_id] = (arrival_dates[0] + timedelta(seconds=r_calendars[r_id].next_available(arrival_dates[0])))

    for t_id in traces:
        traces[t_id][0].task_name = 0
        ev_queue.insert(traces[t_id][0], arrival_dates[t_id])

    while not ev_queue.is_empty():
        c_event, enabled_at = ev_queue.pop_min()
        c_event.enaled_at = enabled_at

        enabled_at = max(enabled_at, r_last_avail[c_event.resource_id])

        r_available = enabled_at + timedelta(seconds=r_calendars[c_event.resource_id].next_available(enabled_at))

        c_event.started_at = r_available

        real_time = r_calendars[c_event.resource_id].adjust_duration(c_event.started_at, c_event.processing_time)

        c_event.completed_at = c_event.started_at + timedelta(seconds=real_time)
        r_last_avail[c_event.resource_id] = max(c_event.completed_at, r_last_avail[c_event.resource_id])

        if c_event.task_name < len(traces[c_event.p_case]) - 1:
            next_ev = traces[c_event.p_case][c_event.task_name + 1]
            next_ev.task_name = c_event.task_name + 1
            ev_queue.insert(next_ev, c_event.completed_at)

    _save_event_log(
        out_log_path=get_file_path(is_fuzzy=True, proc_name=proc_name, file_type=FileType.ORIGINAL_CSV_LOG,
                                   calendar_type=calendar_type, even=is_even),
        traces=traces,
        is_fuzzy=True)


def get_arrival_time_distribution(total_cases, duration_min):
    lambd = 1 / duration_min
    arrival_durations = [0]
    for i in range(1, total_cases):
        arrival_durations.append(int(random.expovariate(lambd) * 60))
    return arrival_durations


def get_arrival_times(total_cases, start_datetime, mean_x_day, calendar: WorkingCalendar, is_fuzzy):
    if is_fuzzy:
        return get_arrival_times_fuzzy(total_cases, start_datetime, mean_x_day, calendar)
    return get_arrival_times_multi(total_cases, start_datetime, mean_x_day)


def get_arrival_times_multi(total_cases, start_datetime, mean_x_day):
    # arrival_dates = list()
    # d_generator = start_datetime.replace(hour=00, minute=00, second=00, microsecond=00)
    # delta = mean_x_day // 24
    # while total_cases > 0:
    #     arrival_dates.append(d_generator)
    #     d_generator = d_generator + timedelta(minutes=delta)
    #     total_cases -= 1
    i_arrival = total_cases
    arrival_dates = list()
    rng = np.random.default_rng()
    dates = [start_datetime.replace(hour=8, minute=00, second=00, microsecond=00),
             start_datetime.replace(hour=13, minute=00, second=00, microsecond=00)]
    while total_cases > 0:
        c_count = min(int(rng.poisson(lam=mean_x_day)), total_cases)
        r_count = [c_count - c_count // 2, c_count // 2]
        for d in range(0, len(dates)):
            for i in range(0, r_count[d]):
                arrival_dates.append(dates[d])
            dates[d] += timedelta(hours=24)
        total_cases -= c_count
    arrival_dates.sort()
    if len(arrival_dates) != i_arrival:
        print(len(arrival_dates))
    return arrival_dates


def get_arrival_times_fuzzy(total_cases, start_datetime, mean_x_day, calendar: WorkingCalendar):
    arrival_dates = list()
    rng = np.random.default_rng()
    from_m, a_minutes = calendar.available_minutes()
    c_date = start_datetime.replace(hour=8, minute=00, second=00, microsecond=00)
    for j in range(0, total_cases):
        c_count = rng.poisson(lam=mean_x_day)
        for i in range(0, c_count):
            arrival_dates.append(c_date + timedelta(minutes=(a_minutes[random.randrange(len(a_minutes))] - from_m)))
        c_date = c_date.replace(hour=17, minute=5, second=0, microsecond=0)
        c_date += timedelta(seconds=calendar.next_available(c_date))
    arrival_dates.sort()
    return arrival_dates


def adjust_to_calendar(arrival_durations, starting_at: datetime, calendar: WorkingCalendar):
    c_date = starting_at
    arrival_dates = list()
    for duration in arrival_durations:
        c_date += timedelta(seconds=duration)
        c_date += timedelta(seconds=calendar.next_available(c_date))
        arrival_dates.append(c_date)

    return arrival_dates


class PoolQueue:
    def __init__(self, r_pool):
        self.p_queue = deque()
        while True:
            stop = True
            for r_id in r_pool:
                if r_pool[r_id] > 0:
                    self.p_queue.append(r_id)
                    r_pool[r_id] -= 1
                    stop = False
            if stop:
                break

    def next_resource(self):
        r_id = self.p_queue.popleft()
        self.p_queue.append(r_id)
        return r_id


def get_task_resource_workload(p_info: ProcInfo, is_even: bool, single=None):
    pool_list = list()
    task_pool = dict()
    taken = dict()

    factors = [6, 5, 4, 3, 2, 1] if not is_even else [1, 1, 1, 1, 1, 1]
    for t_id in p_info.task_resources:
        r_pool = dict()
        f_ind = 0
        p_index = len(pool_list)
        for r_id in p_info.task_resources[t_id]:
            if r_id not in taken:
                r_pool[r_id] = factors[f_ind]
                taken[r_id] = p_index
                f_ind = ((f_ind + 1) % 6)
            else:
                p_index = taken[r_id]
                break
        task_pool[t_id] = p_index
        if p_index == len(pool_list):
            pool_list.append(PoolQueue(r_pool))

    task_res_allocation = dict()
    for t_id in task_pool:
        task_res_allocation[t_id] = pool_list[task_pool[t_id]]

    return task_res_allocation


# def get_task_single_resource_workload(task_res: dict, r_id):
#     for t_id in task_res:
#         for r_id in task_res[t_id]:
#
#
#
#
#     task_res_allocation = dict()
#     for t_id in task_pool:
#         task_res_allocation[t_id] = pool_list[task_pool[t_id]]
#
#     return task_res_allocation
#



def _get_task_resource_workload(p_info: ProcInfo, equitative):
    r_workloads = _get_workload_vector(p_info, equitative)
    task_resource_workload = dict()
    for t_id in p_info.task_resources:
        total_r = 0
        for r_id in p_info.task_resources[t_id]:
            total_r += (2 * r_workloads[r_id])
        min_prob = 1 / total_r

        probability_list = list()
        r_alloc = list()
        for r_id in p_info.task_resources[t_id]:
            r_alloc.append(r_id)
            probability_list.append(min_prob * 2 if r_workloads[r_id] == 1 else min_prob)
        task_resource_workload[t_id] = Choice(r_alloc, probability_list)
    return task_resource_workload


def _get_workload_vector(p_info: ProcInfo, equitative):
    r_workloads = dict()
    pending = len(p_info.resource_tasks) if equitative else len(p_info.resource_tasks) // 2
    while pending > 0:
        for t_id in p_info.task_resources:
            for r_id in p_info.task_resources[t_id]:
                if r_id not in r_workloads:
                    r_workloads[r_id] = 1.0
                    pending -= 1
                    break
            if pending == 0:
                break
    for r_id in p_info.resource_tasks:
        if r_id not in r_workloads:
            r_workloads[r_id] = 0.5
    return r_workloads


def get_task_res_distsibution(p_info: ProcInfo):
    t_res_distributions = dict()
    for r_id in p_info.r_t_events:
        for t_id in p_info.r_t_events[r_id]:
            if t_id not in t_res_distributions:
                t_res_distributions[t_id] = dict()
            if r_id not in t_res_distributions[t_id]:
                t_res_distributions[t_id][r_id] = list()
            durations = list()
            for ev in p_info.r_t_events[r_id][t_id]:
                durations.append((ev.completed_at - ev.started_at).total_seconds())
            t_res_distributions[t_id][r_id] = best_fit_distribution(durations)
    return t_res_distributions


def parse_gateway_probabilities(gateways_prob):
    gateways_distribution = dict()
    for g_id in gateways_prob:
        probability_list = list()
        out_arc = list()

        fixed_prob = list()
        needed = 0.0
        exceeded = 0.0
        to_adjust = list()
        for f_arc in gateways_prob[g_id]:
            out_arc.append(f_arc)
            if gateways_prob[g_id][f_arc] < 0.1:
                fixed_prob.append(0.1)
                needed += 0.1 - gateways_prob[g_id][f_arc]
                to_adjust.append(False)
            else:
                fixed_prob.append(gateways_prob[g_id][f_arc])
                exceeded += 1
                to_adjust.append(True)
        if needed > 0:
            to_deduct = needed / exceeded
            while True:
                unchanged = True
                for i in range(0, len(fixed_prob)):
                    if to_adjust[i] and fixed_prob[i] - to_deduct < 0.1:
                        exceeded -= 1
                        unchanged = False
                        to_adjust[i] = False
                if unchanged:
                    break
                to_deduct = needed / exceeded

            for i in range(0, len(fixed_prob)):
                if to_adjust[i]:
                    fixed_prob[i] -= to_deduct

        for prob in fixed_prob:
            probability_list.append(prob)

        gateways_distribution[g_id] = Choice(out_arc, probability_list)

    return gateways_distribution


def generate_trace(p_case, bpmn_graph: BPMNGraph, task_res_distr: dict, task_res_allocation: dict, fix_res=None):
    p_state = ProcessState(bpmn_graph)

    bpmn_graph.last_datetime[bpmn_graph.starting_event][p_case] = None
    ev_queue = bpmn_graph.update_process_state(p_case, bpmn_graph.starting_event, p_state,
                                               CustomDatetimeAndSeconds(0, datetime.now()))[0]
    i = 0
    event_list = []
    while i < len(ev_queue):
        c_event = ev_queue[i].task_id
        t_name = bpmn_graph.element_info[c_event].name
        if fix_res is None:
            t_event = TaskEvent(p_case, t_name, task_res_allocation[t_name].next_resource())
        else:
            t_event = TaskEvent(p_case, t_name, fix_res)
        t_event.processing_time = generate_number_from(task_res_distr[t_name][t_event.resource_id]['distribution_name'],
                                                       task_res_distr[t_name][t_event.resource_id][
                                                           'distribution_params'])
        event_list.append(t_event)
        enabled_tasks = bpmn_graph.update_process_state(p_case, c_event, p_state,
                                                        CustomDatetimeAndSeconds(0, datetime.now()))[0]
        for ev in enabled_tasks:
            ev_queue.append(ev)
        i += 1
    return event_list


def _save_event_log(out_log_path, traces, is_fuzzy):
    if not is_fuzzy:
        _print_stats(traces)
    with open(out_log_path, mode='w', newline='', encoding='utf-8') as log_csv_file:
        f_writer = csv.writer(log_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # add_simulation_event_log_header(f_writer)
        log_writer = FileManager(10000, f_writer)
        for c_id in traces:
            for ev in traces[c_id]:
                log_writer.add_csv_row([ev.p_case, ev.task_id, '', ev.started_at, ev.completed_at, ev.resource_id])
        log_writer.force_write()


def _print_stats(traces):
    resource_events = dict()
    index = 0
    total_waiting = 0
    for t_id in traces:
        for ev in traces[t_id]:
            if ev.resource_id not in resource_events:
                resource_events[ev.resource_id] = list()
            resource_events[ev.resource_id].append((ev.started_at, 'start', index))
            resource_events[ev.resource_id].append((ev.completed_at, 'end', index))
            index += 1
            total_waiting += (ev.started_at - ev.enaled_at).total_seconds()
    print("Total tasks: " + str(index))
    print("Ave Waiting Time: " + str(total_waiting / index))
    print("................ Max Resource Multitasking ..................")

    resource_multitask_info = dict()

    for r_id in resource_events:
        resource_multitask_info[r_id] = 0
        resource_events[r_id].sort(key=lambda x: x[0])
        active_tasks = set()
        for time, event_type, index in resource_events[r_id]:
            if event_type == 'start':
                active_tasks.add(index)
                resource_multitask_info[r_id] = max(resource_multitask_info[r_id], len(active_tasks))
            else:
                active_tasks.remove(index)
        print(f"Resource {r_id}: {resource_multitask_info[r_id]}")
    print("-------------------------------------------------------------")


def add_simulation_event_log_header(log_fwriter):
    if log_fwriter:
        log_fwriter.writerow([
            'case_id', 'activity', 'enable_time', 'start_time', 'end_time', 'resource', ])
