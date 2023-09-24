import sys
import datetime
import pytz
from prosimos.control_flow_manager import BPMN

from pix_framework.discovery.resource_calendar_and_performance.crisp.resource_calendar import Interval

from prosimos.execution_info import TaskEvent, Trace
from prosimos.simulation_setup import SimDiffSetup


class KPIInfo:
    def __init__(self):
        self.min = sys.float_info.max
        self.max = 0
        self.avg = 0
        self.total = 0
        self.count = 0

    def set_values(self, min_val, max_val, avg_val, total=0, count=0):
        self.min = min_val
        self.max = max_val
        self.avg = avg_val
        self.total = total
        self.count = count

    def set_average(self):
        self.avg = self.total / self.count if self.count > 0 else 0

    def add_value(self, new_value):
        if new_value >= 0:
            self.min = min(self.min, new_value)
            self.max = max(self.max, new_value)
            self.total += new_value
            self.count += 1
            self.avg = self.total / self.count


class KPIMap:
    def __init__(self):
        self.cycle_time = KPIInfo()
        self.processing_time = KPIInfo()
        self.waiting_time = KPIInfo()

        self.idle_cycle_time = KPIInfo()
        self.idle_processing_time = KPIInfo()
        self.idle_time = KPIInfo()

        self.duration = KPIInfo()
        self.cost = KPIInfo()


class ResourceKPI:
    def __init__(self, r_profile, task_allocated, available_time, worked_time, utilization):
        self.r_profile = r_profile
        self.task_allocated = task_allocated
        self.available_time = available_time
        self.worked_time = worked_time
        self.utilization = utilization


class LogInfo:
    def __init__(self, sim_setup: SimDiffSetup):
        self.started_at = pytz.UTC.localize(datetime.datetime.max)
        self.ended_at = pytz.UTC.localize(datetime.datetime.min)
        self.trace_list = list()
        self.task_exec_info = dict()
        self.sim_setup = sim_setup

    def trace_info(self, p_case: int):
        return self.trace_list[p_case]

    def event_info(self, p_case: int, event_index: int):
        return self.trace_list[p_case].event_list[event_index]

    def add_event_info(self, p_case: int, event_info: TaskEvent, task_cost: float):
        trace_info = self.trace_list[p_case]
        trace_info.completed_at = max(trace_info.completed_at, event_info.completed_datetime)
        trace_info.event_list.append(event_info)
        self._update_global_task_stats(event_info, task_cost)

    def compute_execution_times(self, trace_info: Trace, process_kpi: KPIMap, model_type="CRISP"):
        processing_intervals = list()
        waiting_intervals = list()
        real_work_intervals = list()
        for event_info in trace_info.event_list:
            if (event_info.type == BPMN.TASK):
                r_calendar = self.sim_setup.calendars_map[self.sim_setup.resources_map[event_info.resource_id].calendar_id]

                if model_type == "FUZZY":
                    for interval in event_info.worked_intervals:
                        real_work_intervals.append(interval)
                else:
                    r_calendar.remove_idle_times(event_info.started_datetime, event_info.completed_datetime,
                                                 real_work_intervals)
            processing_intervals.append(Interval(event_info.started_datetime, event_info.completed_datetime))
            waiting_intervals.append(Interval(event_info.enabled_datetime, event_info.started_datetime))

        idle_cycle_time = (trace_info.completed_at - trace_info.started_at).total_seconds()
        idle_processing_time = sum_interval_union(processing_intervals)
        processing_time = sum_interval_union(real_work_intervals)
        waiting_time = sum_interval_union(waiting_intervals)
        idle_time = round(idle_processing_time - processing_time, 6)

        process_kpi.idle_cycle_time.add_value(idle_cycle_time)
        process_kpi.idle_processing_time.add_value(idle_processing_time)
        process_kpi.processing_time.add_value(processing_time)
        process_kpi.waiting_time.add_value(waiting_time)
        process_kpi.idle_time.add_value(idle_time)
        process_kpi.cycle_time.add_value(idle_cycle_time - idle_time)

    def _update_global_task_stats(self, event_info: TaskEvent, cost_per_hour: float):
        self.started_at = min(self.started_at, event_info.started_datetime)
        self.ended_at = max(self.ended_at, event_info.completed_datetime)
        task_cost = cost_per_hour * event_info.processing_time / 3600
        t_id = event_info.task_id

        if t_id not in self.task_exec_info:
            self.task_exec_info[t_id] = KPIMap()

        self.task_exec_info[t_id].waiting_time.add_value(event_info.waiting_time)
        self.task_exec_info[t_id].processing_time.add_value(event_info.processing_time)
        self.task_exec_info[t_id].idle_time.add_value(event_info.idle_time)
        self.task_exec_info[t_id].cycle_time.add_value(event_info.cycle_time)
        self.task_exec_info[t_id].idle_processing_time.add_value(event_info.idle_processing_time)
        self.task_exec_info[t_id].idle_cycle_time.add_value(event_info.idle_cycle_time)
        self.task_exec_info[t_id].cost.add_value(task_cost)

    def save_joint_statistics(self, bpm_env):
        self.save_start_end_dates(bpm_env.stat_fwriter)
        save_resource_utilization(bpm_env)
        self.compute_individual_task_stats(bpm_env.stat_fwriter)
        bpm_env.stat_fwriter.writerow([""])
        self.compute_full_simulation_statistics(bpm_env.stat_fwriter, bpm_env.sim_setup.model_type)

    def compute_process_kpi(self, bpm_env):
        process_kpi = KPIMap()
        for trace_info in self.trace_list:
            self.compute_execution_times(trace_info, process_kpi, bpm_env.sim_setup.model_type)

        return [process_kpi, self.task_exec_info, compute_resource_utilization(bpm_env), self.started_at, self.ended_at]

    def save_start_end_dates(self, stat_fwriter):
        stat_fwriter.writerow(["started_at", str(self.started_at)])
        stat_fwriter.writerow(["completed_at", str(self.ended_at)])
        stat_fwriter.writerow([""])

    def compute_individual_task_stats(self, stat_fwriter):
        stat_fwriter.writerow(['Individual Task Statistics'])
        stat_fwriter.writerow(['Name', 'Count', 'Min Duration', 'Max Duration', 'Avg Duration', 'Total Duration',
                               'Min Waiting Time', 'Max Waiting Time', 'Avg Waiting Time', 'Total Waiting Time',
                               'Min Processing Time', 'Max Processing Time', 'Avg Processing Time',
                               'Total Processing Time', 'Min Cycle Time', 'Max Cycle Time', 'Avg Cycle Time',
                               'Total Cycle Time', 'Min Idle Time', 'Max Idle Time', 'Avg Idle Time', 'Total Idle Time',
                               'Min Idle Cycle Time', 'Max Idle Cycle Time', 'Avg Idle Cycle Time',
                               'Total Idle Cycle Time', 'Min Idle Processing Time', 'Max Idle Processing Time',
                               'Avg Idle Processing Time', 'Total Idle Processing Time', 'Min Cost', 'Max Cost',
                               'Avg Cost', 'Total Cost'])
        for t_name in self.task_exec_info:
            t_info: KPIMap = self.task_exec_info[t_name]
            stat_fwriter.writerow([self.sim_setup.bpmn_graph.element_info[t_name].name,
                                   t_info.cycle_time.count, t_info.duration.min, t_info.duration.max,
                                   t_info.duration.avg, t_info.duration.total, t_info.waiting_time.min,
                                   t_info.waiting_time.max, t_info.waiting_time.avg, t_info.waiting_time.total,
                                   t_info.processing_time.min, t_info.processing_time.max, t_info.processing_time.avg,
                                   t_info.processing_time.total, t_info.cycle_time.min, t_info.cycle_time.max,
                                   t_info.cycle_time.avg, t_info.cycle_time.total, t_info.idle_time.min,
                                   t_info.idle_time.max, t_info.idle_time.avg, t_info.idle_time.total,
                                   t_info.idle_cycle_time.min, t_info.idle_cycle_time.max, t_info.idle_cycle_time.avg,
                                   t_info.idle_cycle_time.total, t_info.idle_processing_time.min,
                                   t_info.idle_processing_time.max, t_info.idle_processing_time.avg,
                                   t_info.idle_processing_time.total, t_info.cost.min, t_info.cost.max,
                                   t_info.cost.avg, t_info.cost.total])

    def compute_full_simulation_statistics(self, stat_fwriter, model_type):
        process_kpi = KPIMap()
        for trace_info in self.trace_list:
            self.compute_execution_times(trace_info, process_kpi, model_type)

        kpi_map = {"cycle_time": process_kpi.cycle_time,
                   "processing_time": process_kpi.processing_time,
                   "idle_cycle_time": process_kpi.idle_cycle_time,
                   "idle_processing_time": process_kpi.idle_processing_time,
                   "waiting_time": process_kpi.waiting_time,
                   "idle_time": process_kpi.idle_time}

        stat_fwriter.writerow(['Overall Scenario Statistics'])
        stat_fwriter.writerow(['KPI', 'Min', 'Max', 'Average', 'Accumulated Value', 'Trace Ocurrences'])
        for kpi_name in kpi_map:
            stat_fwriter.writerow([kpi_name,
                                   kpi_map[kpi_name].min,
                                   kpi_map[kpi_name].max,
                                   kpi_map[kpi_name].avg,
                                   kpi_map[kpi_name].total,
                                   len(self.trace_list)])


def compute_resource_utilization(bpm_env):
    if bpm_env.sim_setup.model_type == "FUZZY":
        compute_fuzzy_availability(bpm_env)
    else:
        compute_resorce_availability(bpm_env)
    resource_info = dict()

    available_time = dict()
    started_at = bpm_env.log_info.started_at
    completed_at = bpm_env.log_info.ended_at

    for r_id in bpm_env.sim_resources:
        # r_utilization = bpm_env.get_utilization_for(r_id)
        # r_info = bpm_env.sim_setup.resources_map[r_id]
        resource_info[r_id] = ResourceKPI(bpm_env.sim_setup.resources_map[r_id],
                                          bpm_env.sim_resources[r_id].allocated_tasks,
                                          bpm_env.sim_resources[r_id].worked_time,
                                          bpm_env.sim_resources[r_id].available_time,
                                          bpm_env.get_utilization_for(r_id))

    # for r_id in bpm_env.sim_setup.resources_map:
    #     calendar_info = bpm_env.sim_setup.get_resource_calendar(r_id)
    #     if calendar_info.calendar_id not in available_time:
    #         available_time[calendar_info.calendar_id] = calendar_info.find_working_time(started_at, completed_at)
    #     bpm_env.sim_resources[r_id].available_time = available_time[calendar_info.calendar_id]
    #
    #     resource_info[r_id] = ResourceKPI(bpm_env.sim_setup.resources_map[r_id],
    #                                       bpm_env.sim_resources[r_id].allocated_tasks,
    #                                       bpm_env.sim_resources[r_id].worked_time,
    #                                       bpm_env.sim_resources[r_id].available_time,
    #                                       bpm_env.get_utilization_for(r_id))
    return resource_info


def save_resource_utilization(bpm_env):
    stat_fwriter = bpm_env.stat_fwriter
    stat_fwriter.writerow(['Resource Utilization'])
    stat_fwriter.writerow(['Resource ID', 'Resource name', 'Utilization Ratio', 'Tasks Allocated',
                           'Worked Time (seconds)', 'Available Time (seconds)', 'Pool ID', 'Pool name'])

    compute_resorce_availability(bpm_env)

    for r_id in bpm_env.sim_resources:
        r_utilization = bpm_env.get_utilization_for(r_id)
        r_info = bpm_env.sim_setup.resources_map[r_id]
        stat_fwriter.writerow([r_id,
                               r_info.resource_name,
                               str(r_utilization),
                               bpm_env.sim_resources[r_id].allocated_tasks,
                               bpm_env.sim_resources[r_id].worked_time,
                               bpm_env.sim_resources[r_id].available_time,
                               r_info.pool_info.pool_id,
                               r_info.pool_info.pool_name])

    stat_fwriter.writerow([""])


def compute_resorce_availability(bpm_env):
    if bpm_env.sim_setup.model_type == "FUZZY":
        compute_fuzzy_availability(bpm_env)
    else:
        available_time = dict()
        started_at = bpm_env.log_info.started_at
        completed_at = bpm_env.log_info.ended_at
        for r_id in bpm_env.sim_setup.resources_map:
            calendar_info = bpm_env.sim_setup.get_resource_calendar(r_id)
            if calendar_info.calendar_id not in available_time:
                available_time[calendar_info.calendar_id] = calendar_info.find_working_time(started_at, completed_at)
            bpm_env.sim_resources[r_id].available_time = available_time[calendar_info.calendar_id]


def compute_fuzzy_availability(bpm_env):
    trace_list = bpm_env.log_info.trace_list
    r_working_intervals = dict()
    # Grouping the timeintervals each resource was working on the allocated tasks
    for trace in trace_list:
        for ev in trace.event_list:
            if ev.resource_id not in r_working_intervals:
                r_working_intervals[ev.resource_id] = []
            for interval in ev.worked_intervals:
                r_working_intervals[ev.resource_id].append(interval)

    #
    started_at = bpm_env.log_info.started_at
    completed_at = bpm_env.log_info.ended_at
    full_simulation_duration = (completed_at - started_at).total_seconds()

    # Sorting (by starting time) the working intervals of each resource to perform a sweep-line
    cumulative_worked_time = dict()
    cumulative_available_time = dict()
    for r_id in r_working_intervals:
        f_calendar = bpm_env.sim_setup.get_resource_calendar(r_id)
        i_len = len(r_working_intervals[r_id])
        if i_len == 0:
            cumulative_worked_time[r_id] = 0
            cumulative_available_time[r_id] = f_calendar.estimate_available_time(started_at, completed_at)
            bpm_env.sim_resources[r_id].available_time = cumulative_available_time[r_id]
            continue

        r_working_intervals[r_id].sort(key=lambda i_info: i_info.start)
        worked_time = r_working_intervals[r_id][0].duration

        available_time = f_calendar.estimate_available_time(started_at, r_working_intervals[r_id][0].start)
        available_time += r_working_intervals[r_id][0].duration
        available_time += f_calendar.estimate_available_time(r_working_intervals[r_id][i_len - 1].end, completed_at)

        last_date = r_working_intervals[r_id][0].end
        for i in range(1, i_len):
            c_interval = r_working_intervals[r_id][i]
            if last_date > c_interval.end:
                continue
            if last_date < c_interval.start:
                available_time += f_calendar.estimate_available_time(last_date, c_interval.start)
            if last_date > c_interval.start:
                i_duration = (c_interval.end - last_date).total_seconds()
                worked_time += i_duration
                available_time += i_duration
            else:
                worked_time += c_interval.duration
                available_time += c_interval.duration
            last_date = c_interval.end
        cumulative_worked_time[r_id] = worked_time
        cumulative_available_time[r_id] = available_time
        bpm_env.sim_resources[r_id].available_time = available_time


def update_min_max(trace_info, duration_array, case_duration):
    duration_array[0] = case_duration
    duration_array[1] = trace_info.started_at
    duration_array[2] = trace_info.completed_at
    duration_array[3] = "Case %d:" % trace_info.p_case


def print_event_state(state, e_step, bpm_env, resource_id, to_print):
    print("(%d) - %s %s at - %s by %s --- %d %d" % (e_step.trace_info.p_case,
                                                    bpm_env.sim_setup.bpmn_graph.element_info[e_step.task_id].name,
                                                    state,
                                                    str(bpm_env.current_simulation_date()),
                                                    bpm_env.sim_setup.name_from_id(resource_id),
                                                    to_print, bpm_env.simpy_env.now))


def sum_interval_union(interval_list):
    if interval_list is None or len(interval_list) == 0:
        return 0
    interval_list = sorted(interval_list, key=lambda interval: interval.start)
    t_duration = interval_list[0].duration
    for i in range(1, len(interval_list)):
        i_interval = interval_list[i - 1].intersection(interval_list[i])
        for j in range(i - 2, -1, -1):
            intersection = interval_list[j].intersection(i_interval)
            if intersection is None:
                break
            i_interval = intersection
        t_duration += interval_list[i].duration - i_interval.duration if i_interval else interval_list[i].duration
    return round(t_duration, 6)


def _compute_times(trace_info, env, event_index, with_idle):
    duration = trace_info.event_list[event_index].idle_processing_time() if with_idle else \
        trace_info.event_list[event_index].processing_time()
    yield env.timeout(duration)
    for next_task in trace_info.next_parallel_tasks[event_index]:
        env.process(_compute_times(trace_info, env, next_task, with_idle))
