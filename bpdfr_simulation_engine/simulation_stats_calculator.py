import sys
import datetime
from datetime import timedelta
import pytz

import simpy
from bpdfr_simulation_engine.resource_calendar import Interval

from bpdfr_simulation_engine.execution_info import TaskEvent, Trace
from bpdfr_simulation_engine.resource_profile import ResourceProfile


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


class LogInfo:
    def __init__(self, sim_setup):
        self.started_at = pytz.UTC.localize(datetime.datetime.max)
        self.ended_at = pytz.UTC.localize(datetime.datetime.min)
        self.trace_list = list()
        self.last_post_processed_trace = 0
        self.task_exec_info = dict()
        self.sim_setup = sim_setup

    def compute_execution_times(self, trace_info, process_kpi):
        processing_intervals = list()
        waiting_intervals = list()
        real_work_intervals = list()
        for event_info in trace_info.event_list:
            r_calendar = self.sim_setup.calendars_map[self.sim_setup.resources_map[event_info.resource_id].calendar_id]
            r_calendar.remove_idle_times(event_info.started_at, event_info.completed_at, real_work_intervals)
            processing_intervals.append(Interval(event_info.started_at, event_info.completed_at))
            waiting_intervals.append(Interval(event_info.enabled_at, event_info.started_at))

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

        # These conditional are for debugging, remove after testing all the models
        if idle_time < 0:
            print('===========================================================')
            for xx in processing_intervals:
                print("%s -> %s (%f)" % (str(xx.start), str(xx.end), xx.duration))
            print('===========================================================')
        if idle_cycle_time != round(idle_processing_time + waiting_time, 6):
            calc = idle_processing_time + waiting_time
            print('trace_duration %s - %s idle_cycle_time (calculated)' % (idle_cycle_time, calc))

    def register_completed_task(self, event_info: TaskEvent, res_prof: ResourceProfile):
        self.started_at = min(self.started_at, event_info.started_at)
        self.ended_at = max(self.ended_at, event_info.completed_at)
        task_duration = (event_info.completed_at - event_info.started_at).total_seconds()
        waiting_time, processing_time = event_info.waiting_time(), event_info.processing_time()
        task_cost = res_prof.cost_per_hour * processing_time / 3600
        t_name = event_info.task_name

        if t_name not in self.task_exec_info:
            self.task_exec_info[t_name] = KPIMap()

        self.task_exec_info[t_name].duration.add_value(task_duration)
        self.task_exec_info[t_name].waiting_time.add_value(event_info.waiting_time())
        self.task_exec_info[t_name].processing_time.add_value(event_info.processing_time())
        self.task_exec_info[t_name].idle_time.add_value(event_info.idle_time)
        self.task_exec_info[t_name].cycle_time.add_value(event_info.cycle_time())
        self.task_exec_info[t_name].idle_processing_time.add_value(event_info.idle_processing_time())
        self.task_exec_info[t_name].idle_cycle_time.add_value(event_info.idle_cycle_time())
        self.task_exec_info[t_name].cost.add_value(task_cost)

    def post_process_completed_trace(self, process_kpi):
        if self.last_post_processed_trace < len(self.trace_list):
            self.compute_execution_times(self.trace_list[self.last_post_processed_trace], process_kpi)
            self.last_post_processed_trace += 1
            return True
        return False

    def save_joint_statistics(self, bpm_env):
        self.save_start_end_dates(bpm_env.stat_fwriter)
        compute_resource_utilization(bpm_env)
        self.compute_individual_task_stats(bpm_env.stat_fwriter)
        bpm_env.stat_fwriter.writerow([""])
        self.compute_full_simulation_statistics(bpm_env.stat_fwriter)

    def save_start_end_dates(self, stat_fwriter):
        stat_fwriter.writerow(["started_at", str(self.started_at)])
        stat_fwriter.writerow(["completed_at", str(self.ended_at)])
        stat_fwriter.writerow([""])

    def compute_individual_task_stats(self, stat_fwriter):
        stat_fwriter.writerow(['Individual Task Statistics'])
        stat_fwriter.writerow(['Name', 'Count', 'Min Duration', 'Max Duration', 'Avg Duration', 'Total Duration',
                               'Min Waiting Time', 'Max Waiting Time', 'Ave Waiting Time', 'Total Waiting Time',
                               'Min Processing Time', 'Max Processing Time', 'Ave Processing Time',
                               'Total Processing Time', 'Min Cycle Time', 'Max Cycle Time', 'Ave Cycle Time',
                               'Total Cycle Time', 'Min Idle Time', 'Max Idle Time', 'Ave Idle Time', 'Total Idle Time',
                               'Min Idle Cycle Time', 'Max Idle Cycle Time', 'Ave Idle Cycle Time',
                               'Total Idle Cycle Time', 'Min Idle Processing Time', 'Max Idle Processing Time',
                               'Ave Idle Processing Time', 'Total Idle Processing Time', 'Min Cost', 'Max Cost',
                               'Ave Cost', 'Total Cost'])
        for t_name in self.task_exec_info:
            t_info: KPIMap = self.task_exec_info[t_name]
            stat_fwriter.writerow([t_name,
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

    def compute_full_simulation_statistics(self, stat_fwriter):
        process_kpi = KPIMap()
        while self.post_process_completed_trace(process_kpi):
            continue

        kpi_map = {"cycle_time": process_kpi.cycle_time,
                   "processing_time": process_kpi.processing_time,
                   "idle_cycle_time": process_kpi.idle_cycle_time,
                   "idle_processing_time": process_kpi.idle_processing_time,
                   "waiting_time": process_kpi.waiting_time,
                   "idle_time": process_kpi.idle_time}

        stat_fwriter.writerow(['Overall Scenario Statistics'])
        stat_fwriter.writerow(['KPI', 'Min', 'Max', 'Average', 'Accumulated Value', 'Trace Ocurrences'])
        for kpi_name in kpi_map:
            # print("%s:      %s" % (kpi_name, str(timedelta(seconds=(kpi_average[kpi_name])))))
            stat_fwriter.writerow([kpi_name,
                                   kpi_map[kpi_name].min,
                                   kpi_map[kpi_name].max,
                                   kpi_map[kpi_name].avg,
                                   kpi_map[kpi_name].total,
                                   len(self.trace_list)])


def compute_resource_utilization(bpm_env):
    stat_fwriter = bpm_env.stat_fwriter
    stat_fwriter.writerow(['Resource Utilization'])
    stat_fwriter.writerow(['Resource ID', 'Resource name', 'Utilization Ratio', 'Tasks Allocated',
                           'Worked Time (seconds)', 'Available Time (seconds)', 'Pool ID', 'Pool name'])

    available_time = dict()
    started_at = bpm_env.simulation_started_at()
    completed_at = bpm_env.simulation_completed_at()
    for r_id in bpm_env.sim_setup.resources_map:
        calendar_info = bpm_env.sim_setup.get_resource_calendar(r_id)
        if calendar_info.calendar_id not in available_time:
            available_time[calendar_info.calendar_id] = calendar_info.find_working_time(started_at, completed_at)
        bpm_env.sim_resources[r_id].available_time = available_time[calendar_info.calendar_id]

    for r_id in bpm_env.sim_resources:
        r_utilization = bpm_env.get_utilization_for(r_id)
        r_info = bpm_env.sim_setup.resources_map[r_id]
        stat_fwriter.writerow([r_id,
                               r_info.resource_name,
                               str(r_utilization),
                               bpm_env.resource_total_allocated_tasks[r_id],
                               bpm_env.sim_resources[r_id].worked_time,
                               bpm_env.sim_resources[r_id].available_time,
                               r_info.pool_info.pool_id,
                               r_info.pool_info.pool_name])

        # if r_utilization > 0:
        #     print("Ideal: %s" % str(datetime.timedelta(seconds=bpm_env.real_duration[r_id])))
        #     print("Sum:   %s" % str(datetime.timedelta(seconds=bpm_env.sim_resources[r_id].worked_time)))
        #     print("Full:  %s" % str(datetime.timedelta(seconds=bpm_env.sim_resources[r_id].available_time)))
        #     print("%s -> Utilization: %f" % (r_id, r_utilization))
        #     print('-------------------------------------------')
    stat_fwriter.writerow([""])


def update_min_max(trace_info, duration_array, case_duration):
    duration_array[0] = case_duration
    duration_array[1] = trace_info.started_at
    duration_array[2] = trace_info.completed_at
    duration_array[3] = "Case %d:" % trace_info.p_case


def print_event_state(state, e_step, bpm_env, resource_id):
    if bpm_env.sim_setup.bpmn_graph.element_info[e_step.task_id].name == 'Check credit history':
        print("(%d) - %s %s at - %s by %s" % (e_step.trace_info.p_case,
                                              bpm_env.sim_setup.bpmn_graph.element_info[e_step.task_id].name, state,
                                              str(bpm_env.current_simulation_date()), resource_id))


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


def compute_execution_times_by_simulation(trace_info):
    trace_info.idle_cycle_time = (trace_info.completed_at - trace_info.started_at).total_seconds()
    for with_idle in [True, False]:
        env = simpy.Environment()
        env.process(_compute_times(trace_info, env, 0, with_idle))
        env.run()
        if with_idle:
            trace_info.idle_processing_time = env.now
        else:
            trace_info.processing_time = env.now
    trace_info.idle_time = trace_info.idle_processing_time - trace_info.processing_time
    trace_info.cycle_time = trace_info.idle_cycle_time - trace_info.idle_time
    trace_info.waiting_time = trace_info.cycle_time - trace_info.processing_time


def _compute_times(trace_info, env, event_index, with_idle):
    duration = trace_info.event_list[event_index].idle_processing_time() if with_idle else \
        trace_info.event_list[event_index].processing_time()
    yield env.timeout(duration)
    for next_task in trace_info.next_parallel_tasks[event_index]:
        env.process(_compute_times(trace_info, env, next_task, with_idle))
