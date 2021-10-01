import sys
import datetime
from datetime import timedelta

import simpy

from bpdfr_simulation_engine.execution_info import TaskEvent
from bpdfr_simulation_engine.resource_profile import ResourceProfile


class LogInfo:
    def __init__(self):
        self.trace_list = list()
        self.last_post_processed_trace = 0
        self.task_exec_info = dict()

    def register_completed_task(self, event_info: TaskEvent, res_prof: ResourceProfile):
        task_duration = (event_info.completed_at - event_info.started_at).total_seconds()
        waiting_time, processing_time = event_info.waiting_time(), event_info.processing_time()
        cycle_time, idle_time = event_info.cycle_time(), event_info.idle_time
        task_cost = res_prof.cost_per_hour * processing_time / 3600
        t_id = event_info.task_id
        if event_info.task_id not in self.task_exec_info:
            self.task_exec_info[t_id] = {
                'name': event_info.task_name,
                'count': 1,
                'total_duration': task_duration,
                'min_duration': task_duration,
                'max_duration': task_duration,
                'total_wait_time': waiting_time,
                'min_wait_time': waiting_time,
                'max_wait_time': waiting_time,
                'total_processing_time': processing_time,
                'min_processing_time': processing_time,
                'max_processing_time': processing_time,
                'total_cycle_time': cycle_time,
                'min_cycle_time': cycle_time,
                'max_cycle_time': cycle_time,
                'total_idle_time': idle_time,
                'min_idle_time': idle_time,
                'max_idle_time': idle_time,
                'total_cost': task_cost,
                'min_cost': task_cost,
                'max_cost': task_cost
            }
        else:
            self.task_exec_info[t_id]['count'] += 1
            self.task_exec_info[t_id]['total_duration'] += task_duration
            self.task_exec_info[t_id]['min_duration'] += min(task_duration, self.task_exec_info[t_id]['min_duration'])
            self.task_exec_info[t_id]['max_duration'] += max(task_duration, self.task_exec_info[t_id]['max_duration'])
            self.task_exec_info[t_id]['total_wait_time'] += waiting_time
            self.task_exec_info[t_id]['min_wait_time'] += min(waiting_time, self.task_exec_info[t_id]['min_wait_time'])
            self.task_exec_info[t_id]['max_wait_time'] += max(waiting_time, self.task_exec_info[t_id]['max_wait_time'])
            self.task_exec_info[t_id]['total_processing_time'] += processing_time
            self.task_exec_info[t_id]['min_processing_time'] += min(processing_time,
                                                                    self.task_exec_info[t_id]['min_processing_time'])
            self.task_exec_info[t_id]['max_processing_time'] += max(processing_time,
                                                                    self.task_exec_info[t_id]['max_processing_time'])
            self.task_exec_info[t_id]['total_cycle_time'] += cycle_time
            self.task_exec_info[t_id]['min_cycle_time'] += min(cycle_time, self.task_exec_info[t_id]['min_cycle_time'])
            self.task_exec_info[t_id]['max_cycle_time'] += max(cycle_time, self.task_exec_info[t_id]['max_cycle_time'])
            self.task_exec_info[t_id]['total_idle_time'] += idle_time
            self.task_exec_info[t_id]['min_idle_time'] += min(idle_time, self.task_exec_info[t_id]['min_idle_time'])
            self.task_exec_info[t_id]['max_idle_time'] += max(idle_time, self.task_exec_info[t_id]['max_idle_time'])
            self.task_exec_info[t_id]['total_cost'] += task_cost
            self.task_exec_info[t_id]['min_cost'] += min(task_cost, self.task_exec_info[t_id]['min_cost'])
            self.task_exec_info[t_id]['max_cost'] += max(task_cost, self.task_exec_info[t_id]['max_cost'])

    def post_process_completed_trace(self):
        if self.last_post_processed_trace < len(self.trace_list):
            compute_execution_times(self.trace_list[self.last_post_processed_trace])
            self.last_post_processed_trace += 1
            return True
        return False

    def save_joint_statistics(self, bpm_env):
        compute_resource_utilization(bpm_env)
        bpm_env.stat_fwriter.writerow([""])
        self.compute_individual_task_stats(bpm_env.stat_fwriter)
        bpm_env.stat_fwriter.writerow([""])
        self.compute_full_simulation_statistics(bpm_env.stat_fwriter)

    def compute_individual_task_stats(self, stat_fwriter):
        stat_fwriter.writerow(['Individual Task Statistics'])
        stat_fwriter.writerow(['Name', 'Count', 'Avg Duration', 'Min Duration', 'Max Duration', 'Ave Waiting Time',
                               'Min Waiting Time', 'Max Waiting Time', 'Ave Processing Time', 'Min Processing Time',
                               'Max Processing Time', 'Ave Cycle Time', 'Min Cycle Time', 'Max Cycle Time',
                               'Ave Idle Time', 'Min Idle Time', 'Max Idle Time', 'Ave Cost', 'Min Cost', 'Max Cost'])
        for t_id in self.task_exec_info:
            count = self.task_exec_info[t_id]['count']
            stat_fwriter.writerow([self.task_exec_info[t_id]['name'],
                                   count,
                                   self.task_exec_info[t_id]['total_duration'] / count,
                                   self.task_exec_info[t_id]['min_duration'],
                                   self.task_exec_info[t_id]['max_duration'],
                                   self.task_exec_info[t_id]['total_wait_time'] / count,
                                   self.task_exec_info[t_id]['min_wait_time'],
                                   self.task_exec_info[t_id]['max_wait_time'],
                                   self.task_exec_info[t_id]['total_processing_time'] / count,
                                   self.task_exec_info[t_id]['min_processing_time'],
                                   self.task_exec_info[t_id]['max_processing_time'],
                                   self.task_exec_info[t_id]['total_cycle_time'] / count,
                                   self.task_exec_info[t_id]['min_cycle_time'],
                                   self.task_exec_info[t_id]['max_cycle_time'],
                                   self.task_exec_info[t_id]['total_idle_time'] / count,
                                   self.task_exec_info[t_id]['min_idle_time'],
                                   self.task_exec_info[t_id]['max_idle_time'],
                                   self.task_exec_info[t_id]['total_cost'] / count,
                                   self.task_exec_info[t_id]['min_cost'],
                                   self.task_exec_info[t_id]['max_cost']])

    def compute_full_simulation_statistics(self, stat_fwriter):
        while self.post_process_completed_trace():
            continue

        kpi_map = {"cycle_times": {"min": sys.float_info.max, "max": 0, "total": 0},
                   "processing_times": {"min": sys.float_info.max, "max": 0, "total": 0},
                   "idle_cycle_times": {"min": sys.float_info.max, "max": 0, "total": 0},
                   "idle_processing_times": {"min": sys.float_info.max, "max": 0, "total": 0},
                   "waiting_time": {"min": sys.float_info.max, "max": 0, "total": 0},
                   "idle_time": {"min": sys.float_info.max, "max": 0, "total": 0}}

        for trace_info in self.trace_list:
            add_time_duration(kpi_map["cycle_times"], trace_info.cycle_time)
            add_time_duration(kpi_map["idle_cycle_times"], trace_info.idle_cycle_time)
            add_time_duration(kpi_map["processing_times"], trace_info.processing_time)
            add_time_duration(kpi_map["idle_processing_times"], trace_info.idle_processing_time)
            add_time_duration(kpi_map["waiting_time"], trace_info.waiting_time)
            add_time_duration(kpi_map["idle_time"], trace_info.idle_time)

        kpi_average = {kpi: val["total"] / len(self.trace_list) for kpi, val in kpi_map.items()}

        stat_fwriter.writerow(['Overall Scenario Statistics'])
        stat_fwriter.writerow(['KPI', 'Average', 'Min', 'Max'])
        for kpi_name in kpi_map:
            print("%s:      %s" % (kpi_name, str(timedelta(seconds=(kpi_average[kpi_name])))))
            stat_fwriter.writerow([kpi_name, kpi_average[kpi_name], kpi_map[kpi_name]['min'], kpi_map[kpi_name]["max"]])


def add_time_duration(time_map, new_value):
    time_map["total"] += new_value
    if time_map["min"] > new_value:
        time_map["min"] = new_value
    if time_map["max"] < new_value:
        time_map["max"] = new_value


def compute_resource_utilization(bpm_env):
    stat_fwriter = bpm_env.stat_fwriter
    stat_fwriter.writerow(['Resource Utilization'])
    stat_fwriter.writerow(['Resource', 'Utilization Ratio'])

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
        stat_fwriter.writerow([r_id, str(r_utilization)])

        # if r_utilization > 0:
        #     print("Ideal: %s" % str(datetime.timedelta(seconds=bpm_env.real_duration[r_id])))
        #     print("Sum:   %s" % str(datetime.timedelta(seconds=bpm_env.sim_resources[r_id].worked_time)))
        #     print("Full:  %s" % str(datetime.timedelta(seconds=bpm_env.sim_resources[r_id].available_time)))
        #     print("%s -> Utilization: %f" % (r_id, r_utilization))
        #     print('-------------------------------------------')


def update_min_max(trace_info, duration_array, case_duration):
    duration_array[0] = case_duration
    duration_array[1] = trace_info.started_at
    duration_array[2] = trace_info.completed_at
    duration_array[3] = "Case %d:" % trace_info.p_case


def print_event_state(state, e_step, bpm_env, resource_id):
    print("(%d) - %s %s at - %s by %s" % (e_step.trace_info.p_case,
                                          bpm_env.sim_setup.bpmn_graph.element_info[e_step.task_id].name, state,
                                          str(bpm_env.current_simulation_date()), resource_id))


def compute_execution_times(trace_info):
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
