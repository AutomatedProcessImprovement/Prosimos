import sys

from bpdfr_discovery.log_parser import event_list_from_xes_log, event_list_from_csv
from prosimos.execution_info import Trace, TaskEvent
from prosimos.simulation_properties_parser import parse_simulation_model

import numpy as np
from scipy.optimize import linear_sum_assignment


def log_dl_distance(real_log_path, simulated_log_path, bpmn_path):
    max_waiting = dict()
    max_processing = dict()
    bpmn_graph = parse_simulation_model(bpmn_path)

    real_log_traces = sort_event_log(event_list_from_xes_log(real_log_path), bpmn_graph, max_waiting, max_processing)
    simulated_traces = sort_event_log(event_list_from_csv(simulated_log_path), bpmn_graph, max_waiting, max_processing)

    normalize_times(real_log_traces, max_waiting, max_processing)
    normalize_times(simulated_traces, max_waiting, max_processing)

    parralel_rel = compute_paralell_relations(real_log_traces, max_waiting)

    bptd_graph = list()
    for r_trace in real_log_traces:
        dl_costs = list()
        for s_trace in simulated_traces:
            dl_costs.append(trace_dl_distance(r_trace, s_trace, parralel_rel))
        bptd_graph.append(dl_costs)

    return linear_sum_assignment(np.array(bptd_graph)).sum() / len(real_log_traces)


def sort_event_log(log_path, bpmn_graph, max_waiting, max_processing):
    trace_list = event_list_from_xes_log(log_path)
    for trace in trace_list:
        trace.sort_by_completion_date(True)
        compute_enabling_processing_times(trace, bpmn_graph, max_waiting, max_processing)
    return trace_list


def compute_enabling_processing_times(trace_info, bpmn_graph, max_waiting, max_processing):
    flow_arcs_frequency = dict()
    task_sequence = list()
    for ev_info in trace_info.event_list:
        if ev_info.task_id not in max_waiting:
            max_waiting[ev_info.task_id] = 0
            max_processing[ev_info.task_id] = 0
        task_sequence.append(ev_info.task_id)

    _, _, _, enabling_times = bpmn_graph.reply_trace(task_sequence, flow_arcs_frequency, True, trace_info.event_list)
    for i in range(0, len(enabling_times)):
        ev_info = trace_info.event_list[i]
        if ev_info.started_at < enabling_times[i]:
            fix_enablement_from_incorrect_models(i, enabling_times, trace_info.event_list)
        ev_info.update_enabling_times(enabling_times[i])
        max_waiting[ev_info.task_id] = max(max_waiting[ev_info.task_id], ev_info.waiting_time)
        max_processing[ev_info.task_id] = max(max_processing[ev_info.task_id], ev_info.waiting_time)


def fix_enablement_from_incorrect_models(from_i: int, task_enablement: list, trace: list):
    started_at = trace[from_i].started_at
    enabled_at = task_enablement[from_i]
    i = from_i
    while i > 0:
        i -= 1
        if enabled_at == trace[i].completed_at:
            task_enablement[from_i] = started_at
            return True
    return False


def normalize_times(trace_list, max_waiting, max_processing):
    for t_info in trace_list:
        event_list = t_info.event_list
        for e_info in event_list:
            e_info.normalized_waiting = e_info.waiting_time / max_waiting[e_info.task_id]
            e_info.normalized_processing = e_info.processing_time / max_processing[e_info.task_id]


def compute_paralell_relations(trace_list, tasks_map):
    dependency_matrix = dict()
    parallel_rel = dict()
    for i_task in tasks_map:
        dependency_matrix[i_task] = dict()
        parallel_rel[i_task] = dict()
        for j_task in tasks_map:
            dependency_matrix[i_task][j_task] = 0
            parallel_rel[i_task][j_task] = False

    for t_info in trace_list:
        event_list = t_info.event_list
        for i in range(1, len(event_list)):
            dependency_matrix[event_list[i - 1].task_id][event_list[i].task_id] += 1

    for i_task in dependency_matrix:
        for j_task in dependency_matrix[i_task]:
            f_sum = dependency_matrix[i_task][j_task] + dependency_matrix[j_task][i_task]
            if f_sum > 0 and 0.3 <= dependency_matrix[i_task][j_task] / f_sum <= 0.7:
                parallel_rel[i_task][j_task] = True

    return parallel_rel


def trace_dl_distance(trace_1: Trace, trace_2: Trace, parallel_rel):
    return dl_distance(trace_1.event_list,
                       trace_2.event_list,
                       len(trace_1.event_list) - 1,
                       len(trace_2.event_list) - 1,
                       parallel_rel) / max(len(trace_1.event_list), len(trace_2.event_list))


def dl_distance(evs_i, evs_j, i, j, parallel_r):
    if i == j == 0:
        return 0
    dist = sys.float_info.max
    if i > 0:
        dist = min(dist, dl_distance(evs_i, evs_j, i - 1, j, parallel_r) + penalty(evs_i[i], evs_j[j], parallel_r))
    if j > 0:
        dist = min(dist, dl_distance(evs_i, evs_j, i, j - 1, parallel_r) + penalty(evs_i[i], evs_j[j], parallel_r))
    if i > 0 and j > 0:
        dist = min(dist, dl_distance(evs_i, evs_j, i - 1, j - 1, parallel_r) + penalty(evs_i[i], evs_j[j], parallel_r))
    if i > 1 and j > 1 and evs_i[i].task_id == evs_j[j - 1].task_id and evs_i[i - 1].task_id == evs_j[j].task_id:
        dist = min(dist, dl_distance(evs_i, evs_j, i - 2, j - 2, parallel_r) + penalty(evs_i[i], evs_j[j], parallel_r))
    return dist


def penalty(ev_1: TaskEvent, ev_2: TaskEvent, parallel_rel, weight=0.5):
    if ev_1.task_id == ev_2.task_id or parallel_rel[ev_1.task_id][ev_2.task_id]:
        return weight * abs(ev_2.normalized_processing - ev_1.normalized_processing) \
               + (1 - weight) * abs(ev_2.normalized_waiting - ev_1.normalized_waiting)
    return 1
