import math
import os
import csv

import pytz
import sys
from enum import Enum
from datetime import datetime, timedelta

import pandas as pd

from bpdfr_discovery.log_parser import event_list_from_csv, preprocess_xes_log
from prosimos.file_manager import FileManager
from prosimos.simulation_engine import run_simulation, SimBPMEnv
from prosimos.simulation_stats import SimulationResult
from prosimos.simulation_stats_calculator import KPIMap
from fuzzy_engine.event_log_analyser import get_starting_datetimes
from log_distance_measures.circadian_event_distribution import circadian_event_distribution_distance
from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance
from log_distance_measures.relative_event_distribution import relative_event_distribution_distance
from testing_scripts.fuzzy_scripts.fuzzy_discovery_script import split_event_log, discover_model_from_csv_log, \
    transform_log_datetimes_to_utc, localize_datetimes
from testing_scripts.fuzzy_scripts.fuzzy_test_files import test_processes, get_file_path, \
    FileType, crisp_discovery_params, is_syntetic

from log_distance_measures.absolute_event_distribution import absolute_event_distribution_distance
from log_distance_measures.config import AbsoluteTimestampType, DEFAULT_CSV_IDS, discretize_to_hour, EventLogIDs, \
    DistanceMetric
from testing_scripts.fuzzy_scripts.syntetic_logs_generator import generate_syntetic_log


class SimulationType(Enum):
    CRISP_CALENDAR = 1
    FUZZY_CALENDAR = 2


class Steps(Enum):
    SPLIT_LOG = 0
    FUZZY_DISCOVERY = 1
    FUZZY_SIMULATION = 2
    TO_UTC = 3
    CRISP_DISCOVERY = 4
    CRISP_SIMULATION = 5
    METRICS = 6
    SYNTETIC_LOG_GEN = 7
    BAYESIAN_OPTIMIZER = 8
    LOCALIZE = 9
    LOG_INFO = 10


granule_sizes = [60]
trapezoidal_angles = [0, 0.25, 0.5, 0.75, 1.0]

active_steps = {
    Steps.LOG_INFO: False,
    Steps.TO_UTC: False,
    Steps.LOCALIZE: False,
    Steps.SYNTETIC_LOG_GEN: False,
    Steps.SPLIT_LOG: False,
    Steps.FUZZY_DISCOVERY: False,
    Steps.FUZZY_SIMULATION: False,
    Steps.CRISP_DISCOVERY: False,
    Steps.CRISP_SIMULATION: False,
    Steps.METRICS: False
}

print_stats = False


def main():
    for i in range(0, len(test_processes)):
        proc_name = test_processes[i]
        synthetic = is_syntetic[proc_name]

        if active_steps[Steps.LOG_INFO]:
            # get_log_info()
            print_log_sequences(proc_name)

        if active_steps[Steps.LOCALIZE]:
            localize_datetimes(get_file_path(proc_name, FileType.TESTING_CSV_LOG))

        if active_steps[Steps.SYNTETIC_LOG_GEN]:
            generate_syntetic_log(proc_name=proc_name, total_cases=2000)

        if active_steps[Steps.TO_UTC]:
            if synthetic:
                transform_log_datetimes_to_utc(get_file_path(proc_name, FileType.GENERATOR_LOG))
            else:
                # transform_log_datetimes_to_utc(get_file_path(proc_name, FileType.ORIGINAL_CSV_LOG))
                transform_log_datetimes_to_utc(get_file_path(proc_name, FileType.TRAINING_CSV_LOG))
                transform_log_datetimes_to_utc(get_file_path(proc_name, FileType.TESTING_CSV_LOG))

        if active_steps[Steps.SPLIT_LOG]:
            if synthetic:
                split_synthetic_log(proc_name)
            else:
                split_event_log(get_file_path(proc_name, FileType.ORIGINAL_CSV_LOG),
                                get_file_path(proc_name, FileType.TRAINING_CSV_LOG),
                                get_file_path(proc_name, FileType.TESTING_CSV_LOG),
                                0.5)

        if active_steps[Steps.CRISP_DISCOVERY]:
            discover_crisp_calendars(proc_name)

        if active_steps[Steps.CRISP_SIMULATION]:
            simulate_and_save_crisp_model(proc_name, 5)

        if active_steps[Steps.FUZZY_DISCOVERY]:
            discover_fuzzy_parameters(proc_name)

        if active_steps[Steps.FUZZY_SIMULATION]:
            simulate_and_save_results(proc_name, 5)

        if active_steps[Steps.METRICS]:
            compute_log_distance_metric(proc_name, 5)

        break

    # simulate_process(process_files['loan_SC_LU'], SimulationType.FUZZY_CALENDAR, 5)
    os._exit(0)


def get_log_info():
    for i in test_processes:
        proc_name = test_processes[i]
        if i == 3:
            for even in [True, False]:
                for c_type in range(1, 5):
                    print("Process: %s, Calendar Type: %d, Balanced: %s" % (proc_name, c_type, str(even)))
                    file_path = get_file_path(proc_name, FileType.ORIGINAL_CSV_LOG, 60, 0, 1, c_type, even)
                    print_log_info(file_path)
        elif i == 6:
            print_joint_train_test(proc_name)
        else:
            print("Process: %s" % proc_name)
            print_log_info(get_file_path(proc_name, FileType.ORIGINAL_CSV_LOG, 60, 0, 1, 1, True))


def print_joint_train_test(proc_name):
    count_traces = 0
    count_events = 0
    resources = set()
    activities = set()
    evt_x_trace = 0

    for is_train in [True, False]:
        f_type = FileType.TRAINING_CSV_LOG if is_train else FileType.TESTING_CSV_LOG
        file_path = get_file_path(proc_name, f_type, 60, 0, 1, 1, True)
        traces = event_list_from_csv(file_path)
        count_traces += len(traces)
        for trace in traces:
            evt_x_trace += len(trace.event_list)
            for ev in trace.event_list:
                count_events += 1
                resources.add(ev.resource_id)
                activities.add(ev.task_id)
    print("Traces: %d" % count_traces)
    print("Total Events: %d" % count_events)
    print("Ave Events per Trace: %f" % (evt_x_trace / count_traces))
    print("Total Activities: %d" % len(activities))
    print("Total Resources: %d" % len(resources))
    print('----------------------------------------------------------')


def print_log_info(file_path):
    traces = event_list_from_csv(file_path)
    print("Traces: %d" % len(traces))
    count_events = 0
    evt_x_trace = 0
    resources = set()
    activities = set()
    for trace in traces:
        evt_x_trace += len(trace.event_list)
        for ev in trace.event_list:
            count_events += 1
            resources.add(ev.resource_id)
            activities.add(ev.task_id)
    print("Total Events: %d" % count_events)
    print("Ave Events per Trace: %f" % (evt_x_trace / len(traces)))
    print("Total Activities: %d" % len(activities))
    print("Total Resources: %d" % len(resources))
    print('----------------------------------------------------------')


def print_log_sequences(proc_name):
    file_path = get_file_path(proc_name, FileType.TESTING_CSV_LOG, 60, 0, 1)
    traces = event_list_from_csv(file_path)
    t_map = dict()
    for trace in traces:
        t_str = ""
        for ev in trace.event_list:
            t_str += (str(ev.task_id) + ",")
        if t_str not in t_map:
            t_map[t_str] = 0
        t_map[t_str] += 1
    for x in t_map:
        if t_map[x] > 500:
            print("%d) %s" % (t_map[x], x))


def split_synthetic_log(proc_name):
    for calendar_type in range(1, 5):
        for even in [True, False]:
            split_event_log(get_file_path(proc_name, FileType.ORIGINAL_CSV_LOG, 60, 0, 1, calendar_type, even),
                            get_file_path(proc_name, FileType.TRAINING_CSV_LOG, 60, 0, 1, calendar_type, even),
                            get_file_path(proc_name, FileType.TESTING_CSV_LOG, 60, 0, 1, calendar_type, even),
                            0.5)


def compute_log_distance_metric(proc_name, s_count):
    testing_log = get_file_path(proc_name=proc_name, file_type=FileType.TESTING_CSV_LOG)
    if print_stats:
        print("Crisp Model %s" % proc_name)
    compute_average_log_distance_measures(proc_name, None, FileType.CRISP_LOG, testing_log, 60, 0, s_count)

    for g_size in granule_sizes:
        for angle in trapezoidal_angles:
            if print_stats:
                print("Fuzzy Model %s With Granule Size: %s and Angle: %s " % (proc_name, str(g_size), str(angle)))
            compute_average_log_distance_measures(proc_name, None, FileType.SIMULATED_LOG, testing_log,
                                                  g_size, angle, s_count)


def discover_fuzzy_parameters(proc_name):
    if is_syntetic[proc_name]:
        for calendar_type in range(1, 5):
            for even in [True, False]:
                for g_size in granule_sizes:
                    for angle in trapezoidal_angles:
                        discover_model_from_csv_log(proc_name, g_size, angle, calendar_type, even)
                        if print_stats:
                            print(
                                "Discovery Completed -- Granule %s, Angle: %s, Calendar: %s, Even Resorce Workload: %s"
                                % (str(g_size), str(angle), str(calendar_type), str(even)))
    else:
        for g_size in granule_sizes:
            for angle in trapezoidal_angles:
                discover_model_from_csv_log(proc_name, g_size, angle, None, None)
                if print_stats:
                    print("Discovery Completed -- Granule %s, Angle: %s" % (str(g_size), str(angle)))


def discover_crisp_calendars(proc_name, conf=None, supp=None, part=None):
    if print_stats:
        print('--------------------------------------------------------------------------')
        print("Starting Crisp Calendars Discovery of process %s" % proc_name)
        print('--------------------------------------------------------------------------')

    if conf is None:
        [granule, conf, supp, part, adj_c] = crisp_discovery_params[proc_name]
    else:
        granule = 60
        adj_c = True

    if is_syntetic[proc_name]:
        for c in range(1, 5):
            for even in [True, False]:
                preprocess_xes_log(
                    get_file_path(proc_name=proc_name, file_type=FileType.TRAINING_CSV_LOG, calendar_type=c, even=even),
                    get_file_path(proc_name=proc_name, file_type=FileType.BPMN, calendar_type=c, even=even),
                    get_file_path(proc_name=proc_name, file_type=FileType.CRISP_JSON, calendar_type=c, even=even),
                    granule, conf, supp, part, adj_c, True)
    else:
        preprocess_xes_log(get_file_path(proc_name=proc_name, file_type=FileType.TRAINING_CSV_LOG),
                           get_file_path(proc_name=proc_name, file_type=FileType.BPMN),
                           get_file_path(proc_name=proc_name, file_type=FileType.CRISP_JSON), granule, conf, supp, part,
                           adj_c, True)


def simulate_and_save_results(proc_name, s_count):
    if is_syntetic[proc_name]:
        for c in range(1, 5):
            for even in [True, False]:
                for g_size in granule_sizes:
                    for angle in trapezoidal_angles:
                        if print_stats:
                            print("Simulating Model With Granule Size: %s and Angle: %s " % (str(g_size), str(angle)))
                        run_fuzzy_simulation(proc_name, g_size, angle, s_count, c, even)
    else:
        for g_size in granule_sizes:
            for angle in trapezoidal_angles:
                if print_stats:
                    print("Simulating Model With Granule Size: %s and Angle: %s " % (str(g_size), str(angle)))
                run_fuzzy_simulation(proc_name, g_size, angle, s_count, 1, False)


def compute_average_log_distance_measures(proc_name, sim_info, file_type, testing_log, g_size, angle, s_count,
                                          calendar_type, even):
    if print_stats:
        print('Individual Stats ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    log_metrics = list()
    for i in range(0, s_count):
        out_sim_log = get_file_path(proc_name=proc_name, file_type=file_type, granule=g_size, angle=angle, file_index=i,
                                    calendar_type=calendar_type, even=even)
        if sim_info is not None:
            _save_simulation_log(out_sim_log, sim_info[i])
        log_metrics.append(_compute_log_distance_measures(testing_log, out_sim_log))
        if print_stats:
            print_log_distance_metrics(log_metrics[len(log_metrics) - 1])
    mean_metrics = dict()
    for individual_metrics in log_metrics:
        for m_id in individual_metrics:
            if m_id not in mean_metrics:
                mean_metrics[m_id] = 0
            mean_metrics[m_id] += individual_metrics[m_id]
    for m_id in mean_metrics:
        mean_metrics[m_id] /= len(log_metrics)
    if print_stats:
        print('Average Stats ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print_log_distance_metrics(mean_metrics)
    return log_metrics, mean_metrics


def _mean_by_removing_metric_boundaries(log_metrics: dict, filter_by: str):
    min_v = sys.float_info.max
    max_v = 0
    i_min, i_max = 0, 0
    i = 0

    for individual_metrics in log_metrics:
        val = individual_metrics[filter_by]
        if val < min_v:
            min_v = val
            i_min = i
        elif val > max_v:
            max_v = val
            i_max = i
        i += 1

    mean_metrics = dict()
    for i in range(0, len(log_metrics)):
        if i in [i_min, i_max]:
            continue
        for m_id in log_metrics[i]:
            if m_id not in mean_metrics:
                mean_metrics[m_id] = 0
            mean_metrics[m_id] += log_metrics[i][m_id]
    for m_id in mean_metrics:
        mean_metrics[m_id] /= (len(log_metrics) - 2)
    return mean_metrics


def print_log_distance_metrics(metrics_map):
    names = {'AED': 'AED- Absolute Event Distribution Distance',
             'RED': 'RED- Relative Event Distribution Distance',
             'CED': 'CED- Circadian Event Distribution Distance',
             'CTD': 'CTD- Cycle Time Distribution Distance'}
    for m_id in metrics_map:
        print('%s: %f' % (names[m_id], metrics_map[m_id]))
    print('------------------------------------------------------------')


def simulate_and_save_crisp_model(proc_name: str, s_count):
    if is_syntetic[proc_name]:
        for calendar_type in range(1, 5):
            for even in [True, False]:
                if print_stats:
                    print("Calendar Type %s, Even Resource Workload: %s" % (str(calendar_type), str(even)))
                return run_crisp_simulation(proc_name, s_count, calendar_type, even)
    else:
        return run_crisp_simulation(proc_name, s_count)


def run_crisp_simulation(proc_name: str, s_count: int, c_typ, even):
    if print_stats:
        print("Simulating Crisp Model %s" % proc_name)
    testing_log = get_file_path(proc_name=proc_name, file_type=FileType.TESTING_CSV_LOG, calendar_type=c_typ, even=even)
    total_execution_time = 0
    sim_info = list()
    fixed_arrival_times, starting_datetime = get_starting_datetimes(testing_log)
    for i in range(0, s_count):
        if print_stats:
            print("Starting Simulation: %d" % i)
        s_start = datetime.now()
        sim_info.append(
            run_simulation(
                bpmn_path=get_file_path(proc_name=proc_name, file_type=FileType.BPMN, calendar_type=c_typ, even=even),
                json_path=get_file_path(proc_name=proc_name, file_type=FileType.CRISP_JSON, calendar_type=c_typ,
                                        even=even),
                total_cases=len(fixed_arrival_times),
                stat_out_path=None,
                log_out_path=None,
                starting_at=str(starting_datetime),
                fixed_arrival_times=fixed_arrival_times
            ))
        total_execution_time += (datetime.now() - s_start).total_seconds()
    # if print_stats:
    # print("------------------------------------------------------------")
    print("Simulation Execution Time: %s" % (str(timedelta(seconds=(total_execution_time / s_count)))))
    # print("------------------------------------------------------------")
    return compute_average_log_distance_measures(proc_name, sim_info, FileType.CRISP_LOG, testing_log, 60, 0, s_count,
                                                 c_typ, even)


def run_fuzzy_simulation(proc_name: str, g_size, angle, s_count, c_typ, even):
    testing_log = get_file_path(proc_name=proc_name, file_type=FileType.TESTING_CSV_LOG, calendar_type=c_typ, even=even)
    total_execution_time = 0
    sim_info = list()
    fixed_arrival_times, starting_datetime = get_starting_datetimes(testing_log)
    for i in range(0, s_count):
        # print("Starting Simulation: %d" % i)
        s_start = datetime.now()
        sim_info.append(
            run_simulation(
                bpmn_path=get_file_path(proc_name=proc_name, file_type=FileType.BPMN, calendar_type=c_typ, even=even),
                json_path=get_file_path(proc_name=proc_name, file_type=FileType.SIMULATION_JSON, granule=g_size,
                                        angle=angle, calendar_type=c_typ, even=even),
                total_cases=len(fixed_arrival_times),
                stat_out_path=None,
                log_out_path=None,
                starting_at=str(starting_datetime),
                fixed_arrival_times=fixed_arrival_times
            ))
        total_execution_time += (datetime.now() - s_start).total_seconds()

    # if print_stats:
    # print("------------------------------------------------------------")
    print("Simulation Execution Time: %s" % (str(timedelta(seconds=(total_execution_time / s_count)))))
    # print("------------------------------------------------------------")
    return compute_average_log_distance_measures(proc_name, sim_info, FileType.SIMULATED_LOG, testing_log, g_size,
                                                 angle, s_count, c_typ, even)


def _compute_resource_allocation_metrics(real_log_path, simulated_log_path):
    sim_resources, sim_alloc = _compute_log_resource_stats(event_list_from_csv(simulated_log_path))
    real_resources, real_alloc = _compute_log_resource_stats(event_list_from_csv(real_log_path))

    match_resources = real_resources.intersection(sim_resources)

    mistmatch_index = 1 - len(match_resources) / len(real_resources)

    allocation_ratio_distance = 0.0
    work_time_ratio_distance = 0.0

    for r_id in match_resources:
        allocation_ratio_distance += pow(real_alloc[r_id][0] - sim_alloc[r_id][0], 2)
        work_time_ratio_distance += pow(real_alloc[r_id][1] - sim_alloc[r_id][1], 2)

    return mistmatch_index, math.sqrt(allocation_ratio_distance), math.sqrt(work_time_ratio_distance)


def _compute_log_resource_stats(log_traces):
    bounds = [datetime.max.replace(tzinfo=pytz.UTC), datetime.min.replace(tzinfo=pytz.UTC)]
    resources = set()
    allocations = dict()
    total_tasks = 0
    resource_tasks = dict()
    task_freqs = dict()
    for trace in log_traces:
        for ev in trace.event_list:
            total_tasks += 1
            if ev.task_id not in task_freqs:
                task_freqs[ev.task_id] = 0
            if ev.resource_id not in resources:
                resources.add(ev.resource_id)
                resource_tasks[ev.resource_id] = set()
                allocations[ev.resource_id] = [0.0, 0.0]

            bounds[0] = min(bounds[0], ev.started_at)
            bounds[1] = max(bounds[1], ev.completed_at)
            resource_tasks[ev.resource_id].add(ev.task_id)
            task_freqs[ev.task_id] += 1
            allocations[ev.resource_id][0] += 1
            allocations[ev.resource_id][1] += (ev.completed_at - ev.started_at).total_seconds()
    process_duration = (bounds[1] - bounds[0]).total_seconds()

    possible_allocations = dict()
    for r_id in resource_tasks:
        tot_freq = 0
        for t_id in resource_tasks[r_id]:
            tot_freq += task_freqs[t_id]
        possible_allocations[r_id] = tot_freq

    for r_id in resources:
        allocations[r_id][0] = allocations[r_id][0] / possible_allocations[r_id]
        allocations[r_id][1] = allocations[r_id][1] / process_duration

    return resources, allocations


def _compute_log_distance_measures(real_log_path, simulated_log_path):
    real_log = _parse_log_for_metrics(real_log_path)
    sim_log = _parse_log_for_metrics(simulated_log_path)
    results = dict()

    absolute_event_distribution_dist = absolute_event_distribution_distance(
        real_log, DEFAULT_CSV_IDS,
        sim_log, DEFAULT_CSV_IDS,
        discretize_type=AbsoluteTimestampType.BOTH,
        discretize_event=discretize_to_hour)
    results['AED'] = absolute_event_distribution_dist

    relative_event_distribution_dist = relative_event_distribution_distance(
        real_log, DEFAULT_CSV_IDS,
        sim_log, DEFAULT_CSV_IDS,
        discretize_type=AbsoluteTimestampType.BOTH,
        discretize_event=discretize_to_hour
    )
    results['RED'] = relative_event_distribution_dist

    circadian_event_distribution_dist = circadian_event_distribution_distance(
        real_log, DEFAULT_CSV_IDS,
        sim_log, DEFAULT_CSV_IDS,
        discretize_type=AbsoluteTimestampType.BOTH
    )
    results['CED'] = circadian_event_distribution_dist

    cycle_time_distribution_dist = cycle_time_distribution_distance(
        real_log, DEFAULT_CSV_IDS,
        sim_log, DEFAULT_CSV_IDS,
        bin_size=pd.Timedelta(hours=1)
    )
    results['CTD'] = cycle_time_distribution_dist
    results['MTR'], results['TAR'], results['WTR'] = _compute_resource_allocation_metrics(real_log_path,
                                                                                          simulated_log_path)

    return results


# def _parse_log_for_metrics(log_path):
#     event_log_ids = EventLogIDs(case="case_id", activity="Activity", start_time="start_time", end_time="end_time")
#     event_log = pd.read_csv(log_path)
#
#     _custom_to_datetime(event_log, event_log_ids.start_time)
#     _custom_to_datetime(event_log, event_log_ids.end_time)
#
#     # event_log[event_log_ids.start_time] = pd.to_datetime(event_log[event_log_ids.start_time], utc=True)
#     # event_log[event_log_ids.end_time] = pd.to_datetime(event_log[event_log_ids.end_time], utc=True)
#     return event_log


def _parse_log_for_metrics(log_path):
    event_log_ids = EventLogIDs(case="case_id", activity="Activity", start_time="start_time", end_time="end_time")
    event_log = pd.read_csv(log_path)

    # Parsing start_time column
    event_log[event_log_ids.start_time] = pd.to_datetime(event_log[event_log_ids.start_time], utc=True, errors='coerce')
    mask_start = event_log[event_log_ids.start_time].isna()
    event_log.loc[mask_start, event_log_ids.start_time] = pd.to_datetime(
        event_log.loc[mask_start, event_log_ids.start_time], format="%Y-%m-%d %H:%M:%S%z", utc=True, errors='coerce')

    # Parsing end_time column
    event_log[event_log_ids.end_time] = pd.to_datetime(event_log[event_log_ids.end_time], utc=True, errors='coerce')
    mask_end = event_log[event_log_ids.end_time].isna()
    event_log.loc[mask_end, event_log_ids.end_time] = pd.to_datetime(event_log.loc[mask_end, event_log_ids.end_time],
                                                                     format="%Y-%m-%d %H:%M:%S%z", utc=True,
                                                                     errors='coerce')

    # Removing rows where either start_time or end_time is still NaN after all parsing attempts
    event_log = event_log.dropna(subset=[event_log_ids.start_time, event_log_ids.end_time])

    return event_log

# def _custom_to_datetime(event_log, c_time):
#     event_log[c_time] = pd.to_datetime(event_log[c_time], utc=True, errors='coerce')
#     mask = event_log[c_time].isna()
#     event_log.loc[mask, c_time] = pd.to_datetime(event_log.loc[mask, c_time], format="%Y-%m-%d %H:%M:%S%z", utc=True)


def _find_simulation_median_cycle_times(sim_info: list):
    cycle_times = list()
    sim_kpi = dict()
    for i in range(0, len(sim_info)):
        res = sim_info[i].log_info.compute_process_kpi(sim_info[i])
        process_kpi: KPIMap = res[0]
        cycle_times.append((i, process_kpi.cycle_time.avg))
        sim_kpi[i] = res
    cycle_times.sort(key=lambda c_time: c_time[1])
    m_i = cycle_times[len(cycle_times) // 2][0]
    return m_i, _build_simulation_result(sim_kpi[m_i], sim_info[m_i].sim_setup.bpmn_graph.from_name)


def _build_simulation_result(sim_out: list, tasks_id: dict):
    sim_result = SimulationResult(
        started_at=sim_out[3],
        ended_at=sim_out[4]
    )
    sim_result.process_kpi_map = sim_out[0]
    tasks_kpi = dict()
    for t_name in tasks_id:
        if tasks_id[t_name] in sim_out[1]:
            tasks_kpi[t_name] = sim_out[1][tasks_id[t_name]]
    sim_result.tasks_kpi_map = tasks_kpi
    r_info = dict()
    r_utilization = dict()
    for r_name in sim_out[2]:
        r_utilization[r_name] = sim_out[2][r_name].utilization
        r_info[r_name] = [sim_out[2][r_name].task_allocated,
                          sim_out[2][r_name].available_time,
                          sim_out[2][r_name].worked_time,
                          "None"]
    sim_result.resource_utilization = r_utilization
    sim_result.resource_info = r_info
    return sim_result


def _save_simulation_log(out_csv_log_path, bpm_env: SimBPMEnv):
    with open(out_csv_log_path, mode='w', newline='', encoding='utf-8') as log_csv_file:
        f_writer = csv.writer(log_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # add_simulation_event_log_header(f_writer)
        log_writer = FileManager(10000, f_writer)

        for trace in bpm_env.log_info.trace_list:
            for ev in trace.event_list:
                log_writer.add_csv_row([ev.p_case,
                                        bpm_env.sim_setup.bpmn_graph.element_info[ev.task_id].name,
                                        ev.enabled_datetime,
                                        ev.started_datetime,
                                        ev.completed_datetime,
                                        bpm_env.sim_setup.resources_map[ev.resource_id].resource_name])
        log_writer.force_write()


def add_simulation_event_log_header(log_fwriter):
    if log_fwriter:
        log_fwriter.writerow([
            'case_id', 'activity', 'enable_time', 'start_time', 'end_time', 'resource', ])


if __name__ == "__main__":
    main()
