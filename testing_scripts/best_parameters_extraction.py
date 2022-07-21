import json
import sys
from datetime import datetime

import pytz

from bpdfr_discovery.log_parser import sort_by_completion_times, discover_arrival_calendar, discover_arrival_time_distribution, discover_resource_calendars, \
    discover_resource_task_duration_distribution, map_task_id_from_names
from bpdfr_simulation_engine.execution_info import Trace
from bpdfr_simulation_engine.resource_calendar import CalendarFactory, parse_datetime
from bpdfr_simulation_engine.simulation_properties_parser import parse_simulation_model
from pm4py.objects.log.importer.xes import importer as xes_importer

from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation
from testing_scripts.bpm_2022_testing_files import process_files
from bpdfr_discovery.emd_metric import read_and_preprocess_log, absolute_hour_emd, trace_duration_emd, \
    discretize_to_hour, SimStats, discretize_to_day


def discover_simulation_parameters(model_name, log_path, bpmn_path, out_f_path):
    print('Parsing Event Log %s ...' % model_name)
    bpmn_graph = parse_simulation_model(bpmn_path)

    log_traces = xes_importer.apply(log_path)
    print('Total Traces in Log: %d' % len(log_traces))

    completed_events = list()
    task_events = dict()
    task_resource_events = dict()
    initial_events = dict()
    log_info = dict()
    flow_arcs_frequency = dict()

    for trace in log_traces:
        caseid = trace.attributes['concept:name']
        started_events = dict()
        trace_info = Trace(caseid)

        initial_events[caseid] = datetime(9999, 12, 31, tzinfo=pytz.UTC)
        log_info[caseid] = trace_info
        for event in trace:
            task_name = event['concept:name']
            if 'org:resource' not in event:
                resource = task_name
            else:
                resource = event['org:resource']
            state = event['lifecycle:transition'].lower()
            timestamp = event['time:timestamp']

            initial_events[caseid] = min(initial_events[caseid], timestamp)

            if task_name not in task_resource_events:
                task_resource_events[task_name] = dict()
                task_events[task_name] = list()
            if resource not in task_resource_events[task_name]:
                task_resource_events[task_name][resource] = list()

            if state in ["start", "assign"]:
                started_events[task_name] = trace_info.start_event(task_name, task_name, timestamp, resource)
            elif state == "complete":
                if task_name in started_events:
                    c_event = trace_info.complete_event(started_events.pop(task_name), timestamp)
                    task_events[task_name].append(c_event)
                    task_resource_events[task_name][resource].append(c_event)
                    completed_events.append(c_event)
        trace_info.filter_incomplete_events()
        task_sequence = sort_by_completion_times(trace_info)
        bpmn_graph.reply_trace(task_sequence, flow_arcs_frequency, True, trace_info.event_list)

    [[best_granule, best_conf, best_supp, best_part, adj_c],
     [best_granule_t, best_conf_t, best_supp_t, best_part_t, adj_c_t]] = find_best_parameters(model_name,
                                                                                              bpmn_path,
                                                                                              log_info,
                                                                                              initial_events,
                                                                                              task_resource_events,
                                                                                              flow_arcs_frequency,
                                                                                              out_f_path)
    print('Best Parameters ----------------------------------------------')
    print('Best EMD_Hour  -> GSize: %d Conf: %.1f, Supp: %.1f, Part: %.1f, Adj_Calendar: %s' % (
        best_granule, best_conf, best_supp, best_part, str(adj_c)
    ))
    print('Best EMD_Trace -> GSize: %d Conf: %.1f, Supp: %.1f, Part: %.1f, Adj_Calendar: %s' % (
        best_granule_t, best_conf_t, best_supp_t, best_part_t, str(adj_c_t)
    ))
    return [[best_granule, best_conf, best_supp, best_part, adj_c],
            [best_granule_t, best_conf_t, best_supp_t, best_part_t, adj_c_t]]


def find_best_parameters(model_name, bpmn_path, log_info, initial_events, task_res_evt, flow_arcs_freq, out_f_path):
    bpmn_graph = parse_simulation_model(bpmn_path)
    real_log = read_and_preprocess_log(process_files[model_name]['real_csv_log'])
    # # Discovering Gateways Branching Probabilities
    gateways_branching = bpmn_graph.compute_branching_probability(flow_arcs_freq)

    best_emd = sys.float_info.max
    best_emd_trace = sys.float_info.max
    best_supp_emd_hour, best_conf_emd_hour, best_part_emd_hour, best_granule_emd_hour, best_bin = 0, 0, 0, 0, 0
    best_supp_emd_trace, best_conf_emd_trace, best_part_emd_trace, best_granule_emd_trace = 0, 0, 0, 0
    with_fit_c = True
    with_fit_c_trace = True
    for granule_size in [60]:
        # # Discovering Arrival Calendar
        arrival_calendar = discover_arrival_calendar(initial_events, 60, 0.1, 1.0)
        json_arrival_calendar = arrival_calendar.to_json()

        # # Discovering Arrival Time Distribution
        arrival_time_dist = discover_arrival_time_distribution(initial_events, arrival_calendar)

        calendar_factory = CalendarFactory(granule_size)
        for case_id in log_info:
            for e_info in log_info[case_id].event_list:
                calendar_factory.check_date_time(e_info.resource_id, e_info.task_id, e_info.started_at)
                calendar_factory.check_date_time(e_info.resource_id, e_info.task_id, e_info.completed_at)
        for min_conf in [0.5]:
            for min_supp in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                for min_part in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    # # Discovering Resource Calendars
                    res_calendars, task_res, joint_res_evts, pools, _ = discover_resource_calendars(calendar_factory,
                                                                                                    task_res_evt,
                                                                                                    min_conf,
                                                                                                    min_supp,
                                                                                                    min_part)
                    res_json_calendar = dict()
                    for r_id in res_calendars:
                        res_json_calendar[r_id] = res_calendars[r_id].to_json()

                    # # Discovering Task-Duration Distributions per resource
                    for fit_c in [True]:
                        task_resource_dist = discover_resource_task_duration_distribution(task_res_evt,
                                                                                          res_calendars,
                                                                                          task_res,
                                                                                          joint_res_evts,
                                                                                          fit_c,
                                                                                          50)

                        to_save = {
                            "resource_profiles": map_task_id_from_names(pools, bpmn_graph.from_name),
                            "arrival_time_distribution": arrival_time_dist,
                            "arrival_time_calendar": json_arrival_calendar,
                            "gateway_branching_probabilities": gateways_branching,
                            "task_resource_distribution": map_task_id_from_names(task_resource_dist,
                                                                                 bpmn_graph.from_name),
                            "resource_calendars": res_json_calendar,
                        }
                        with open(out_f_path, 'w') as file_writter:
                            json.dump(to_save, file_writter)

                        emd_index, _, emd_trace = compute_median_simulation_emd(model_name, len(log_info), bpmn_path,
                                                                                out_f_path, real_log, 'temp_log.csv')
                        if emd_index < best_emd:
                            best_emd, best_granule_emd_hour = emd_index, granule_size
                            best_conf_emd_hour, best_supp_emd_hour, best_part_emd_hour = min_conf, min_supp, min_part
                            with_fit_c = fit_c

                        if emd_trace < best_emd_trace:
                            best_emd_trace, best_granule_emd_trace = emd_trace, granule_size
                            best_supp_emd_trace, best_conf_emd_trace, best_part_emd_trace = min_supp, min_conf, min_part
                            with_fit_c_trace = fit_c

                        print('GSize: %d Conf: %.1f, Supp: %.1f, Part: %.1f --> EMD: %.2f, T_EMD: %.2f' % (
                            granule_size, min_conf, min_supp, min_part, emd_index, emd_trace
                        ))
    return [[best_granule_emd_hour, best_conf_emd_hour, best_supp_emd_hour, best_part_emd_hour, with_fit_c],
            [best_granule_emd_trace, best_conf_emd_trace, best_supp_emd_trace, best_part_emd_trace, with_fit_c_trace]]


def compute_median_simulation_emd(model_name, p_cases, bpmn_path, json_path, real_log, sim_log_path):
    emd_list = list()
    i = 0
    bin_size = max(
        [events['end_time'].max() - events['start_time'].min()
         for case, events in real_log.groupby(['case_id'])]
    ) / 100

    sim_duration = 0

    while i < 5:
        # try:
        sim_duration, _ = run_diff_res_simulation(parse_datetime(process_files[model_name]['start_datetime'], True),
                                                  p_cases, bpmn_path, json_path, None, sim_log_path)

        simulated_log = read_and_preprocess_log(sim_log_path)

        emd_list.append(SimStats(absolute_hour_emd(real_log, simulated_log, discretize_to_hour),
                                 absolute_hour_emd(real_log, simulated_log, discretize_to_day),
                                 trace_duration_emd(real_log, simulated_log, bin_size)))
        i += 1
    print("Mean Simulation Time: %.2f" % (sim_duration / 5))
    # except:
    #     print('Simulation Limit exceeded: %d' % i)
    #     continue

    emd_list.sort(key=lambda x: x.hour_emd_index)

    return emd_list[2].hour_emd_index, emd_list[2].day_emd_index, emd_list[2].emd_trace
