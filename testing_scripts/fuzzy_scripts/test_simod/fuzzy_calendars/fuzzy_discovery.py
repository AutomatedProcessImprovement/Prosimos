import json
from pathlib import Path

import pandas as pd
from datetime import datetime, timedelta

from pix_framework.io.bpm_graph import BPMNGraph


from pix_framework.discovery.gateway_probabilities import compute_gateway_probabilities, \
    GatewayProbabilitiesDiscoveryMethod

from pix_framework.discovery.case_arrival import discover_case_arrival_model

from pix_framework.io.event_log import PROSIMOS_LOG_IDS, EventLogIDs

from pix_framework.discovery.resource_calendar_and_performance.fuzzy.discovery import \
    discovery_fuzzy_resource_calendars_and_performances

from pix_framework.discovery.resource_calendar_and_performance.calendar_discovery_parameters import \
    CalendarDiscoveryParameters

from bpdfr_discovery.log_parser import event_list_from_csv, discover_arrival_calendar, \
    discover_arrival_time_distribution
from prosimos.simulation_properties_parser import parse_simulation_model

from testing_scripts.fuzzy_scripts.test_simod.fuzzy_calendars.fuzzy_factory import FuzzyFactory
from testing_scripts.fuzzy_scripts.test_simod.fuzzy_calendars.intervals_frequency_calculator import ProcInfo, Method


def build_fuzzy_calendars(csv_log_path, bpmn_path, json_path=None, i_size_minutes=15, angle=0.0, min_prob=0.1):
    # log_df = pd.read_csv(csv_log_path)
    # _add_enabled_times(log_df, PROSIMOS_LOG_IDS)
    # bpmn_graph = BPMNGraph.from_bpmn_path(Path(bpmn_path))

    # 1) Discovering Resource Availability (Fuzzy Calendars)
    # 2) Discovering Resource Performance (resource-task distributions ajusted from the fuzzy calendars)
    r_calendar, res_task_distr = discovery_fuzzy_resource_calendars_and_performances(log=log_df,
                                                                                     log_ids=PROSIMOS_LOG_IDS,
                                                                                     granularity=i_size_minutes,
                                                                                     fuzzy_angle=angle)
    # 3) Discovering Arrival Time Calendar
    arrival_calend, arrival_dist = discover_case_arrival_model(event_log=log_df, log_ids=PROSIMOS_LOG_IDS,
                                                               granularity=60)

    # 5) Discovering Gateways Branching Probabilities
    gateways_branching = compute_gateway_probabilities(event_log=log_df,
                                                       log_ids=PROSIMOS_LOG_IDS,
                                                       bpmn_graph=bpmn_graph,
                                                       discovery_method=GatewayProbabilitiesDiscoveryMethod.DISCOVERY)




def build_fuzzy_calendars_old(csv_log_path, bpmn_path, json_path=None, i_size_minutes=15, angle=0.0, min_prob=0.1):
    traces = event_list_from_csv(csv_log_path)
    bpmn_graph = parse_simulation_model(bpmn_path)

    p_info = ProcInfo(traces, bpmn_graph, i_size_minutes, True, Method.TRAPEZOIDAL, angle=angle)
    f_factory = FuzzyFactory(p_info)

    # 1) Discovering Resource Availability (Fuzzy Calendars)
    p_info.fuzzy_calendars = f_factory.compute_resource_availability_calendars(min_impact=min_prob)

    # 2) Discovering Resource Performance (resource-task distributions ajusted from the fuzzy calendars)
    res_task_distr = f_factory.compute_processing_times(p_info.fuzzy_calendars)

    # 3) Discovering Arrival Time Calendar -- Nothing New, just re-using the original Prosimos approach
    arrival_calend = discover_arrival_calendar(p_info.initial_events, 15, 0.1, 1.0)

    # 4) Discovering Arrival Time Distribution -- Nothing New, just re-using the original Prosimos approach
    arrival_dist = discover_arrival_time_distribution(p_info.initial_events, arrival_calend).to_prosimos_distribution()

    # 5) Discovering Gateways Branching Probabilities -- Nothing New, just re-using the original Prosimos approach
    gateways_branching = bpmn_graph.compute_branching_probability(p_info.flow_arcs_frequency)

    simulation_params = {
        "resource_profiles": build_resource_profiles(p_info),
        "arrival_time_distribution": arrival_dist,
        "arrival_time_calendar": arrival_calend.intervals_to_json(),
        "gateway_branching_probabilities": gateway_branching_to_json(gateways_branching),
        "task_resource_distribution": processing_times_json(res_task_distr, p_info.task_resources, p_info.bpmn_graph),
        "resource_calendars": join_fuzzy_calendar_intervals(p_info.fuzzy_calendars, p_info.i_size),
        "granule_size": {
            "value": i_size_minutes,
            "time_unit": "MINUTES"
        }
    }

    if json_path is not None:
        with open(json_path, 'w') as file_writter:
            json.dump(simulation_params, file_writter)

    return simulation_params


def processing_times_json(res_task_distr, task_resources, bpmn_graph):
    distributions = []
    for t_name in task_resources:
        resources = []
        for r_id in task_resources[t_name]:
            if r_id not in res_task_distr:
                continue

            resources.append({
                "resource_id": r_id,
                "distribution_name": res_task_distr[r_id][t_name]["distribution_name"],
                "distribution_params": res_task_distr[r_id][t_name]["distribution_params"]
            })
        distributions.append({
            "task_id": bpmn_graph.from_name[t_name],
            "resources": resources
        })
    return distributions


def join_fuzzy_calendar_intervals(fuzzy_calendars, i_size):
    resource_calendars = []
    for r_id in fuzzy_calendars:
        resource_calendars.append({
            "id": "%s_timetable" % r_id,
            "name": r_id,
            "time_periods": to_prosimos_calendar(sweep_line_intervals(fuzzy_calendars[r_id].res_absolute_prob, i_size)),
            "workload_ratio": to_prosimos_calendar(
                sweep_line_intervals(fuzzy_calendars[r_id].res_relative_prob, i_size))
        })
    return resource_calendars


def to_prosimos_calendar(weekly_intervals):
    p_calendars = []

    for wd_info in weekly_intervals:
        wd = wd_info["week_day"]
        for p_info in wd_info['fuzzy_intervals']:
            p_calendars.append({
                "from": wd,
                "to": wd,
                "beginTime": p_info["begin_time"],
                "endTime": p_info["end_time"],
                "probability": p_info["probability"]
            })
    return p_calendars


def sweep_line_intervals(prob_map, i_size):
    days_str = {0: "MONDAY", 1: "TUESDAY", 2: "WEDNESDAY", 3: "THURSDAY", 4: "FRIDAY", 5: "SATURDAY", 6: "SUNDAY"}
    weekly_intervals = []
    for w_day in days_str:
        joint_intervals = []
        c_prob = prob_map[w_day][0]
        first_i = 0
        for i in range(1, len(prob_map[w_day])):
            if c_prob != prob_map[w_day][i]:
                if c_prob != 0:
                    joint_intervals.append((first_i, i))
                first_i = i
                c_prob = prob_map[w_day][i]
        if c_prob != 0:
            joint_intervals.append((first_i, 0))
        time_periods = []
        for from_i, to_i in joint_intervals:
            time_periods.append({
                "begin_time": str(interval_index_to_time(from_i, i_size, True).time()),
                "end_time": str(interval_index_to_time(to_i, i_size, True).time()),
                "probability": prob_map[w_day][from_i]
            })
        weekly_intervals.append({
            "week_day": days_str[w_day],
            "fuzzy_intervals": time_periods
        })
    return weekly_intervals


def interval_index_to_time(i_index, i_size, is_start):
    from_time = datetime.strptime("00:00:00", '%H:%M:%S') + timedelta(minutes=(i_index * i_size))
    return from_time if is_start else from_time + timedelta(minutes=i_size)


def build_resource_profiles(p_info: ProcInfo):
    resource_profiles = []
    for t_name in p_info.task_resources:
        t_id = p_info.bpmn_graph.from_name[t_name]
        resource_list = []
        for r_id in p_info.task_resources[t_name]:
            if r_id not in p_info.fuzzy_calendars:
                continue
            resource_list.append({
                "id": r_id,
                "name": r_id,
                "cost_per_hour": 1,
                "amount": 1,
                "calendar": "%s_timetable" % r_id,
                "assigned_tasks": [p_info.bpmn_graph.from_name[t_n] for t_n in p_info.resource_tasks[r_id]]
            })
        resource_profiles.append({
            "id": t_id,
            "name": t_name,
            "resource_list": resource_list
        })
    return resource_profiles


def distribution_to_json(distribution):
    distribution_params = []
    for d_param in distribution["distribution_params"]:
        distribution_params.append({
            "value": d_param
        })
    return {
        "distribution_name": distribution["distribution_name"],
        "distribution_params": distribution_params
    }


def gateway_branching_to_json(gateways_branching):
    gateways_json = []
    for g_id in gateways_branching:
        probabilities = []
        g_prob = gateways_branching[g_id]
        for flow_arc in g_prob:
            probabilities.append({
                "path_id": flow_arc,
                "value": g_prob[flow_arc]
            })

        gateways_json.append({
            "gateway_id": g_id,
            "probabilities": probabilities
        })
    return gateways_json


def _check_probabilities_range(fuzzy_calendars):
    for r_id in fuzzy_calendars:
        i_fuzzy = fuzzy_calendars[r_id]
        for wd in i_fuzzy.res_relative_prob:
            for p in i_fuzzy.res_relative_prob[wd]:
                if p < 0 or p > 1:
                    print("Wrong Relative")
        for wd in i_fuzzy.res_absolute_prob:
            for p in i_fuzzy.res_absolute_prob[wd]:
                if p < 0 or p > 1:
                    print("Wrong Absolute")
