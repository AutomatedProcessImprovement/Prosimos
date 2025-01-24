import json

import pandas as pd
from pathlib import Path

from pix_framework.discovery.resource_profiles import discover_differentiated_resource_profiles

from pix_framework.discovery.gateway_probabilities import compute_gateway_probabilities, \
    GatewayProbabilitiesDiscoveryMethod

from pix_framework.discovery.case_arrival import discover_case_arrival_model

from pix_framework.discovery.resource_calendar_and_performance.fuzzy.discovery import \
    discovery_fuzzy_resource_calendars_and_performances

from pix_framework.io.bpm_graph import BPMNGraph
from pix_framework.io.event_log import PROSIMOS_LOG_IDS, EventLogIDs


def build_fuzzy_calendars(log_df, bpmn_graph, json_path=None, i_size_minutes=15, angle=0.0):
    # log_df = pd.read_csv(csv_log_path)
    # _add_enabled_times(log_df, PROSIMOS_LOG_IDS)
    # bpmn_graph = BPMNGraph.from_bpmn_path(Path(bpmn_path))

    # 1) Discovering Resource Availability (Fuzzy Calendars)
    # 2) Discovering Resource Performance (resource-task distributions ajusted from the fuzzy calendars)
    r_calendars, res_task_distr = discovery_fuzzy_resource_calendars_and_performances(log=log_df,
                                                                                      log_ids=PROSIMOS_LOG_IDS,
                                                                                      granularity=i_size_minutes,
                                                                                      fuzzy_angle=angle)
    # 3) Discovering Arrival Time Calendar
    arrival_model = discover_case_arrival_model(event_log=log_df, log_ids=PROSIMOS_LOG_IDS, granularity=60).to_dict()

    # 5) Discovering Gateways Branching Probabilities
    gateways_branching = compute_gateway_probabilities(event_log=log_df,
                                                       log_ids=PROSIMOS_LOG_IDS,
                                                       bpmn_graph=bpmn_graph,
                                                       discovery_method=GatewayProbabilitiesDiscoveryMethod.DISCOVERY)

    resource_profiles = discover_differentiated_resource_profiles(event_log=log_df, log_ids=PROSIMOS_LOG_IDS)

    simulation_params = {
        "resource_profiles": to_profiles_with_task_id(resource_profiles, bpmn_graph),
        "arrival_time_distribution": arrival_model["arrival_time_distribution"],
        "arrival_time_calendar": arrival_model["arrival_time_calendar"],
        "gateway_branching_probabilities": [g_prob.to_dict() for g_prob in gateways_branching],
        "task_resource_distribution": to_distribution_with_task_id(res_task_distr, bpmn_graph),
        "resource_calendars": [r_calendar.to_prosimos() for r_calendar in r_calendars],
        "model_type": "FUZZY",
        "granule_size": {
            "value": i_size_minutes,
            "time_unit": "MINUTES"
        }
    }
    if json_path is not None:
        with open(json_path, 'w') as file_writter:
            json.dump(simulation_params, file_writter)

    return simulation_params


def to_profiles_with_task_id(resource_profiles: list, bpmn_graph: BPMNGraph):
    result = []
    for r_p in resource_profiles:
        r_profile = r_p.to_dict()
        for r_info in r_profile["resource_list"]:
            task_ids = []
            for t_name in r_info["assignedTasks"]:
                task_ids.append(bpmn_graph.from_name[t_name])
            r_info["assignedTasks"] = task_ids
        result.append(r_profile)
    return result


def to_distribution_with_task_id(res_task_distr: list, bpmn_graph: BPMNGraph):
    result = []
    for rt_d in res_task_distr:
        rt_dist = rt_d.to_dict()
        rt_dist["task_id"] = bpmn_graph.from_name[rt_dist["task_id"]]
        result.append(rt_dist)
    return result
