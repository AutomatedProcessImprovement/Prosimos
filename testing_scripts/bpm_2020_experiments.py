import enum
import json
import math
import os
import sys
from dataclasses import dataclass

from scipy.stats import wasserstein_distance

from bpdfr_discovery.support_modules.log_parser import preprocess_xes_log, transform_xes_to_csv, \
    discover_aggregated_task_distributions
from bpdfr_simulation_engine.resource_calendar import parse_datetime, RCalendar, build_full_time_calendar, \
    CalendarFactory
from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation
import pandas as pd

experiment_logs = {0: 'production',
                   1: 'purchasing_example',
                   2: 'consulta_data_mining',
                   3: 'insurance',
                   4: 'call_centre',
                   5: 'bpi_2012',
                   6: 'bpi_2017_filtered',
                   7: 'bpi_2017'}

process_files = {
    'purchasing_example': {'xes_log': './../input_files/xes_files/PurchasingExample.xes',
                           'real_csv_log': './../output_files/real_csv_logs/purchasing_example.csv',
                           'bpmn': './../input_files/bpmn_simod_models/purchasing_example.bpmn',
                           'json': './../output_files/discovery/purchasing_example.json',
                           'sim_log': './../output_files/diffsim_logs/purchasing_example.csv',
                           'start_datetime': '2011-01-01T00:00:00.000000-05:00',
                           'total_cases': 608
                           },
    'production': {'xes_log': './../input_files/xes_files/production.xes',
                   'real_csv_log': './../output_files/real_csv_logs/production.csv',
                   'bpmn': './../input_files/bpmn_simod_models/Production.bpmn',
                   'json': './../output_files/discovery/production.json',
                   'sim_log': './../output_files/diffsim_logs/production.csv',
                   'start_datetime': '2012-01-02T07:00:00.000000+02:00',
                   'total_cases': 225
                   },
    'insurance': {'xes_log': './../input_files/xes_files/insurance.xes',
                  'real_csv_log': './../output_files/real_csv_logs/insurance.csv',
                  'bpmn': './../input_files/bpmn_simod_models/insurance.bpmn',
                  'json': './../output_files/discovery/insurance.json',
                  'sim_log': './../output_files/diffsim_logs/insurance.csv'
                  },
    'call_centre': {'xes_log': './../input_files/xes_files/callcentre.xes',
                    'real_csv_log': './../output_files/real_csv_logs/callcentre.csv',
                    'bpmn': './../input_files/bpmn_simod_models/callcentre.bpmn'},
    'bpi_2012': {'xes_log': './../input_files/xes_files/BPI_Challenge_2012_W_Two_TS.xes',
                 'real_csv_log': './../output_files/real_csv_logs/BPI_Challenge_2012_W_Two_TS.csv',
                 'bpmn': './../input_files/bpmn_simod_models/BPI_Challenge_2012_W_Two_TS.bpmn'},
    'bpi_2017_filtered': {'xes_log': './../input_files/xes_files/BPI_Challenge_2017_W_Two_TS_filtered.xes',
                          'real_csv_log': './../output_files/real_csv_logs/BPI_Challenge_2017_W_Two_TS_filtered.csv',
                          'bpmn': './../input_files/bpmn_simod_models/BPI_Challenge_2017_W_Two_TS_filtered.bpmn'},
    'bpi_2017': {'xes_log': './../input_files/xes_files/BPI_Challenge_2017_W_Two_TS.xes',
                 'real_csv_log': './../output_files/real_csv_logs/BPI_Challenge_2017_W_Two_TS.csv',
                 'bpmn': './../input_files/bpmn_simod_models/BPI_Challenge_2017_W_Two_TS.bpmn'},
    'consulta_data_mining': {'xes_log': './../input_files/xes_files/ConsultaDataMining201618.xes',
                             'real_csv_log': './../output_files/real_csv_logs/consulta_data_mining.csv',
                             'bpmn': './../input_files/bpmn_simod_models/consulta_data_mining.bpmn'}
}

canonical_json = {
    'purchasing_example': './../output_files/canonical_json_simod/PurchasingExample_canon.json',
    'production': './../output_files/canonical_json_simod/Production_canon.json',
    # 'insurance': './../output_files/canonical_json_simod/PurchasingExample_canon.json',
    # 'call_centre': {'xes_log': './../input_files/xes_files/callcentre.xes',
    #                 'real_csv_log': './../output_files/real_csv_logs/callcentre.csv',
    #                 'bpmn': './../input_files/bpmn_simod_models/callcentre.bpmn'},
    # 'bpi_2012': {'xes_log': './../input_files/xes_files/BPI_Challenge_2012_W_Two_TS.xes',
    #              'real_csv_log': './../output_files/real_csv_logs/BPI_Challenge_2012_W_Two_TS.csv',
    #              'bpmn': './../input_files/bpmn_simod_models/BPI_Challenge_2012_W_Two_TS.bpmn'},
    # 'bpi_2017_filtered': {'xes_log': './../input_files/xes_files/BPI_Challenge_2017_W_Two_TS_filtered.xes',
    #                       'real_csv_log': './../output_files/real_csv_logs/BPI_Challenge_2017_W_Two_TS_filtered.csv',
    #                       'bpmn': './../input_files/bpmn_simod_models/BPI_Challenge_2017_W_Two_TS_filtered.bpmn'},
    # 'bpi_2017': {'xes_log': './../input_files/xes_files/BPI_Challenge_2017_W_Two_TS.xes',
    #              'real_csv_log': './../output_files/real_csv_logs/BPI_Challenge_2017_W_Two_TS.csv',
    #              'bpmn': './../input_files/bpmn_simod_models/BPI_Challenge_2017_W_Two_TS.bpmn'},
    # 'consulta_data_mining': {'xes_log': './../input_files/xes_files/ConsultaDataMining201618.xes',
    #                          'real_csv_log': './../output_files/real_csv_logs/consulta_data_mining.csv',
    #                          'bpmn': './../input_files/bpmn_simod_models/consulta_data_mining.bpmn'}

}

out_folder = './../output_files/exp_cases/json_in/'


def main():
    for i in range(1, 8):

        model_name = experiment_logs[i]
        # transform_xes_to_csv(process_files[model_name]['xes_log'], process_files[model_name]['real_csv_log'])

        # Extracting the simulation parameters from event-log (it saves them results to JSON files)
        xes_path = process_files[model_name]['xes_log']
        bpmn_path = process_files[model_name]['bpmn']
        sim_log_path = process_files[model_name]['sim_log']
        real_log = read_and_preprocess_log(process_files[model_name]['real_csv_log'])

        discover_from_xes_log(model_name)

        p_cases = 10

        if "total_cases" in process_files[model_name]:
            p_cases = process_files[model_name]["total_cases"]

        json_paths = []
        for j in range(1, 6):
            json_paths.append('%s%s_c_%d.json' % (out_folder, model_name, j))
        json_paths.append(process_files[model_name]['json'])
        c = 1
        for json_path in json_paths:
            print("Case %d:" % c)

            print(compute_median_simulation_emd(model_name, p_cases, bpmn_path, json_path, real_log, sim_log_path))
            print('--------------------------------')
            c += 1

        break

    os._exit(0)


def compute_median_simulation_emd(model_name, p_cases, bpmn_path, json_path, real_log, sim_log_path):
    emd_list = list()
    i = 0
    while i < 15:
        try:
            diff_sim_result = run_diff_res_simulation(parse_datetime(process_files[model_name]['start_datetime'], True),
                                                      p_cases, bpmn_path, json_path, None, sim_log_path)

            simulated_log = read_and_preprocess_log(process_files[model_name]['sim_log'])

            emd_list.append(absolute_hour_emd(real_log, simulated_log))
            i += 1
        except:
            print('Simulation Limit exceeded: %d' % i)
            continue

    emd_list.sort()
    return emd_list[8]


def discover_from_xes_log(model_name):
    [pools_json,
     arrival_time_dist,
     json_arrival_calendar,
     gateways_branching,
     task_res_dist,
     diff_task_res_calendar,
     task_resources,
     res_calendars,
     task_events,
     task_resource_events,
     id_from_name] = preprocess_xes_log(process_files[model_name]['xes_log'],
                                        process_files[model_name]['bpmn'],
                                        process_files[model_name]['json'],
                                        15, 0.1, 0.8, 0.3)

    task_distributions = dict()
    for t_name in task_resources:
        t_id = id_from_name[t_name]
        print('Discovering Aggregated Task-Duration for task: %s' % t_name)
        task_distributions[t_id] = discover_aggregated_task_distributions(task_events[t_name])

    [non_diff_res_profiles,
     non_diff_res_calendars,
     full_time_calendar,
     non_diff_task_res_dist,
     diff_task_res_dist] = build_non_diff(canonical_json[model_name], task_resource_events,
                                          15, 0.1, 0.8, 0.3, id_from_name, task_distributions, task_res_dist)

    diff_res_non_diff_tasks = dict()
    for t_name in task_resources:
        t_id = id_from_name[t_name]
        diff_res_non_diff_tasks[t_id] = dict()
        for r_name in task_resources[t_name]:
            diff_res_non_diff_tasks[t_id][r_name] = task_distributions[t_id]

    print('Generating Scenarios ...............................')
    # Case Scenario 1: Non differentiated resources working full time.
    #                  Non differentiated task-duration distribution, i.e., resources share same task-duration function.
    save_json_file('%s%s_c_1.json' % (out_folder, model_name), non_diff_res_profiles, arrival_time_dist,
                   json_arrival_calendar, gateways_branching, non_diff_task_res_dist, full_time_calendar)

    # Case Scenario 2: Non differentiated resources with aggregated calendar.
    #                  Non differentiated task-duration distribution, i.e., resources share same task-duration function.
    save_json_file('%s%s_c_2.json' % (out_folder, model_name), non_diff_res_profiles, arrival_time_dist,
                   json_arrival_calendar, gateways_branching, non_diff_task_res_dist, non_diff_res_calendars)

    # Case Scenario 3: Differentiated resource calendars.
    #                  Non differentiated task-duration distribution, i.e., resources share same task-duration function.
    save_json_file('%s%s_c_3.json' % (out_folder, model_name), pools_json, arrival_time_dist,
                   json_arrival_calendar, gateways_branching, diff_res_non_diff_tasks, diff_task_res_calendar)

    # Case Scenario 4: Non differentiated resources working full time.
    #                  Differentiated task-duration distribution, i.e., each resource with own task-duration function.
    save_json_file('%s%s_c_4.json' % (out_folder, model_name), non_diff_res_profiles, arrival_time_dist,
                   json_arrival_calendar, gateways_branching, diff_task_res_dist, full_time_calendar)

    # Case Scenario 5: Non differentiated resources with aggregated calendar.
    #                  Differentiated task-duration distribution, i.e., each resource with own task-duration function.
    save_json_file('%s%s_c_5.json' % (out_folder, model_name), non_diff_res_profiles, arrival_time_dist,
                   json_arrival_calendar, gateways_branching, diff_task_res_dist, non_diff_res_calendars)


def save_json_file(out_f_path, res_pools, arrival_dist, arrival_calendar, gateway_branching, task_dist, res_calendars):
    to_save = {
        "resource_profiles": res_pools,
        "arrival_time_distribution": arrival_dist,
        "arrival_time_calendar": arrival_calendar,
        "gateway_branching_probabilities": gateway_branching,
        "task_resource_distribution": task_dist,
        "resource_calendars": res_calendars,
    }
    with open(out_f_path, 'w') as file_writter:
        json.dump(to_save, file_writter)


def build_non_diff(json_path, task_res_events, minutes_x_granule, min_conf, min_supp, min_cov, from_name, task_distr,
                   task_res_dist):
    with open(json_path) as json_file:
        json_data = json.load(json_file)
    task_resources = dict()

    resource_pools = dict()
    shared_pools = dict()

    for pool_info in json_data['resource_pool']:
        shared_pools[pool_info['@id']] = list()
        if pool_info['@id'] == 'QBP_DEFAULT_RESOURCE':
            resource_pools[pool_info['@id']] = ['SYSTEM']
        else:
            resource_pools[pool_info['@id']] = json_data['rol_user'][pool_info['@name']]

    for element_info in json_data['elements_data']:
        shared_pools[element_info['resource']].append(element_info['name'])
        task_resources[element_info['name']] = element_info['resource']

    calendar_factory = CalendarFactory(minutes_x_granule)

    for p_id in shared_pools:
        for t_name in shared_pools[p_id]:
            for r_name in resource_pools[p_id]:
                if t_name in ['Start', 'End']:
                    r_name = t_name
                if t_name in task_res_events and r_name in task_res_events[t_name]:
                    for ev_info in task_res_events[t_name][r_name]:
                        calendar_factory.check_date_time(p_id, t_name, ev_info.started_at)
                        calendar_factory.check_date_time(p_id, t_name, ev_info.completed_at)
    res_calendars = calendar_factory.build_weekly_calendars(min_conf, min_supp, min_cov)

    non_diff_res_profiles = dict()
    non_diff_res_calendars = dict()
    non_diff_task_res_dist = dict()
    diff_task_res_dist = dict()
    full_time_calendar = dict()
    for p_id in res_calendars:
        for t_name in shared_pools[p_id]:
            t_id = from_name[t_name]
            non_diff_res_profiles[t_id] = {
                "name": t_name,
                "resource_list": list()
            }
            non_diff_task_res_dist[t_id] = dict()
            diff_task_res_dist[t_id] = dict()
            for i in range(1, len(resource_pools[p_id]) + 1):
                r_name = '%s' % (resource_pools[p_id][i - 1])
                non_diff_res_profiles[t_id]['resource_list'].append({
                    "id": r_name,
                    "name": r_name,
                    "cost_per_hour": 1,
                    "amount": 1
                })
                non_diff_task_res_dist[t_id][r_name] = task_distr[t_id]

                diff_task_res_dist[t_id][r_name] = task_res_dist[t_id][r_name] \
                    if r_name in task_res_dist else task_distr[t_id]
                if r_name not in non_diff_res_calendars:
                    non_diff_res_calendars[r_name] = res_calendars[p_id].to_json()
                    full_time_calendar[r_name] = build_full_time_calendar('%s timetable' % r_name).to_json()

    return [non_diff_res_profiles, non_diff_res_calendars, full_time_calendar, non_diff_task_res_dist,
            diff_task_res_dist]


def read_and_preprocess_log(event_log_path: str) -> pd.DataFrame:
    # Read from CSV
    event_log = pd.read_csv(event_log_path)
    # Transform to Timestamp bot start and end columns

    event_log['StartTimestamp'] = pd.to_datetime(event_log['StartTimestamp'], utc=True)
    event_log['EndTimestamp'] = pd.to_datetime(event_log['EndTimestamp'], utc=True)

    # Sort by end timestamp, then by start timestamp, and then by activity name
    event_log = event_log.sort_values(
        ['EndTimestamp', 'StartTimestamp', 'Activity', 'CaseID', 'Resource']
    )
    # Reset the index
    event_log.reset_index(drop=True, inplace=True)
    return event_log


def absolute_hour_emd(
        event_log_1: pd.DataFrame,
        event_log_2: pd.DataFrame,
) -> float:
    # Get the first and last dates of the log

    interval_start = min(event_log_1['StartTimestamp'].min(), event_log_2['StartTimestamp'].min())
    interval_start = interval_start.replace(minute=0, second=0, microsecond=0, nanosecond=0)
    # Discretize each instant to its corresponding "bin"
    discretized_instants_1 = []

    discretized_instants_1 += [
        discretize(difference.total_seconds()) for difference in (event_log_1['StartTimestamp'] - interval_start)
    ]
    discretized_instants_1 += [
        discretize(difference.total_seconds()) for difference in (event_log_1['EndTimestamp'] - interval_start)
    ]
    # Discretize each instant to its corresponding "bin"
    discretized_instants_2 = []

    discretized_instants_2 += [
        discretize(difference.total_seconds()) for difference in (event_log_2['StartTimestamp'] - interval_start)
    ]
    discretized_instants_2 += [
        discretize(difference.total_seconds()) for difference in (event_log_2['EndTimestamp'] - interval_start)
    ]
    # Return EMD metric

    return wasserstein_distance(discretized_instants_1, discretized_instants_2)


def discretize(seconds: int):
    return math.floor(seconds / 3600)


if __name__ == "__main__":
    main()
