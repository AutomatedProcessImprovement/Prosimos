import json
import math

import os

from bpdfr_discovery.log_parser import discover_aggregated_task_distributions, preprocess_xes_log, \
    discover_resource_task_duration_distribution, save_prosimos_json
from bpdfr_simulation_engine.probability_distributions import create_default_distribution
from bpdfr_simulation_engine.resource_calendar import build_full_time_calendar, CalendarFactory
from testing_scripts.best_parameters_extraction import compute_median_simulation_emd

from testing_scripts.bpm_2022_testing_files import experiment_logs, process_files, out_folder, canonical_json
from bpdfr_discovery.emd_metric import read_and_preprocess_log


def main():

    for i in range(0, 9):
        model_name = experiment_logs[i]
        # transform_xes_to_csv(process_files[model_name]['xes_log'], process_files[model_name]['real_csv_log'])

        # Extracting the simulation parameters from event-log (it saves them results to JSON files)
        print("Discovering MP-DP-DA Model: Unpooled Model, Differentiated Performance, Differentiated Availability")
        print("Process Name: %s" % model_name)
        discover_from_xes_log(model_name)

        # break

    os._exit(0)


# def simulate_from_existing_files(model_name):
#     bpmn_path = process_files[model_name]['bpmn']
#     sim_log_path = process_files[model_name]['sim_log']
#     json_path = process_files[model_name]['json']
#     start_datetime = process_files[model_name]['start_datetime']
#     p_cases = 10
#
#     if "total_cases" in process_files[model_name]:
#         p_cases = process_files[model_name]["total_cases"]
#
#     total_time = 0
#     for i in range(0, 5):
#         sim_time, _ = run_diff_res_simulation(start_datetime,
#                                               p_cases, bpmn_path, json_path, None, sim_log_path)
#         total_time += sim_time
#     print("Mean Simulation Time: %.2f" % (total_time / 5))


def discover_from_xes_log(model_name):
    [granule, conf, supp, part, adj_calendar] = process_files[model_name]['disc_params']

    # best_params = discover_simulation_parameters(model_name, process_files[model_name]['xes_log'],
    #                                              process_files[model_name]['bpmn'],
    #                                              process_files[model_name]['json'])
    [diff_resource_profiles,
     arrival_time_dist,
     json_arrival_calendar,
     gateways_branching,
     task_res_dist,
     task_resources,
     diff_res_calendars,
     task_events,
     task_resource_events,
     id_from_name] = preprocess_xes_log(process_files[model_name]['xes_log'],
                                        process_files[model_name]['bpmn'],
                                        process_files[model_name]['json'], granule, conf, supp, part,
                                        adj_calendar)

    print('Generating Experimental Scenarios ...............................')
    task_distributions = dict()

    for t_name in task_resources:
        t_id = id_from_name[t_name]
        # print('Discovering Aggregated Task-Duration for task: %s' % t_name)
        task_distributions[t_id] = discover_aggregated_task_distributions(task_events[t_name], adj_calendar, None)

    [aggregated_res_profiles,
     aggregated_res_calendars,
     aggregated_task_res_dist,
     diff_res_calendar,
     diff_task_res_dist] = build_non_diff(canonical_json[model_name], task_resource_events, task_events,
                                          granule, conf, supp, part, id_from_name, adj_calendar,
                                          task_res_dist, diff_resource_profiles, diff_res_calendars)

    cases_label = [
        "SP-NP-NA: Unpooled Model, Undifferentiated Performance, Undifferentiated Availability",
        "MP-NP-NA: Pooled Model, Undifferentiated Performance, Undifferentiated Availability",
        "MP-DP-NA: Pooled Model, Differentiated Performance, Undifferentiated Availability",
        "MP-NP-DA: Pooled Model, Undifferentiated Performance, Differentiated Availability",
        "MP-DP-DA: Unpooled Model, Differentiated Performance, Differentiated Availability"
    ]

    # Case Scenario 1: Single pool sharing an aggregated calendar built from entire event log.
    #                  Aggregated task-duration function for each task in the log
    print("Generating SP-NP-NA: Unpooled Model, Undifferentiated Performance, Undifferentiated Availability")
    [c1_res_profiles, c1_task_distr, c1_calendars] = create_single_pool_model(task_resource_events, granule,
                                                                              id_from_name, conf, supp, part,
                                                                              adj_calendar)
    save_json_file('%s%s_c_1.json' % (out_folder, model_name), c1_res_profiles, arrival_time_dist,
                   json_arrival_calendar, gateways_branching, c1_task_distr, c1_calendars)

    # Case Scenario 2: Aggregated Resource Profiles/Calendars (Simod approach)
    #                  Aggregated task-duration, i.e., resources share estimated joint task-duration distribution
    print("Generating MP-NP-NA: Pooled Model, Undifferentiated Performance, Undifferentiated Availability")
    save_json_file('%s%s_c_2.json' % (out_folder, model_name), aggregated_res_profiles, arrival_time_dist,
                   json_arrival_calendar, gateways_branching, aggregated_task_res_dist, aggregated_res_calendars)

    # Case Scenario 3: Aggregated Resource Profiles/Calendars (Simod approach)
    #                  Differentiated Task-Resource Distribution
    print("Generating MP-DP-NA: Pooled Model, Differentiated Performance, Undifferentiated Availability")
    save_json_file('%s%s_c_3.json' % (out_folder, model_name), aggregated_res_profiles, arrival_time_dist,
                   json_arrival_calendar, gateways_branching, diff_task_res_dist, aggregated_res_calendars)

    # Case Scenario 4: Differentiated Resource Profiles/Calendars (Simod approach)
    #                   Aggregated task-duration, i.e., resources share estimated joint task-duration distribution
    print("Generating MP-NP-DA: Pooled Model, Undifferentiated Performance, Differentiated Availability")
    save_json_file('%s%s_c_4.json' % (out_folder, model_name), aggregated_res_profiles, arrival_time_dist,
                   json_arrival_calendar, gateways_branching, aggregated_task_res_dist, diff_res_calendar)

    p_cases = 10
    if "total_cases" in process_files[model_name]:
        p_cases = process_files[model_name]["total_cases"]

    json_paths = []
    for j in range(1, 5):
        json_paths.append('%s%s_c_%d.json' % (out_folder, model_name, j))
    json_paths.append(process_files[model_name]['json'])
    c = 1
    bpmn_path = process_files[model_name]['bpmn']
    sim_log_path = process_files[model_name]['sim_log']
    real_log = read_and_preprocess_log(process_files[model_name]['csv_log'])
    print('------------------------------------------------------------------------')

    for json_path in json_paths:
        print("Case %s:" % cases_label[c - 1])
        hour_emd_index, day_emd_index, emd_trace = compute_median_simulation_emd(model_name, p_cases, bpmn_path,
                                                                                 json_path,
                                                                                 real_log, sim_log_path)
        print("   EMD-CT --------------- %.3f" % emd_trace)
        print("   EMD-WR --------------- %.3f" % hour_emd_index)
        print('--------------------------------------------------')
        c += 1


def create_single_pool_model(task_resource_events, granule_size, id_from_name, min_conf, min_supp, min_part, adj_cal):
    joint_res_profiles = dict()
    joint_res_calendars = dict()
    joint_task_distr = dict()
    joint_calendar_factory = CalendarFactory(granule_size)
    task_events = dict()
    for t_name in task_resource_events:
        t_id = id_from_name[t_name]
        resource_list = list()
        joint_res_profiles[t_id] = dict()
        joint_task_distr[t_id] = dict()
        task_events[t_name] = list()

        for r_name in task_resource_events[t_name]:
            resource_list.append(build_profile_entry(r_name))
            for ev_info in task_resource_events[t_name][r_name]:
                task_events[t_name].append(ev_info)
                joint_calendar_factory.check_date_time('joint_res', t_name, ev_info.started_at)
                joint_calendar_factory.check_date_time('joint_res', t_name, ev_info.completed_at)

        joint_res_profiles[t_id] = {
            "name": t_name,
            "resource_list": resource_list
        }
    s_calendar = joint_calendar_factory.build_weekly_calendars(min_conf, min_supp, min_part)['joint_res']
    json_calendar = s_calendar.to_json()

    for t_name in task_resource_events:
        t_id = id_from_name[t_name]
        s_distribution = discover_aggregated_task_distributions(task_events[t_name], adj_cal, s_calendar)
        for r_name in task_resource_events[t_name]:
            joint_task_distr[t_id][r_name] = s_distribution
            if r_name not in joint_res_calendars:
                joint_res_calendars[r_name] = json_calendar

    return [joint_res_profiles, joint_task_distr, joint_res_calendars]


def create_naive_resource_profiles(task_resource_events, observed_task_resources, min_max_task_duration, id_from_name,
                                   adj_c):
    res_profiles = dict()
    res_calendars = dict()
    to_disc_res_calendar = dict()
    task_res_distribution = dict()
    naive_task_dist = dict()
    naive_diff_task_resource_distr = dict()

    n_task_res_evt = dict()
    task_res = dict()
    for t_name in task_resource_events:
        t_id = id_from_name[t_name]
        resource_list = list()
        task_res_distribution[t_id] = dict()
        naive_task_dist[t_id] = dict()
        naive_diff_task_resource_distr[t_id] = dict()
        n_task_res_evt[t_name] = dict()
        task_res[t_name] = list()

        i = 0

        for r_obs in observed_task_resources[t_name]:
            r_name = '%s_%d' % (t_name, i)
            task_res[t_name].append(r_name)
            n_task_res_evt[t_name][r_name] = task_resource_events[t_name][r_obs]
            resource_list.append(build_profile_entry(r_name))
            to_disc_res_calendar[r_name] = build_full_time_calendar('%s timetable' % r_name)
            res_calendars[r_name] = build_full_time_calendar('%s timetable' % r_name).to_json()
            naive_task_dist[t_id] = create_default_distribution(min_max_task_duration[t_name][0],
                                                                min_max_task_duration[t_name][1])
            task_res_distribution[t_id][r_name] = naive_task_dist[t_id]
            i += 1
        res_profiles[t_id] = {
            "name": t_name,
            "resource_list": resource_list
        }
    n_dist = discover_resource_task_duration_distribution(n_task_res_evt, to_disc_res_calendar, task_res, dict(), adj_c)
    for t_name in n_dist:
        naive_diff_task_resource_distr[id_from_name[t_name]] = n_dist[t_name]

    return [res_profiles, task_res_distribution, res_calendars, naive_task_dist, naive_diff_task_resource_distr]


def update_task_resource_distribution(res_profiles, naive_task_dist):
    task_res_distribution = dict()
    for t_id in res_profiles:
        task_res_distribution[t_id] = dict()
        for p_info in res_profiles[t_id]["resource_list"]:
            task_res_distribution[t_id][p_info["name"]] = naive_task_dist[t_id]
    return task_res_distribution


def build_profile_entry(res_name):
    return {
        "id": res_name,
        "name": res_name,
        "cost_per_hour": 1,
        "amount": 1
    }


def save_json_file(out_f_path, res_pools, arrival_dist, arrival_calendar, gateway_branching, task_dist, res_calendars):
    to_save = {
        "resource_profiles": res_pools,
        "arrival_time_distribution": arrival_dist,
        "arrival_time_calendar": arrival_calendar,
        "gateway_branching_probabilities": gateway_branching,
        "task_resource_distribution": task_dist,
        "resource_calendars": res_calendars,
    }
    save_prosimos_json(to_save, out_f_path)


def build_non_diff(json_path, task_res_events, task_events, minutes_x_granule, min_conf, min_supp, min_cov, from_name,
                   with_fit_c, discovered_task_res_dist, discovered_pools_json, discovered_res_calendars):
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

    task_pools_events = dict()
    diff_res_calendar = dict()

    for p_id in shared_pools:
        for t_name in shared_pools[p_id]:
            task_pools_events[t_name] = {p_id: list()}
            for r_name in resource_pools[p_id]:
                if r_name == 'SYSTEM':
                    r_name = t_name
                if t_name in task_res_events and r_name in task_res_events[t_name]:
                    task_pools_events[t_name][p_id] += task_res_events[t_name][r_name]
                    for ev_info in task_res_events[t_name][r_name]:
                        calendar_factory.check_date_time(p_id, t_name, ev_info.started_at)
                        calendar_factory.check_date_time(p_id, t_name, ev_info.completed_at)
    res_calendars = calendar_factory.build_weekly_calendars(min_conf, min_supp, min_cov)

    aggregated_res_profiles = dict()
    aggregated_res_calendars = dict()
    aggregated_task_distribution = dict()

    diff_task_res_dist = dict()

    for p_id in res_calendars:
        for t_name in shared_pools[p_id]:
            t_id = from_name[t_name]
            t_pool_distr = discover_aggregated_task_distributions(task_pools_events[t_name][p_id], with_fit_c,
                                                                  res_calendars[p_id])

            if t_id not in aggregated_res_profiles:
                aggregated_res_profiles[t_id] = {
                    "name": t_name,
                    "resource_list": list()
                }
                aggregated_task_distribution[t_id] = dict()
                diff_task_res_dist[t_id] = dict()

            for r_name in resource_pools[p_id]:
                if r_name == 'SYSTEM':
                    r_name = t_name
                aggregated_res_profiles[t_id]['resource_list'].append({
                    "id": r_name,
                    "name": r_name,
                    "cost_per_hour": 1,
                    "amount": 1
                })

                aggregated_task_distribution[t_id][r_name] = t_pool_distr
                if r_name not in aggregated_res_calendars:
                    aggregated_res_calendars[r_name] = res_calendars[p_id].to_json()
                    if r_name in discovered_res_calendars:
                        diff_res_calendar[r_name] = discovered_res_calendars[r_name].to_json()
                    else:
                        diff_res_calendar[r_name] = aggregated_res_calendars[r_name]

                if r_name not in diff_task_res_dist[t_id]:
                    if r_name in discovered_task_res_dist[t_id]:
                        diff_task_res_dist[t_id][r_name] = discovered_task_res_dist[t_id][r_name]
                    else:
                        diff_task_res_dist[t_id][r_name] = aggregated_task_distribution[t_id][r_name]

    # Fixing Missing tasks in the BPMN model discovered by Simod
    for t_name in task_events:
        t_id = from_name[t_name]

        if t_id not in aggregated_res_profiles:
            c_factory = CalendarFactory(minutes_x_granule)
            for ev_info in task_events[t_name]:
                c_factory.check_date_time('aggregated', t_name, ev_info.started_at)
                c_factory.check_date_time('aggregated', t_name, ev_info.completed_at)
            agg_calendar = c_factory.build_weekly_calendars(min_conf, min_supp, min_cov)['aggregated']
            agg_distr = discover_aggregated_task_distributions(task_events[t_name], with_fit_c, agg_calendar)

            aggregated_res_profiles[t_id] = discovered_pools_json[t_id]
            if t_id not in aggregated_task_distribution:
                aggregated_task_distribution[t_id] = dict()
                diff_task_res_dist[t_id] = dict()

            for r_info in aggregated_res_profiles[t_id]["resource_list"]:
                r_name = r_info['name']
                aggregated_task_distribution[t_id][r_name] = agg_distr
                if r_name not in aggregated_res_calendars:
                    aggregated_res_calendars[r_name] = agg_calendar.to_json()
                if r_name not in diff_task_res_dist[t_id]:
                    diff_task_res_dist[t_id][r_name] = discovered_task_res_dist[t_id][r_name]
                if r_name not in diff_res_calendar:
                    if t_id in discovered_res_calendars in discovered_res_calendars and r_name in \
                            discovered_res_calendars[t_id]:
                        diff_res_calendar[r_name] = discovered_res_calendars[t_id][r_name]
                    else:
                        diff_res_calendar[r_name] = agg_calendar.to_json()

    return [aggregated_res_profiles, aggregated_res_calendars, aggregated_task_distribution,
            diff_res_calendar, diff_task_res_dist]


def in_resource_list(t_id, r_id, res_profiles):
    for r_info in res_profiles[t_id]["resource_list"]:
        if r_id in [r_info["name"], r_info["id"]]:
            return True
    return False


def median_absolute_deviation(emd_list):
    emd_list.sort(key=lambda x: x.hour_emd_index)
    i = math.ceil(len(emd_list) / 2)
    median = emd_list[i].hour_emd_index
    med_diff = list()
    for e in emd_list:
        med_diff.append(abs(median - e.hour_emd_index))
    med_diff.sort()
    return emd_list[i].hour_emd_index, emd_list[i].day_emd_index, med_diff[i]


if __name__ == "__main__":
    main()
