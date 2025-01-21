import copy
import os
import random
import shutil
from pathlib import Path
import pandas as pd

from bpdfr_discovery.log_parser import event_list_from_csv
from pix_framework.discovery.probabilistic_multitasking.discovery import calculate_multitasking, MultiType, \
    MultitaskInfo
from pix_framework.discovery.probabilistic_multitasking.model_serialization import extend_prosimos_json
from prosimos.simulation_engine import run_simulation
from testing_scripts.fuzzy_scripts.bayesian_optimizer import start_optimizer, start_single_optimizer
from testing_scripts.fuzzy_scripts.fuzzy_discovery_script import split_event_log
from testing_scripts.fuzzy_scripts.icpm22_fuzzy_experiments_script import split_synthetic_log, Steps, \
    discover_fuzzy_parameters, granule_sizes, trapezoidal_angles
from testing_scripts.fuzzy_scripts.syntetic_logs_generator import generate_syntetic_multitasking_log, get_file_path, \
    generate_syntetic_log_vacation

from testing_scripts.multitasking_scripts.multitasking_files import get_file_path_multi, FileType, test_processes_multi, \
    is_syntetic

assets_dir = Path(__file__).parent.parent / "assets/multitasking"
bpmn = assets_dir / "bpmn_models"
json = assets_dir / "json_params"
out_log = assets_dir / "out_csv_log"

active_steps = {
    Steps.LOG_INFO: False,
    Steps.TO_UTC: False,
    Steps.SYNTETIC_LOG_GEN: False,
    Steps.VACATION_SYNTETIC: False,
    Steps.VACATION_IMPACT: True,
    Steps.SPLIT_LOG: False,
    Steps.FUZZY_DISCOVERY: False,
    Steps.FUZZY_SIMULATION: False,
    Steps.METRICS: False,
    Steps.MULTITASKING_DISCOVERY: False,
    Steps.BAYESIAN_OPTIMIZER: False,

}


def main():
    for i in range(0, len(test_processes_multi)):
        proc_name = test_processes_multi[i]
        synthetic = is_syntetic[proc_name]
        if active_steps[Steps.LOG_INFO]:
            compute_log_stats(proc_name, synthetic)
        if active_steps[Steps.SYNTETIC_LOG_GEN]:
            generate_syntetic_multitasking_log(proc_name, False, 2000)

        if active_steps[Steps.VACATION_IMPACT]:
            evaluate_vacation_impact(proc_name, single_resource=True)
            break
        if active_steps[Steps.VACATION_SYNTETIC]:
            generate_syntetic_log_vacation(proc_name, False, 2000, num_res=1, single_resource=True, arrival_rate=2)
            break

        if active_steps[Steps.SPLIT_LOG]:
            if synthetic:
                split_synthetic_log(proc_name, False)
            else:
                split_event_log(get_file_path(False, proc_name, FileType.ORIGINAL_CSV_LOG),
                                get_file_path(False, proc_name, FileType.TRAINING_CSV_LOG),
                                get_file_path(False, proc_name, FileType.TESTING_CSV_LOG),
                                0.5)
        if active_steps[Steps.FUZZY_DISCOVERY]:
            discover_fuzzy_parameters(proc_name, False)
        if active_steps[Steps.MULTITASKING_DISCOVERY]:
            for m_type in [MultiType.GLOBAL, MultiType.LOCAL]:
                discover_multitasking_probabilities(proc_name, m_type)
        if active_steps[Steps.BAYESIAN_OPTIMIZER]:
            start_optimizer()

    os._exit(0)


def evaluate_vacation_impact(proc_name, single_resource):
    str_method = ["Resource Working All Time - No rest",
                  "Resource Resting in Trainig but not in Testing",
                  "Resource Resting in Testing but Not in Training",
                  "Resource Resting in Training and Testing"]
    traces = event_list_from_csv(
        get_file_path(is_fuzzy=False, proc_name=proc_name, file_type=FileType.ORIGINAL_CSV_LOG,
                      calendar_type=5, even=True))

    # f_res = ['Clerk-000001',
    #          'Appraiser-000001', 'AML Investigator-000001', 'Loan Officer-000001', 'Senior Officer-000001']
    f_res = ["R1"]
    task_res, dates_at = find_datetime_at([0, 0.10, 0.25, 0.5, 0.6, 0.75, 1.0], traces, f_res)

    print("Simulation period tarts in %s" % dates_at[0])
    print("Simulation period ends at %s" % dates_at[6])
    print("-------------------------------------------------")

    print("Resource Working All Time - No rest")
    split_and_optimize(proc_name, traces)

    if single_resource:
        f_res = None
    for w_count in [1, 2, 4, 8]:
        for p_method in range(1, 4):
            print("%s -> Resting Time %d Weeks" % (str_method[p_method], w_count))
            c_traces = copy.deepcopy(traces)
            if p_method == 1:
                c_traces = shift_dates_to_right(c_traces, w_count * 7, dates_at[1], f_res, task_res)
            elif p_method == 2:
                c_traces = shift_dates_to_right(c_traces, w_count * 7, dates_at[4], f_res, task_res)
            elif p_method == 3:
                c_traces = shift_dates_to_right(c_traces, w_count * 7, dates_at[4], f_res, task_res)
                c_traces = shift_dates_to_right(c_traces, w_count * 7, dates_at[1], f_res, task_res)

            split_and_optimize(proc_name, c_traces)


def find_datetime_at(positions: list, traces: list, fix_res: list):
    task_res = dict()
    taken = dict()
    all_dates = []
    for tr in traces:
        for ev in tr.event_list:
            all_dates.append(ev.started_at)
            all_dates.append(ev.completed_at)
            if ev.task_id not in task_res:
                task_res[ev.task_id] = []
                taken[ev.task_id] = set()
            if ev.resource_id not in fix_res and ev.resource_id not in taken[ev.task_id]:
                task_res[ev.task_id].append(ev.resource_id)
                taken[ev.task_id].add(ev.resource_id)
    all_dates.sort()
    dates = []
    for i_ratio in positions:
        i = int(i_ratio * (len(all_dates) - 1))
        dates.append(all_dates[i])
    return task_res, dates


def shift_dates_to_right(traces: list, days: int, from_date, f_res, task_res):
    if f_res is None:
        for tr in traces:
            for ev in tr.event_list:
                if ev.started_at >= from_date:
                    ev.started_at = ev.started_at + pd.Timedelta(days=days)
                    ev.completed_at = ev.completed_at + pd.Timedelta(days=days)
                elif ev.completed_at >= from_date:
                    ev.completed_at = ev.completed_at + pd.Timedelta(days=days)
    else:
        to_date = from_date + pd.Timedelta(days=days)
        for tr in traces:
            for ev in tr.event_list:
                if ev.resource_id in f_res:
                    if from_date <= ev.started_at <= to_date:
                        if ev.resource_id == 'Senior Officer-000001':
                            ev.started_at = ev.started_at + pd.Timedelta(days=days)
                            ev.completed_at = ev.completed_at + pd.Timedelta(days=days)
                        else:
                            r_id = random.choice(task_res[ev.task_id])
                            ev.resource_id = r_id
                    elif from_date <= ev.completed_at <= to_date and ev.resource_id in f_res:
                        ev.completed_at = ev.completed_at + pd.Timedelta(days=days)
    return traces


def split_and_optimize(proc_name, c_traces):
    split_event_log(
        get_file_path(False, proc_name, FileType.ORIGINAL_CSV_LOG, 60, 0, 1, calendar_type=5, even=True),
        get_file_path(False, proc_name, FileType.TRAINING_CSV_LOG, 60, 0, 1, calendar_type=5, even=True),
        get_file_path(False, proc_name, FileType.TESTING_CSV_LOG, 60, 0, 1, calendar_type=5, even=True),
        0.5,
        c_traces)
    start_single_optimizer()


def discover_multitasking_probabilities(proc_name, m_type: MultiType):
    if is_syntetic[proc_name]:
        for c_type in range(1, 5):
            for even in [True, False]:
                for g_size in granule_sizes:
                    for angle in trapezoidal_angles:
                        probabilities = calculate_multitasking(
                            pd.read_csv(get_file_path(is_fuzzy=False, proc_name=proc_name,
                                                      file_type=FileType.TRAINING_CSV_LOG,
                                                      granule=g_size, angle=angle, file_index=0,
                                                      calendar_type=c_type, even=even)),
                            m_type, 60)

                        json_file = get_file_path(is_fuzzy=False, proc_name=proc_name,
                                                  file_type=FileType.SIMULATION_JSON,
                                                  granule=g_size, angle=angle, file_index=0,
                                                  calendar_type=c_type, even=even)

                        directory = os.path.dirname(json_file)
                        base_name = os.path.basename(json_file)
                        name_without_extension, extension = os.path.splitext(base_name)
                        new_file_name = f"{name_without_extension}_{str(m_type)}{extension}"

                        extend_prosimos_json(json_file, os.path.join(directory, new_file_name), probabilities, False)


def compute_log_stats(proc_name: str, syntetic, is_fuzzy: bool = False):
    print('Log Info: %s' % proc_name)
    if syntetic:
        for c_type in range(1, 5):
            for even in [True, False]:
                print('Calendar Type %d, even %s' % (c_type, str(even)))
                training_log = pd.read_csv(
                    get_file_path(is_fuzzy, proc_name, FileType.TRAINING_CSV_LOG, 60, 0, 0, c_type, even))
                testing_log = pd.read_csv(
                    get_file_path(is_fuzzy, proc_name, FileType.TESTING_CSV_LOG, 60, 0, 0, c_type, even))
                _calculate_task_intersections(pd.concat([training_log, testing_log], ignore_index=True))
                print('-------------------------------------------------')
    else:
        training_log = pd.read_csv(get_file_path(is_fuzzy, proc_name, FileType.TRAINING_CSV_LOG))
        testing_log = pd.read_csv(get_file_path(is_fuzzy, proc_name, FileType.TESTING_CSV_LOG))
        _calculate_task_intersections(pd.concat([training_log, testing_log], ignore_index=True))
        print('-------------------------------------------------')


def _calculate_task_intersections(event_log: pd.DataFrame):
    resource_multitask_info = dict()
    total_events = 0
    task_groups = list()
    initial_multi = dict()
    cases_set = set()
    tasks_set = set()
    resource_set = set()

    for resource, group in event_log.groupby('resource'):
        resource_multitask_info[resource] = MultitaskInfo()
        events = []

        for index, row in group.iterrows():
            events.append((row['start_time'], 'start', index))
            events.append((row['end_time'], 'end', index))
            cases_set.add(row['case_id'])
            tasks_set.add(row['activity'])
            resource_set.add(row['resource'])

        # Sort the events by time
        events.sort(key=lambda x: x[0])

        active_tasks = set()

        for time, event_type, index in events:
            if event_type == 'start':
                active_tasks.add(index)
                while len(task_groups) <= len(active_tasks):
                    task_groups.append(0)
                initial_multi[index] = len(active_tasks)
            else:
                total_events += 1
                task_groups[max(initial_multi[index], len(active_tasks))] += 1
                active_tasks.remove(index)
    print('Total Cases: %d' % len(cases_set))
    print('Total Activities: %d' % len(tasks_set))
    print('Total Resources: %d' % len(resource_set))
    print('Total Events: %d' % total_events)
    sum_m, tot_m = 0, 0
    for i in range(0, len(task_groups)):
        if task_groups[i] > 0:
            sum_m += (i * task_groups[i])
            tot_m += task_groups[i]
    count = 0
    for i in range(0, len(task_groups)):
        if count + task_groups[i] >= tot_m / 2:
            print('Median Multitasking: %f' % i)
            break
        count += task_groups[i]
    print('Average Multitasking: %f' % (sum_m / tot_m))
    print('Max Multitasking: %f' % (len(task_groups)))
    print('Ratio Multitasking: %f' % ((tot_m - task_groups[1]) / tot_m))

    return resource_multitask_info


def execute_prosimos_simulation(process_name: str, with_miltitasking: bool = True):
    result = run_simulation(
        bpmn_path=get_file_path_multi("sequential", FileType.BPMN),
        json_path=get_file_path_multi("sequential_global", FileType.SIMULATION_JSON),
        total_cases=100)


if __name__ == "__main__":
    main()
