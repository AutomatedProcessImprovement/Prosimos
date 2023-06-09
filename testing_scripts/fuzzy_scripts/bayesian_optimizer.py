import os
from datetime import datetime, timedelta

from bayes_opt import BayesianOptimization, UtilityFunction
import warnings

from bpdfr_discovery.log_parser import preprocess_xes_log
from simod.fuzzy_calendars.fuzzy_discovery import build_fuzzy_calendars
from testing_scripts.fuzzy_scripts.fuzzy_discovery_script import discover_model_from_csv_log
from testing_scripts.fuzzy_scripts.fuzzy_test_files import test_processes, is_syntetic, FileType, get_file_path
from testing_scripts.fuzzy_scripts.icpm22_fuzzy_experiments_script import discover_crisp_calendars, \
    simulate_and_save_crisp_model, simulate_and_save_results, discover_fuzzy_parameters, run_fuzzy_simulation, \
    _mean_by_removing_metric_boundaries, run_crisp_simulation

current_process = [""]
op_parameter = ['']
calendar_type = [1]
is_even = [True]


def main():
    for param in ['RED']:
        op_parameter[0] = param
        print("%s - Hyperparameter Optimization" % op_parameter[0])

        for i in range(6, len(test_processes)):
            proc_name = test_processes[i]
            print("STARTING PROCESS: %s " % proc_name)
            if is_syntetic[proc_name]:
                for c_type in range(1, 5):
                    calendar_type[0] = c_type
                    for even in [True, False]:
                        is_even[0] = even
                        naive_crisp_model(proc_name)
                        for is_fuzzy in [True, False]:
                            print('++++++++++++++ %s HYPERPARAMETER TUNNING - %s process +++++++++++++++++++++++'
                                  % ('PROBABILISTIC' if is_fuzzy else 'CRISP', test_processes[i]))
                            execute_optimizer(proc_name, is_fuzzy)
            else:
                # naive_crisp_model(proc_name)
                for is_fuzzy in [True, False]:
                    print('++++++++++++++ %s HYPERPARAMETER TUNNING - %s process +++++++++++++++++++++++'
                          % ('PROBABILISTIC' if is_fuzzy else 'CRISP', test_processes[i]))
                    execute_optimizer(proc_name, is_fuzzy)

            break
    os._exit(0)


def execute_optimizer(proc_name, is_fuzzy):
    if is_syntetic[proc_name]:
        print("Calendar Type %s, Even Resource Workload: %s" % (calendar_type[0], is_even[0]))
    if not is_fuzzy:
        crisp_bayesian_optimizer(proc_name)
    else:
        fuzzy_bayesian_optimizer(proc_name)


def naive_crisp_model(proc_name):
    current_process[0] = proc_name
    print('++++++++++++++ NAIVE-CRISP process model +++++++++++++++++++++++')
    if is_syntetic[proc_name]:
        print("Calendar Type %s, Even Resource Workload: %s" % (calendar_type[0], is_even[0]))
    crisp_discovery_function(confidence=0.0, support=0.0, participation=0.0)
    print()


def crisp_bayesian_optimizer(proc_name):
    current_process[0] = proc_name
    pbounds = {"confidence": (0.1, 1.0),
               "support": (0.1, 1.0),
               "participation": (0.1, 1.0)}

    optimizer = BayesianOptimization(f=crisp_discovery_function, pbounds=pbounds, verbose=2, random_state=4)
    optimizer.maximize(init_points=5, n_iter=25)

    print("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))


def fuzzy_bayesian_optimizer(proc_name):
    current_process[0] = proc_name
    pbounds = {"angle": (0.0, 1.0)}

    optimizer = BayesianOptimization(f=fuzzy_discovery_function, pbounds=pbounds, verbose=2, random_state=4)
    optimizer.maximize(init_points=5, n_iter=25)

    print("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))


def crisp_discovery_function(confidence, support, participation):
    proc_name = current_process[0]

    d_start = datetime.now()
    preprocess_xes_log(
        get_file_path(
            proc_name=proc_name, file_type=FileType.TRAINING_CSV_LOG, calendar_type=calendar_type[0], even=is_even[0]),
        get_file_path(
            proc_name=proc_name, file_type=FileType.BPMN, calendar_type=calendar_type[0], even=is_even[0]),
        get_file_path(
            proc_name=proc_name, file_type=FileType.CRISP_JSON, calendar_type=calendar_type[0], even=is_even[0]),
        60, confidence, support, participation, True, True)

    print('')
    print("Discovery Execution Time: %s" % (str(timedelta(seconds=((datetime.now() - d_start).total_seconds())))))

    log_metrics, _ = run_crisp_simulation(proc_name, s_count=5, c_typ=calendar_type[0], even=is_even[0])
    return _get_bayesian_iteration_info(log_metrics)


def fuzzy_discovery_function(angle):
    proc_name = current_process[0]

    d_start = datetime.now()
    build_fuzzy_calendars(
        csv_log_path=get_file_path(proc_name, FileType.TRAINING_CSV_LOG, 60, angle, 0, calendar_type[0], is_even[0]),
        bpmn_path=get_file_path(proc_name, FileType.BPMN, 60, angle, 0, calendar_type[0], is_even[0]),
        json_path=get_file_path(proc_name, FileType.SIMULATION_JSON, 60, angle, 0, calendar_type[0], is_even[0]),
        i_size_minutes=60,
        angle=angle)

    # discover_model_from_csv_log(proc_name, 60, angle, calendar_type[0], is_even[0])
    print('')
    print("Discovery Execution Time: %s" % (str(timedelta(seconds=((datetime.now() - d_start).total_seconds())))))

    log_metrics, _ = run_fuzzy_simulation(proc_name, 60, angle, s_count=5, c_typ=calendar_type[0], even=is_even[0])
    return _get_bayesian_iteration_info(log_metrics)


def _get_bayesian_iteration_info(log_metrics):
    mean_metrics = _mean_by_removing_metric_boundaries(log_metrics, op_parameter[0])
    aed, red = str(round(mean_metrics['AED'], 2)), str(round(mean_metrics['RED'], 2))
    ced, ctd = str(round(mean_metrics['CED'], 2)), str(round(mean_metrics['CTD'], 2))
    mtr = str(round(mean_metrics['MTR'], 3))
    tar, wtr = str(round(mean_metrics['TAR'], 3)), str(round(mean_metrics['WTR'], 3))
    print(f'|           |MTR: {mtr:6}|TAR: {tar:6}|WTR: {wtr:6}|')
    print(f'|           |AED: {aed:6}|RED: {red:6}|CED: {ced:6}|CTD: {ctd:6}|')

    return -1 * mean_metrics[op_parameter[0]]


if __name__ == "__main__":
    main()
