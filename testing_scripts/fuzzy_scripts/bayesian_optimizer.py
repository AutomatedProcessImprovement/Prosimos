import os
from datetime import datetime, timedelta

import pandas as pd
from bayes_opt import BayesianOptimization
from pathlib import Path

from enum import Enum

from pix_framework.discovery.probabilistic_multitasking.discovery import calculate_multitasking, MultiType
from pix_framework.discovery.probabilistic_multitasking.model_serialization import extend_prosimos_json
from pix_framework.io.bpm_graph import BPMNGraph

from pix_framework.io.event_log import PROSIMOS_LOG_IDS, EventLogIDs

from bpdfr_discovery.log_parser import preprocess_xes_log

from testing_scripts.fuzzy_scripts.fuzzy_test_files import test_processes_fuzzy, is_syntetic, FileType
from testing_scripts.multitasking_scripts.fuzzy_model_discovery import build_fuzzy_calendars
from testing_scripts.multitasking_scripts.multitasking_files import test_processes_multi
from testing_scripts.fuzzy_scripts.syntetic_logs_generator import get_file_path

from testing_scripts.fuzzy_scripts.icpm22_fuzzy_experiments_script import run_fuzzy_simulation, \
    _mean_by_removing_metric_boundaries, run_crisp_simulation

from pix_framework.enhancement.start_time_estimator.config import Configuration as StartTimeEstimatorConfiguration
from pix_framework.enhancement.concurrency_oracle import OverlappingConcurrencyOracle
from pix_framework.enhancement.start_time_estimator.config import ConcurrencyThresholds


class Log(Enum):
    TRAINING = 0
    TESTING = 1


class ModelType(Enum):
    MULTI_GLOBAL = 0
    MULTI_LOCAL = 1
    SEQUENTIAL = 2


current_process = [""]
op_parameter = ['']
calendar_type = [1]
is_even = [True]
is_fuzzy_file = [False]
experiment_logs = {
    Log.TRAINING: None,
    Log.TESTING: None
}
multi_info = [ModelType.SEQUENTIAL]
bpmn_model = [None]

start_proc_msg = "STARTING PROCESS: %s (%s)"
tunning_msg = '++++++++++++++ %s HYPERPARAMETER TUNNING - %s process +++++++++++++++++++++++'


def start_single_optimizer():
    is_fuzzy_file[0] = False
    op_parameter[0] = 'RED'
    print("%s - Hyperparameter Optimization" % op_parameter[0])
    proc_name = (test_processes_fuzzy if is_fuzzy_file[0] else test_processes_multi)[0]
    for mt in [ModelType.MULTI_GLOBAL, ModelType.MULTI_LOCAL]:
        print(start_proc_msg % (proc_name, str(mt)))
        multi_info[0] = mt
        calendar_type[0] = 5
        is_even[0] = True
        for is_fuzzy_discovery in [True]:
            print(tunning_msg % ('PROBABILISTIC' if is_fuzzy_discovery else 'CRISP', proc_name))
            execute_optimizer(proc_name, is_fuzzy_discovery)
            if not is_fuzzy_file[0]:
                break


def start_optimizer():
    is_fuzzy_file[0] = False
    for param in ['RED']:
        op_parameter[0] = param
        print("%s - Hyperparameter Optimization" % op_parameter[0])

        test_processes = test_processes_fuzzy if is_fuzzy_file[0] else test_processes_multi

        for i in range(0, len(test_processes)):
            proc_name = test_processes[i]
            if is_syntetic[proc_name]:
                _execute_syntetic(proc_name, test_processes, i)
            else:
                _execute_real(proc_name, test_processes, i)

    os._exit(0)


def _execute_syntetic(proc_name, test_processes, index):
    for c_type in range(4, 5):
        for mt in [ModelType.MULTI_GLOBAL, ModelType.MULTI_LOCAL, ModelType.SEQUENTIAL]:
            print(start_proc_msg % (proc_name, str(mt)))
            multi_info[0] = mt
            calendar_type[0] = c_type
            _try_calendar_types(proc_name, test_processes, index)


def _try_calendar_types(proc_name, test_processes, index):
    for even in [True, False]:
        is_even[0] = even
        if is_fuzzy_file[0]:
            naive_crisp_model(proc_name)
        for is_fuzzy_discovery in [True]:
            print(tunning_msg
                  % ('PROBABILISTIC' if is_fuzzy_discovery else 'CRISP', test_processes[index]))
            execute_optimizer(proc_name, is_fuzzy_discovery)
            if not is_fuzzy_file[0]:
                break



def _execute_real(proc_name, test_processes, index):
    # naive_crisp_model(proc_name)
    for mt in [ModelType.SEQUENTIAL, ModelType.MULTI_GLOBAL, ModelType.MULTI_LOCAL]:
        print(start_proc_msg % (proc_name, str(mt)))
        multi_info[0] = mt
        for is_fuzzy_discovery in [True]:
            print(tunning_msg % ('PROBABILISTIC' if is_fuzzy_discovery else 'CRISP', test_processes[index]))
            execute_optimizer(proc_name, is_fuzzy_discovery)


def execute_optimizer(proc_name, is_fuzzy_disc):
    if is_syntetic[proc_name]:
        print("Calendar Type %s, Even Resource Workload: %s" % (calendar_type[0], is_even[0]))
    if not is_fuzzy_disc:
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

    bpmn_model[0] = BPMNGraph.from_bpmn_path(Path(get_file_path(is_fuzzy_file[0],
                                                                proc_name,
                                                                FileType.BPMN,
                                                                60, 0, 0, calendar_type[0], is_even[0])))

    experiment_logs[Log.TRAINING] = parse_and_add_enabled_times(
        get_file_path(is_fuzzy_file[0], proc_name, FileType.TRAINING_CSV_LOG, 60, 0, 0, calendar_type[0], is_even[0]))

    experiment_logs[Log.TESTING] = parse_and_add_enabled_times(
        get_file_path(is_fuzzy_file[0], proc_name, FileType.TESTING_CSV_LOG, 60, 0, 0, calendar_type[0], is_even[0]))

    optimizer = BayesianOptimization(f=fuzzy_discovery_function, pbounds=pbounds, verbose=2, random_state=4)
    optimizer.maximize(init_points=5, n_iter=25)

    print("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))


def parse_and_add_enabled_times(csv_log_path):
    log_df = pd.read_csv(csv_log_path)
    _add_enabled_times(log_df, PROSIMOS_LOG_IDS)

    log_df[PROSIMOS_LOG_IDS.enabled_time] = pd.to_datetime(log_df[PROSIMOS_LOG_IDS.enabled_time], utc=True,
                                                           format='ISO8601')
    log_df[PROSIMOS_LOG_IDS.start_time] = pd.to_datetime(log_df[PROSIMOS_LOG_IDS.start_time], utc=True,
                                                         format='ISO8601')
    log_df[PROSIMOS_LOG_IDS.end_time] = pd.to_datetime(log_df[PROSIMOS_LOG_IDS.end_time], utc=True,
                                                       format='ISO8601')

    return log_df


def _add_enabled_times(log: pd.DataFrame, log_ids: EventLogIDs):
    configuration = StartTimeEstimatorConfiguration(
        log_ids=log_ids,
        concurrency_thresholds=ConcurrencyThresholds(df=0.75),
        consider_start_times=True,
    )
    # The start times are the original ones, so use overlapping concurrency oracle
    OverlappingConcurrencyOracle(log, configuration).add_enabled_times(log)


def crisp_discovery_function(confidence, support, participation):
    proc_name = current_process[0]

    d_start = datetime.now()
    preprocess_xes_log(
        get_file_path(is_fuzzy=True, proc_name=proc_name, file_type=FileType.TRAINING_CSV_LOG,
                      calendar_type=calendar_type[0], even=is_even[0]),
        get_file_path(is_fuzzy=True, proc_name=proc_name, file_type=FileType.BPMN,
                      calendar_type=calendar_type[0], even=is_even[0]),
        get_file_path(is_fuzzy=True, proc_name=proc_name, file_type=FileType.CRISP_JSON,
                      calendar_type=calendar_type[0], even=is_even[0]),
        60, confidence, support, participation, True, True)

    print('')
    print("Discovery Execution Time: %s" % (str(timedelta(seconds=((datetime.now() - d_start).total_seconds())))))

    log_metrics, _ = run_crisp_simulation(proc_name, s_count=5, c_typ=calendar_type[0], even=is_even[0])
    return _get_bayesian_iteration_info(log_metrics)


def fuzzy_discovery_function(angle):
    proc_name = current_process[0]

    d_start = datetime.now()
    build_fuzzy_calendars(
        log_df=experiment_logs[Log.TRAINING],
        bpmn_graph=bpmn_model[0],
        json_path=get_file_path(is_fuzzy_file[0], proc_name, FileType.SIMULATION_JSON, 60, angle, 0, calendar_type[0],
                                is_even[0]),
        i_size_minutes=60,
        angle=angle)

    if multi_info[0] in [ModelType.MULTI_GLOBAL, ModelType.MULTI_LOCAL]:
        m_type = MultiType.GLOBAL if multi_info[0] is ModelType.MULTI_GLOBAL else MultiType.LOCAL
        (probabilities, workloads) = calculate_multitasking(
            event_log=experiment_logs[Log.TRAINING],
            m_type=m_type)

        json_file = get_file_path(is_fuzzy_file[0], proc_name, FileType.SIMULATION_JSON, 60, angle, 0, calendar_type[0],
                                  is_even[0])
        extend_prosimos_json(json_file, json_file, probabilities, workloads, m_type is MultiType.LOCAL)

    print('')
    print("Discovery Execution Time: %s" % (str(timedelta(seconds=((datetime.now() - d_start).total_seconds())))))

    log_metrics, _ = run_fuzzy_simulation(proc_name, 60, angle, s_count=5, c_typ=calendar_type[0], even=is_even[0],
                                          is_fuzzy_file=is_fuzzy_file[0])
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
    start_optimizer()
