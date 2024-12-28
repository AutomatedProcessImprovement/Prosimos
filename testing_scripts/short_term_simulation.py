import os
import json
import datetime

import pytest
from pix_framework.io.event_log import read_csv_log, PROSIMOS_LOG_IDS

from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation

test_short_term_simulation_exact_cases = [
    {
        'test_id': "Simple",
        'bps_model_path': "./assets/short-term/STsim_test_simple_exact.bpmn",
        'sim_params_path': "./assets/short-term/STsim_test_simple_exact.json",
        'simulation_horizon': "2024-01-01T07:59:59.000Z",
        'ongoing_cases_path': "./assets/short-term/STsim_test_simple_exact__executed.csv",
        'ground_truth_path': "./assets/short-term/STsim_test_simple_exact__continuation.csv",
    },
    # {
    #     'test_id': "Simple with ongoing activities",
    #     'bps_model_path': "./assets/short-term/STsim_test_simple_exact.bpmn",
    #     'sim_params_path': "./assets/short-term/STsim_test_simple_exact.json",
    #     'simulation_horizon': "2024-01-01T07:59:59.000Z",
    #     'ongoing_cases_path': "./assets/short-term/STsim_test_simple_exact__executed_wOngoingAct.csv",
    #     'ground_truth_path': "./assets/short-term/STsim_test_simple_exact__continuation_wOngoingAct.csv"
    # }
]


@pytest.mark.parametrize(
    "test_data",
    test_short_term_simulation_exact_cases,
    ids=[test_data['test_id'] for test_data in test_short_term_simulation_exact_cases]
)
def test_short_term_simulation_exact(test_data):
    # Discover the ongoing process state
    log_ids = PROSIMOS_LOG_IDS
    ongoing_cases = read_csv_log(test_data['ongoing_cases_path'], log_ids)
    # simulation_starting_point = max(
    #     max(ongoing_cases[log_ids.enabled_time]),
    #     max(ongoing_cases[log_ids.start_time]),
    #     max(ongoing_cases[log_ids.end_time])
    # )
    with open('./assets/short-term/output.json', 'r') as f:
        process_state = json.load(f)

    process_state = parse_process_state(process_state)
    # Run Prosimos in short-term mode
    output_path = "./assets/short-term/out/output_log.csv"
    _ = run_diff_res_simulation(
        '2024-01-01 01:59:59.000000+02:00',
        30,  # Short-term, Prosimos will simulate based on time
        test_data['bps_model_path'],
        test_data['sim_params_path'],
        None,  # No simulation stats needed
        output_path,
        process_state=process_state,
        simulation_horizon=parse_datetime(test_data['simulation_horizon']),
    )
    # Assert expected result
    simulated_continuation = read_csv_log(output_path, log_ids)
    ground_truth = read_csv_log(test_data['ground_truth_path'], log_ids)
    assert simulated_continuation.equals(ground_truth)
    # Remove intermediate files
    os.remove(output_path)

def parse_process_state(process_state):
    process_state['last_case_arrival'] = parse_datetime(process_state['last_case_arrival'])

    for resource_id, end_time_str in process_state.get('resource_last_end_times', {}).items():
        process_state['resource_last_end_times'][resource_id] = parse_datetime(end_time_str)

    for case_id, case_data in process_state.get('cases', {}).items():
        # Convert enabled activity times
        for activity in case_data.get('enabled_activities', []):
            activity['enabled_time'] = parse_datetime(activity['enabled_time'])
        # Convert ongoing activity times
        for activity in case_data.get('ongoing_activities', []):
            activity['start_time'] = parse_datetime(activity['start_time'])
    return process_state

def parse_datetime(datetime_str):
    return datetime.datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))