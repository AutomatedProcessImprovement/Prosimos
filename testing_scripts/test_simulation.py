import datetime
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation
from testing_scripts.test_update_state import _setup_sim_scenario_file


@pytest.fixture
def assets_path(request) -> Path:
    entry_path: Path
    if os.path.basename(os.getcwd()) == 'testing_scripts':
        entry_path = Path('assets')
    else:
        entry_path = Path('testing_scripts/assets')

    def teardown():
        files_to_delete = [
            'timer_with_task_stats.csv',
            'timer_with_task_logs.csv'
        ]

        for file in files_to_delete:
            output_path = entry_path / file
            if output_path.exists():
                os.remove(output_path)

    request.addfinalizer(teardown)

    return entry_path


def test_timer_event_correct_duration_in_sim_logs(assets_path):
    """
    Input: run simulation with writting events to the log file

    Output:
    1) validate that the file does include both task and event per each case
    2) validate the duration of the logged task (30 min)
    3) validate the duration of the logged timer event (15 min)
    """
    # ====== ARRANGE ======

    model_path = assets_path / 'timer_with_task.bpmn'

    json_path = assets_path / 'timer_with_task.json'
    _setup_and_write_case_attributes(json_path, [])

    sim_stats = assets_path / 'timer_with_task_stats.csv'
    sim_logs = assets_path / 'timer_with_task_logs.csv'

    start_string = '2022-06-21 13:22:30.035185+03:00'

    # ====== ACT ======
    _, _ = run_diff_res_simulation(start_string,
                                   5,
                                   model_path,
                                   json_path,
                                   sim_stats,
                                   sim_logs,
                                   True)

    # ====== ASSERT ======
    df = pd.read_csv(sim_logs)
    grouped_by_case_id = df.groupby(by="case_id")["case_id"]
    assert grouped_by_case_id.count().size == 5, \
        "The total number of simulated cases does not equal to the setup number"

    for name, group in grouped_by_case_id:
        assert group.size == 2, \
            f"The case '{name}' does not have the required number of logged simulated activities"

    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')

    only_timer_events = df[df['activity'] == '15m']
    expected_task_timedelta = datetime.timedelta(minutes=15)
    expected_task_count = 5
    _verify_activity_count_and_duration(only_timer_events, expected_task_count, expected_task_timedelta)

    # other events should include only task 
    # with the fixed distribution of 30 minutes
    df = df[df['activity'] != '15m']
    expected_task_timedelta = datetime.timedelta(minutes=30)
    expected_task_count = 5
    _verify_activity_count_and_duration(df, expected_task_count, expected_task_timedelta)



def test_histogram_sampling_arrival_distribution_in_sim_logs(assets_path):
    """
    Input: run simulation with an arrival distribution sampling from a custom histogram

    Output:
    1) validate that the simulation does not fail
    """
    # ====== ARRANGE ======

    model_path = assets_path / 'LoanApp_sequential_9-5.bpmn'
    json_path = assets_path / 'LoanApp_arrival_fix_10.json'
    sim_logs = assets_path / 'LoanApp_arrival_fix_10_sim_log.csv'

    # ====== ACT ======
    _, _ = run_diff_res_simulation(start_date="2022-06-21 13:22:30.035185+03:00",
                                   total_cases=100,
                                   bpmn_model=model_path,
                                   json_sim_params=json_path,
                                   out_stats_csv_path=None,
                                   out_log_csv_path=sim_logs)

    # ====== ASSERT ======
    df = pd.read_csv(sim_logs)
    assert len(df['case_id'].unique()) == 100


def test_timer_event_no_events_in_logs(assets_path):
    """
    Input: run simulation without writting events to the log file

    Output:
    1) validate that the file does include only tasks (automatically means no event)
    2) validate the duration of the logged task (30 min)
    """

    # ====== ARRANGE ======

    model_path = assets_path / 'timer_with_task.bpmn'

    json_path = assets_path / 'timer_with_task.json'
    _setup_and_write_case_attributes(json_path, [])

    sim_stats = assets_path / 'timer_with_task_stats.csv'
    sim_logs = assets_path / 'timer_with_task_logs.csv'

    start_string = '2022-06-21 13:22:30.035185+03:00'

    # ====== ACT ======
    _, _ = run_diff_res_simulation(start_string,
                                   5,
                                   model_path,
                                   json_path,
                                   sim_stats,
                                   sim_logs,
                                   False)

    # ====== ASSERT ======
    df = pd.read_csv(sim_logs)
    grouped_by_case_id = df.groupby(by="case_id")["case_id"]
    assert grouped_by_case_id.count().size == 5, \
        "The total number of simulated cases does not equal to the setup number"

    for name, group in grouped_by_case_id:
        assert group.size == 1, \
            f"The case '{name}' does not have the required number of logged simulated activities"

    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')

    # events should include only task 
    # with the fixed distribution of 30 minutes
    expected_task_timedelta = datetime.timedelta(minutes=30)
    expected_task_count = 5
    _verify_activity_count_and_duration(df, expected_task_count, expected_task_timedelta)



def test_event_based_gateway_correct(assets_path):
    """
    Input:      BPMN model consists timer event and event-based gateway.
                Event-based gateway contains three alternative directions.

    Expected:   The event with the lowest duration time ('Timer Event' in our case) is being executed for all cases.
                Duration of 'Timer Event' is being verified (equals 3 hours).
    """

    # ====== ARRANGE ======
    model_path = assets_path / 'stock_replenishment.bpmn'
    json_path = assets_path / 'stock_replenishment_logs.json'
    sim_stats = assets_path / 'with_event_gateway_stats.csv'
    sim_logs = assets_path / 'with_event_gateway_logs.csv'

    event_distr_array = [
        {
            "event_id": "Event_0761x5g",
            "distribution_name": "fix",
            "distribution_params": [
                {
                    "value": 14400
                }
            ]
        },
        {
            "event_id": "Event_1qclhcl",
            "distribution_name": "fix",
            "distribution_params": [
                {
                    "value": 18000
                }
            ]
        },
        {
            "event_id": "Event_052kspk",
            "distribution_name": "fix",
            "distribution_params": [
                {
                    "value": 14400
                }
            ]
        },
        {
            "event_id": "Event_0bsdbzb",
            "distribution_name": "fix",
            "distribution_params": [
                {
                    "value": 10800
                }
            ]
        }
    ]

    _setup_sim_scenario_file(json_path, event_distr_array)

    start_string = '2022-06-21 13:22:30.035185+03:00'

    # ====== ACT ======
    _, _ = run_diff_res_simulation(start_string,
                                   5,
                                   model_path,
                                   json_path,
                                   sim_stats,
                                   sim_logs,
                                   True)

    # ====== ASSERT ======
    df = pd.read_csv(sim_logs)

    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')

    # verify that occurence of 'Timer Event' event is 5 
    #   (that means it was executed in 100% executed cases)
    # verify that duration of the 'Timer Event' event is 3 hours
    expected_timer_timedelta = datetime.timedelta(hours=3)
    only_timer_events = df[df['activity'] == 'Timer Event']
    _verify_activity_count_and_duration(only_timer_events, 5, expected_timer_timedelta)

    # verify that occurence of '4h' event is 5 
    #   (that means it was executed in 100% executed cases)
    # verify that duration of the '4h' event is 4 hours
    expected_timer_timedelta = datetime.timedelta(hours=4)
    only_four_h_events = df[df['activity'] == '4h']
    _verify_activity_count_and_duration(only_four_h_events, 5, expected_timer_timedelta)


def _verify_activity_count_and_duration(activities, count, expected_activity_timedelta):
    assert activities.shape[0] == count, \
        f"The total number of activities in the log file should be equal to {count}"
    
    end_start_diff_for_task = activities['end_time'] - activities['start_time']
    for diff in end_start_diff_for_task:
        assert diff == expected_activity_timedelta, \
            f"The duration of the activity does not equal to {expected_activity_timedelta}"


def _setup_and_write_case_attributes(json_path, case_attributes):
    with open(json_path, "r") as f:
        json_dict = json.load(f)

    json_dict["case_attributes"] = case_attributes

    with open(json_path, "w+") as json_file:
        json.dump(json_dict, json_file)
