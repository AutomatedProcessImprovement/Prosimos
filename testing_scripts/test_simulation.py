import os
import pytest
import pandas as pd

import datetime
from pathlib import Path
from bpdfr_simulation_engine.resource_calendar import parse_datetime

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
    sim_stats = assets_path / 'timer_with_task_stats.csv'
    sim_logs = assets_path / 'timer_with_task_logs.csv'

    start_string = '2022-06-21 13:22:30.035185+03:00'
    start_date = parse_datetime(start_string, True)

    # ====== ACT ======
    _, _ = run_diff_res_simulation(start_date,
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
    
    expected_timer_timedelta = datetime.timedelta(minutes=15)
    _verify_event_count_and_duration(df, '15m', 5, expected_timer_timedelta)

    # other events should include only task 
    # with the fixed distribution of 30 minutes
    df = df[df['activity'] != '15m']
    end_start_diff_for_other_events = df['end_time'] - df['start_time']

    assert df.shape[0] == 5, \
        "The total number of task events in the log file should be equal to 5"

    expected_task_timedelta = datetime.timedelta(minutes=30)
    for diff in end_start_diff_for_other_events:
        assert diff == expected_task_timedelta, \
            f"The duration of the task does not equal to 30 min"

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
    sim_stats = assets_path / 'timer_with_task_stats.csv'
    sim_logs = assets_path / 'timer_with_task_logs.csv'

    start_string = '2022-06-21 13:22:30.035185+03:00'
    start_date = parse_datetime(start_string, True)

    # ====== ACT ======
    _, _ = run_diff_res_simulation(start_date,
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
    end_start_diff_for_other_events = df['end_time'] - df['start_time']

    assert df.shape[0] == 5, \
        "The total number of task events in the log file should be equal to 5"

    expected_task_timedelta = datetime.timedelta(minutes=30)
    for diff in end_start_diff_for_other_events:
        assert diff == expected_task_timedelta, \
            f"The duration of the task does not equal to 30 min"

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
    start_date = parse_datetime(start_string, True)

    # ====== ACT ======
    _, _ = run_diff_res_simulation(start_date,
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
    _verify_event_count_and_duration(df, 'Timer Event', 5, expected_timer_timedelta)

    # verify that occurence of '4h' event is 5 
    #   (that means it was executed in 100% executed cases)
    # verify that duration of the '4h' event is 4 hours
    expected_timer_timedelta = datetime.timedelta(hours=4)
    _verify_event_count_and_duration(df, '4h', 5, expected_timer_timedelta)

def _verify_event_count_and_duration(df, event_name, expected_occurences, expected_timer_timedelta):
    only_timer_events = df[df['activity'] == event_name]
    end_start_diff_for_timer = only_timer_events['end_time'] - only_timer_events['start_time']

    assert only_timer_events.shape[0] == expected_occurences, \
        "The total number of timer events in the log file should be equal to {expected_occurences}"

    for diff in end_start_diff_for_timer:
        assert diff == expected_timer_timedelta, \
            f"The duration of the timer does not equal to {expected_timer_timedelta}"
        