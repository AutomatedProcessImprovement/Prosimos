import os
import pytest
import pandas as pd

import datetime
from pathlib import Path
from bpdfr_simulation_engine.resource_calendar import parse_datetime

from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation

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
    expected_task_timedelta = datetime.timedelta(minutes=30)
    expected_task_count = 5
    _verify_activity_count_and_duration(df, expected_task_count, expected_task_timedelta)


def _verify_activity_count_and_duration(activities, count, expected_activity_timedelta):
    assert activities.shape[0] == count, \
        f"The total number of activities in the log file should be equal to {count}"
    
    end_start_diff_for_task = activities['end_time'] - activities['start_time']
    for diff in end_start_diff_for_task:
        assert diff == expected_activity_timedelta, \
            f"The duration of the activity does not equal to {expected_activity_timedelta}"
