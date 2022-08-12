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
    end_start_diff_for_timer = only_timer_events['end_time'] - only_timer_events['start_time']

    assert only_timer_events.shape[0] == 5, \
        "The total number of timer events in the log file should be equal to 5"

    expected_timer_timedelta = datetime.timedelta(minutes=15)
    for diff in end_start_diff_for_timer:
        assert diff == expected_timer_timedelta, \
            f"The duration of the timer does not equal to 15 min"

    # other events should include only task 
    # with the fixed distribution of 30 minutes
    other_than_timer_events = df[df['activity'] != '15m']
    end_start_diff_for_other_events = other_than_timer_events['end_time'] - other_than_timer_events['start_time']

    assert other_than_timer_events.shape[0] == 5, \
        "The total number of task events in the log file should be equal to 5"

    expected_task_timedelta = datetime.timedelta(minutes=30)
    for diff in end_start_diff_for_other_events:
        assert diff == expected_task_timedelta, \
            f"The duration of the task does not equal to 30 min"

    # TODO: add new test for validating that event record doesn't appear in the log file
