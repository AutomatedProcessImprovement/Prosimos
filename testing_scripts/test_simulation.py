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
    model_path = assets_path / 'timer_with_task.bpmn'
    json_path = assets_path / 'timer_with_task.json'
    sim_stats = assets_path / 'timer_with_task_stats.csv'
    sim_logs = assets_path / 'timer_with_task_logs.csv'

    start_string = '2022-06-21 13:22:30.035185+03:00'
    start_date = parse_datetime(start_string, True)
    sim_time, diff_sim_result = run_diff_res_simulation(start_date,
                                                 5,
                                                 model_path,
                                                 json_path,
                                                 sim_stats,
                                                 sim_logs,
                                                 True)


    df = pd.read_csv(sim_logs)
    print(df.to_string())

    grouped_by_case_id = df.groupby(by="case_id")["case_id"]
    assert grouped_by_case_id.count().size == 5, \
        "The total number of simulated cases does not equal to the setup number"
    
    for name, group in grouped_by_case_id:
        assert group.size == 2, \
            f"The case '{name}' does not have the required number of logged simulated activities"

    # TODO: validate the duration of the timer event

    # TODO: add new test for validating that event record doesn't appear in the log file
