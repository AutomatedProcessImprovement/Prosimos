import pandas as pd
import datetime
import os
from pathlib import Path
import pytest

from bpdfr_simulation_engine.resource_calendar import parse_datetime
from bpdfr_simulation_engine.simulation_properties_parser import parse_json_sim_parameters
from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation
from testing_scripts.test_simulation import _verify_activity_count_and_duration

@pytest.fixture
def assets_path(request) -> Path:
    entry_path: Path
    if os.path.basename(os.getcwd()) == 'testing_scripts':
        entry_path = Path('assets')
    else:
        entry_path = Path('testing_scripts/assets')

    def teardown():
        output_paths = [
            entry_path / 'batch_stats.csv',
            entry_path / 'batch_logs.csv'
        ]
        for output_path in output_paths:
            if output_path.exists():
                os.remove(output_path)
    request.addfinalizer(teardown)

    return entry_path


def test_seq_batch_count_firing_rule_correct_duration(assets_path):
    # ====== ARRANGE ======
    model_path = assets_path / 'batch-example-end-task.bpmn'
    json_path = assets_path / 'batch-example-with-batch.json'
    sim_stats = assets_path / 'batch_stats.csv'
    sim_logs = assets_path / 'batch_logs.csv'

    start_string = '2022-06-21 13:22:30.035185+03:00'
    start_date = parse_datetime(start_string, True)

    # ====== ACT ======
    _, diff_sim_result = run_diff_res_simulation(start_date,
                                    6,
                                    model_path,
                                    json_path,
                                    sim_stats,
                                    sim_logs,
                                    True)

    diff_sim_result.print_simulation_results()

    # ====== ASSERT ======
    # verify stats
    task_d_sim_result = diff_sim_result.tasks_kpi_map['D']
    expected_average_processing_time = 96.0 # 80% from the initially defined task performance
    assert task_d_sim_result.processing_time.avg == expected_average_processing_time
    assert task_d_sim_result.processing_time.count == 6

    # make sure the next tasks after batching were executed
    task_d_sim_result = diff_sim_result.tasks_kpi_map['E']
    assert task_d_sim_result.processing_time.count == 6

    # verify logs
    df = pd.read_csv(sim_logs)
    
    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')

    # verify that there are 6 instances of D activity executed
    # and that theirs duration is exactly 96 seconds
    logs_d_task = df[df['activity'] == 'D']
    expected_activity_timedelta = datetime.timedelta(seconds=96)
    _verify_activity_count_and_duration(logs_d_task, 6, expected_activity_timedelta)
