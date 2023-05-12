import json
from unittest import mock
import pandas as pd
from prosimos.batch_processing import FiringSubRule
from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation
from testing_scripts.test_batching import (
    JSON_FILENAME,
    MODEL_FILENAME,
    SIM_LOGS_FILENAME,
    SIM_STATS_FILENAME,
    _setup_arrival_distribution,
    _setup_sim_scenario_file,
    _verify_start_time_num_tasks,
    assets_path
)
from testing_scripts.test_batching_daily_hour import _arrange_and_act


def test_distr_one_choice_correct(assets_path):
    # ====== ARRANGE ======
    model_path = assets_path / MODEL_FILENAME
    json_path = assets_path / JSON_FILENAME
    sim_stats = assets_path / SIM_STATS_FILENAME
    sim_logs = assets_path / SIM_LOGS_FILENAME

    start_string = "2022-06-21 13:22:30.035185+03:00"

    with open(json_path, "r") as f:
        json_dict = json.load(f)

    size_distr = [
        { "key": "3", "value": 1}
    ]
    _setup_sim_scenario_file(json_dict, None, None, "Parallel", [], size_distr)

    arrival_distr = {
        "distribution_name": "fix",
        "distribution_params": [{"value": 120}, {"value": 0}, {"value": 1}],
    }
    _setup_arrival_distribution(json_dict, arrival_distr)
    
    with open(json_path, "w+") as json_file:
        json.dump(json_dict, json_file)

    # ====== ACT ======
    _ = run_diff_res_simulation(
        start_string, 5, model_path, json_path, sim_stats, sim_logs
    )

    df = pd.read_csv(sim_logs)

    logs_d_tasks = df[df["activity"] == "D"]
    grouped_by_start = logs_d_tasks.groupby(by=["start_time"])

    expected_start_time_keys = [
        ("2022-06-21 13:30:30.035185+03:00", 3),
        ("2022-06-21 13:35:18.035185+03:00", 2),
    ]
    _verify_start_time_num_tasks(grouped_by_start, expected_start_time_keys)


@mock.patch('bpdfr_simulation_engine.batch_processing.BatchConfigPerTask.get_new_subrule_rule')
def test_distr_multi_choice_correct(mock_choice, assets_path):
    # ====== ARRANGE ======
    # provide the sequence of the mocked random.choices 
    # length should be total_sim_num + 1 
    # because we generate the new rule every time we move tasks to the batch execution
    # distibution of first total_sim_num (15 in this case) is 50%/50%
    sim_logs = assets_path / SIM_LOGS_FILENAME
    start_string = "2022-06-21 13:22:30.035185+03:00"
    total_sim_num = 15

    prob_seq = [2, 2, 2, 3, 3, 3, 2, 3, 2]
    mock_choice.side_effect = [ FiringSubRule("size", "=", i) for i in prob_seq ]
    initial_size_distr = [
        {"key": "2", "value": 0.5},
        {"key": "3", "value": 0.5}
    ]
    _arrange_and_act(assets_path, {}, start_string, total_sim_num, 10800, initial_size_distr)

    # ====== ASSERT ======
    df = pd.read_csv(sim_logs)

    logs_d_tasks = df[df["activity"] == "D"]
    actual_num_d_tasks = len(logs_d_tasks)
    assert total_sim_num == actual_num_d_tasks
    _verify_batched_size_match_distrib(logs_d_tasks, initial_size_distr)


def test_distr_all_ones_correct(assets_path):
    # ====== ARRANGE ======
    sim_logs = assets_path / SIM_LOGS_FILENAME
    total_sim_num = 20
    start_string = "2022-06-21 13:22:30.035185+03:00"
    
    initial_size_distr = [
        {"key": "1", "value": 1}
    ]
    _arrange_and_act(assets_path, {}, start_string, total_sim_num, 10800, initial_size_distr)

    # ====== ASSERT ======
    df = pd.read_csv(sim_logs)

    logs_d_tasks = df[df["activity"] == "D"]
    actual_num_d_tasks = len(logs_d_tasks)
    assert total_sim_num == actual_num_d_tasks
    _verify_batched_size_match_distrib(logs_d_tasks, initial_size_distr)


def _verify_batched_size_match_distrib(logs_d_tasks, initial_size_distr):
    grouped_by_start = logs_d_tasks.groupby(by=["start_time"])
    total_num_batches = len(logs_d_tasks.groupby(by=["start_time"]).groups)

    grouped_by_number_of_rows = grouped_by_start.size().reset_index().groupby(by=0).groups
    actual_row_num_count = [{ "key": str(k), "value": (len(v) / total_num_batches) } for k, v in grouped_by_number_of_rows.items()]
    assert initial_size_distr == actual_row_num_count