import json
import pandas as pd
from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation
from testing_scripts.test_batching import (
    JSON_FILENAME,
    JSON_NEAREST_COEF_FILENAME,
    MODEL_FILENAME,
    SIM_LOGS_FILENAME,
    SIM_STATS_FILENAME,
    _setup_sim_scenario_file,
    _verify_start_time_num_tasks,
    assets_path
)


def test_no_batch_rules_correct(assets_path):
    # ====== ARRANGE ======
    model_path = assets_path / MODEL_FILENAME
    basic_json_path = assets_path / JSON_FILENAME
    json_path = assets_path / JSON_NEAREST_COEF_FILENAME
    sim_stats = assets_path / SIM_STATS_FILENAME
    sim_logs = assets_path / SIM_LOGS_FILENAME

    start_string = "2022-06-21 13:22:30.035185+03:00"

    with open(basic_json_path, "r") as f:
        json_dict = json.load(f)

    size_distr = {
        "3": 100
    }
    _setup_sim_scenario_file(json_dict, None, None, "Parallel", [], size_distr)

    with open(json_path, "w+") as json_file:
        json.dump(json_dict, json_file)

    # ====== ACT ======
    _, diff_sim_result = run_diff_res_simulation(
        start_string, 5, model_path, json_path, sim_stats, sim_logs  # one batch
    )

    df = pd.read_csv(sim_logs)

    logs_d_tasks = df[df["activity"] == "D"]
    grouped_by_start = logs_d_tasks.groupby(by=["start_time"])

    expected_start_time_keys = [
        ("2022-06-21 13:30:30.035185+03:00", 3),
        ("2022-06-21 13:34:30.035185+03:00", 2),
    ]
    _verify_start_time_num_tasks(grouped_by_start, expected_start_time_keys)

