import pandas as pd
from testing_scripts.test_batching import (
    SIM_LOGS_FILENAME, assets_path
)
from testing_scripts.test_batching_daily_hour import _arrange_and_act
from testing_scripts.test_batching_large_wt import ONE_HOUR_IN_SEC, TWO_HOURS_IN_SEC

def test_all_tasks_alone(assets_path):
    """
    Input:      Firing rule of large_wt in [ONE_HOUR_IN_SEC, TWO_HOURS_IN_SEC].
                size_distr defines that all batches tasks should be executed alone 
                (probability of 1.0 for size = 1 cases)
                20 process cases are being generated. A new case arrive every 30 minutes.
                Batched task are executed in parallel.
    Expected:   All batched tasks are being executed one by one
                (without any waiting time for the batch formation).
    Verified:   The start_time and enabled_time should be equal for the D activities.
                The number of tasks (only one) in every executed batch.
                The start_time of all logs files is being sorted by ASC.
    """
    sim_logs = assets_path / SIM_LOGS_FILENAME

    start_string = "2022-06-21 13:22:30.035185+03:00"

    firing_rules = [
        [
            {"attribute": "large_wt", "comparison": ">", "value": ONE_HOUR_IN_SEC},
            {"attribute": "large_wt", "comparison": "<", "value": TWO_HOURS_IN_SEC}
        ]
    ]
    total_num_cases = 20
    size_distr = {
        '1': 1
    }

    _arrange_and_act(assets_path, firing_rules, start_string, total_num_cases, 1800, size_distr)

    df = pd.read_csv(sim_logs)

    df["enable_time"] = pd.to_datetime(df["enable_time"], errors="coerce")
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")

    logs_d_tasks = df[df["activity"] == "D"]
    total_num_batches = len(logs_d_tasks.groupby(by=["start_time","resource"]).groups)

    assert total_num_batches == 20

    expected_diff = 0
    _verify_diff_start_enabled_time(logs_d_tasks, expected_diff)


def _verify_diff_start_enabled_time(df, expected_diff):
    for _, item in df.iterrows():
        actual_diff_start_and_first_enabled = (item['start_time'] - item['enable_time']).seconds
        
        assert (
            actual_diff_start_and_first_enabled == expected_diff
        ), f"Expected the difference to be equal to {expected_diff},\
            but it was {actual_diff_start_and_first_enabled}"
