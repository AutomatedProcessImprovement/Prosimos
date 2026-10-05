import json
import random

import numpy as np
import pandas as pd
import pytest

from prosimos.simulation_engine import run_simulation
from testing_scripts.test_batching import assets_path

MODEL_FILENAME = "batch-example-end-task.bpmn"
JSON_FILENAME = "batch-example-with-batch.json"
BATCHED_TASK = "D"
TOTAL_CASES = 40

# other batching tests overwrite these two sections of the example file in place,
# so they are set back to the committed values here
COMMITTED_ARRIVAL_DISTRIBUTION = {
    "distribution_name": "expon",
    "distribution_params": [{"value": 7200.0}, {"value": 0.0}, {"value": 10000.0}],
}
COMMITTED_BATCH_PROCESSING = [
    {
        "task_id": "sid-503A048D-6344-446A-8D67-172B164CF8FA",
        "type": "Parallel",
        "batch_frequency": 1.0,
        "size_distrib": [{"key": "1", "value": 0}, {"key": "2", "value": 1}],
        "duration_distrib": [{"key": "3", "value": 0.8}],
        "firing_rules": [
            [
                {"attribute": "ready_wt", "comparison": ">", "value": 7200},
                {"attribute": "ready_wt", "comparison": "<", "value": 10800},
            ]
        ],
    }
]


def _committed_example(assets_path):
    with open(assets_path / JSON_FILENAME) as file:
        settings = json.load(file)
    settings["arrival_time_distribution"] = COMMITTED_ARRIVAL_DISTRIBUTION
    settings["batch_processing"] = json.loads(json.dumps(COMMITTED_BATCH_PROCESSING))
    return settings


def _run(assets_path, tmp_path, settings):
    json_path = tmp_path / "settings.json"
    with open(json_path, "w") as file:
        json.dump(settings, file)
    log_path = tmp_path / "log.csv"

    random.seed(1)
    np.random.seed(1)
    run_simulation(assets_path / MODEL_FILENAME, json_path, TOTAL_CASES, None, log_path,
                   "2024-01-01T09:00:00+00:00")

    log = pd.read_csv(log_path)
    return log[log["activity"] == BATCHED_TASK].groupby("case_id").size()


@pytest.mark.parametrize("batch_type", ["Parallel", "Sequential"])
def test_random_durations_run_every_case_through_the_batch_once(assets_path, tmp_path, batch_type):
    # ====== ARRANGE ======
    # random task durations make cases reach the batched task out of the order they were added to its queue,
    # and the queue holds cases that reach it only later on
    settings = _committed_example(assets_path)
    for task in settings["task_resource_distribution"]:
        for resource in task["resources"]:
            resource["distribution_name"] = "expon"
            resource["distribution_params"] = [{"value": 1800}, {"value": 0}, {"value": 18000}]
    settings["batch_processing"][0]["type"] = batch_type

    # ====== ACT ======
    runs_per_case = _run(assets_path, tmp_path, settings)

    # ====== ASSERT ======
    assert len(runs_per_case) == TOTAL_CASES
    assert (runs_per_case == 1).all()
