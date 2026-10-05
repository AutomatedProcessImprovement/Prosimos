import json
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from prosimos.batch_processing import AndFiringRule, FiringSubRule
from prosimos.simulation_engine import run_simulation
from testing_scripts.test_batching import assets_path

MODEL_FILENAME = "batch-example-end-task.bpmn"
JSON_FILENAME = "batch-example-with-batch.json"
BATCHED_TASK = "D"
TOTAL_CASES = 40
BATCH_SIZE_ZERO_WARNING = "batch size for the execution returned to be 0"

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


@pytest.fixture(autouse=True)
def _keep_random_state():
    # these tests seed the global random generators; their states are put back afterwards,
    # so later tests draw the same values as without these tests
    python_state, numpy_state = random.getstate(), np.random.get_state()
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)


def _committed_example(assets_path):
    with open(assets_path / JSON_FILENAME) as file:
        settings = json.load(file)
    settings["arrival_time_distribution"] = COMMITTED_ARRIVAL_DISTRIBUTION
    settings["batch_processing"] = json.loads(json.dumps(COMMITTED_BATCH_PROCESSING))
    return settings


def _run(assets_path, tmp_path, settings, seed=1):
    json_path = tmp_path / "settings.json"
    with open(json_path, "w") as file:
        json.dump(settings, file)
    log_path = tmp_path / "log.csv"

    random.seed(seed)
    np.random.seed(seed)
    run_simulation(assets_path / MODEL_FILENAME, json_path, TOTAL_CASES, None, log_path,
                   "2024-01-01T09:00:00+00:00")

    log = pd.read_csv(log_path)
    return log[log["activity"] == BATCHED_TASK].groupby("case_id").size()


@pytest.mark.parametrize("batch_type", ["Parallel", "Sequential"])
def test_random_durations_run_every_case_through_the_batch_once(assets_path, tmp_path, capsys, batch_type):
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
    assert BATCH_SIZE_ZERO_WARNING not in capsys.readouterr().out


@pytest.mark.parametrize("seed", [1, 8])
@pytest.mark.parametrize("batch_type", ["Parallel", "Sequential"])
def test_fixed_durations_print_no_batch_size_zero_warning(assets_path, tmp_path, capsys, batch_type, seed):
    # ====== ARRANGE ======
    # the example as committed: a single waiting case between the ready_wt boundaries (seed 1),
    # or a wait a fraction of a second past a boundary (seed 8),
    # used to make the firing rule and the batch size disagree
    settings = _committed_example(assets_path)
    settings["batch_processing"][0]["type"] = batch_type

    # ====== ACT ======
    runs_per_case = _run(assets_path, tmp_path, settings, seed)

    # ====== ASSERT ======
    assert len(runs_per_case) == TOTAL_CASES
    assert (runs_per_case == 1).all()
    assert BATCH_SIZE_ZERO_WARNING not in capsys.readouterr().out


@pytest.mark.parametrize(
    "wait_sec, expected_batch_size",
    [
        (7200.6, 0),  # rounded down to 7200: not past "> 7200" yet
        (7201.4, 2),  # rounded down to 7201: past "> 7200", both cases fire
    ],
)
def test_fraction_of_a_second_past_a_boundary_gives_the_same_answer_in_rule_and_batch_size(
    capsys, wait_sec, expected_batch_size
):
    # ====== ARRANGE ======
    # the boundaries are whole seconds, and a wait a fraction of a second past one
    # used to make the firing rule say "fire" while the batch size said 0
    rule = AndFiringRule([FiringSubRule("ready_wt", ">", 7200), FiringSubRule("ready_wt", "<", 10800)])
    rule.init_boundaries()
    first = datetime(2024, 1, 2, 9, 0, 0, tzinfo=timezone.utc)
    last = first + timedelta(minutes=10)
    now = last + timedelta(seconds=wait_sec)

    def element():
        return {
            "size": 2,
            "waiting_times": [(now - first).total_seconds(), (now - last).total_seconds()],
            "enabled_datetimes": [first, last],
            "curr_enabled_at": now,
            "is_triggered_by_batch": True,
            "is_only_one_batch_return": False,
        }

    # ====== ACT ======
    rule_says_fire = all(subrule.is_true(element()) for subrule in rule.rules)
    batch_size, _ = rule.get_firing_batch_size(2, element())
    is_true, batch_spec, _ = rule.is_true(element())

    # ====== ASSERT ======
    assert rule_says_fire == (batch_size > 0)
    assert batch_size == expected_batch_size
    assert is_true == (expected_batch_size > 0)
    assert batch_spec == ([expected_batch_size] if expected_batch_size else None)
    assert BATCH_SIZE_ZERO_WARNING not in capsys.readouterr().out
