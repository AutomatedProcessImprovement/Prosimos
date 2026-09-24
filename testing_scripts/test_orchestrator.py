import csv
import json
from datetime import datetime

import pytest
import pytz

from prosimos.orchestrator import ProcessSpec, run_orchestrator
from testing_scripts.test_batching_stats import get_path

START = pytz.utc.localize(datetime(2024, 1, 1, 9, 30))
SEED = 42


def _batch_process(name):
    assets = get_path()
    return ProcessSpec(name, str(assets / "1_task-batch.bpmn"), str(assets / "1_task-batch.json"), 10)


def _gateway_process(name):
    assets = get_path("gateway_conditions")
    return ProcessSpec(
        name,
        str(assets / "gateway_condition_xor_model.bpmn"),
        str(assets / "gateway_one_true_condition.json"),
        10,
    )


def _random_arrival_gateway_process(name, tmp_path):
    # the stock gateway config has fixed arrivals, so generating them draws no random
    # numbers; exponential arrivals make the order engines are built in observable
    spec = _gateway_process(name)
    with open(spec.json_path) as f:
        config = json.load(f)
    config["arrival_time_distribution"] = {
        "distribution_name": "expon",
        "distribution_params": [{"value": 30.0}, {"value": 0.0}, {"value": 300.0}],
    }
    json_path = tmp_path / f"{name}.json"
    with open(json_path, "w") as f:
        json.dump(config, f)
    return ProcessSpec(name, spec.bpmn_path, str(json_path), spec.total_cases)


def _log_rows(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))[1:]


def test_two_different_models_both_produce_events(tmp_path):
    executed = run_orchestrator([_batch_process("batch"), _gateway_process("gateway")], START, SEED, str(tmp_path))

    assert {name for _, name in executed} == {"batch", "gateway"}
    assert len(_log_rows(tmp_path / "batch.csv")) == 10
    assert len(_log_rows(tmp_path / "gateway.csv")) > 0


def test_events_are_executed_in_global_time_order():
    executed = run_orchestrator([_batch_process("batch"), _gateway_process("gateway")], START, SEED)

    times = [event_time for event_time, _ in executed]
    assert times == sorted(times)


def test_ties_are_broken_alphabetically_by_process_name():
    # the same model twice: fixed daily arrivals make both report the same batch times
    executed = run_orchestrator([_batch_process("zeta"), _batch_process("alpha")], START, SEED)

    ties = [(a, b) for a, b in zip(executed, executed[1:]) if a[0] == b[0]]
    assert ties, "expected the two identical processes to tie at least once"
    for (_, first_name), (_, second_name) in ties:
        assert (first_name, second_name) == ("alpha", "zeta")


def test_same_seed_gives_the_same_run_regardless_of_input_order(tmp_path):
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    gateway_a = _random_arrival_gateway_process("gateway_a", tmp_path)
    gateway_b = _random_arrival_gateway_process("gateway_b", tmp_path)

    first = run_orchestrator([gateway_a, gateway_b], START, SEED, str(first_dir))
    second = run_orchestrator([gateway_b, gateway_a], START, SEED, str(second_dir))

    assert first == second
    # the gateway logs have no random id column, so they can be compared in full
    for name in ("gateway_a", "gateway_b"):
        assert _log_rows(first_dir / f"{name}.csv") == _log_rows(second_dir / f"{name}.csv")


def test_duplicate_process_names_are_rejected():
    with pytest.raises(ValueError):
        run_orchestrator([_batch_process("same"), _gateway_process("same")], START, SEED)


def test_runs_without_a_seed():
    executed = run_orchestrator([_batch_process("batch"), _gateway_process("gateway")], START)

    assert {name for _, name in executed} == {"batch", "gateway"}
    times = [event_time for event_time, _ in executed]
    assert times == sorted(times)
