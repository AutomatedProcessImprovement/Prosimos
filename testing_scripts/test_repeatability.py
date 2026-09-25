import ast
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytz

from prosimos.orchestrator import ProcessSpec, run_orchestrator

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "testing_scripts" / "assets"
START = pytz.utc.localize(datetime(2024, 1, 1, 9, 30))
SEED = 42

GATEWAY_BPMN = "gateway_conditions/gateway_condition_xor_model.bpmn"
GATEWAY_JSON = "gateway_conditions/gateway_one_true_condition.json"

# name -> (bpmn, json, random arrival gaps?). Every process gets random task durations.
PROCESSES = {
    # two identical processes with fixed arrival gaps: their cases arrive at exactly the same
    # moments, so the orchestrator has to break a tie at every arrival
    "tied_a": (GATEWAY_BPMN, GATEWAY_JSON, False),
    "tied_b": (GATEWAY_BPMN, GATEWAY_JSON, False),
    # processes with random arrival gaps, so the order in which engines draw their
    # arrivals matters; one of them groups cases into batches
    "random_a": (GATEWAY_BPMN, GATEWAY_JSON, True),
    "random_b": ("timer_with_task.bpmn", "timer_with_task.json", True),
    "random_batch": ("1_task-batch.bpmn", "1_task-batch.json", True),
}


def _expon(mean_seconds):
    return {
        "distribution_name": "expon",
        "distribution_params": [{"value": mean_seconds}, {"value": 0.0}, {"value": mean_seconds * 10}],
    }


def write_random_configs(config_dir):
    # The stock configs use fixed arrival gaps and task durations, so which engine draws random
    # numbers first would never show in the output and ordering bugs would go unnoticed.
    for name, (_, json_file, random_arrivals) in PROCESSES.items():
        with open(ASSETS / json_file) as f:
            config = json.load(f)
        if random_arrivals:
            config["arrival_time_distribution"] = _expon(1800.0)
        for task in config["task_resource_distribution"]:
            for resource in task["resources"]:
                resource.update(_expon(600.0))
        with open(Path(config_dir) / f"{name}.json", "w") as f:
            json.dump(config, f)


def run_and_write_log(log_path, config_dir, seed=SEED):
    specs = [
        ProcessSpec(name, str(ASSETS / bpmn), str(Path(config_dir) / f"{name}.json"), 10)
        for name, (bpmn, _, _) in PROCESSES.items()
    ]
    return run_orchestrator(specs, START, seed, str(log_path))


def _set_order_of_process_names(hash_seed):
    result = subprocess.run(
        [sys.executable, "-c", f"print(list(set({list(PROCESSES)!r})))"],
        env={**os.environ, "PYTHONHASHSEED": str(hash_seed)},
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _hash_seeds_with_reversed_set_order():
    # Python only changes the order of a set of strings between separate launches, depending
    # on PYTHONHASHSEED. Pick two values that put every pair of process names in opposite
    # orders, so any behaviour that depends on set order must differ between the two runs.
    first = _set_order_of_process_names(0)
    reversed_first = str(list(reversed(ast.literal_eval(first)))) + "\n"
    for other in range(1, 500):
        if _set_order_of_process_names(other) == reversed_first:
            return 0, other
    raise AssertionError("no PYTHONHASHSEED found that reverses the order of a set of process names")


def _run_in_fresh_python(log_path, config_dir, hash_seed):
    code = (
        "from testing_scripts.test_repeatability import run_and_write_log; "
        f"run_and_write_log({str(log_path)!r}, {str(config_dir)!r})"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONHASHSEED": str(hash_seed)},
        check=True,
        capture_output=True,
    )


def test_same_seed_gives_identical_merged_logs(tmp_path):
    # each run gets its own interpreter, with opposite set orders, so that code depending on
    # the order of a set can't pass by producing the same order twice
    write_random_configs(tmp_path)
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    first_hash_seed, second_hash_seed = _hash_seeds_with_reversed_set_order()

    _run_in_fresh_python(first, tmp_path, first_hash_seed)
    _run_in_fresh_python(second, tmp_path, second_hash_seed)

    assert first.read_bytes() == second.read_bytes()


def test_the_repeatability_test_exercises_randomness_and_ties(tmp_path):
    # guards against the test above passing trivially: the runs must really depend on the
    # seed, and the orchestrator must really have to break ties
    write_random_configs(tmp_path)

    executed = run_and_write_log(tmp_path / "first.csv", tmp_path)
    run_and_write_log(tmp_path / "other_seed.csv", tmp_path, seed=SEED + 1)

    assert (tmp_path / "first.csv").read_bytes() != (tmp_path / "other_seed.csv").read_bytes()
    assert any(a[0] == b[0] and a[1] != b[1] for a, b in zip(executed, executed[1:]))
