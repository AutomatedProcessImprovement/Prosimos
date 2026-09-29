import csv
import json
import os
from datetime import datetime

import pytest
import pytz

from prosimos.orchestrator import (
    Message, ProcessSpec, ProsimosEngine, SimulationConfig, SimulationEngine, run_orchestrator,
)
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


def _read_log(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_two_different_models_both_produce_events(tmp_path):
    log_path = tmp_path / "merged.csv"
    executed = run_orchestrator(SimulationConfig([_batch_process("batch"), _gateway_process("gateway")], START, SEED), str(log_path))

    assert {name for _, name in executed} == {"batch", "gateway"}
    header, *rows = _read_log(log_path)
    assert header[0] == "process"
    assert sum(row[0] == "batch" for row in rows) == 10
    assert sum(row[0] == "gateway" for row in rows) == 10


def test_merged_log_is_sorted_by_start_time(tmp_path):
    # with these two models, events are executed in an order where one process's tasks wait
    # for a busy resource while the other carries on, so start times arrive out of order
    assets = get_path()
    log_path = tmp_path / "merged.csv"
    run_orchestrator(
        SimulationConfig(
            [
                ProcessSpec("stock", str(assets / "stock_replenishment.bpmn"), str(assets / "stock_replenishment_logs.json"), 10),
                ProcessSpec("timer", str(assets / "timer_with_task.bpmn"), str(assets / "timer_with_task.json"), 10),
            ],
            START, SEED,
        ),
        str(log_path),
    )

    header, *rows = _read_log(log_path)
    start_times = [datetime.fromisoformat(row[header.index("start_time")]) for row in rows]
    assert {row[0] for row in rows} == {"stock", "timer"}
    assert start_times == sorted(start_times)


def test_events_are_executed_in_global_time_order():
    executed = run_orchestrator(SimulationConfig([_batch_process("batch"), _gateway_process("gateway")], START, SEED))

    times = [event_time for event_time, _ in executed]
    assert times == sorted(times)


def test_ties_are_broken_alphabetically_by_process_name():
    # the same model twice: fixed daily arrivals make both report the same batch times
    executed = run_orchestrator(SimulationConfig([_batch_process("zeta"), _batch_process("alpha")], START, SEED))

    ties = [(a, b) for a, b in zip(executed, executed[1:]) if a[0] == b[0]]
    assert ties, "expected the two identical processes to tie at least once"
    for (_, first_name), (_, second_name) in ties:
        assert (first_name, second_name) == ("alpha", "zeta")


def test_same_seed_gives_the_same_run_regardless_of_input_order(tmp_path):
    gateway_a = _random_arrival_gateway_process("gateway_a", tmp_path)
    gateway_b = _random_arrival_gateway_process("gateway_b", tmp_path)

    first = run_orchestrator(SimulationConfig([gateway_a, gateway_b], START, SEED), str(tmp_path / "first.csv"))
    second = run_orchestrator(SimulationConfig([gateway_b, gateway_a], START, SEED), str(tmp_path / "second.csv"))

    assert first == second
    # the gateway logs have no random id column, so they can be compared in full
    assert _read_log(tmp_path / "first.csv") == _read_log(tmp_path / "second.csv")


def test_duplicate_process_names_are_rejected():
    with pytest.raises(ValueError):
        SimulationConfig([_batch_process("same"), _gateway_process("same")], START, SEED)


def test_runs_without_a_seed():
    executed = run_orchestrator(SimulationConfig([_batch_process("batch"), _gateway_process("gateway")], START))

    assert {name for _, name in executed} == {"batch", "gateway"}
    times = [event_time for event_time, _ in executed]
    assert times == sorted(times)


def test_engine_offers_only_the_interface_methods():
    engine = ProsimosEngine(_batch_process("batch"), START)

    assert isinstance(engine, SimulationEngine)
    assert engine.subscriptions() == []
    assert engine.next_event_time() is not None
    assert engine.step() == []
    assert engine.deliver([], START) == ([], [])


def _spec(name):
    # validation never opens the files, so the paths don't need to exist
    return ProcessSpec(name, f"{name}.bpmn", f"{name}.json", 1)


def test_config_rejects_a_process_missing_from_the_groups():
    with pytest.raises(ValueError, match=r"\['Billing'\] are in no group"):
        SimulationConfig([_spec("Sales"), _spec("Billing")], START, SEED, {"Sales": ["Sales"]})


def test_config_rejects_a_process_listed_in_two_groups():
    with pytest.raises(ValueError, match="listed more than once"):
        SimulationConfig([_spec("Sales"), _spec("Billing")], START, SEED,
                         {"Sales": ["Sales"], "Billing": ["Billing"], "Everyone": ["Sales"]})


def test_config_rejects_a_group_naming_an_unknown_process():
    with pytest.raises(ValueError, match=r"unknown processes \['Shipping'\]"):
        SimulationConfig([_spec("Sales")], START, SEED, {"Sales": ["Sales", "Shipping"]})


def test_config_puts_every_process_in_its_own_group_by_default():
    config = SimulationConfig([_spec("Sales"), _spec("Billing")], START, SEED)

    assert config.consumer_groups == {"Sales": ["Sales"], "Billing": ["Billing"]}


def test_config_accepts_processes_sharing_a_group():
    names = ["Sales", "Billing", "TartuWarehouse", "TallinnWarehouse"]
    groups = {"Sales": ["Sales"], "Billing": ["Billing"], "Warehouses": ["TartuWarehouse", "TallinnWarehouse"]}

    config = SimulationConfig([_spec(name) for name in names], START, SEED, groups)

    assert config.consumer_groups == groups


def test_config_loads_from_a_json_file_and_runs(tmp_path):
    assets = get_path().resolve()
    config_file = tmp_path / "simulation.json"
    with open(config_file, "w") as f:
        json.dump({
            "processes": [
                {"name": "batch", "bpmn_path": os.path.relpath(assets / "1_task-batch.bpmn", tmp_path),
                 "json_path": os.path.relpath(assets / "1_task-batch.json", tmp_path), "total_cases": 10},
                {"name": "gateway",
                 "bpmn_path": os.path.relpath(assets / "gateway_conditions/gateway_condition_xor_model.bpmn", tmp_path),
                 "json_path": os.path.relpath(assets / "gateway_conditions/gateway_one_true_condition.json", tmp_path),
                 "total_cases": 10},
            ],
            "seed": SEED,
            "start_time": "2024-01-01T09:30:00",
            "consumer_groups": {"batch": ["batch"], "gateway": ["gateway"]},
        }, f)

    config = SimulationConfig.from_json(config_file)
    executed = run_orchestrator(config)

    assert config.start_datetime == START
    assert config.seed == SEED
    assert {name for _, name in executed} == {"batch", "gateway"}


def test_config_file_missing_a_required_field_is_rejected(tmp_path):
    config_file = tmp_path / "simulation.json"
    with open(config_file, "w") as f:
        json.dump({"processes": [{"name": "Sales", "bpmn_path": "sales.bpmn", "json_path": "sales.json"}],
                   "start_time": "2024-01-01T09:30:00+00:00"}, f)

    with pytest.raises(ValueError, match=r"Sales is missing \['total_cases'\]"):
        SimulationConfig.from_json(config_file)


def test_message_fields_set_by_the_orchestrator_start_empty():
    first = Message("OrderPlaced", {"order_id": "ord7", "city": "Tartu"})
    second = Message("OrderPlaced")

    assert (first.id, first.source, first.time) == (None, None, None)
    assert second.attributes == {} and second.attributes is not Message("OrderPlaced").attributes
