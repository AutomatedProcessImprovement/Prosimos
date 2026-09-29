"""
The running example (docs/running-example.md) played through the orchestrator loop with scripted
engines, one test per check. ord3 can go to either warehouse depending on the seed, so rules that
always hold are checked for many seeds, and the exact outcome of one seed is recorded separately.
The truck loading rule is scripted in the fake warehouses, so the capacity check tests the scenario,
not the orchestrator.
"""
import random
from collections import Counter

import pytest

from prosimos.orchestrator import Verdict, run_engines
from testing_scripts.protocol_scenario import at, run_scenario
from testing_scripts.scripted_engine import ScriptedEngine

SEEDS = list(range(20))
FIXED_SEED = 1  # sends ord3 to TartuWarehouse


def _label(message):
    key = message.attributes.get("order_id") or message.attributes.get("truck_id")
    return f"{message.type} {key}" if key else message.type


def _messages(report):
    return {message.id: message for message in report.published}


def _claims(report):
    messages = _messages(report)
    return [(_label(messages[message_id]), process, time) for message_id, process, time in report.claims]


def _discards(report):
    messages = _messages(report)
    return [(_label(messages[message_id]), process, time) for message_id, process, time in report.discards]


def _warehouse_of(report, order_id):
    return [process for label, process, _ in _claims(report)
            if label == f"OrderPlaced {order_id}" and process.endswith("Warehouse")]


@pytest.mark.parametrize("seed", SEEDS)
def test_announcement(seed):
    _, report = run_scenario(seed)
    orders = [m.id for m in report.published if m.type == "OrderPlaced"]

    assert len(orders) == 5
    assert sorted(i for i, process, _ in report.claims if process == "Billing") == sorted(orders)
    copies = Counter(message_id for message_id, group in report.copies if group == "Warehouses")
    assert {message_id: copies[message_id] for message_id in orders} == {message_id: 1 for message_id in orders}


@pytest.mark.parametrize("seed", SEEDS)
def test_shared_group(seed):
    _, report = run_scenario(seed)

    for order_id in ("ord1", "ord2", "ord3", "ord4", "ord5"):
        assert len(_warehouse_of(report, order_id)) <= 1
    assert _warehouse_of(report, "ord1") == ["TartuWarehouse"]
    assert _warehouse_of(report, "ord5") == ["TartuWarehouse"]
    assert _warehouse_of(report, "ord2") == ["TallinnWarehouse"]


def test_random_tie():
    outcomes = {}
    for seed in SEEDS:
        first = _warehouse_of(run_scenario(seed)[1], "ord3")
        second = _warehouse_of(run_scenario(seed)[1], "ord3")
        assert first == second and len(first) == 1
        outcomes[seed] = first[0]

    # the tie is genuinely random: some seeds send ord3 to each warehouse
    assert set(outcomes.values()) == {"TartuWarehouse", "TallinnWarehouse"}


@pytest.mark.parametrize("seed", SEEDS)
def test_pending(seed):
    engines, report = run_scenario(seed)
    shipment = next(m for m in report.published if _label(m) == "Shipment ord1")

    assert shipment.time == at("10:30")
    assert ("Shipment ord1", "Sales", at("12:00")) in _claims(report)
    assert engines["Sales"].closed_at["ord1"] == at("12:00")


@pytest.mark.parametrize("seed", SEEDS)
def test_discard(seed):
    engines, report = run_scenario(seed)
    discarded = {(label, process) for label, process, _ in _discards(report)}

    for warehouse in ("TartuWarehouse", "TallinnWarehouse"):
        assert ("OrderPlaced ord4", warehouse) in discarded
        assert ("Truck T0", warehouse) in discarded
    assert ("Shipment ord2", "Sales") in discarded
    assert engines["Sales"].cases["ord2"] == "cancelled"


@pytest.mark.parametrize("seed", SEEDS)
def test_warnings(seed):
    _, report = run_scenario(seed)
    messages = _messages(report)
    warned_about = [_label(messages[text.split()[1]]) for text in report.warnings]

    # Shipment ord2 is discarded by its only recipient (the order was canceled), which the
    # protocol counts as "discarded by every recipient"; ord4 raises none because Billing claimed it
    assert warned_about == ["Newsletter", "Truck T0", "Shipment ord2"]
    assert "has no subscribers" in report.warnings[0]
    assert all("discarded by every recipient" in text for text in report.warnings[1:])


@pytest.mark.parametrize("seed", SEEDS)
def test_capacity(seed):
    engines, _ = run_scenario(seed)
    loads = {truck: orders for name in ("TartuWarehouse", "TallinnWarehouse")
             for truck, orders in engines[name].loads.items()}

    assert all(len(orders) <= 2 for orders in loads.values())
    loaded = [order for orders in loads.values() for order in orders]
    assert len(loaded) == len(set(loaded))
    assert "ord5" not in loads["T1"] and "ord5" in loads["T3"]


@pytest.mark.parametrize("seed", SEEDS)
def test_end_of_run(seed):
    engines, report = run_scenario(seed)

    assert sorted((group, _label(message)) for group, message in report.unclaimed) == [
        ("Sales", "Shipment ord3"), ("Warehouses", "Truck T4"),
    ]
    reported_by_engines = Counter((message.type, name) for name, engine in engines.items()
                                  for message, _ in engine.discarded)
    assert report.discarded_counts == dict(reported_by_engines)
    assert engines["Sales"].waiting() == ["ord4"]


@pytest.mark.parametrize("seed", SEEDS)
def test_time_never_goes_backwards(seed):
    _, report = run_scenario(seed)
    times = [time for time, _ in report.executed]

    assert times == sorted(times)


@pytest.mark.parametrize("seed", SEEDS)
def test_same_seed_gives_identical_runs(seed):
    assert run_scenario(seed)[1] == run_scenario(seed)[1]


def test_orchestrator_leaves_the_global_random_generator_alone():
    # the engines draw from the global random module; if choosing a warehouse drew from it too,
    # every later draw inside the engines would shift
    random.seed(123)
    before = random.getstate()
    run_scenario(FIXED_SEED)

    assert random.getstate() == before


def test_exact_outcome_of_a_fixed_seed():
    engines, report = run_scenario(FIXED_SEED)

    assert [(time.strftime("%H:%M"), source, _label(message)) for message, time, source in
            ((m, m.time, m.source) for m in report.published)] == [
        ("09:00", "Sales", "OrderPlaced ord1"),
        ("09:05", "Sales", "Newsletter"),
        ("09:10", "Sales", "OrderPlaced ord2"),
        ("09:20", "Sales", "OrderPlaced ord3"),
        ("09:30", "Sales", "OrderPlaced ord4"),
        ("09:40", "Sales", "OrderPlaced ord5"),
        ("09:45", "Carrier", "Truck T0"),
        ("10:30", "Carrier", "Truck T1"),
        ("10:30", "TartuWarehouse", "Shipment ord1"),
        ("10:30", "TartuWarehouse", "Shipment ord3"),
        ("12:00", "Carrier", "Truck T2"),
        ("12:00", "TallinnWarehouse", "Shipment ord2"),
        ("13:00", "Carrier", "Truck T3"),
        ("13:00", "TartuWarehouse", "Shipment ord5"),
        ("16:00", "Carrier", "Truck T4"),
    ]
    assert [(label, process, time.strftime("%H:%M")) for label, process, time in _claims(report)] == [
        ("OrderPlaced ord1", "Billing", "09:00"),
        ("OrderPlaced ord1", "TartuWarehouse", "09:00"),
        ("OrderPlaced ord2", "Billing", "09:10"),
        ("OrderPlaced ord2", "TallinnWarehouse", "09:10"),
        ("OrderPlaced ord3", "Billing", "09:20"),
        ("OrderPlaced ord3", "TartuWarehouse", "09:20"),
        ("OrderPlaced ord4", "Billing", "09:30"),
        ("OrderPlaced ord5", "Billing", "09:40"),
        ("OrderPlaced ord5", "TartuWarehouse", "09:40"),
        ("Truck T1", "TartuWarehouse", "10:30"),
        ("Truck T2", "TallinnWarehouse", "12:00"),
        ("Shipment ord1", "Sales", "12:00"),
        ("Truck T3", "TartuWarehouse", "13:00"),
        ("Shipment ord5", "Sales", "13:00"),
    ]
    assert [(label, process, time.strftime("%H:%M")) for label, process, time in _discards(report)] == [
        ("OrderPlaced ord1", "TallinnWarehouse", "09:00"),
        ("OrderPlaced ord2", "TartuWarehouse", "09:10"),
        ("OrderPlaced ord4", "TartuWarehouse", "09:30"),
        ("OrderPlaced ord4", "TallinnWarehouse", "09:30"),
        ("OrderPlaced ord5", "TallinnWarehouse", "09:40"),
        ("Truck T0", "TallinnWarehouse", "09:45"),
        ("Truck T0", "TartuWarehouse", "09:45"),
        ("Truck T1", "TallinnWarehouse", "10:30"),
        ("Shipment ord2", "Sales", "12:00"),
        ("Truck T4", "TallinnWarehouse", "16:00"),
    ]
    assert engines["TartuWarehouse"].loads == {"T1": ["ord1", "ord3"], "T3": ["ord5"]}
    assert engines["TallinnWarehouse"].loads == {"T2": ["ord2"]}


def test_a_member_that_discarded_a_copy_is_not_offered_it_again():
    # not exercised by the running example: A discards, B leaves the copy pending, then B has an
    # event of its own, so the orchestrator offers the pooled copy again, to B only
    publisher, a, b = ScriptedEngine("P"), ScriptedEngine("A"), ScriptedEngine("B")
    publisher.publish_at(at("09:00"), "X")
    a.consume("X", lambda message, now: Verdict.DISCARDED)
    b.consume("X", lambda message, now: Verdict.PENDING)
    b.at(at("10:00"), lambda now: None)

    report = run_engines({"P": publisher, "A": a, "B": b}, {"P": ["P"], "Pair": ["A", "B"]}, FIXED_SEED)

    assert [time for _, time in a.offered] == [at("09:00")]
    assert [time for _, time in b.offered] == [at("09:00"), at("10:00")]
    assert [(group, message.type) for group, message in report.unclaimed] == [("Pair", "X")]


def test_an_engine_answering_with_something_other_than_a_verdict_is_rejected():
    publisher, old_style = ScriptedEngine("P"), ScriptedEngine("Old")
    publisher.publish_at(at("09:00"), "X")
    old_style.consume("X", lambda message, now: Verdict.PENDING)
    old_style.deliver = lambda message, now: ([], [])

    with pytest.raises(TypeError, match="must return a Verdict"):
        run_engines({"P": publisher, "Old": old_style}, None, FIXED_SEED)
