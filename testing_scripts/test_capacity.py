"""
Capacity (docs/messaging.md): one message resumes up to its consume entry's capacity of matching
waiting cases, oldest first, without waiting to fill up.
"""
import json
from datetime import datetime

import pytest
import pytz

from prosimos.exceptions import InvalidSimScenarioException
from prosimos.orchestrator import Message, ProcessSpec, ProsimosEngine, Verdict, run_engines
from prosimos.simulation_setup import SimDiffSetup
from testing_scripts.scripted_engine import ScriptedEngine

ASSETS = "testing_scripts/assets/messaging"
# an order arrives every 10 minutes from 09:00, is packed (1 min) and waits at Catch_Truck for a truck at
# the Tartu dock; a truck takes up to 2 waiting orders and gives each its truck_id
ORDERS = f"{ASSETS}/orders_and_trucks.bpmn"
ORDERS_JSON = f"{ASSETS}/orders_and_trucks.json"
START = pytz.utc.localize(datetime(2024, 1, 1, 9))


def at(hh_mm):
    hours, minutes = map(int, hh_mm.split(":"))
    return START.replace(hour=hours, minute=minutes)


class _Log:
    def __init__(self):
        self.rows = []

    def writerow(self, header):
        self.header = header

    def writerows(self, rows):
        self.rows.extend(rows)

    def loaded(self):
        """case -> when it was loaded on a truck."""
        return {row[0]: _time(row[2]) for row in self.rows if row[1] == "Load"}


def _time(logged):
    return logged if isinstance(logged, datetime) else datetime.fromisoformat(logged)


def _settings(tmp_path, change):
    with open(ORDERS_JSON) as file:
        settings = json.load(file)
    change(settings)
    path = tmp_path / "orders.json"
    path.write_text(json.dumps(settings))
    return str(path)


def _orders(cases, json_path=ORDERS_JSON):
    log = _Log()
    return ProsimosEngine(ProcessSpec("Orders", ORDERS, json_path, cases), START, log, seed=1), log


def _run_until_idle(engine):
    while engine.next_event_time() is not None:
        engine.step()


def truck(case_id="Carrier-0", **attributes):
    return Message("Truck", {"case_id": case_id, "dock": "Tartu", **attributes}, source="Carrier")


def _truck_ids(engine, cases):
    return [engine._env.sim_setup.bpmn_graph.all_attributes[case].get("truck_id") for case in cases]


def test_a_truck_of_capacity_2_resumes_the_two_oldest_of_three_waiting_orders():
    engine, log = _orders(3)
    _run_until_idle(engine)  # orders 0, 1, 2 wait from 09:01, 09:11, 09:21

    assert engine.deliver(truck(), at("12:00")) is Verdict.CLAIMED
    _run_until_idle(engine)

    assert log.loaded() == {0: at("12:00"), 1: at("12:00")}
    assert sorted(engine._env._parked_events) == [(2, "Catch_Truck")]  # the third keeps waiting


def test_a_truck_that_finds_one_order_resumes_one():
    engine, log = _orders(1)
    _run_until_idle(engine)

    assert engine.deliver(truck(), at("12:00")) is Verdict.CLAIMED
    _run_until_idle(engine)

    assert log.loaded() == {0: at("12:00")}


def test_a_truck_that_finds_none_is_pending_and_resumes_the_first_order_to_arrive():
    # the truck comes at 08:30, before any order; it doesn't wait to fill up, so it takes order 0
    # when that order starts waiting at 09:01, and order 1 is left waiting
    orders, log = _orders(2)
    carrier = ScriptedEngine("Carrier")
    carrier.publish_at(START.replace(hour=8, minute=30), "Truck", case_id="Carrier-0", dock="Tartu")

    report = run_engines({"Carrier": carrier, "Orders": orders}, None, 1)

    assert [(process, time) for _, process, time in report.claims] == [("Orders", at("09:01"))]
    assert log.loaded() == {0: at("09:01")}
    assert [case.case_id for _, case in report.stalled] == ["Orders-1"]


def test_capacity_can_be_read_from_the_message(tmp_path):
    from_message = _settings(tmp_path, lambda settings: settings["messages"]["consume"][0].update(
        capacity={"attribute": "capacity"}))
    engine, log = _orders(4, json_path=from_message)
    _run_until_idle(engine)

    engine.deliver(truck(capacity=3), at("12:00"))
    _run_until_idle(engine)

    assert log.loaded() == {0: at("12:00"), 1: at("12:00"), 2: at("12:00")}


def test_a_missing_or_invalid_capacity_in_the_message_counts_as_1_with_one_warning(tmp_path):
    from_message = _settings(tmp_path, lambda settings: settings["messages"]["consume"][0].update(
        capacity={"attribute": "capacity"}))
    engine, log = _orders(3, json_path=from_message)
    _run_until_idle(engine)

    engine.deliver(truck("Carrier-0"), at("12:00"))  # no capacity
    engine.deliver(truck("Carrier-1", capacity=0), at("13:00"))  # not at least 1
    _run_until_idle(engine)

    assert log.loaded() == {0: at("12:00"), 1: at("13:00")}
    assert engine.finish().warnings == [
        "Truck message accepted at Catch_Truck has no valid capacity (got None); it resumes one waiting case"]


def test_every_resumed_order_gets_the_trucks_id():
    engine, _ = _orders(3)
    _run_until_idle(engine)

    engine.deliver(truck("Carrier-0"), at("12:00"))
    engine.deliver(truck("Carrier-1"), at("13:00"))

    assert _truck_ids(engine, (0, 1, 2)) == ["Carrier-0", "Carrier-0", "Carrier-1"]


def test_a_truck_for_another_dock_resumes_nobody():
    engine, _ = _orders(2)
    _run_until_idle(engine)

    assert engine.deliver(truck(dock="Tallinn"), at("12:00")) is Verdict.DISCARDED
    assert sorted(engine._env._parked_events) == [(0, "Catch_Truck"), (1, "Catch_Truck")]


@pytest.mark.parametrize("capacity, reason", [
    ({"value": 0}, r"'capacity' value must be a whole number of at least 1, got 0"),
    ({"value": 1.5}, r"'capacity' value must be a whole number of at least 1, got 1.5"),
    ({"value": "2"}, r"'capacity' value must be a whole number of at least 1, got '2'"),
    ({"value": 2, "attribute": "capacity"}, r"'capacity' must be either"),
    ({"attribute": ""}, r"'capacity' attribute must be a message attribute name"),
])
def test_an_invalid_capacity_is_rejected_when_the_model_is_loaded(tmp_path, capacity, reason):
    settings = _settings(tmp_path, lambda settings: settings["messages"]["consume"][0].update(capacity=capacity))

    with pytest.raises(InvalidSimScenarioException, match=reason):
        SimDiffSetup(ORDERS, settings, False, 1, START)


def test_a_start_event_takes_no_capacity(tmp_path):
    with open(f"{ASSETS}/tartu_warehouse.json") as file:
        settings = json.load(file)
    settings["messages"]["consume"][0]["capacity"] = {"value": 2}
    path = tmp_path / "warehouse.json"
    path.write_text(json.dumps(settings))

    with pytest.raises(InvalidSimScenarioException, match="a start event takes no capacity"):
        SimDiffSetup(f"{ASSETS}/tartu_warehouse.bpmn", str(path), False, None, START)
