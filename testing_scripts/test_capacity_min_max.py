"""
Capacity (min, max) (docs/messaging.md): one message resumes between min and max waiting cases. While fewer than min
are waiting, it waits (PENDING) and is offered again at the engine's next step; with min waiting, it resumes them and
any others waiting, up to max, oldest first.
"""
import json

import pytest

from prosimos.exceptions import InvalidSimScenarioException
from prosimos.orchestrator import Verdict, run_engines
from prosimos.simulation_setup import SimDiffSetup
from testing_scripts.scripted_engine import ScriptedEngine
from testing_scripts.test_capacity import (ASSETS, ORDERS, START, _orders, _run_until_idle, _settings, _truck_ids, at,
                                           truck)
from testing_scripts.test_race import AFTER_TIES, TRUCK_RACE, TRUCK_RACE_JSON, _race_orders
from testing_scripts.test_race import _settings as _race_settings


def _capacity(capacity):
    return lambda settings: settings["messages"]["consume"][0].update(capacity=capacity)


def _with_carrier(tmp_path, capacity, cases, trucks):
    """Orders (one every 10 minutes from 09:00, waiting at Catch_Truck from 09:01, 09:11, ...) and a Carrier
    publishing a truck at each (time, attributes) of trucks; the report and the log."""
    orders, log = _orders(cases, json_path=_settings(tmp_path, _capacity(capacity)))
    carrier = ScriptedEngine("Carrier")
    for number, (time, attributes) in enumerate(trucks):
        carrier.publish_at(at(time), "Truck", case_id=f"Carrier-{number}", dock="Tartu", **attributes)
    return run_engines({"Carrier": carrier, "Orders": orders}, None, 1), log


def test_2_4_with_1_order_waiting_stays_pending_then_resumes_2_when_the_2nd_order_parks(tmp_path):
    # the truck comes at 08:30: order 0 parks at 09:01, order 1 at 09:11; order 2, parking at 09:21, is too late
    report, log = _with_carrier(tmp_path, {"min": 2, "max": 4}, 3, [("08:30", {})])

    assert [(process, time) for _, process, time in report.claims] == [("Orders", at("09:11"))]  # one truck
    assert log.loaded() == {0: at("09:11"), 1: at("09:11")}
    assert [case.case_id for _, case in report.stalled] == ["Orders-2"]


def test_2_4_answers_pending_while_fewer_than_2_are_waiting(tmp_path):
    engine, log = _orders(1, json_path=_settings(tmp_path, _capacity({"min": 2, "max": 4})))
    _run_until_idle(engine)  # order 0 waits from 09:01

    assert engine.deliver(truck(), at("12:00")) is Verdict.PENDING
    _run_until_idle(engine)

    assert log.loaded() == {}
    assert sorted(engine._env._parked_events) == [(0, "Catch_Truck")]


def test_2_4_with_5_waiting_resumes_the_4_oldest(tmp_path):
    engine, log = _orders(5, json_path=_settings(tmp_path, _capacity({"min": 2, "max": 4})))
    _run_until_idle(engine)  # orders 0..4 wait from 09:01, 09:11, ..., 09:41

    assert engine.deliver(truck(), at("12:00")) is Verdict.CLAIMED
    _run_until_idle(engine)

    assert log.loaded() == {case: at("12:00") for case in range(4)}
    assert sorted(engine._env._parked_events) == [(4, "Catch_Truck")]
    assert _truck_ids(engine, range(5)) == ["Carrier-0"] * 4 + [None]


def test_1_2_behaves_exactly_like_capacity_2(tmp_path):
    # one truck before any order (it takes the first to arrive), then one for 3 waiting orders (it takes 2)
    trucks = [("08:30", {}), ("12:00", {})]

    report, log = _with_carrier(tmp_path, {"min": 1, "max": 2}, 4, trucks)
    expected_report, expected_log = _with_carrier(tmp_path, 2, 4, trucks)

    assert report.claims == expected_report.claims
    assert log.rows == expected_log.rows
    assert log.loaded() == {0: at("09:01"), 1: at("12:00"), 2: at("12:00")}


def test_a_message_still_waiting_at_the_end_is_reported_unclaimed(tmp_path):
    report, log = _with_carrier(tmp_path, {"min": 3, "max": 4}, 2, [("08:30", {})])

    assert report.claims == [] and log.loaded() == {}
    assert [(group, message.type) for group, message in report.unclaimed] == [("Orders", "Truck")]
    assert [case.case_id for _, case in report.stalled] == ["Orders-0", "Orders-1"]


def test_a_min_without_max_resumes_every_waiting_case(tmp_path):
    engine, log = _orders(5, json_path=_settings(tmp_path, _capacity({"min": 2})))
    _run_until_idle(engine)

    engine.deliver(truck(), at("12:00"))
    _run_until_idle(engine)

    assert log.loaded() == {case: at("12:00") for case in range(5)}


def test_both_bounds_can_be_read_from_the_message(tmp_path):
    capacity = {"min": {"attribute": "fewest"}, "max": {"attribute": "capacity"}}
    engine, log = _orders(4, json_path=_settings(tmp_path, _capacity(capacity)))
    _run_until_idle(engine)  # orders 0..3 wait

    assert engine.deliver(truck("Carrier-0", fewest=5, capacity=6), at("12:00")) is Verdict.PENDING
    assert engine.deliver(truck("Carrier-1", fewest=2, capacity=3), at("12:00")) is Verdict.CLAIMED
    _run_until_idle(engine)

    assert log.loaded() == {0: at("12:00"), 1: at("12:00"), 2: at("12:00")}


def test_a_min_of_2_and_a_max_from_the_message(tmp_path):
    engine, log = _orders(4, json_path=_settings(tmp_path, _capacity({"min": 2, "max": {"attribute": "capacity"}})))
    _run_until_idle(engine)

    engine.deliver(truck(capacity=3), at("12:00"))
    _run_until_idle(engine)

    assert log.loaded() == {0: at("12:00"), 1: at("12:00"), 2: at("12:00")}


def test_an_invalid_min_in_the_message_counts_as_1_with_one_warning(tmp_path):
    engine, log = _orders(2, json_path=_settings(tmp_path, _capacity({"min": {"attribute": "fewest"}, "max": 1})))
    _run_until_idle(engine)

    engine.deliver(truck("Carrier-0"), at("12:00"))  # no fewest
    engine.deliver(truck("Carrier-1", fewest=0), at("13:00"))
    _run_until_idle(engine)

    assert log.loaded() == {0: at("12:00"), 1: at("13:00")}
    assert engine.finish().warnings == [
        "Truck message accepted at Catch_Truck has no valid fewest (got None); its min counts as 1"]


@pytest.mark.parametrize("capacity, attributes, warning", [
    ({"min": 2, "max": {"attribute": "capacity"}}, {"capacity": 1},
     "Truck message accepted at Catch_Truck has no valid capacity (got 1, min 2); it resumes no more than 2 waiting "
     "cases"),
    ({"min": {"attribute": "fewest"}, "max": 2}, {"fewest": 3},
     "Truck message accepted at Catch_Truck has no valid max (got 2, min 3); it resumes no more than 3 waiting cases"),
])
def test_a_max_below_min_counts_as_min_with_one_warning(tmp_path, capacity, attributes, warning):
    engine, log = _orders(4, json_path=_settings(tmp_path, _capacity(capacity)))
    _run_until_idle(engine)

    engine.deliver(truck(**attributes), at("12:00"))
    _run_until_idle(engine)

    min_ = capacity["min"] if isinstance(capacity["min"], int) else attributes["fewest"]
    assert log.loaded() == {case: at("12:00") for case in range(min_)}
    assert engine.finish().warnings == [warning]


@pytest.mark.parametrize("cases, loaded, by_courier", [
    # the third order reaches the gateway at 09:21: the truck takes all three then, and wins their races
    (3, {0: at("09:21"), 1: at("09:21"), 2: at("09:21")}, {}),
    # never three waiting: the truck waits, and each order's 4-hour deadline wins its race
    (2, {}, {0: at("13:01") + AFTER_TIES, 1: at("13:11") + AFTER_TIES}),
])
def test_in_a_race_a_truck_waits_for_min_orders_while_their_deadlines_run(tmp_path, cases, loaded, by_courier):
    json_path = _race_settings(tmp_path, _capacity({"min": 3, "max": 3}), base=TRUCK_RACE_JSON)
    orders, log = _race_orders(TRUCK_RACE, json_path, cases)
    carrier = ScriptedEngine("Carrier")
    carrier.publish_at(at("08:30"), "Truck", dock="Tartu", case_id="Carrier-0")

    report = run_engines({"Orders": orders, "Carrier": carrier}, None, 1)

    assert log.times("Load") == loaded
    assert log.times("Send by courier") == by_courier
    assert report.stalled == []


# load errors

@pytest.mark.parametrize("capacity, reason", [
    ({"min": 3, "max": 2}, r"'capacity' min 3 is above max 2"),
    ({"min": 0, "max": 2}, r"'capacity' min must be a whole number of at least 1 or \{\"attribute\": <message "
                           r"attribute>\}, got 0"),
    ({"min": 1, "max": 0}, r"'capacity' max must be a whole number of at least 1"),
    ({"min": 1.5}, r"'capacity' min must be a whole number of at least 1"),
    ({"min": {"atribute": "fewest"}}, r"capacity.min: unknown key 'atribute'; allowed: attribute"),
    ({"min": 1, "most": 3}, r"capacity: unknown key 'most'; allowed: max, min"),
    ({"max": 4}, r"'capacity' max needs a min"),
    (0, r"'capacity' must be either a whole number of at least 1"),
    (2.5, r"'capacity' must be either"),
    ("all", r"'capacity' must be either"),
])
def test_an_invalid_capacity_is_rejected(tmp_path, capacity, reason):
    with pytest.raises(InvalidSimScenarioException, match=r"messages.consume\[0\](: |\.)" + reason):
        SimDiffSetup(ORDERS, _settings(tmp_path, _capacity(capacity)), False, 1, START)


@pytest.mark.parametrize("capacity", [2, {"min": 2, "max": 4}])
def test_a_start_event_takes_no_capacity(tmp_path, capacity):
    with open(f"{ASSETS}/tartu_warehouse.json") as file:
        settings = json.load(file)
    settings["messages"]["consume"][0]["capacity"] = capacity
    path = tmp_path / "warehouse.json"
    path.write_text(json.dumps(settings))

    with pytest.raises(InvalidSimScenarioException, match="a start event takes no capacity"):
        SimDiffSetup(f"{ASSETS}/tartu_warehouse.bpmn", str(path), False, None, START)


@pytest.mark.parametrize("capacity", [2, {"min": 2, "max": 4}])
def test_capacity_with_collect_is_rejected(tmp_path, capacity):
    def with_collect(settings):
        settings["messages"]["consume"][0].update(capacity=capacity, collect=2)

    with pytest.raises(InvalidSimScenarioException, match="collect and capacity can't be combined"):
        SimDiffSetup(ORDERS, _settings(tmp_path, with_collect), False, 1, START)
