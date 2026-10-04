"""
Collecting (docs/messaging.md): a case waits at a catch event until it has claimed collect matching
messages, the number fixed or read from a case attribute when the case reaches the event.
"""
import csv
import json
from datetime import datetime

import pytest
import pytz

from cli.diff_res_bpsim import run_summary
from prosimos.exceptions import InvalidSimScenarioException
from prosimos.orchestrator import Message, ProcessSpec, ProsimosEngine, Verdict, run_engines
from prosimos.simulation_engine import run_simulation
from prosimos.simulation_setup import SimDiffSetup
from testing_scripts.scripted_engine import ScriptedEngine

ASSETS = "testing_scripts/assets/messaging"
# an order arrives every hour from 09:00; after Pick (10 min) it waits at Catch_Items for one ItemReady
# per item (case attribute items, 3), each with its order_id, copying item_id into last_item; then Pack
ORDERS = f"{ASSETS}/orders_and_items.bpmn"
ORDERS_JSON = f"{ASSETS}/orders_and_items.json"
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

    def packed(self):
        """case -> when Pack was enabled, i.e. when the case continued after collecting."""
        return {row[0]: _time(row[2]) for row in self.rows if row[1] == "Pack"}


def _time(logged):
    return logged if isinstance(logged, datetime) else datetime.fromisoformat(logged)


def _settings(tmp_path, change):
    with open(ORDERS_JSON) as file:
        settings = json.load(file)
    change(settings)
    path = tmp_path / "orders.json"
    path.write_text(json.dumps(settings))
    return str(path)


def _items(value):
    return lambda settings: settings["case_attributes"][0]["values"].update(distribution_params=[{"value": value}])


def _orders(cases=1, json_path=ORDERS_JSON):
    log = _Log()
    return ProsimosEngine(ProcessSpec("Orders", ORDERS, json_path, cases), START, log, seed=1), log


def _run_until_idle(engine):
    while engine.next_event_time() is not None:
        engine.step()


def item(order_id="Orders-0", item_id="i1"):
    return Message("ItemReady", {"order_id": order_id, "item_id": item_id}, source="Picking")


def test_an_order_with_3_items_continues_at_the_time_of_its_third_item_not_before():
    engine, log = _orders()
    _run_until_idle(engine)  # order 0 waits at Catch_Items from 09:10

    for hh_mm, item_id in (("10:00", "i1"), ("10:30", "i2")):
        assert engine.deliver(item(item_id=item_id), at(hh_mm)) is Verdict.CLAIMED
        _run_until_idle(engine)
        assert log.packed() == {}  # still collecting
    assert engine.deliver(item(item_id="i3"), at("11:00")) is Verdict.CLAIMED
    _run_until_idle(engine)

    assert log.packed() == {0: at("11:00")}


def test_items_that_arrive_before_the_order_reaches_the_event_are_claimed_when_it_parks():
    # two items come at 08:30 and 08:40, before the order arrives (09:00) and reaches the event (09:10);
    # both are claimed in the step where it parks, the third comes at 12:00
    orders, log = _orders()
    picking = ScriptedEngine("Picking")
    for hh_mm, item_id in (("08:30", "i1"), ("08:40", "i2"), ("12:00", "i3")):
        picking.publish_at(at(hh_mm), "ItemReady", order_id="Orders-0", item_id=item_id)

    report = run_engines({"Picking": picking, "Orders": orders}, None, 1)

    assert [time for _, _, time in report.claims] == [at("09:10"), at("09:10"), at("12:00")]
    assert log.packed() == {0: at("12:00")}


def test_an_order_with_0_items_passes_straight_on(tmp_path):
    engine, log = _orders(json_path=_settings(tmp_path, _items(0)))
    _run_until_idle(engine)

    assert log.packed() == {0: at("09:10")}  # the moment Pick ended, no message needed
    assert engine.finish().stalled == []


def test_with_0_items_the_catch_event_is_logged_with_zero_duration(tmp_path):
    log_path = tmp_path / "log.csv"
    run_simulation(ORDERS, _settings(tmp_path, _items(0)), 1, None, log_path, START.isoformat(),
                   is_event_added_to_log=True)

    with open(log_path) as file:
        [catch] = [row for row in csv.DictReader(file) if row["activity"] == "All items ready"]
    assert catch["enable_time"] == catch["start_time"] == catch["end_time"]
    assert _time(catch["start_time"]) == at("09:10")


def test_an_order_that_gets_only_2_of_3_items_is_stalled_with_2_of_3():
    engine, _ = _orders()
    _run_until_idle(engine)
    engine.deliver(item(item_id="i1"), at("10:00"))
    engine.deliver(item(item_id="i2"), at("10:30"))
    _run_until_idle(engine)

    [stalled] = engine.finish().stalled
    assert (stalled.case_id, stalled.collected, stalled.needed) == ("Orders-0", 2, 3)


def test_the_summary_shows_how_many_a_stalled_case_collected():
    orders, _ = _orders()
    picking = ScriptedEngine("Picking")
    for hh_mm, item_id in (("10:00", "i1"), ("10:30", "i2")):
        picking.publish_at(at(hh_mm), "ItemReady", order_id="Orders-0", item_id=item_id)

    report = run_engines({"Picking": picking, "Orders": orders}, None, 1)

    assert "  Orders: 1 (Orders-0 collected 2 of 3)" in run_summary(report).splitlines()


def test_the_oldest_case_that_still_needs_messages_gets_the_next_one(tmp_path):
    # without a condition every item fits every order: order 0 (waiting from 09:10) takes the first two,
    # then order 1 (waiting from 10:10) takes the next ones
    any_order = _settings(tmp_path, lambda settings: (settings["messages"]["consume"][0].pop("condition"),
                                                      _items(2)(settings)))
    engine, log = _orders(cases=2, json_path=any_order)
    _run_until_idle(engine)

    for hh_mm in ("11:00", "11:30", "12:00"):
        engine.deliver(item(order_id="any"), at(hh_mm))
    _run_until_idle(engine)

    assert log.packed() == {0: at("11:30")}
    [stalled] = engine.finish().stalled
    assert (stalled.case_id, stalled.collected, stalled.needed) == ("Orders-1", 1, 2)


def test_copy_is_applied_at_every_claim_the_last_message_winning():
    engine, _ = _orders()
    _run_until_idle(engine)
    values = engine._env.sim_setup.bpmn_graph.all_attributes[0]

    seen = []
    for hh_mm, item_id in (("10:00", "i1"), ("10:30", "i2"), ("11:00", "i3")):
        engine.deliver(item(item_id=item_id), at(hh_mm))
        seen.append(values["last_item"])

    assert seen == ["i1", "i2", "i3"]


def test_a_number_to_collect_that_isnt_a_whole_number_counts_as_1_with_one_warning(tmp_path):
    engine, log = _orders(cases=2, json_path=_settings(tmp_path, _items(2.5)))
    _run_until_idle(engine)

    engine.deliver(item("Orders-0"), at("12:00"))
    engine.deliver(item("Orders-1"), at("12:00"))
    _run_until_idle(engine)

    assert log.packed() == {0: at("12:00"), 1: at("12:00")}
    assert engine.finish().warnings == [
        "case 0 reaches Catch_Items with no valid items (got 2.5); it collects one message"]


@pytest.mark.parametrize("change, reason", [
    (lambda entry: entry.update(collect={"value": -1}), r"'collect' value must be a whole number of at least 0, got -1"),
    (lambda entry: entry.update(collect={"value": 1.5}), r"'collect' value must be a whole number of at least 0, got 1.5"),
    (lambda entry: entry.update(collect={"value": 2, "case_attribute": "items"}), r"'collect' must be either"),
    (lambda entry: entry.update(capacity={"value": 2}),
     r"collect and capacity can't be combined on one entry.*Use two steps through an intermediate case"),
])
def test_an_invalid_collect_is_rejected_when_the_model_is_loaded(tmp_path, change, reason):
    settings = _settings(tmp_path, lambda settings: change(settings["messages"]["consume"][0]))

    with pytest.raises(InvalidSimScenarioException, match=reason):
        SimDiffSetup(ORDERS, settings, False, 1, START)


def test_collect_is_only_allowed_on_a_catch_event_with_exactly_one_consume_entry(tmp_path):
    # a second entry at Catch_Items, even without collect, makes it unclear which messages the case counts
    def second_entry(settings):
        settings["messages"]["consume"].append({"event_id": "Catch_Items", "type": "ItemCancelled"})

    with pytest.raises(InvalidSimScenarioException,
                       match=r"messages.consume\[0\]: collect is only allowed on a catch event with exactly one consume "
                             r"entry, but Catch_Items has 2: a case waiting there keeps one count. To accept several "
                             r"variants of one type, use one entry with alternatives in its condition; to wait for "
                             r"several kinds, use one catch event per type"):
        SimDiffSetup(ORDERS, _settings(tmp_path, second_entry), False, 1, START)


def test_a_catch_event_with_several_entries_and_no_collect_still_loads(tmp_path):
    def two_entries_without_collect(settings):
        settings["messages"]["consume"][0].pop("collect")
        settings["messages"]["consume"].append({"event_id": "Catch_Items", "type": "ItemCancelled"})

    setup = SimDiffSetup(ORDERS, _settings(tmp_path, two_entries_without_collect), False, 1, START)

    assert [point.type for point in setup.messaging.consume] == ["ItemReady", "ItemCancelled"]


def test_a_start_event_cant_collect_and_the_error_points_to_start_then_collect(tmp_path):
    with open(f"{ASSETS}/tartu_warehouse.json") as file:
        settings = json.load(file)
    settings["messages"]["consume"][0]["collect"] = {"value": 4}
    path = tmp_path / "warehouse.json"
    path.write_text(json.dumps(settings))

    with pytest.raises(InvalidSimScenarioException,
                       match=r"a start event can't collect.*Start on the first message, then collect the rest at a "
                             r"catch event right after the start"):
        SimDiffSetup(f"{ASSETS}/tartu_warehouse.bpmn", str(path), False, None, START)
