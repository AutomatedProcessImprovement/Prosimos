"""
Message events in the OCEL file (docs/orchestrator.md, "Output"): a throw, catch, start or end event whose entry has
a qualifier writes an event of its own per case, element and moment it sends or takes messages, linked to all their
objects, named after the element, enabled when the case reached it, with no resource; the CSV log stays a log of
tasks.
"""
import json
from datetime import datetime, timedelta

import pytz

from prosimos.orchestration.config import ProcessSpec, SimulationConfig
from prosimos.orchestration.simulation import run_multiprocess_simulation
from testing_scripts.multiprocess.messaging.test_item_verdicts import (PICKING, PICKING_JSON, SALES, SALES_JSON,
                                                                       picking_in)
from testing_scripts.multiprocess.messaging.test_collect import ORDERS, settings_with
from testing_scripts.multiprocess.orchestrator_settings import one_organization, with_qualifiers
from testing_scripts.multiprocess.scripted_engine import ScriptedEngine

START = pytz.utc.localize(datetime(2024, 1, 1, 9))
HOUR = timedelta(hours=1)
ROUTING = "testing_scripts/assets/order_routing"
ASSETS = "testing_scripts/assets/messaging"


def _ocel(tmp_path, specs, extra=None):
    config = SimulationConfig(one_organization(specs), START, 1, extra_processes=list(extra or {}))
    report = run_multiprocess_simulation(config, None, extra or {}, ocel_out_path=tmp_path / "log.json")
    with open(tmp_path / "log.json") as file:
        return report, json.load(file)["events"]


def _verdicts(tmp_path, picking_json=PICKING_JSON):
    """One order with 3 items (docs/messaging.md, "Races"), every entry with a qualifier: Sales places the order
    (10 minutes) and sends an ItemOrdered per item, each starting an item in Picking; one picker picks each in turn
    (an hour, or picking_json's time) and sends ItemReady; Sales collects them in a race against 2 days, then sends
    each item Shipped, or Cancelled."""
    return _ocel(tmp_path, [ProcessSpec("Sales", SALES, with_qualifiers(SALES_JSON), 1, "order"),
                            ProcessSpec("Picking", PICKING, with_qualifiers(picking_json), None, "item")])


def _of(events, activity):
    """(own object, time, enable_time, linked objects) of each event of the activity."""
    def attribute(event, name):
        return next(attribute["value"] for attribute in event["attributes"] if attribute["name"] == name)

    return [(event["relationships"][0]["objectId"], event["time"], attribute(event, "enable_time"),
             [link["objectId"] for link in event["relationships"][1:]])
            for event in events if event["type"] == activity]


def _at(moment):
    return moment.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def test_a_fan_out_of_3_writes_one_event_with_3_links_and_each_message_start_its_own(tmp_path):
    _, events = _verdicts(tmp_path)

    placed, items = _at(START + timedelta(minutes=10)), ["item-0", "item-1", "item-2"]
    assert _of(events, "Items ordered") == [("order-0", placed, placed, items)]
    assert _of(events, "Item ordered") == [(item, placed, placed, ["order-0"]) for item in items]


def test_a_collect_with_arrivals_at_different_times_writes_one_event_each_enabled_when_the_case_began_waiting(
        tmp_path):
    # the order waits for its items from 09:10; they are picked one after the other, by 10:10, 11:10, and 12:10
    _, events = _verdicts(tmp_path)

    waiting = _at(START + timedelta(minutes=10))
    assert _of(events, "All items ready") == [
        ("order-0", _at(START + timedelta(minutes=10) + (number + 1) * HOUR), waiting, [f"item-{number}"])
        for number in range(3)]
    assert all(event["attributes"][0]["name"] == "start_time" for event in events if event["type"] == "All items ready")


def test_a_message_taken_on_a_race_branch_that_lost_is_still_written(tmp_path):
    # picking takes 20 hours each: two items are ready within the 2 days, then the deadline wins and cancels
    _, events = _verdicts(tmp_path, picking_in(tmp_path, 20))

    assert [linked for _, _, _, linked in _of(events, "All items ready")] == [["item-0"], ["item-1"]]
    assert [(own, linked) for own, _, _, linked in _of(events, "Items cancelled")] == [
        ("order-0", ["item-0", "item-1", "item-2"])]
    assert not _of(events, "Items shipped")


def test_a_collect_taking_2_waiting_messages_at_once_writes_one_event(tmp_path):
    # Picking sends two ItemReady at 09:05, before order-0 waits for its 3 items at 09:10, and a third at 09:30
    def qualified(settings):
        settings["messages"]["consume"][0]["qualifier"] = "item"

    picking = ScriptedEngine("Picking")
    for minutes in (5, 5, 30):
        picking.publish_at(START.replace(minute=minutes), "ItemReady", order_id="order-0")
    _, events = _ocel(tmp_path, [ProcessSpec("Orders", ORDERS, settings_with(tmp_path, qualified), 1, "order")],
                      {"Picking": picking})

    waiting = _at(START.replace(minute=10))
    assert _of(events, "All items ready") == [("order-0", waiting, waiting, []),
                                              ("order-0", _at(START.replace(minute=30)), waiting, [])]


def test_a_message_end_event_writes_an_event_when_the_case_ends(tmp_path):
    # Order Routing's Sales and TartuWarehouse; the warehouse's end event Shipped sends each Shipment
    report, events = _ocel(tmp_path, [
        ProcessSpec("Sales", f"{ROUTING}/sales.bpmn", f"{ROUTING}/sales.json", 20, "order"),
        ProcessSpec("TartuWarehouse", f"{ROUTING}/tartu_warehouse.bpmn",
                    with_qualifiers(f"{ROUTING}/tartu_warehouse.json"), None, "shipment")])

    shipments = [message for message in report.published if message.type == "Shipment"]
    shipped = _of(events, "Shipped")
    assert len(shipped) == len(shipments) > 0
    assert sorted((time, order) for _, time, _, [order] in shipped) == sorted(
        (_at(message.time), message.attributes["order_id"]) for message in shipments)
    assert all(time == enabled for _, time, enabled, _ in shipped)


def test_with_capacity_each_resumed_case_writes_its_own_event(tmp_path):
    # two orders wait at the dock from 09:00 and 09:10 (orders_and_trucks); the Carrier's truck at 09:30 takes both
    carrier = ScriptedEngine("Carrier")
    carrier.publish_at(START.replace(minute=30), "Truck", dock="Tartu", object_id="Carrier-0")

    report, events = _ocel(tmp_path, [ProcessSpec("Orders", f"{ASSETS}/orders_and_trucks.bpmn",
                                                  with_qualifiers(f"{ASSETS}/orders_and_trucks.json"), 3, "order")],
                           {"Carrier": carrier})

    [truck] = [record for record in report.message_records if record.type == "Truck"]
    assert [(own, time, links) for own, time, _, links in _of(events, "Truck at the dock")] == [
        (claimer.object_id, _at(START.replace(minute=30)), []) for claimer in truck.claimers]
    assert len(truck.claimers) == 2
