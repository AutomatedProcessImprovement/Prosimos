"""
One event linked to many objects (docs/orchestrator.md, "OCEL output"): a message end whose entry has a qualifier
writes an event of its own per message, which links the objects at the other end of the message; with carry on the
entry, every later event of the case, or those it names, links them too.
"""
import dataclasses
import json
from collections import defaultdict
from datetime import datetime

import pandas as pd
import pytest
import pytz

from prosimos.exceptions import InvalidSimScenarioException
from prosimos.orchestration.config import ProcessSpec, SimulationConfig
from prosimos.orchestration.ocel_writer import MessageEvent, OcelProcess, write_ocel
from prosimos.orchestration.prosimos_engine import ProsimosEngine
from prosimos.orchestration.protocol import CaseObject
from prosimos.orchestration.simulation import run_multiprocess_simulation
from testing_scripts.multiprocess.examples.test_order_management import order_management_config, package_items
from testing_scripts.multiprocess.messaging.test_append import run as _run_orders
from testing_scripts.multiprocess.messaging.test_item_verdicts import PICKING, PICKING_JSON, SALES, SALES_JSON
from testing_scripts.multiprocess.orchestrator_settings import one_organization, with_qualifiers

ASSETS = "testing_scripts/assets/messaging"
GATEWAYS = "testing_scripts/assets/ocel_links"
START = pytz.utc.localize(datetime(2024, 1, 1, 9))


@pytest.fixture(scope="module")
def order_management(tmp_path_factory):
    """A 100-order run of the Order Management example written as OCEL, and the items each package took."""
    path = tmp_path_factory.mktemp("ocel_links") / "log.json"
    report = run_multiprocess_simulation(order_management_config(100), None, ocel_out_path=path)
    with open(path) as file:
        ocel = json.load(file)
    return report, ocel, {package: set(items) for package, items in package_items(report).items()}


def _events(ocel, activity):
    return [event for event in ocel["events"] if event["type"] == activity]


def _owner(event):
    """The event's own case's object, and the qualifier of that link: always the first relationship."""
    own = event["relationships"][0]
    return own["objectId"], own["qualifier"]


def _linked(event, qualifier):
    return {link["objectId"] for link in event["relationships"][1:] if link["qualifier"] == qualifier}


def item_ids(report):
    """(order id, item index) -> the item object, from the run report: the OCEL file writes neither attribute."""
    def at_creation(case_object, name):
        return next(value for attribute, value, _ in case_object.attributes if attribute == name)

    return {(at_creation(case_object, "order_id"), at_creation(case_object, "item_index")): case_object.object_id
            for process, case_object in report.objects if process == "Warehouse"}


def items_of_orders(report):
    items = defaultdict(set)
    for (order, _), item in item_ids(report).items():
        items[order].add(item)
    return items


def _without_products(event):
    return [link for link in event["relationships"] if link["qualifier"] != "product"]


def test_every_items_ordered_links_exactly_its_orders_items_and_place_order_none(order_management):
    # Sales's ItemOrdered entry has a qualifier: its throw event writes one event for the messages it sends at once
    report, ocel, _ = order_management
    items_of_order = items_of_orders(report)

    events = _events(ocel, "items ordered")
    assert len(events) == 100
    for event in events:
        order, qualifier = _owner(event)
        assert qualifier == "order"
        assert _linked(event, "item") == items_of_order[order]
    assert sum(len(items) for items in items_of_order.values()) > 300  # with this seed, orders have items
    assert all(_linked(event, "item") == set() for event in _events(ocel, "place order"))


def test_every_create_package_is_linked_to_exactly_the_items_its_package_collected(order_management):
    # by carry from its items picked, one event per item it collected
    _, ocel, items_of_package = order_management

    events = _events(ocel, "create package")
    assert events
    for event in events:
        package, qualifier = _owner(event)
        assert qualifier == "creates"
        assert _linked(event, "item") == items_of_package[package]
        assert len(_without_products(event)) == 1 + len(items_of_package[package])
    assert any(len(items_of_package[_owner(event)[0]]) > 1 for event in events)


def test_carried_the_later_events_of_a_package_link_its_items_too(order_management):
    _, ocel, items_of_package = order_management

    for activity in ("send package", "package delivered", "failed delivery"):
        events = _events(ocel, activity)
        assert events
        for event in events:
            package, qualifier = _owner(event)
            assert qualifier == "shipped package"
            assert _linked(event, "item") == items_of_package[package]


def test_carried_the_later_events_of_an_order_link_its_items_too(order_management):
    report, ocel, _ = order_management
    items_of_order = items_of_orders(report)

    for activity in ("confirm order", "payment reminder", "pay order"):
        for event in _events(ocel, activity):
            order, _ = _owner(event)
            assert _linked(event, "item") == items_of_order[order]


def test_not_carried_an_event_links_only_what_its_own_messages_linked(order_management):
    # the Warehouse's entries don't carry: a reorder takes and sends no message, so it links only its own item, and the
    # item's product, from the reference that picks it
    _, ocel, _ = order_management

    events = _events(ocel, "reorder item")
    assert events
    assert all([relation["qualifier"] for relation in event["relationships"]] == ["item", "product"]
               for event in events)


def test_an_items_events_link_neither_its_order_nor_its_package(order_management):
    # the order links its items from place order and the package from create package; the Warehouse's ItemOrdered
    # and ItemPicked entries have no qualifier, so the item's end of those messages writes no link
    _, ocel, _ = order_management
    item_events = [event for event in ocel["events"] if event["type"] in ("item out of stock", "reorder item", "pick item")]

    object_type = {obj["id"]: obj["type"] for obj in ocel["objects"]}
    assert item_events
    assert {object_type[link["objectId"]] for event in item_events for link in event["relationships"]} == {"item", "product"}


def test_a_process_without_objects_writes_no_message_events_and_warns_nothing():
    # Relay, beside Order Management, has no object type and no tasks: it claims every ItemPicked and publishes
    # Noted at once
    config = order_management_config(20)
    relay = ProcessSpec("Relay", f"{GATEWAYS}/relay.bpmn", f"{GATEWAYS}/relay.json", None)
    report = run_multiprocess_simulation(dataclasses.replace(config,
                                                             organizations=one_organization([*config.processes, relay],
                                                                                            config.catalogue),
                                                             consumer_groups=None), None)

    elements = [element for record in report.message_records
                for element in [*record.publishers, *record.claimers] if element.process == "Relay"]
    assert any(publisher.process == "Relay" for record in report.message_records for publisher in record.publishers)
    assert any(claimer.process == "Relay" for record in report.message_records for claimer in record.claimers)
    assert all(element.at is None for element in elements)
    assert not any(process == "Relay" for process, _ in report.engine_warnings)


def test_a_case_still_waiting_at_the_end_writes_an_event_for_the_message_it_took(tmp_path):
    # order 0 waits at Catch_Items from 09:10 for 3 items and gets one, at 09:20; it never reaches Pack
    def appending_with_links(settings):
        settings["messages"]["consume"][0].update(append={"item_ids": "item_id"}, qualifier="item")

    report, _, ocel = _run_orders(tmp_path, [{"item_id": "i17"}], appending_with_links)

    assert [case.object_id for _, case in report.stalled] == ["order-0"]
    assert report.engine_warnings == []
    [claimer] = [claimer for record in report.message_records for claimer in record.claimers]
    assert (claimer.at, claimer.enabled) == (START.replace(minute=20), START.replace(minute=10))
    assert [event["type"] for event in ocel["events"] if event["attributes"][0]["name"] != "resource"] == [
        claimer.activity]


def _ocel_of_verdicts(tmp_path, sales_json, picking_json, picking_type="item"):
    """The OCEL file and the CSV log of one order with 3 items: Sales publishes ItemOrdered per item after place
    order, each starts an item case in Picking (pick item), which publishes ItemReady; Sales collects them in a race
    against 2 days and then sends each item Shipped (or Canceled), which the item takes in its own race. Sales's
    orders are of type order, Picking's items of picking_type."""
    config = SimulationConfig(one_organization([ProcessSpec("Sales", SALES, sales_json, 1, "order"),
                                                ProcessSpec("Picking", PICKING, picking_json, None, picking_type)]),
                              START, 1)
    run_multiprocess_simulation(config, tmp_path / "log.csv", ocel_out_path=tmp_path / "log.json")
    with open(tmp_path / "log.json") as file:
        return json.load(file), pd.read_csv(tmp_path / "log.csv")


def test_an_entry_writes_its_ends_events_only_with_a_qualifier(tmp_path):
    # only Sales's ItemOrdered entry has one: its throw, Items ordered, writes one event, linking the 3 items; the
    # tasks link only their own case's object
    with open(SALES_JSON) as file:
        settings = json.load(file)
    next(entry for entry in settings["messages"]["publish"] if entry["type"] == "ItemOrdered")["qualifier"] = "item"
    (tmp_path / "sales.json").write_text(json.dumps(settings))

    ocel, _ = _ocel_of_verdicts(tmp_path, str(tmp_path / "sales.json"), PICKING_JSON)

    [ordered] = _events(ocel, "Items ordered")
    assert sorted(_linked(ordered, "item")) == ["item-0", "item-1", "item-2"]
    assert all(len(event["relationships"]) == 1 for event in ocel["events"] if event["type"] != "Items ordered")
    assert {event["type"] for event in ocel["events"]} == {"Place order", "Items ordered", "Pick item"}


def test_a_process_without_an_object_type_writes_nothing(tmp_path):
    # both ends of every message would link, but Picking has no object type: no item objects, no events of Picking,
    # and no link to its cases
    ocel, log = _ocel_of_verdicts(tmp_path, with_qualifiers(SALES_JSON), with_qualifiers(PICKING_JSON), None)

    assert {obj["type"] for obj in ocel["objects"]} == {"order"}
    assert {link["objectId"] for event in ocel["events"] for link in event["relationships"]} == {"order-0"}
    # Sales's tasks, and its message ends: Items ordered and Items shipped, 3 each, and All items ready
    tasks = [event for event in ocel["events"] if event["attributes"][0]["name"] == "resource"]
    assert len(tasks) == (log["process"] == "Sales").sum() < len(log)
    assert {event["type"] for event in ocel["events"]} - {event["type"] for event in tasks} == {
        "Items ordered", "All items ready", "Items shipped"}


def _row(case, activity, end):
    """An instant task of the case, enabled, started and completed at minute end."""
    at = START.replace(minute=end)
    return "P", {"case_id": case, "activity": activity, "resource": "R", "enable_time": at, "start_time": at,
                 "end_time": at}


def _message_event(minute, links):
    """Case 0 of P sends a message at element M at the minute, linking links, as (object, qualifier, carry)."""
    at = START.replace(minute=minute)
    return MessageEvent("P", "p-0", "M", "m", at, at, links)


def _relationships(tmp_path, process, carry=False):
    """Case 0 of P logs A then B, and between them sends a message linking object q-0, its entry carrying it with
    carry. Each event's (object, qualifier) links."""
    rows = [_row(0, "a", 1), _row(0, "b", 2), _row(1, "b", 3)]
    objects = [(name, CaseObject(f"{object_type}-{case}", START.replace(minute=case)))
               for name, object_type in (("P", "p"), ("Q", "q")) for case in range(2)]
    write_ocel(tmp_path / "log.json", {"P": process, "Q": OcelProcess("q")}, objects, rows, {"P": ["A", "B", "B"]},
               [_message_event(1, [(("Q", "q-0"), "item", carry)])])
    with open(tmp_path / "log.json") as file:
        return [[(link["objectId"], link["qualifier"]) for link in event["relationships"]]
                for event in json.load(file)["events"]]


def test_without_carry_a_link_stays_on_its_message_event(tmp_path):
    # A completes at 09:01, and the message is sent then, after it
    assert _relationships(tmp_path, OcelProcess("p")) == [
        [("p-0", "p")], [("p-0", "p"), ("q-0", "item")], [("p-0", "p")], [("p-1", "p")]]


def test_carry_passes_an_entry_s_links_from_its_message_event_to_the_later_events_of_its_case_only(tmp_path):
    assert _relationships(tmp_path, OcelProcess("p"), True) == [
        [("p-0", "p")], [("p-0", "p"), ("q-0", "item")], [("p-0", "p"), ("q-0", "item")], [("p-1", "p")]]


def _carried(tmp_path, links):
    """Case 0 of P logs A, B, C and D, and sends a message after A, linking links, as (object, qualifier, carry).
    The links to Q's objects of each event after the message's."""
    rows = [_row(0, activity, minute) for minute, activity in enumerate("abcd", start=1)]
    objects = [("P", CaseObject("p-0", START))] + [("Q", CaseObject(f"q-{case}", START)) for case in range(2)]
    write_ocel(tmp_path / "log.json", {"P": OcelProcess("p"), "Q": OcelProcess("q")}, objects, rows,
               {"P": ["A", "B", "C", "D"]}, [_message_event(1, links)])
    with open(tmp_path / "log.json") as file:
        return [[(link["objectId"], link["qualifier"]) for link in event["relationships"][1:]]
                for event in json.load(file)["events"]][2:]


def test_a_case_with_two_message_entries_carries_only_the_marked_one(tmp_path):
    # the message went to q-0 (items, carried) and came from q-1 (a shipment, not carried)
    assert _carried(tmp_path, [(("Q", "q-0"), "item", True), (("Q", "q-1"), "shipment", False)]) == [
        [("q-0", "item")], [("q-0", "item")], [("q-0", "item")]]


def test_a_list_of_activities_carries_only_to_those_events(tmp_path):
    # by name or by task id, as in link_events
    assert _carried(tmp_path, [(("Q", "q-0"), "item", ("b",)), (("Q", "q-1"), "shipment", ("D",))]) == [
        [("q-0", "item")], [], [("q-1", "shipment")]]


def test_the_own_qualifier_defaults_to_the_object_type_and_can_be_set_per_activity(tmp_path):
    assert _relationships(tmp_path, OcelProcess("p", "case", {"B": "finishes"})) == [
        [("p-0", "case")], [("p-0", "case"), ("q-0", "item")], [("p-0", "finishes")], [("p-1", "finishes")]]


def test_qualifier_by_activity_must_name_tasks_of_the_model():
    spec = ProcessSpec("Orders", f"{ASSETS}/orders_and_trucks.bpmn", f"{ASSETS}/orders_and_trucks.json", 1,
                       qualifier_by_activity=(("Catch_Truck", "waits"), ("Load", "loads")))

    with pytest.raises(ValueError, match="Orders: qualifier_by_activity names Catch_Truck, which is not a task"):
        ProsimosEngine(spec, START, None, seed=1)


def test_a_message_qualifier_must_be_a_non_empty_string(tmp_path):
    with open(f"{ASSETS}/orders_and_trucks.json") as file:
        settings = json.load(file)
    settings["messages"]["consume"][0]["qualifier"] = ""
    (tmp_path / "orders.json").write_text(json.dumps(settings))
    spec = ProcessSpec("Orders", f"{ASSETS}/orders_and_trucks.bpmn", str(tmp_path / "orders.json"), 1)

    with pytest.raises(InvalidSimScenarioException,
                       match=r"messages.consume\[0\]: 'qualifier' must be a non-empty string"):
        ProsimosEngine(spec, START, None, seed=1)
