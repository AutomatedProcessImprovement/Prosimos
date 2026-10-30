"""
One event linked to many objects (docs/orchestrator.md, "OCEL output"): the task next to a message's throw or
catch event links the objects on the other end of the message, and with carry_links every later event of the
case links them too.
"""
import dataclasses
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
import pytz

from prosimos.exceptions import InvalidSimScenarioException
from prosimos.orchestration.config import ProcessSpec, SimulationConfig
from prosimos.orchestration.ocel_writer import OcelProcess, write_ocel
from prosimos.orchestration.orchestrator import orchestrate
from prosimos.orchestration.output import _message_links
from prosimos.orchestration.prosimos_engine import ProsimosEngine
from prosimos.orchestration.protocol import CaseElement, CaseObject, MessageRecord, Verdict
from prosimos.orchestration.simulation import run_multiprocess_simulation
from testing_scripts.multiprocess.examples.test_order_management import _config, package_items
from testing_scripts.multiprocess.messaging.test_append import _run as _run_orders
from testing_scripts.multiprocess.messaging.test_item_verdicts import (PICKING, PICKING_JSON, SALES, SALES_JSON,
                                                                       _picking_in)
from testing_scripts.multiprocess.orchestrator_settings import one_organization, with_qualifiers
from testing_scripts.multiprocess.scripted_engine import ScriptedEngine

ASSETS = "testing_scripts/assets/messaging"
GATEWAYS = "testing_scripts/assets/ocel_links"
START = pytz.utc.localize(datetime(2024, 1, 1, 9))
DROPPED = "the OCEL links of its messages there are dropped"


@pytest.fixture(scope="module")
def order_management(tmp_path_factory):
    """A 100-order run of the Order Management example written as OCEL, and the items each package took."""
    path = tmp_path_factory.mktemp("ocel_links") / "log.json"
    report = run_multiprocess_simulation(_config(100), None, ocel_out_path=path)
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


def _items(report):
    """(order id, item index) -> the item object, from the run report: the OCEL file writes neither attribute."""
    def at_creation(case_object, name):
        return next(value for attribute, value, _ in case_object.attributes if attribute == name)

    return {(at_creation(case_object, "order_id"), at_creation(case_object, "item_index")): case_object.object_id
            for process, case_object in report.objects if process == "Warehouse"}


def _items_of_orders(report):
    items = defaultdict(set)
    for (order, _), item in _items(report).items():
        items[order].add(item)
    return items


def _without_products(event):
    return [link for link in event["relationships"] if link["qualifier"] != "product"]


def test_every_place_order_is_linked_to_exactly_its_orders_items(order_management):
    report, ocel, _ = order_management
    items_of_order = _items_of_orders(report)

    events = _events(ocel, "place order")
    assert len(events) == 100
    for event in events:
        order, qualifier = _owner(event)
        assert qualifier == "order"
        assert _linked(event, "item") == items_of_order[order]
        # and its customer; the items' products (tag_along) are checked with the Order Management example
        assert len(_without_products(event)) == 1 + len(items_of_order[order]) + 1
    assert sum(len(items) for items in items_of_order.values()) > 300  # with this seed, orders have items


def test_every_create_package_is_linked_to_exactly_the_items_its_package_collected(order_management):
    _, ocel, items_of_package = order_management

    events = _events(ocel, "create package")
    assert events
    for event in events:
        package, qualifier = _owner(event)
        assert qualifier == "creates"
        assert _linked(event, "item") == items_of_package[package]
        assert len(_without_products(event)) == 1 + len(items_of_package[package])
    assert any(len(items_of_package[_owner(event)[0]]) > 1 for event in events)


def test_with_carry_links_the_later_events_of_a_package_link_its_items_too(order_management):
    _, ocel, items_of_package = order_management

    for activity in ("send package", "package delivered", "failed delivery"):
        events = _events(ocel, activity)
        assert events
        for event in events:
            package, qualifier = _owner(event)
            assert qualifier == "shipped package"
            assert _linked(event, "item") == items_of_package[package]


def test_with_carry_links_the_later_events_of_an_order_link_its_items_too(order_management):
    report, ocel, _ = order_management
    items_of_order = _items_of_orders(report)

    for activity in ("confirm order", "payment reminder", "pay order"):
        for event in _events(ocel, activity):
            order, _ = _owner(event)
            assert _linked(event, "item") == items_of_order[order]


def test_without_carry_links_an_event_links_only_what_its_own_messages_linked(order_management):
    # Warehouse has no carry_links: a reorder takes and sends no message, so it links only its own item, and the
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


def test_a_process_without_objects_attaches_nothing_and_warns_nothing():
    # Relay, beside Order Management, has no object type and no tasks: it claims every ItemPicked and publishes
    # Noted at once; attached, every claim and publish would warn
    config = _config(20)
    relay = ProcessSpec("Relay", f"{GATEWAYS}/relay.bpmn", f"{GATEWAYS}/relay.json", None)
    report = run_multiprocess_simulation(dataclasses.replace(config,
                                                             organizations=one_organization([*config.processes, relay],
                                                                                            config.catalogue),
                                                             consumer_groups=None), None)

    elements = [element for record in report.message_records
                for element in [record.publisher, *record.claimers] if element and element.process == "Relay"]
    assert any(record.publisher and record.publisher.process == "Relay" for record in report.message_records)
    assert any(claimer.process == "Relay" for record in report.message_records for claimer in record.claimers)
    assert all(element.task_rows == () for element in elements)
    assert not any(process == "Relay" for process, _ in report.engine_warnings)


def test_a_case_still_waiting_at_the_end_drops_its_links_with_a_warning(tmp_path):
    # order 0 waits for 3 items and gets one: its link would go to Pack, the task after Catch_Items
    def appending_with_links(settings):
        settings["messages"]["consume"][0].update(append={"item_ids": "item_id"}, qualifier="item")

    report, _, _ = _run_orders(tmp_path, [{"item_id": "i17"}], appending_with_links)

    assert [case.object_id for _, case in report.stalled] == ["order-0"]
    assert report.engine_warnings == [
        ("Orders", f"case order-0 reached no task after Catch_Items before the run ended; {DROPPED}")]


def _verdicts(picking_json=PICKING_JSON):
    """One order with 3 items: Sales publishes ItemOrdered per item after place order, each starts an item
    case in Picking (pick item), which publishes ItemReady; Sales collects them in a race against 2 days and
    then sends each item Shipped (or Canceled), which the item takes in its own race."""
    sales = ProsimosEngine(ProcessSpec("Sales", SALES, with_qualifiers(SALES_JSON), 1, "order"), START, None, seed=1)
    picking = ProsimosEngine(ProcessSpec("Picking", PICKING, with_qualifiers(picking_json), None, "item"), START, None,
                             seed=1)
    report = orchestrate({"Sales": sales, "Picking": picking}, None, 1)
    return report, {"Sales": sales, "Picking": picking}


def _ocel_of_verdicts(tmp_path, sales_json, picking_json, picking_type="item"):
    """The OCEL file and the CSV log of the run of _verdicts, Sales's orders of type order, Picking's items of
    picking_type."""
    config = SimulationConfig(one_organization([ProcessSpec("Sales", SALES, sales_json, 1, "order"),
                                                ProcessSpec("Picking", PICKING, picking_json, None, picking_type)]),
                              START, 1)
    run_multiprocess_simulation(config, tmp_path / "log.csv", ocel_out_path=tmp_path / "log.json")
    with open(tmp_path / "log.json") as file:
        return json.load(file), pd.read_csv(tmp_path / "log.csv")


def test_an_entry_writes_its_ends_links_only_with_a_qualifier(tmp_path):
    # only Sales's ItemOrdered entry has one: place order links the 3 items, and their pick item doesn't link the order
    with open(SALES_JSON) as file:
        settings = json.load(file)
    next(entry for entry in settings["messages"]["publish"] if entry["type"] == "ItemOrdered")["qualifier"] = "item"
    (tmp_path / "sales.json").write_text(json.dumps(settings))

    ocel, _ = _ocel_of_verdicts(tmp_path, str(tmp_path / "sales.json"), PICKING_JSON)

    [place_order] = _events(ocel, "Place order")
    assert sorted(_linked(place_order, "item")) == ["item-0", "item-1", "item-2"]
    picks = _events(ocel, "Pick item")
    assert len(picks) == 3 and all(len(event["relationships"]) == 1 for event in picks)  # only its own item
    assert {link["qualifier"] for event in ocel["events"] for link in event["relationships"][1:]} == {"item"}


def test_a_process_without_an_object_type_writes_nothing(tmp_path):
    # both ends of every message would link, but Picking has no object type: no item objects, no events of Picking,
    # and no link to its cases
    ocel, log = _ocel_of_verdicts(tmp_path, with_qualifiers(SALES_JSON), with_qualifiers(PICKING_JSON), None)

    assert {obj["type"] for obj in ocel["objects"]} == {"order"}
    assert {link["objectId"] for event in ocel["events"] for link in event["relationships"]} == {"order-0"}
    assert len(ocel["events"]) == (log["process"] == "Sales").sum() < len(log)


def _attached(engines, element):
    """(case id, element id) of each row a publisher or claimer is attached to."""
    env = engines[element.process]._env
    return [(env.object_id(env.logged_rows[row][0]), env.logged_rows[row][1]) for row in element.task_rows]


def test_a_publish_attaches_to_the_last_task_before_it_and_a_message_start_to_the_next_task_after_it():
    report, engines = _verdicts()

    for record in report.message_records:
        if record.type == "ItemOrdered":
            assert _attached(engines, record.publisher) == [("order-0", "Place_Order")]
            [claimer] = record.claimers
            assert _attached(engines, claimer) == [(claimer.object_id, "Pick_Item")]
        if record.type == "Shipped":  # the last task before it is place order, back across the race
            assert _attached(engines, record.publisher) == [("order-0", "Place_Order")]


def test_an_element_with_no_task_next_to_it_drops_its_links_with_one_warning():
    # Catch_Items and Catch_Shipped lead straight to the end: 3 messages each, one warning each
    report, _ = _verdicts()

    claimers = [claimer for record in report.message_records for claimer in record.claimers
                if claimer.element_id in ("Catch_Items", "Catch_Shipped")]
    assert len(claimers) == 6 and all(claimer.task_rows == () for claimer in claimers)
    assert report.engine_warnings == [
        ("Picking", f"Catch_Shipped has no task after it in the model; {DROPPED}"),
        ("Sales", f"Catch_Items has no task after it in the model; {DROPPED}")]


def test_the_winning_race_branch_attaches_to_its_own_next_task(tmp_path):
    # items take 20 hours to pick, so the order's deadline wins and each item takes its Canceled
    report, engines = _verdicts(_picking_in(tmp_path, 20))

    cancelled = [record.claimers[0] for record in report.message_records if record.type == "Cancelled"]
    assert len(cancelled) == 3
    assert [_attached(engines, claimer) for claimer in cancelled] == [
        [(claimer.object_id, "Return_To_Stock")] for claimer in cancelled]


def test_capacity_attaches_each_resumed_case_to_its_own_next_task():
    orders = ProsimosEngine(ProcessSpec("Orders", f"{ASSETS}/orders_and_trucks.bpmn",
                                        with_qualifiers(f"{ASSETS}/orders_and_trucks.json"), 3, "order"), START, None,
                            seed=1)
    carrier = ScriptedEngine("Carrier")
    carrier.publish_at(START.replace(minute=30), "Truck", dock="Tartu", object_id="Carrier-0")

    report = orchestrate({"Orders": orders, "Carrier": carrier}, None, 1)

    [record] = [record for record in report.message_records if record.type == "Truck"]
    assert [_attached({"Orders": orders}, claimer) for claimer in record.claimers] == [
        [("order-0", "Load")], [("order-1", "Load")]]


def _row(case, activity, end):
    """An instant task of the case, enabled, started and completed at minute end."""
    at = START.replace(minute=end)
    return "P", {"case_id": case, "activity": activity, "resource": "R", "enable_time": at, "start_time": at,
                 "end_time": at}


def _relationships(tmp_path, process):
    """Case 0 of P logs A then B; A's message linked object q-0. Each event's (object, qualifier) links."""
    rows = [_row(0, "a", 1), _row(0, "b", 2), _row(1, "b", 3)]
    objects = [(name, CaseObject(f"{object_type}-{case}", START.replace(minute=case)))
               for name, object_type in (("P", "p"), ("Q", "q")) for case in range(2)]
    write_ocel(tmp_path / "log.json", {"P": process, "Q": OcelProcess("q")}, objects, rows, {"P": ["A", "B", "B"]},
               {("P", 0): [(("Q", "q-0"), "item")]})
    with open(tmp_path / "log.json") as file:
        return [[(link["objectId"], link["qualifier"]) for link in event["relationships"]]
                for event in json.load(file)["events"]]


def test_carry_links_is_off_by_default(tmp_path):
    assert _relationships(tmp_path, OcelProcess("p")) == [
        [("p-0", "p"), ("q-0", "item")], [("p-0", "p")], [("p-1", "p")]]


def test_carry_links_passes_a_cases_links_to_its_later_events_only(tmp_path):
    assert _relationships(tmp_path, OcelProcess("p", carry_links=True)) == [
        [("p-0", "p"), ("q-0", "item")], [("p-0", "p"), ("q-0", "item")], [("p-1", "p")]]


def test_the_own_qualifier_defaults_to_the_object_type_and_can_be_set_per_activity(tmp_path):
    assert _relationships(tmp_path, OcelProcess("p", "case", {"B": "finishes"})) == [
        [("p-0", "case"), ("q-0", "item")], [("p-0", "finishes")], [("p-1", "finishes")]]


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


def _gateway_run(model, settings=None, seed=1, **messages):
    """One case of a gateway model: Prepare (10 minutes) from 09:00, then the model's gateways. A scripted
    process publishes each given message type at its time (or times) after 09:00 and takes every Done.
    settings: the settings file without .json, in the model's folder unless it is a path."""
    settings = settings if isinstance(settings, Path) else f"{GATEWAYS}/{settings or model}"
    engine = ProsimosEngine(ProcessSpec("P", f"{GATEWAYS}/{model}.bpmn", with_qualifiers(f"{settings}.json"), 1, "p"),
                            START, None, seed=seed)
    other = ScriptedEngine("S")
    for message_type, after in messages.items():
        for each in after if isinstance(after, list) else [after]:
            other.publish_at(START + each, message_type)
    other.consume("Done", lambda message, now: Verdict.CLAIMED)
    report = orchestrate({"P": engine, "S": other}, None, seed)
    return report, engine


def _rows_of(report, engine, message_type, element_id):
    """For each message of the type, the (element id, completion) of every row its end at element_id attaches
    to, in time order."""
    rows = engine._env.logged_rows
    ends = [end for record in report.message_records if record.type == message_type
            for end in [record.publisher, *record.claimers] if end is not None and end.element_id == element_id]
    return [sorted((rows[row][1], rows[row][3]) for row in end.task_rows) for end in ends]


def _tasks(attached):
    return [sorted(task for task, _ in rows) for rows in attached]


MINUTES = timedelta(minutes=1)


def test_after_a_parallel_split_a_claim_links_every_task_it_enables():
    # Go at 09:30 resumes the case, whose parallel split enables A and B at once
    report, engine = _gateway_run("and_split_after_catch", Go=30 * MINUTES)

    assert _tasks(_rows_of(report, engine, "Go", "Catch_Go")) == [["Task_A", "Task_B"]]
    assert report.engine_warnings == []


def test_before_a_parallel_join_a_publish_links_the_last_task_of_every_branch():
    # A ends at 09:20 and B at 09:40; the join then passes Done, which links both, not Prepare before the split
    report, engine = _gateway_run("and_join_before_throw")

    [attached] = _rows_of(report, engine, "Done", "Throw_Done")
    assert attached == [("Task_A", START.replace(minute=20)), ("Task_B", START.replace(minute=40))]
    [done] = [message for message in report.published if message.type == "Done"]
    assert done.time == START.replace(minute=40)


def test_after_a_parallel_split_a_claim_links_a_task_its_branch_enables_later():
    # Go at 09:30: A is enabled at once, B only after its branch's 10-minute timer, at 09:40
    report, engine = _gateway_run("and_split_with_timer_after_catch", Go=30 * MINUTES)

    [claimer] = [claimer for record in report.message_records for claimer in record.claimers]
    rows = engine._env.logged_rows
    assert sorted((rows[row][1], rows[row][2]) for row in claimer.task_rows) == [
        ("Task_A", START.replace(minute=30)), ("Task_B", START.replace(minute=40))]


def test_messages_collected_before_a_parallel_split_each_link_every_task_it_enables(tmp_path):
    # the case collects two Go, at 09:20 and 09:30, before it moves on to the split
    with open(f"{GATEWAYS}/and_split_after_catch.json") as file:
        settings = json.load(file)
    settings["messages"]["consume"][0]["collect"] = 2
    (tmp_path / "collect_2.json").write_text(json.dumps(settings))

    report, engine = _gateway_run("and_split_after_catch", tmp_path / "collect_2", Go=[20 * MINUTES, 30 * MINUTES])

    assert _tasks(_rows_of(report, engine, "Go", "Catch_Go")) == [["Task_A", "Task_B"], ["Task_A", "Task_B"]]


def test_in_a_loop_a_claim_links_only_the_tasks_of_its_own_round():
    # seed 4: a Go every hour from 09:30 starts a round; round 1 does A alone, rounds 2 and 3 do A and B. The
    # first B after round 1's Go is round 2's, which is not linked to it
    report, engine = _gateway_run("or_split_in_a_loop", seed=4, Go=[(30 + 60 * hour) * MINUTES for hour in range(5)])

    attached = _rows_of(report, engine, "Go", "Catch_Go")
    assert _tasks(attached) == [["Task_A"], ["Task_A", "Task_B"], ["Task_A", "Task_B"]]
    claimed = sorted(time for message_id, process, time in report.claims if process == "P")
    for rows, since, until in zip(attached, claimed, claimed[1:] + [None]):
        assert all(since <= completed and (until is None or completed <= until) for _, completed in rows)


@pytest.mark.parametrize("settings, taken", [
    ("or_split_after_catch_several", ["Task_A", "Task_B"]),  # A and B always, C never
    ("or_split_after_catch_single", ["Task_B"]),  # B only
])
def test_after_an_inclusive_split_a_claim_links_only_the_branches_it_took(settings, taken):
    report, engine = _gateway_run("or_split_after_catch", settings, Go=30 * MINUTES)

    assert _tasks(_rows_of(report, engine, "Go", "Catch_Go")) == [taken]


def test_before_an_inclusive_join_a_publish_links_the_last_task_of_the_branches_taken():
    # the split takes A and B, never C
    report, engine = _gateway_run("or_join_before_throw")

    assert _tasks(_rows_of(report, engine, "Done", "Throw_Done")) == [["Task_A", "Task_B"]]


def test_in_a_loop_a_publish_links_only_the_tasks_of_its_own_round():
    # seed 1: three rounds, A and B, A and B, then A alone; B of round 2 is not linked to round 3's Done
    report, engine = _gateway_run("or_join_in_a_loop")

    assert _tasks(_rows_of(report, engine, "Done", "Throw_Done")) == [
        ["Task_A", "Task_B"], ["Task_A", "Task_B"], ["Task_A"]]
    published = [message.time for message in report.published if message.type == "Done"]
    for rows, since, until in zip(_rows_of(report, engine, "Done", "Throw_Done"), [None] + published, published):
        assert all((since is None or since < completed) and completed <= until for _, completed in rows)


@pytest.mark.parametrize("messages, linked", [
    ({"Go": 30 * MINUTES, "Ok": 40 * MINUTES}, "Task_Ok"),  # Ok wins the race
    ({"Go": 30 * MINUTES}, "Task_Late"),  # the hour's timer wins
])
def test_after_an_event_based_gateway_a_claim_links_only_the_winning_branchs_task(messages, linked):
    report, engine = _gateway_run("event_gateway_after_catch", **messages)

    assert _tasks(_rows_of(report, engine, "Go", "Catch_Go")) == [[linked]]
    assert _tasks(_rows_of(report, engine, "Ok", "Catch_Ok")) == ([[linked]] if "Ok" in messages else [])


def test_a_message_attached_to_several_tasks_links_its_objects_from_each():
    record = MessageRecord("m1", "Go", CaseElement("S", "S-0", "Throw_Go", "go", (4,)),
                           [CaseElement("P", "P-0", "Catch_Go", "sender", (1, 2))])

    assert _message_links([record], {"S": OcelProcess("s"), "P": OcelProcess("p")}) == {
        ("S", 4): [(("P", "P-0"), "go")], ("P", 1): [(("S", "S-0"), "sender")], ("P", 2): [(("S", "S-0"), "sender")]}
