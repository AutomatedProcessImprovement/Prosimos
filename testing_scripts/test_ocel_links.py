"""
One event linked to many objects (docs/orchestrator.md, "OCEL output"): the task next to a message's throw or
catch event links the objects on the other end of the message, and with carry_links every later event of the
case links them too.
"""
import json
from collections import defaultdict
from datetime import datetime, timedelta

import pytest
import pytz

from prosimos.exceptions import InvalidSimScenarioException
from prosimos.ocel_writer import OcelProcess, write_ocel
from prosimos.orchestrator import (CaseElement, MessageRecord, ProcessSpec, ProsimosEngine, Verdict, _message_links,
                                   run_engines, run_orchestrator)
from testing_scripts.scripted_engine import ScriptedEngine
from testing_scripts.test_item_verdicts import PICKING, PICKING_JSON, SALES, SALES_JSON, _picking_in
from testing_scripts.test_order_management import _config, _PackageContents

ASSETS = "testing_scripts/assets/messaging"
GATEWAYS = "testing_scripts/assets/ocel_links"
START = pytz.utc.localize(datetime(2024, 1, 1, 9))
DROPPED = "the OCEL links of its messages there are dropped"


@pytest.fixture(scope="module")
def order_management(tmp_path_factory):
    """A 100-order run of the Order Management example written as OCEL, and what each package took."""
    monkeypatch = pytest.MonkeyPatch()
    contents = _PackageContents(monkeypatch)
    path = tmp_path_factory.mktemp("ocel_links") / "log.json"
    try:
        report = run_orchestrator(_config(100), None, ocel_out_path=path)
    finally:
        monkeypatch.undo()
    with open(path) as file:
        ocel = json.load(file)
    return report, ocel, contents.items


def _events(ocel, activity):
    return [event for event in ocel["events"] if event["type"] == activity]


def _owner(event):
    """The event's own case's object, and the qualifier of that link: always the first relationship."""
    own = event["relationships"][0]
    return own["objectId"], own["qualifier"]


def _linked(event, qualifier):
    return {link["objectId"] for link in event["relationships"][1:] if link["qualifier"] == qualifier}


def _items(ocel):
    """(order id, item index) -> the item object."""
    def at_creation(obj, name):
        return next(attribute["value"] for attribute in obj["attributes"] if attribute["name"] == name)

    return {(at_creation(obj, "order_id"), at_creation(obj, "item_index")): obj["id"]
            for obj in ocel["objects"] if obj["type"] == "items"}


def _items_of_orders(ocel):
    items = defaultdict(set)
    for (order, _), item in _items(ocel).items():
        items[order].add(item)
    return items


def _items_of_packages(ocel, contents):
    items = _items(ocel)
    return {f"Packaging-{package}": {items[(order, index)] for order, index, _ in took}
            for package, took in contents.items()}


def test_every_place_order_is_linked_to_exactly_its_orders_items(order_management):
    _, ocel, _ = order_management
    items_of_order = _items_of_orders(ocel)

    events = _events(ocel, "place order")
    assert len(events) == 100
    for event in events:
        order, qualifier = _owner(event)
        assert qualifier == "order"
        assert _linked(event, "item") == items_of_order[order]
        assert len(event["relationships"]) == 1 + len(items_of_order[order])  # nothing else
    assert sum(len(items) for items in items_of_order.values()) > 300  # with this seed, orders have items


def test_every_create_package_is_linked_to_exactly_the_items_its_package_collected(order_management):
    _, ocel, contents = order_management
    items_of_package = _items_of_packages(ocel, contents)

    events = _events(ocel, "create package")
    assert events
    for event in events:
        package, qualifier = _owner(event)
        assert qualifier == "creates"
        assert _linked(event, "item") == items_of_package[package]
        assert len(event["relationships"]) == 1 + len(items_of_package[package])
    assert any(len(items_of_package[_owner(event)[0]]) > 1 for event in events)


def test_with_carry_links_the_later_events_of_a_package_link_its_items_too(order_management):
    _, ocel, contents = order_management
    items_of_package = _items_of_packages(ocel, contents)

    for activity, own_qualifier in [("send package", "shipped package"), ("package delivered", "package"),
                                    ("failed delivery", "package")]:
        events = _events(ocel, activity)
        assert events
        for event in events:
            package, qualifier = _owner(event)
            assert qualifier == own_qualifier
            assert _linked(event, "item") == items_of_package[package]


def test_with_carry_links_the_later_events_of_an_order_link_its_items_too(order_management):
    _, ocel, _ = order_management
    items_of_order = _items_of_orders(ocel)

    for activity in ("confirm order", "payment reminder", "pay order"):
        for event in _events(ocel, activity):
            order, _ = _owner(event)
            assert _linked(event, "item") == items_of_order[order]


def test_without_carry_links_an_event_links_only_what_its_own_messages_linked(order_management):
    # Warehouse has no carry_links: a reorder takes and sends no message, so it links only its own item
    _, ocel, _ = order_management

    events = _events(ocel, "reorder item")
    assert events
    assert all(len(event["relationships"]) == 1 and _owner(event)[1] == "item" for event in events)


def test_an_item_links_its_order_at_its_first_task_and_its_package_at_its_last(order_management):
    _, ocel, _ = order_management
    order_of_item = {item: order for (order, _), item in _items(ocel).items()}
    first_task = {}
    for event in ocel["events"]:
        if event["type"] in ("item out of stock", "pick item"):
            first_task.setdefault(_owner(event)[0], event)

    for item, event in first_task.items():
        assert _linked(event, "order") == {order_of_item[item]}
    for event in _events(ocel, "pick item"):
        assert len(_linked(event, "package")) == 1


def test_a_won_race_links_its_next_task_and_a_won_timer_links_nothing(order_management):
    # Sales races Payment against a 20-day reminder: a payment claimed at Catch_Payment attaches to the
    # order's pay order; a reminder is never the task a claim attaches to
    report, _, _ = order_management
    elements = report.logged_elements["Sales"]

    attached = [row for record in report.message_records for claimer in record.claimers
                if claimer.element_id == "Catch_Payment" for row in claimer.task_rows]
    assert attached
    assert {elements[row] for row in attached} == {"Pay_Order"}
    assert len(set(attached)) == len(attached)  # each payment its own order's
    assert "Payment_Reminder" in elements


def test_a_process_without_objects_attaches_nothing_and_warns_nothing(order_management):
    # Customer (object_type null) has no tasks at all: attached, every claim and publish would warn
    report, _, _ = order_management

    customer = [element for record in report.message_records
                for element in [record.publisher, *record.claimers] if element and element.process == "Customer"]
    assert customer
    assert all(element.task_rows == () for element in customer)
    assert not any(process == "Customer" for process, _ in report.engine_warnings)


def test_a_case_still_waiting_at_the_end_drops_its_links_with_a_warning(order_management):
    report, _, _ = order_management
    stalled = {case.case_id for process, case in report.stalled if process == "Packaging"}

    dropped = [warning for _, warning in report.engine_warnings if DROPPED in warning]
    assert stalled and dropped
    for warning in dropped:  # once per element, naming the first such case
        assert warning.split()[1] in stalled
        assert "before the run ended" in warning


def _verdicts(picking_json=PICKING_JSON):
    """One order with 3 items: Sales publishes ItemOrdered per item after place order, each starts an item
    case in Picking (pick item), which publishes ItemReady; Sales collects them in a race against 2 days and
    then sends each item Shipped (or Canceled), which the item takes in its own race."""
    sales = ProsimosEngine(ProcessSpec("Sales", SALES, SALES_JSON, 1), START, None, seed=1)
    picking = ProsimosEngine(ProcessSpec("Picking", PICKING, picking_json, None), START, None, seed=1)
    report = run_engines({"Sales": sales, "Picking": picking}, None, 1)
    return report, {"Sales": sales, "Picking": picking}


def _attached(engines, element):
    """(case id, element id) of each row a publisher or claimer is attached to."""
    env = engines[element.process]._env
    return [(env.case_id(env.logged_rows[row][0]), env.logged_rows[row][1]) for row in element.task_rows]


def test_a_publish_attaches_to_the_last_task_before_it_and_a_message_start_to_the_next_task_after_it():
    report, engines = _verdicts()

    for record in report.message_records:
        if record.type == "ItemOrdered":
            assert _attached(engines, record.publisher) == [("Sales-0", "Place_Order")]
            [claimer] = record.claimers
            assert _attached(engines, claimer) == [(claimer.case_id, "Pick_Item")]
        if record.type == "Shipped":  # the last task before it is place order, back across the race
            assert _attached(engines, record.publisher) == [("Sales-0", "Place_Order")]


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
        [(claimer.case_id, "Return_To_Stock")] for claimer in cancelled]


def test_capacity_attaches_each_resumed_case_to_its_own_next_task():
    orders = ProsimosEngine(ProcessSpec("Orders", f"{ASSETS}/orders_and_trucks.bpmn",
                                        f"{ASSETS}/orders_and_trucks.json", 3), START, None, seed=1)
    carrier = ScriptedEngine("Carrier")
    carrier.publish_at(START.replace(minute=30), "Truck", dock="Tartu", case_id="Carrier-0")

    report = run_engines({"Orders": orders, "Carrier": carrier}, None, 1)

    [record] = [record for record in report.message_records if record.type == "Truck"]
    assert [_attached({"Orders": orders}, claimer) for claimer in record.claimers] == [
        [("Orders-0", "Load")], [("Orders-1", "Load")]]


def _row(case, activity, end):
    return "P", {"case_id": case, "activity": activity, "resource": "R", "end_time": START.replace(minute=end)}


def _relationships(tmp_path, process):
    """Case 0 of P logs A then B; A's message linked object Q-1. Each event's (object, qualifier) links."""
    rows = [_row(0, "a", 1), _row(0, "b", 2), _row(1, "b", 3)]
    write_ocel(tmp_path / "log.json", {"P": process}, [], rows, {"P": ["A", "B", "B"]},
               {("P", 0): [("Q-1", "item")]})
    with open(tmp_path / "log.json") as file:
        return [[(link["objectId"], link["qualifier"]) for link in event["relationships"]]
                for event in json.load(file)["events"]]


def test_carry_links_is_off_by_default(tmp_path):
    assert _relationships(tmp_path, OcelProcess("p")) == [
        [("P-0", "p"), ("Q-1", "item")], [("P-0", "p")], [("P-1", "p")]]


def test_carry_links_passes_a_cases_links_to_its_later_events_only(tmp_path):
    assert _relationships(tmp_path, OcelProcess("p", carry_links=True)) == [
        [("P-0", "p"), ("Q-1", "item")], [("P-0", "p"), ("Q-1", "item")], [("P-1", "p")]]


def test_the_own_qualifier_defaults_to_the_object_type_and_can_be_set_per_activity(tmp_path):
    assert _relationships(tmp_path, OcelProcess("p", "case", {"B": "finishes"})) == [
        [("P-0", "case"), ("Q-1", "item")], [("P-0", "finishes")], [("P-1", "finishes")]]


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
    process publishes each given message type at its time after 09:00 and takes every Done."""
    engine = ProsimosEngine(ProcessSpec("P", f"{GATEWAYS}/{model}.bpmn", f"{GATEWAYS}/{settings or model}.json", 1),
                            START, None, seed=seed)
    other = ScriptedEngine("S")
    for message_type, after in messages.items():
        other.publish_at(START + after, message_type)
    other.consume("Done", lambda message, now: Verdict.CLAIMED)
    report = run_engines({"P": engine, "S": other}, None, seed)
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
        ("S", 4): [("P-0", "go")], ("P", 1): [("S-0", "sender")], ("P", 2): [("S-0", "sender")]}
