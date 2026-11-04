"""
Fan-in (docs/messaging.md, "Fan-in"): each item of an order hands in its part of OrderPicked when it is picked and
moves on; the part that completes the order's pile publishes one OrderPicked, which starts a package.
"""
import json
import math
import re
from datetime import datetime

import pandas as pd
import pytest
import pytz

from prosimos.orchestration.config import ProcessSpec, SimulationConfig
from prosimos.orchestration.simulation import run_multiprocess_simulation
from testing_scripts.multiprocess.orchestrator_settings import one_organization

ASSETS = "testing_scripts/assets/messaging"
# an order every hour from 09:00; after Take order (10 minutes) it publishes one ItemOrdered per item (3)
SALES, SALES_JSON = f"{ASSETS}/orders_with_items.bpmn", f"{ASSETS}/orders_with_items.json"
# an item case per ItemOrdered: Pick item (random) -> throw OrderPicked (fan-in per order_id) -> Put on shelf
WAREHOUSE, WAREHOUSE_JSON = f"{ASSETS}/picked_items.bpmn", f"{ASSETS}/picked_items.json"
# a package per OrderPicked
PACKAGING, PACKAGING_JSON = f"{ASSETS}/order_packages.bpmn", f"{ASSETS}/order_packages.json"
START = pytz.utc.localize(datetime(2024, 1, 1, 9))
ORDERS = 20


def _settings(tmp_path, path, change, name):
    with open(path) as file:
        settings = json.load(file)
    change(settings)
    (tmp_path / name).write_text(json.dumps(settings))
    return str(tmp_path / name)


def _sends_items(settings):
    settings["messages"]["publish"][0]["attributes"].append("items")


def _run(tmp_path, change=lambda settings: None, warehouse=WAREHOUSE):
    """20 orders of 3 items through the Warehouse, its settings changed by change; the report and the log."""
    config = SimulationConfig(one_organization([
        ProcessSpec("Sales", SALES, _settings(tmp_path, SALES_JSON, _sends_items, "sales.json"), ORDERS, "order"),
        ProcessSpec("Warehouse", warehouse, _settings(tmp_path, WAREHOUSE_JSON, change, "warehouse.json"), None,
                    "item"),
        ProcessSpec("Packaging", PACKAGING, PACKAGING_JSON, None, "package")]), START, 1)
    report = run_multiprocess_simulation(config, tmp_path / "log.csv")
    log = pd.read_csv(tmp_path / "log.csv", parse_dates=["enable_time", "start_time", "end_time"])
    return report, log


def _picked(log):
    """Item id -> (when it was picked, its order, its weight), from the log."""
    picks = log[(log["process"] == "Warehouse") & (log["activity"] == "Pick item")]
    return {f"item-{row.case_id}": (row.end_time, row.order_id, row.weight) for row in picks.itertuples()}


def _order_picked(report):
    return [(message.time, message.attributes) for message in report.published if message.type == "OrderPicked"]


@pytest.fixture(scope="module")
def items_run(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("fan_in"))


def test_every_order_sends_one_order_picked_at_its_last_items_time(items_run):
    report, log = items_run
    picked = _picked(log)

    expected = []
    for order in (f"order-{number}" for number in range(ORDERS)):
        items = sorted((time, item) for item, (time, of, _) in picked.items() if of == order)
        weights = [picked[item][2] for _, item in items]
        expected.append((items[-1][0], order, [item for _, item in items], sum(weights)))
    sent = [(time, attributes["order_id"], attributes["item_ids"], attributes["total_weight"])
            for time, attributes in _order_picked(report)]
    assert len({order for _, order, _, _ in expected}) == ORDERS
    assert [(time, order, items) for time, order, items, _ in sorted(sent)] == [
        (time, order, items) for time, order, items, _ in sorted(expected)]
    assert all(math.isclose(total, weight) for (*_, total), (*_, weight) in zip(sorted(sent), sorted(expected)))
    # the items of an order are picked at different times, so the last one isn't always the last created
    assert any(items != sorted(items, key=lambda item: int(item.split("-")[1])) for _, _, items, _ in sent)
    assert report.engine_warnings == [] and report.stalled == []


def test_an_item_doesnt_wait_at_the_throw_event(items_run):
    _, log = items_run
    rows = log[log["process"] == "Warehouse"]
    picked = rows[rows["activity"] == "Pick item"].set_index("case_id")["end_time"]
    shelved = rows[rows["activity"] == "Put on shelf"].set_index("case_id")["enable_time"]

    assert len(picked) == 3 * ORDERS and (shelved.sort_index() == picked.sort_index()).all()


def test_each_package_starts_at_its_order_picked_with_all_its_items(items_run):
    report, log = items_run
    packs = log[log["process"] == "Packaging"]

    assert sorted((row.enable_time, row.order_id, json.loads(row.item_ids)) for row in packs.itertuples()) == sorted(
        (time, attributes["order_id"], attributes["item_ids"]) for time, attributes in _order_picked(report))


def test_without_same_any_three_parts_make_a_message(tmp_path):
    def any_three(settings):
        entry = settings["messages"]["publish"][0]
        entry["fan_in"] = 3
        del entry["same"]
        entry["attributes"] = [{"item_ids": "object_id"}]

    report, log = _run(tmp_path, any_three)

    in_time = sorted((time, item) for item, (time, _, _) in _picked(log).items())
    groups = [in_time[start:start + 3] for start in range(0, len(in_time), 3)]
    assert [(time, attributes["item_ids"]) for time, attributes in _order_picked(report)] == [
        (group[-1][0], [item for _, item in group]) for group in groups]


def test_a_part_after_its_group_was_sent_opens_a_new_group(tmp_path):
    def pairs(settings):
        settings["messages"]["publish"][0]["fan_in"] = 2

    report, log = _run(tmp_path, pairs)

    picked = _picked(log)
    expected, never_sent = [], []
    for order in (f"order-{number}" for number in range(ORDERS)):
        items = sorted((time, item) for item, (time, of, _) in picked.items() if of == order)
        expected.append((items[1][0], order, [item for _, item in items[:2]]))
        never_sent.append(f"fan-in OrderPicked for order_id {order}: 1 of 2 parts, never sent")
    assert sorted((time, attributes["order_id"], attributes["item_ids"])
                  for time, attributes in _order_picked(report)) == sorted(expected)
    # one warning per pile left at the end, in the order the piles were opened
    assert sorted(warning for _, warning in report.engine_warnings) == sorted(never_sent)


def _terminated_warehouse(tmp_path):
    """The Warehouse with a second branch from the start: Check stock (40 minutes) -> terminate end event."""
    with open(WAREHOUSE) as file:
        model = file.read()
    model = model.replace(
        '<bpmn:sequenceFlow id="Flow_1" sourceRef="Start_Item" targetRef="Pick_Item" />',
        '<bpmn:parallelGateway id="Split"><bpmn:incoming>Flow_1</bpmn:incoming><bpmn:outgoing>Flow_1a</bpmn:outgoing>'
        '<bpmn:outgoing>Flow_5</bpmn:outgoing></bpmn:parallelGateway>'
        '<bpmn:task id="Check_Stock" name="Check stock"><bpmn:incoming>Flow_5</bpmn:incoming>'
        '<bpmn:outgoing>Flow_6</bpmn:outgoing></bpmn:task>'
        '<bpmn:endEvent id="Out_Of_Stock"><bpmn:incoming>Flow_6</bpmn:incoming><bpmn:terminateEventDefinition />'
        '</bpmn:endEvent>'
        '<bpmn:sequenceFlow id="Flow_1" sourceRef="Start_Item" targetRef="Split" />'
        '<bpmn:sequenceFlow id="Flow_1a" sourceRef="Split" targetRef="Pick_Item" />'
        '<bpmn:sequenceFlow id="Flow_5" sourceRef="Split" targetRef="Check_Stock" />'
        '<bpmn:sequenceFlow id="Flow_6" sourceRef="Check_Stock" targetRef="Out_Of_Stock" />')
    path = tmp_path / "warehouse.bpmn"
    path.write_text(model)
    return str(path)


def _checks_stock(settings):
    settings["task_resource_distribution"].append(
        {"task_id": "Check_Stock", "resources": [{"resource_id": "Picker", "distribution_name": "fix",
                                                  "distribution_params": [{"value": 2400}]}]})
    settings["resource_profiles"][0]["resource_list"][0]["assigned_tasks"].append("Check_Stock")


def test_an_item_terminated_before_its_part_never_counts(tmp_path):
    report, log = _run(tmp_path, _checks_stock, _terminated_warehouse(tmp_path))

    # an item picked after its terminate at 40 minutes hands in no part, though its pick is logged as it ran
    checked = log[log["activity"] == "Check stock"].set_index("case_id")["end_time"]
    picked = {item: pick for item, pick in _picked(log).items() if pick[0] < checked[int(item.split("-")[1])]}
    expected, never_sent = [], []
    for order in (f"order-{number}" for number in range(ORDERS)):
        items = sorted((time, item) for item, (time, of, _) in picked.items() if of == order)
        if len(items) == 3:
            expected.append((items[-1][0], order, [item for _, item in items]))
        elif items:
            never_sent.append(f"fan-in OrderPicked for order_id {order}: {len(items)} of 3 parts, never sent")
    assert expected and never_sent  # both kinds of order occur
    assert sorted((time, attributes["order_id"], attributes["item_ids"])
                  for time, attributes in _order_picked(report)) == sorted(expected)
    assert sorted(warning for _, warning in report.engine_warnings) == sorted(never_sent)


def test_an_invalid_size_counts_as_1_with_one_warning(tmp_path):
    def half_items(settings):
        del settings["messages"]["consume"][0]["copy"]["items"]
        settings["object_attributes"].append({"name": "items", "type": "continuous", "values": {
            "distribution_name": "fix", "distribution_params": [{"value": 2.5}]}})

    report, log = _run(tmp_path, half_items)

    picked = _picked(log)
    sent = sorted(attributes["item_ids"] for _, attributes in _order_picked(report))
    assert sent == sorted([item] for item in picked)
    first = min(picked, key=lambda item: picked[item][0]).split("-")[1]
    assert report.engine_warnings == [("Warehouse", f"case {first} passes Throw_OrderPicked with no valid items (got "
                                                    f"2.5); its part gives a group size of 1")]


def test_the_first_parts_size_decides_and_another_size_gets_one_warning(tmp_path):
    def index_as_size(settings):
        # item 1 of an order gives a size of 1, item 2 of 2, item 3 of 3
        settings["messages"]["consume"][0]["copy"]["items"] = "index"

    report, log = _run(tmp_path, index_as_size)

    size_of = log[log["process"] == "Warehouse"].groupby("case_id")["items"].first()
    sent = [attributes["item_ids"] for _, attributes in _order_picked(report)]
    assert sent and all(len(items) == size_of[int(items[0].split("-")[1])] for items in sent)
    size_warnings = [warning for _, warning in report.engine_warnings if "from its first part" in warning]
    assert len(size_warnings) == 1
    assert re.fullmatch(r"case \d+ passes Throw_OrderPicked with items \d, but its fan-in group has \d from its first "
                        r"part; the group keeps \d", size_warnings[0])


def test_a_value_that_isnt_a_number_is_left_out_of_a_sum_with_one_warning(tmp_path):
    def weight_as_text(settings):
        settings["object_attributes"][0] = {"name": "weight", "type": "discrete", "values": [
            {"key": "light", "value": 1}]}

    report, _ = _run(tmp_path, weight_as_text)

    assert {attributes["total_weight"] for _, attributes in _order_picked(report)} == {0}
    assert report.engine_warnings == [("Warehouse", "case 2 passes Throw_OrderPicked with weight 'light', which isn't "
                                                    "a number to sum into total_weight; its part is left out of "
                                                    "total_weight")]


# several publishers: every item that handed in a part publishes the message, for its records and OCEL links

def _with_links(tmp_path):
    def picked_for(settings):
        settings["messages"]["publish"][0].update(qualifier="picked for", o2o="part of", carry=["Put on shelf"])

    def contains(settings):
        settings["messages"]["consume"][0].update(qualifier="contains", o2o="holds")

    config = SimulationConfig(one_organization([
        ProcessSpec("Sales", SALES, _settings(tmp_path, SALES_JSON, _sends_items, "sales.json"), ORDERS, "order"),
        ProcessSpec("Warehouse", WAREHOUSE, _settings(tmp_path, WAREHOUSE_JSON, picked_for, "warehouse.json"), None,
                    "item"),
        ProcessSpec("Packaging", PACKAGING, _settings(tmp_path, PACKAGING_JSON, contains, "packaging.json"), None,
                    "package")]), START, 1)
    report = run_multiprocess_simulation(config, None, ocel_out_path=tmp_path / "ocel.json")
    with open(tmp_path / "ocel.json") as file:
        return report, json.load(file)


@pytest.fixture(scope="module")
def linked_run(tmp_path_factory):
    return _with_links(tmp_path_factory.mktemp("fan_in_links"))


def _packages(report):
    """Package id -> the items of the OrderPicked that started it."""
    item_ids = {message.id: message.attributes["item_ids"] for message in report.published
                if message.type == "OrderPicked"}
    return {record.claimers[0].object_id: item_ids[record.message_id] for record in report.message_records
            if record.type == "OrderPicked"}


def test_every_member_is_a_publisher_of_the_message(linked_run):
    report, _ = linked_run
    item_ids = {message.id: message.attributes["item_ids"] for message in report.published
                if message.type == "OrderPicked"}

    records = [record for record in report.message_records if record.type == "OrderPicked"]
    assert len(records) == ORDERS
    for record in records:
        assert [(publisher.object_id, publisher.element_id) for publisher in record.publishers] == [
            (item, "Throw_OrderPicked") for item in item_ids[record.message_id]]
        assert [claimer.process for claimer in record.claimers] == ["Packaging"]


def test_each_items_pick_and_shelving_link_its_package_and_the_pack_links_all_its_items(linked_run):
    report, ocel = linked_run
    package_of = {item: package for package, items in _packages(report).items() for item in items}

    def linked(event, qualifier):
        return [link["objectId"] for link in event["relationships"] if link["qualifier"] == qualifier]

    for event in ocel["events"]:
        own = event["relationships"][0]["objectId"]
        if event["type"] in ("Pick item", "Put on shelf"):  # Put on shelf by carry
            assert linked(event, "picked for") == [package_of[own]]
        if event["type"] == "Pack":
            assert linked(event, "contains") == _packages(report)[own]


def test_o2o_links_each_item_and_its_package_both_ways(linked_run):
    report, ocel = linked_run
    links = {obj["id"]: [(link["objectId"], link["qualifier"]) for link in obj.get("relationships", [])]
             for obj in ocel["objects"]}

    for package, items in _packages(report).items():
        assert links[package] == [(item, "holds") for item in items]
        assert all(links[item] == [(package, "part of")] for item in items)
