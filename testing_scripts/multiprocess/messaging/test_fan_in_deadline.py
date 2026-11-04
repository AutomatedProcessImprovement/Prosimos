"""
A fan-in deadline (docs/messaging.md, "Fan-in"): the clock starts at a group's first part; if the group isn't
complete by then, its message goes out at the deadline with the parts so far.
"""
import json
from datetime import timedelta

import pytest

from prosimos.orchestration.config import ProcessSpec, SimulationConfig
from prosimos.orchestration.orchestrator import orchestrate
from prosimos.orchestration.prosimos_engine import ProsimosEngine
from prosimos.orchestration.simulation import run_multiprocess_simulation
from testing_scripts.multiprocess.messaging.test_fan_in import (ORDERS, START, _order_picked, _picked, _run,
                                                              _specs)
from testing_scripts.multiprocess.orchestrator_settings import one_organization
from testing_scripts.multiprocess.scripted_engine import ScriptedEngine

HOUR = timedelta(hours=1)
MICROSECOND = timedelta(microseconds=1)


def _within_an_hour(settings):
    settings["messages"]["publish"][0]["deadline"] = {"hours": 1}


@pytest.fixture(scope="module")
def deadline_run(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("fan_in_deadline"), _within_an_hour)


def _expected(picked):
    """(time, order, items) of each message: a group opens at its first part and is sent at its third part or an
    hour after its first (1 µs late, after every part of that instant), whichever comes first; a later part opens
    the next group."""
    expected = []
    for order in (f"order-{number}" for number in range(ORDERS)):
        parts = sorted((time, item) for item, (time, of, _) in picked.items() if of == order)
        while parts:
            due = parts[0][0] + HOUR
            group = [part for part in parts[:3] if part[0] <= due]
            sent = group[-1][0] if len(group) == 3 else due + MICROSECOND
            expected.append((sent, order, [item for _, item in group]))
            parts = parts[len(group):]
    return sorted(expected)


def test_a_group_is_sent_when_complete_or_at_its_deadline_with_the_parts_so_far(deadline_run):
    report, log = deadline_run

    expected = _expected(_picked(log))
    sent = sorted((time, attributes["order_id"], attributes["item_ids"]) for time, attributes in _order_picked(report))
    assert sent == expected
    assert any(len(items) == 3 for _, _, items in expected) and any(len(items) < 3 for _, _, items in expected)
    # a part after its group was sent at its deadline opens a new group, with its own deadline
    assert len(expected) > ORDERS
    assert report.engine_warnings == []


def test_a_message_sent_at_its_deadline_has_every_part_handed_in_before_it_as_publishers(deadline_run):
    report, _ = deadline_run
    item_ids = {message.id: message.attributes["item_ids"] for message in report.published
                if message.type == "OrderPicked"}

    for record in (record for record in report.message_records if record.type == "OrderPicked"):
        assert [publisher.object_id for publisher in record.publishers] == item_ids[record.message_id]


def test_a_part_handed_in_exactly_at_the_deadline_is_in_its_message(tmp_path):
    def one_picker(settings):
        # one picker, 30 minutes an item: the items of the order at 09:10 are picked at 09:40, 10:10 and 10:40
        pickers = settings["resource_profiles"][0]["resource_list"]
        pickers[0].update(amount=1, assigned_tasks=["Pick_Item"])
        pickers.append({**pickers[0], "id": "Shelver", "name": "Shelver", "assigned_tasks": ["Put_On_Shelf"]})
        settings["task_resource_distribution"][0]["resources"][0].update(
            distribution_name="fix", distribution_params=[{"value": 1800}])
        settings["task_resource_distribution"][1]["resources"][0]["resource_id"] = "Shelver"
        settings["messages"]["publish"][0]["deadline"] = {"hours": 0.5}

    report, log = _run(tmp_path, one_picker, orders=1)

    in_time = [item for _, item in sorted((time, item) for item, (time, _, _) in _picked(log).items())]
    assert sorted(time for time, _, _ in _picked(log).values()) == [at("09:40"), at("10:10"), at("10:40")]
    # the deadline of the group opened at 09:40 is 10:10, when the second item is handed in
    assert [(time, attributes["item_ids"]) for time, attributes in _order_picked(report)] == [
        (at("10:10") + MICROSECOND, in_time[:2]), (at("11:10") + MICROSECOND, in_time[2:])]


def at(hh_mm):
    hours, minutes = map(int, hh_mm.split(":"))
    return START.replace(hour=hours, minute=minutes)


RELEASED_ITEMS = """<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">
  <bpmn:process id="Warehouse">
    <bpmn:startEvent id="Start_Item"><bpmn:outgoing>Flow_1</bpmn:outgoing>
      <bpmn:messageEventDefinition /></bpmn:startEvent>
    <bpmn:intermediateCatchEvent id="Catch_Released"><bpmn:incoming>Flow_1</bpmn:incoming>
      <bpmn:outgoing>Flow_2</bpmn:outgoing>
      <bpmn:messageEventDefinition /></bpmn:intermediateCatchEvent>
    <bpmn:intermediateThrowEvent id="Throw_OrderPicked"><bpmn:incoming>Flow_2</bpmn:incoming>
      <bpmn:outgoing>Flow_3</bpmn:outgoing>
      <bpmn:messageEventDefinition /></bpmn:intermediateThrowEvent>
    <bpmn:endEvent id="End"><bpmn:incoming>Flow_3</bpmn:incoming></bpmn:endEvent>
    <bpmn:sequenceFlow id="Flow_1" sourceRef="Start_Item" targetRef="Catch_Released" />
    <bpmn:sequenceFlow id="Flow_2" sourceRef="Catch_Released" targetRef="Throw_OrderPicked" />
    <bpmn:sequenceFlow id="Flow_3" sourceRef="Throw_OrderPicked" targetRef="End" />
  </bpmn:process>
</bpmn:definitions>"""


def test_a_part_whose_member_another_process_resumes_exactly_at_the_deadline_is_in_its_message(tmp_path):
    # each item waits for Dispatch to release it, then hands in its part at once
    (tmp_path / "released_items.bpmn").write_text(RELEASED_ITEMS)
    (tmp_path / "released_items.json").write_text(json.dumps({
        "resource_profiles": [], "gateway_branching_probabilities": [], "task_resource_distribution": [],
        "resource_calendars": [], "messages": {
            "consume": [{"event_id": "Start_Item", "type": "ItemOrdered", "copy": {"order_id": "object_id"}},
                        {"event_id": "Catch_Released", "type": "Released",
                         "condition": [[{"attribute": "item", "comparison": "=", "object_attribute": "object_id"}]]}],
            "publish": [{"event_id": "Throw_OrderPicked", "type": "OrderPicked", "fan_in": 3, "same": ["order_id"],
                         "deadline": {"hours": 0.5}, "attributes": ["order_id", {"item_ids": "object_id"}]}]}}))
    dispatch = ScriptedEngine("Dispatch")
    for hh_mm, item in (("09:40", "item-0"), ("10:10", "item-1"), ("10:40", "item-2")):
        dispatch.publish_at(at(hh_mm), "Released", item=item)
    sales, _, _ = _specs(tmp_path, orders=1)
    config = SimulationConfig(one_organization([
        sales, ProcessSpec("Warehouse", str(tmp_path / "released_items.bpmn"), str(tmp_path / "released_items.json"),
                           None, "item")]), START, 1, extra_processes=["Dispatch"])

    report = run_multiprocess_simulation(config, None, {"Dispatch": dispatch})

    # the group opened at 09:40 is due at 10:10, when Dispatch releases item-1
    assert [(time, attributes["item_ids"]) for time, attributes in _order_picked(report)] == [
        (at("10:10") + MICROSECOND, ["item-0", "item-1"]), (at("11:10") + MICROSECOND, ["item-2"])]


def test_a_deadline_no_longer_due_doesnt_keep_the_warehouse_going(tmp_path):
    def within_30_days(settings):
        settings["messages"]["publish"][0]["deadline"] = {"days": 30}

    engines = _engines(tmp_path, within_30_days)
    announced = []
    next_event_time = engines["Warehouse"].next_event_time
    engines["Warehouse"].next_event_time = lambda: _noted(announced, next_event_time())

    report = orchestrate(engines, None, 1)

    # each group completes long before its deadline, which is taken back then
    assert len(_order_picked(report)) == ORDERS
    assert max(time for time in announced if time is not None) < START + timedelta(days=2)


def _noted(announced, time):
    announced.append(time)
    return time


def _engines(tmp_path, change):
    return {spec.name: ProsimosEngine(spec, START, None, seed=1) for spec in _specs(tmp_path, change)}
