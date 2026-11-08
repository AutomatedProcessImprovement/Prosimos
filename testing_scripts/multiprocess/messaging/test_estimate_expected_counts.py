"""
Expected counts over exclusive gateways (docs/messaging.md): each item not yet at the catch event gets its chance to
reach it and its average time to get there, branches weighted by their probability; slot k is when the expected
count there reaches k. A split the item's data decides is exact.
"""
import json
from datetime import datetime, timedelta

import pytz

from prosimos.orchestration.config import ProcessSpec
from prosimos.orchestration.prosimos_engine import ProsimosEngine
from prosimos.orchestration.protocol import Line, Message, Quote, QuoteKind

START = pytz.utc.localize(datetime(2024, 1, 1, 9))
MINUTE, HOUR = timedelta(minutes=1), timedelta(hours=1)
ALWAYS = [{"from": "MONDAY", "to": "SUNDAY", "beginTime": "00:00:00", "endTime": "23:59:59"}]
# the warehouse makes an item every hour from 09:00 (10 minutes) and checks it: 80% go to the shelf, 20% are fixed
# (1 hour) and then scrapped, shelved, or checked again
MODEL = """<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL"><bpmn:process id="Warehouse">
<bpmn:startEvent id="Start"><bpmn:outgoing>F1</bpmn:outgoing></bpmn:startEvent>
<bpmn:task id="Make_Item"><bpmn:incoming>F1</bpmn:incoming><bpmn:outgoing>F2</bpmn:outgoing></bpmn:task>
<bpmn:exclusiveGateway id="Check"><bpmn:incoming>F2</bpmn:incoming>CHECK_IN<bpmn:outgoing>Good</bpmn:outgoing>
  <bpmn:outgoing>Bad</bpmn:outgoing></bpmn:exclusiveGateway>
<bpmn:task id="Fix"><bpmn:incoming>Bad</bpmn:incoming><bpmn:outgoing>Fixed</bpmn:outgoing></bpmn:task>
<bpmn:exclusiveGateway id="Merge"><bpmn:incoming>Good</bpmn:incoming>MERGE_IN<bpmn:outgoing>F5</bpmn:outgoing>
  </bpmn:exclusiveGateway>
<bpmn:intermediateCatchEvent id="Shelf"><bpmn:incoming>F5</bpmn:incoming><bpmn:outgoing>F6</bpmn:outgoing>
  <bpmn:messageEventDefinition/></bpmn:intermediateCatchEvent>
<bpmn:endEvent id="End"><bpmn:incoming>F6</bpmn:incoming></bpmn:endEvent>
<bpmn:sequenceFlow id="F1" sourceRef="Start" targetRef="Make_Item"/>
<bpmn:sequenceFlow id="F2" sourceRef="Make_Item" targetRef="Check"/>
<bpmn:sequenceFlow id="Good" sourceRef="Check" targetRef="Merge"/>
<bpmn:sequenceFlow id="Bad" sourceRef="Check" targetRef="Fix"/>
FIXED
<bpmn:sequenceFlow id="F5" sourceRef="Merge" targetRef="Shelf"/>
<bpmn:sequenceFlow id="F6" sourceRef="Shelf" targetRef="End"/>
</bpmn:process></bpmn:definitions>"""


def _warehouse(tmp_path, items=5, fixed="Scrapped", graded=False):
    """The warehouse's engine; Fix sends the item on to fixed: Scrapped, Merge (the shelf), or Check. graded draws
    each item's grade, A or B, and shelves the A ones only."""
    fixed_in = "<bpmn:incoming>Fixed</bpmn:incoming>"
    model = MODEL.replace("FIXED", f'<bpmn:sequenceFlow id="Fixed" sourceRef="Fix" targetRef="{fixed}"/>' + (
        f'<bpmn:endEvent id="Scrapped">{fixed_in}</bpmn:endEvent>' if fixed == "Scrapped" else ""))
    model = model.replace("MERGE_IN", fixed_in if fixed == "Merge" else "").replace(
        "CHECK_IN", fixed_in if fixed == "Check" else "")
    (tmp_path / "warehouse.bpmn").write_text(model)
    probabilities = [{"path_id": "Good", "value": 0.8}, {"path_id": "Bad", "value": 0.2}]
    if graded:
        for probability, grade in zip(probabilities, "AB"):
            probability["condition_id"] = grade
    tasks = [("Make_Item", 600), ("Fix", 3600)]
    settings = {
        "resource_profiles": [{"id": "Workers", "name": "Workers", "resource_list": [
            {"id": "Worker", "name": "Worker", "cost_per_hour": 1, "amount": 5, "calendar": "Always",
             "assigned_tasks": [task for task, _ in tasks]}]}],
        "arrival_time_distribution": {"distribution_name": "fix", "distribution_params": [{"value": 3600}]},
        "arrival_time_calendar": ALWAYS,
        "gateway_branching_probabilities": [{"gateway_id": "Check", "probabilities": probabilities}],
        "task_resource_distribution": [{"task_id": task, "resources": [
            {"resource_id": "Worker", "distribution_name": "fix", "distribution_params": [{"value": seconds}]}]}
            for task, seconds in tasks],
        "resource_calendars": [{"id": "Always", "name": "Always", "time_periods": ALWAYS}],
        "messages": {"consume": [{"event_id": "Shelf", "type": "ItemQuote"}]}}
    if graded:
        settings["object_attributes"] = [{"name": "grade", "type": "discrete", "values": [
            {"key": "A", "value": 0.5}, {"key": "B", "value": 0.5}]}]
        settings["branch_rules"] = [
            {"id": grade, "rules": [[{"attribute": "grade", "comparison": "=", "value": grade}]]} for grade in "AB"]
    (tmp_path / "warehouse.json").write_text(json.dumps(settings))
    spec = ProcessSpec("Warehouse", str(tmp_path / "warehouse.bpmn"), str(tmp_path / "warehouse.json"), items, "item")
    return ProsimosEngine(spec, START, seed=1)


def _quote(message_id, request, lines=1):
    """A stamped quote of request from Sales with lines identical lines: m1 asks m1.1, m1.2, ..."""
    return Message("ItemQuote", {"object_id": request}, message_id, "Sales", START, Quote(
        QuoteKind.QUOTE, "ItemQuote", request, [Line({"object_id": request}, f"{message_id}.{number}")
                                                for number in range(1, lines + 1)],
        timedelta(days=3), "QuoteAccepted", "QuoteDeclined"))


def _slots(engine):
    return engine.estimate(_quote("m9", "order-9"), START)["m9.1"]


def made(index):
    """When item index is made: Make_Item from its arrival, for 10 minutes."""
    return START + index * HOUR + 10 * MINUTE


def test_a_20_percent_scrap_gives_the_expected_slots(tmp_path):
    # five items, each 80% sure: the expected count reaches 1 with item-1 (1.6), 2 with item-2 (2.4), and so on
    assert _slots(_warehouse(tmp_path)) == [made(1), made(2), made(3), made(4)]


def test_open_lines_take_the_earliest_expected_items_first(tmp_path):
    # five open lines take an expected count of 5: items 0 to 5 (4.8) and 0.2 of item-6. What is left: 0.6 with
    # item-6, 1.4 with item-7, 2.2 with item-8, 3.0 with item-9
    engine = _warehouse(tmp_path, items=10)
    engine.reserve(_quote("m1", "order-7", lines=5), [f"m1.{number}" for number in range(1, 6)], START)

    assert _slots(engine) == [made(7), made(8), made(9)]


def test_a_branch_on_the_way_is_weighted_by_its_probability(tmp_path):
    # every item reaches the shelf; 20% after an hour of fixing: 12 minutes later on average
    assert _slots(_warehouse(tmp_path, fixed="Merge")) == [made(index) + 12 * MINUTE for index in range(5)]


def test_a_loop_back_is_followed_until_its_chance_is_negligible(tmp_path):
    # a fixed item is checked again: on average 0.25 fixes, 15 minutes; the ways cut off shift it by microseconds
    [slot] = _slots(_warehouse(tmp_path, items=1, fixed="Check"))

    assert abs(slot - (made(0) + 15 * MINUTE)) < timedelta(milliseconds=1)


def test_a_split_the_items_data_decides_is_exact(tmp_path):
    engine = _warehouse(tmp_path, graded=True)
    grades = [engine._env.sim_setup.bpmn_graph.all_attributes[p_case]["grade"] for p_case in range(5)]
    assert set(grades) == {"A", "B"}

    assert _slots(engine) == [made(index) for index, grade in enumerate(grades) if grade == "A"]
