"""
Parallel splits in estimates (docs/messaging.md): an item meeting a parallel split goes on from its join when the
longest branch would get there; an item already inside a split goes on from the join when its latest branch gets
there, a branch whose task has run at that task's end. Nested splits go on from the inner join first.
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
# an item arrives every hour from 09:00 and is received (10 minutes); then it is polished (1 hour) while it is cleaned
# (10 minutes) and tested (30 minutes), and packed (30 minutes) for the shelf
MODEL = """<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL"><bpmn:process id="Warehouse">
<bpmn:startEvent id="Start"><bpmn:outgoing>F1</bpmn:outgoing></bpmn:startEvent>
<bpmn:task id="Receive"><bpmn:incoming>F1</bpmn:incoming><bpmn:outgoing>F2</bpmn:outgoing></bpmn:task>
<bpmn:parallelGateway id="Split"><bpmn:incoming>F2</bpmn:incoming><bpmn:outgoing>F3</bpmn:outgoing>
  <bpmn:outgoing>F4</bpmn:outgoing></bpmn:parallelGateway>
<bpmn:task id="Polish"><bpmn:incoming>F3</bpmn:incoming><bpmn:outgoing>F5</bpmn:outgoing></bpmn:task>
<bpmn:task id="Clean"><bpmn:incoming>F4</bpmn:incoming><bpmn:outgoing>F6</bpmn:outgoing></bpmn:task>
<bpmn:task id="Test"><bpmn:incoming>F6</bpmn:incoming><bpmn:outgoing>F7</bpmn:outgoing></bpmn:task>
<bpmn:parallelGateway id="Join"><bpmn:incoming>F5</bpmn:incoming><bpmn:incoming>F7</bpmn:incoming>
  <bpmn:outgoing>F8</bpmn:outgoing></bpmn:parallelGateway>
<bpmn:task id="Pack"><bpmn:incoming>F8</bpmn:incoming><bpmn:outgoing>F9</bpmn:outgoing></bpmn:task>
<bpmn:intermediateCatchEvent id="Shelf"><bpmn:incoming>F9</bpmn:incoming><bpmn:outgoing>F10</bpmn:outgoing>
  <bpmn:messageEventDefinition/></bpmn:intermediateCatchEvent>
<bpmn:endEvent id="End"><bpmn:incoming>F10</bpmn:incoming></bpmn:endEvent>
<bpmn:sequenceFlow id="F1" sourceRef="Start" targetRef="Receive"/>
<bpmn:sequenceFlow id="F2" sourceRef="Receive" targetRef="Split"/>
<bpmn:sequenceFlow id="F3" sourceRef="Split" targetRef="Polish"/>
<bpmn:sequenceFlow id="F4" sourceRef="Split" targetRef="Clean"/>
<bpmn:sequenceFlow id="F5" sourceRef="Polish" targetRef="Join"/>
<bpmn:sequenceFlow id="F6" sourceRef="Clean" targetRef="Test"/>
<bpmn:sequenceFlow id="F7" sourceRef="Test" targetRef="Join"/>
<bpmn:sequenceFlow id="F8" sourceRef="Join" targetRef="Pack"/>
<bpmn:sequenceFlow id="F9" sourceRef="Pack" targetRef="Shelf"/>
<bpmn:sequenceFlow id="F10" sourceRef="Shelf" targetRef="End"/>
</bpmn:process></bpmn:definitions>"""
# the same, with Receive's two outgoing flows as the split: Prosimos's loader puts a hidden parallel split after it
SPLIT_BY_RECEIVE = MODEL
for old, new in [
        ("<bpmn:outgoing>F2</bpmn:outgoing></bpmn:task>",
         "<bpmn:outgoing>F3</bpmn:outgoing><bpmn:outgoing>F4</bpmn:outgoing></bpmn:task>"),
        ('<bpmn:parallelGateway id="Split"><bpmn:incoming>F2</bpmn:incoming><bpmn:outgoing>F3</bpmn:outgoing>\n'
         '  <bpmn:outgoing>F4</bpmn:outgoing></bpmn:parallelGateway>\n', ""),
        ('<bpmn:sequenceFlow id="F2" sourceRef="Receive" targetRef="Split"/>\n', ""),
        ('sourceRef="Split"', 'sourceRef="Receive"')]:
    SPLIT_BY_RECEIVE = SPLIT_BY_RECEIVE.replace(old, new)
# the same split, then inspected (20 minutes), inside an outer one: meanwhile the item is labeled (10 minutes) and
# printed on (40 minutes)
NESTED = """<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL"><bpmn:process id="Warehouse">
<bpmn:startEvent id="Start"><bpmn:outgoing>F1</bpmn:outgoing></bpmn:startEvent>
<bpmn:task id="Receive"><bpmn:incoming>F1</bpmn:incoming><bpmn:outgoing>F2</bpmn:outgoing></bpmn:task>
<bpmn:parallelGateway id="Outer_Split"><bpmn:incoming>F2</bpmn:incoming><bpmn:outgoing>G1</bpmn:outgoing>
  <bpmn:outgoing>G2</bpmn:outgoing></bpmn:parallelGateway>
<bpmn:parallelGateway id="Split"><bpmn:incoming>G1</bpmn:incoming><bpmn:outgoing>F3</bpmn:outgoing>
  <bpmn:outgoing>F4</bpmn:outgoing></bpmn:parallelGateway>
<bpmn:task id="Polish"><bpmn:incoming>F3</bpmn:incoming><bpmn:outgoing>F5</bpmn:outgoing></bpmn:task>
<bpmn:task id="Clean"><bpmn:incoming>F4</bpmn:incoming><bpmn:outgoing>F6</bpmn:outgoing></bpmn:task>
<bpmn:task id="Test"><bpmn:incoming>F6</bpmn:incoming><bpmn:outgoing>F7</bpmn:outgoing></bpmn:task>
<bpmn:parallelGateway id="Join"><bpmn:incoming>F5</bpmn:incoming><bpmn:incoming>F7</bpmn:incoming>
  <bpmn:outgoing>F8</bpmn:outgoing></bpmn:parallelGateway>
<bpmn:task id="Inspect"><bpmn:incoming>F8</bpmn:incoming><bpmn:outgoing>G3</bpmn:outgoing></bpmn:task>
<bpmn:task id="Label"><bpmn:incoming>G2</bpmn:incoming><bpmn:outgoing>G4</bpmn:outgoing></bpmn:task>
<bpmn:task id="Print"><bpmn:incoming>G4</bpmn:incoming><bpmn:outgoing>G5</bpmn:outgoing></bpmn:task>
<bpmn:parallelGateway id="Outer_Join"><bpmn:incoming>G3</bpmn:incoming><bpmn:incoming>G5</bpmn:incoming>
  <bpmn:outgoing>G6</bpmn:outgoing></bpmn:parallelGateway>
<bpmn:task id="Pack"><bpmn:incoming>G6</bpmn:incoming><bpmn:outgoing>F9</bpmn:outgoing></bpmn:task>
<bpmn:intermediateCatchEvent id="Shelf"><bpmn:incoming>F9</bpmn:incoming><bpmn:outgoing>F10</bpmn:outgoing>
  <bpmn:messageEventDefinition/></bpmn:intermediateCatchEvent>
<bpmn:endEvent id="End"><bpmn:incoming>F10</bpmn:incoming></bpmn:endEvent>
<bpmn:sequenceFlow id="F1" sourceRef="Start" targetRef="Receive"/>
<bpmn:sequenceFlow id="F2" sourceRef="Receive" targetRef="Outer_Split"/>
<bpmn:sequenceFlow id="G1" sourceRef="Outer_Split" targetRef="Split"/>
<bpmn:sequenceFlow id="G2" sourceRef="Outer_Split" targetRef="Label"/>
<bpmn:sequenceFlow id="F3" sourceRef="Split" targetRef="Polish"/>
<bpmn:sequenceFlow id="F4" sourceRef="Split" targetRef="Clean"/>
<bpmn:sequenceFlow id="F5" sourceRef="Polish" targetRef="Join"/>
<bpmn:sequenceFlow id="F6" sourceRef="Clean" targetRef="Test"/>
<bpmn:sequenceFlow id="F7" sourceRef="Test" targetRef="Join"/>
<bpmn:sequenceFlow id="F8" sourceRef="Join" targetRef="Inspect"/>
<bpmn:sequenceFlow id="G3" sourceRef="Inspect" targetRef="Outer_Join"/>
<bpmn:sequenceFlow id="G4" sourceRef="Label" targetRef="Print"/>
<bpmn:sequenceFlow id="G5" sourceRef="Print" targetRef="Outer_Join"/>
<bpmn:sequenceFlow id="G6" sourceRef="Outer_Join" targetRef="Pack"/>
<bpmn:sequenceFlow id="F9" sourceRef="Pack" targetRef="Shelf"/>
<bpmn:sequenceFlow id="F10" sourceRef="Shelf" targetRef="End"/>
</bpmn:process></bpmn:definitions>"""
TASKS = [("Receive", 600), ("Polish", 3600), ("Clean", 600), ("Test", 1800), ("Pack", 1800), ("Inspect", 1200),
         ("Label", 600), ("Print", 2400)]


def _warehouse(tmp_path, model=MODEL):
    """The warehouse's engine, for two items; enough staff that no task waits."""
    (tmp_path / "warehouse.bpmn").write_text(model)
    (tmp_path / "warehouse.json").write_text(json.dumps({
        "resource_profiles": [{"id": "Workers", "name": "Workers", "resource_list": [
            {"id": "Worker", "name": "Worker", "cost_per_hour": 1, "amount": 5, "calendar": "Always",
             "assigned_tasks": [task for task, _ in TASKS]}]}],
        "arrival_time_distribution": {"distribution_name": "fix", "distribution_params": [{"value": 3600}]},
        "arrival_time_calendar": ALWAYS, "gateway_branching_probabilities": [],
        "task_resource_distribution": [{"task_id": task, "resources": [
            {"resource_id": "Worker", "distribution_name": "fix", "distribution_params": [{"value": seconds}]}]}
            for task, seconds in TASKS],
        "resource_calendars": [{"id": "Always", "name": "Always", "time_periods": ALWAYS}],
        "messages": {"consume": [{"event_id": "Shelf", "type": "ItemQuote"}]}}))
    spec = ProcessSpec("Warehouse", str(tmp_path / "warehouse.bpmn"), str(tmp_path / "warehouse.json"), 2, "item")
    return ProsimosEngine(spec, START, seed=1)


def _slots(engine, now):
    """Steps the engine through everything due before now, then its slots for a one-line quote."""
    while engine.next_event_time() is not None and engine.next_event_time() < now:
        engine.step()
    asked = Message("ItemQuote", {"object_id": "order-9"}, "m9", "Sales", now,
                    Quote(QuoteKind.QUOTE, "ItemQuote", "order-9", [Line({"object_id": "order-9"}, "l1")],
                          timedelta(days=3), "QuoteAccepted", "QuoteDeclined"))
    return engine.estimate(asked, now)["l1"]


def test_two_branches_ahead_go_on_from_the_join_when_the_longest_gets_there(tmp_path):
    # received by 09:10, polished by 10:10 (cleaned and tested by 09:50), packed by 10:40
    assert _slots(_warehouse(tmp_path), START) == [START + 100 * MINUTE, START + HOUR + 100 * MINUTE]


def test_a_task_with_two_outgoing_flows_splits_like_a_parallel_gateway(tmp_path):
    engine = _warehouse(tmp_path, SPLIT_BY_RECEIVE)
    assert "and_split_Receive" in engine._env.sim_setup.bpmn_graph.element_info

    assert _slots(engine, START) == [START + 100 * MINUTE, START + HOUR + 100 * MINUTE]


def test_an_item_inside_a_split_goes_on_when_its_latest_branch_gets_there(tmp_path):
    # at 09:15 item-0's Polish runs until 10:10, and its Test is queued at 09:20 (done by 09:50): the join at 10:10,
    # packed by 10:40. item-1 is still ahead
    assert _slots(_warehouse(tmp_path), START + 15 * MINUTE) == [START + 100 * MINUTE, START + HOUR + 100 * MINUTE]


def test_nested_splits_ahead_go_on_from_the_inner_join_then_the_outer_one(tmp_path):
    # received by 09:10; polished by 10:10, inspected by 10:30, while labeled and printed on by 10:00; packed by 11:00
    assert _slots(_warehouse(tmp_path, NESTED), START) == [START + 2 * HOUR, START + 3 * HOUR]


def test_an_item_inside_both_splits_goes_on_from_the_inner_join_first(tmp_path):
    # at 09:15 item-0's Polish runs until 10:10 and its Test is queued at 09:20 (done by 09:50): the inner join at
    # 10:10, inspected by 10:30; Print is queued at 09:20 (done by 10:00): the outer join at 10:30, packed by 11:00
    assert _slots(_warehouse(tmp_path, NESTED), START + 15 * MINUTE) == [START + 2 * HOUR, START + 3 * HOUR]
