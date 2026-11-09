"""
Batched tasks in estimates (docs/messaging.md): a batched item starts when its batch does, so the task's measured
wait includes waiting for the batch to fill. An item waiting for its batch goes on after the task's average wait
since it got there, or now if it has waited longer, and then the task's average duration.
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
# an item arrives every hour from 09:00 and is made (10 minutes); it is packed (10 minutes) in batches of 3, one
# item after the other, for the shelf
FLOW = [("startEvent", "Start", False), ("task", "Make", False), ("task", "Pack", False),
        ("intermediateCatchEvent", "Shelf", True), ("endEvent", "End", False)]
TASKS = [("Make", 600), ("Pack", 600)]


def _warehouse(tmp_path):
    """The warehouse's engine, for six items; one worker for each task."""
    elements = "".join(
        f'<bpmn:{kind} id="{element}">' + (f'<bpmn:incoming>F{index}</bpmn:incoming>' if index else "")
        + (f'<bpmn:outgoing>F{index + 1}</bpmn:outgoing>' if index < len(FLOW) - 1 else "")
        + ("<bpmn:messageEventDefinition/>" if message else "") + f'</bpmn:{kind}>'
        for index, (kind, element, message) in enumerate(FLOW))
    flows = "".join(f'<bpmn:sequenceFlow id="F{index + 1}" sourceRef="{FLOW[index][1]}" '
                    f'targetRef="{FLOW[index + 1][1]}"/>' for index in range(len(FLOW) - 1))
    (tmp_path / "warehouse.bpmn").write_text(
        '<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">'
        f'<bpmn:process id="Warehouse">{elements}{flows}</bpmn:process></bpmn:definitions>')
    (tmp_path / "warehouse.json").write_text(json.dumps({
        "resource_profiles": [{"id": task, "name": task, "resource_list": [
            {"id": f"{task}_Worker", "name": f"{task}_Worker", "cost_per_hour": 1, "amount": 1,
             "calendar": "Always", "assigned_tasks": [task]}]} for task, _ in TASKS],
        "arrival_time_distribution": {"distribution_name": "fix", "distribution_params": [{"value": 3600}]},
        "arrival_time_calendar": ALWAYS, "gateway_branching_probabilities": [],
        "task_resource_distribution": [{"task_id": task, "resources": [
            {"resource_id": f"{task}_Worker", "distribution_name": "fix", "distribution_params": [{"value": seconds}]}]}
            for task, seconds in TASKS],
        "resource_calendars": [{"id": "Always", "name": "Always", "time_periods": ALWAYS}],
        "batch_processing": [{"task_id": "Pack", "type": "Sequential", "duration_distrib": [{"key": "3", "value": 1}],
                              "size_distrib": [{"key": "1", "value": 0}, {"key": "3", "value": 1}],
                              "firing_rules": [[{"attribute": "size", "comparison": "=", "value": 3}]]}],
        "messages": {"consume": [{"event_id": "Shelf", "type": "ItemQuote"}]}}))
    spec = ProcessSpec("Warehouse", str(tmp_path / "warehouse.bpmn"), str(tmp_path / "warehouse.json"), 6, "item")
    return ProsimosEngine(spec, START, seed=1)


def _slots(engine, now):
    """Steps the engine through everything due before now, then its slots for a one-line quote."""
    while engine.next_event_time() is not None and engine.next_event_time() < now:
        engine.step()
    asked = Message("ItemQuote", {"object_id": "order-9"}, "m9", "Sales", now,
                    Quote(QuoteKind.QUOTE, "ItemQuote", "order-9", [Line({"object_id": "order-9"}, "l1")],
                          timedelta(days=3), "QuoteAccepted", "QuoteDeclined"))
    return engine.estimate(asked, now)["l1"]


def test_a_batched_item_starts_when_its_batch_does(tmp_path):
    # items 0, 1, and 2 reach Pack at 09:10, 10:10, and 11:10; the batch starts at 11:10, one item after the other
    engine = _warehouse(tmp_path)
    _slots(engine, START + 2 * HOUR + 15 * MINUTE)

    assert [(event.enabled_datetime, event.started_datetime) for trace in engine._env.log_info.trace_list
            for event in trace.event_list if event.task_id == "Pack"] == [
        (START + 10 * MINUTE, START + 2 * HOUR + 10 * MINUTE),
        (START + HOUR + 10 * MINUTE, START + 2 * HOUR + 20 * MINUTE),
        (START + 2 * HOUR + 10 * MINUTE, START + 2 * HOUR + 30 * MINUTE)]
    # so far only item-0 has started: it waited 2 hours
    assert engine._env._average_wait("Pack", START + 2 * HOUR + 15 * MINUTE) == 2 * 3600


def test_an_item_waiting_for_its_batch_goes_on_after_the_average_wait(tmp_path):
    # at 12:05 the three have waited 2 hours, 70 minutes, and 20 minutes: 70 on average. item-3 waits in a new batch
    # since 12:10: packed by 13:30; item-4 and item-5 are still ahead
    engine = _warehouse(tmp_path)
    shelved = [START + 2 * HOUR + minutes * MINUTE for minutes in (20, 30, 40)]

    assert _slots(engine, START + 3 * HOUR + 5 * MINUTE) == shelved + [
        START + 4 * HOUR + 30 * MINUTE, START + 5 * HOUR + 30 * MINUTE, START + 6 * HOUR + 30 * MINUTE]
    assert list(engine._env.sim_setup.bpmn_graph.batch_waiting_processes["Pack"]) == [3]

    # at 13:45 item-3 has waited 95 minutes, longer than the average: packed by 13:55. item-4 waits since 13:10
    assert _slots(engine, START + 4 * HOUR + 45 * MINUTE) == shelved + [
        START + 4 * HOUR + 55 * MINUTE, START + 5 * HOUR + 30 * MINUTE, START + 6 * HOUR + 30 * MINUTE]
