"""
Message catch events in estimates (docs/messaging.md): a catch event on the way, other than the one the quote is
for, adds the average time cases have waited there for a message so far, in clock time; 0 until measured. A case
waiting there goes on after that average since it got there, or now if it has waited longer.
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
# an item arrives every hour from 09:00 and is made (10 minutes); it waits for quality's approval, and is packed
# (30 minutes) for the shelf
FLOW = [("startEvent", "Start", False), ("task", "Make", False), ("intermediateCatchEvent", "Approve", True),
        ("task", "Pack", False), ("intermediateCatchEvent", "Shelf", True), ("endEvent", "End", False)]
LINE = ('<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL"><bpmn:process id="Warehouse">'
        + "".join(f'<bpmn:{kind} id="{element}">' + (f'<bpmn:incoming>F{index}</bpmn:incoming>' if index else "")
                  + (f'<bpmn:outgoing>F{index + 1}</bpmn:outgoing>' if index < len(FLOW) - 1 else "")
                  + ("<bpmn:messageEventDefinition/>" if message else "") + f'</bpmn:{kind}>'
                  for index, (kind, element, message) in enumerate(FLOW))
        + "".join(f'<bpmn:sequenceFlow id="F{index + 1}" sourceRef="{FLOW[index][1]}" '
                  f'targetRef="{FLOW[index + 1][1]}"/>' for index in range(len(FLOW) - 1))
        + "</bpmn:process></bpmn:definitions>")
ASSETS = "testing_scripts/assets/quotes"


def _warehouse(tmp_path, split=False):
    """The warehouse's engine, for four items; enough staff that no task waits. split polishes the item (10 minutes)
    while it waits for approval (approval_with_polish)."""
    if split:
        spec = ProcessSpec("Warehouse", f"{ASSETS}/approval_with_polish.bpmn", f"{ASSETS}/approval_with_polish.json",
                           4, "item")
        return ProsimosEngine(spec, START, seed=1)
    (tmp_path / "warehouse.bpmn").write_text(LINE)
    tasks = [("Make", 600), ("Pack", 1800)]
    (tmp_path / "warehouse.json").write_text(json.dumps({
        "resource_profiles": [{"id": "Workers", "name": "Workers", "resource_list": [
            {"id": "Worker", "name": "Worker", "cost_per_hour": 1, "amount": 5, "calendar": "Always",
             "assigned_tasks": [task for task, _ in tasks]}]}],
        "arrival_time_distribution": {"distribution_name": "fix", "distribution_params": [{"value": 3600}]},
        "arrival_time_calendar": ALWAYS, "gateway_branching_probabilities": [],
        "task_resource_distribution": [{"task_id": task, "resources": [
            {"resource_id": "Worker", "distribution_name": "fix", "distribution_params": [{"value": seconds}]}]}
            for task, seconds in tasks],
        "resource_calendars": [{"id": "Always", "name": "Always", "time_periods": ALWAYS}],
        "messages": {"consume": [{"event_id": "Approve", "type": "Approved"},
                                 {"event_id": "Shelf", "type": "ItemQuote"}]}}))
    spec = ProcessSpec("Warehouse", str(tmp_path / "warehouse.bpmn"), str(tmp_path / "warehouse.json"), 4, "item")
    return ProsimosEngine(spec, START, seed=1)


def _until(engine, now):
    """Steps the engine through everything due before now."""
    while engine.next_event_time() is not None and engine.next_event_time() < now:
        engine.step()


def _approve(engine, now):
    """Quality approves the item waiting longest at now."""
    _until(engine, now)
    engine.deliver(Message("Approved", {}, f"m{now.hour}", "Quality", now), now)


def _slots(engine, now):
    _until(engine, now)
    asked = Message("ItemQuote", {"object_id": "order-9"}, "m9", "Sales", now,
                    Quote(QuoteKind.QUOTE, "ItemQuote", "order-9", [Line({"object_id": "order-9"}, "l1")],
                          timedelta(days=3), "QuoteAccepted", "QuoteDeclined"))
    return engine.estimate(asked, now)["l1"]


def test_with_no_wait_measured_yet_a_catch_event_on_the_way_adds_nothing(tmp_path):
    # each item: made by its arrival + 10 minutes, then packed by + 40
    assert _slots(_warehouse(tmp_path), START) == [START + index * HOUR + 40 * MINUTE for index in range(4)]


def test_a_catch_event_on_the_way_adds_the_average_wait_so_far(tmp_path):
    # item-0 waits for approval from 09:10 to 09:40, item-1 from 10:10 to 11:10
    engine = _warehouse(tmp_path)
    _approve(engine, START + 40 * MINUTE)
    _approve(engine, START + 2 * HOUR + 10 * MINUTE)

    # at 11:10 item-1 is approved, its resume queued: it goes on now. Only item-0's 30 minutes are measured yet
    assert _slots(engine, START + 2 * HOUR + 10 * MINUTE) == [
        START + HOUR + 10 * MINUTE,  # item-0, on the shelf since 10:10
        START + 2 * HOUR + 40 * MINUTE,  # item-1, packed by 11:40
        START + 3 * HOUR + 10 * MINUTE,  # item-2, made by 11:10: approved by 11:40, packed by 12:10
        START + 4 * HOUR + 10 * MINUTE]  # item-3, made by 12:10: approved by 12:40, packed by 13:10

    # at 11:15 both waits are measured: 45 minutes on average
    assert _slots(engine, START + 2 * HOUR + 15 * MINUTE) == [
        START + HOUR + 10 * MINUTE,  # item-0, on the shelf since 10:10
        START + 2 * HOUR + 40 * MINUTE,  # item-1, packed by 11:40
        START + 3 * HOUR + 25 * MINUTE,  # item-2, waiting since 11:10: approved by 11:55, packed by 12:25
        START + 4 * HOUR + 25 * MINUTE]  # item-3, made by 12:10: approved by 12:55, packed by 13:25

    # at 13:00 item-2 and item-3 have waited longer than the average: they go on now
    assert _slots(engine, START + 4 * HOUR)[2:] == [START + 4 * HOUR + 30 * MINUTE] * 2


def test_a_branch_waiting_at_a_catch_event_is_forecast_from_there(tmp_path):
    # at 09:15 item-0 waits for approval since 09:10, nothing measured yet: it goes on now, and its Polish ends at
    # 09:20: the join at 09:20, packed by 09:50
    assert _slots(_warehouse(tmp_path, split=True), START + 15 * MINUTE)[0] == START + 50 * MINUTE
