"""
Timers and timed events in estimates (docs/messaging.md): an intermediate event on the way that doesn't wait for a
message adds the average of its drawn duration, a timer's delay exactly, in clock time: it runs at night too.
"""
import json
from datetime import datetime, timedelta

import pytest
import pytz

from prosimos.orchestration.config import ProcessSpec
from prosimos.orchestration.prosimos_engine import ProsimosEngine
from prosimos.orchestration.protocol import Line, Message, Quote, QuoteKind

MONDAY = pytz.utc.localize(datetime(2024, 1, 1, 9))
MINUTE, HOUR = timedelta(minutes=1), timedelta(hours=1)
ALWAYS = [{"from": "MONDAY", "to": "SUNDAY", "beginTime": "00:00:00", "endTime": "23:59:59"}]
WEEKDAYS = [{"from": "MONDAY", "to": "FRIDAY", "beginTime": "09:00:00", "endTime": "17:00:00"}]
# the warehouse paints an item (1 hour), lets it dry, and packs it (30 minutes); staff work Monday to Friday 9 to 17
FLOW = [("startEvent", "Start", ""), ("task", "Paint", ""), ("intermediateCatchEvent", "Dry", "DRYING"),
        ("task", "Pack", ""), ("intermediateCatchEvent", "Shelf", "<bpmn:messageEventDefinition/>"),
        ("endEvent", "End", "")]
FIXED_2_HOURS = {"distribution_name": "fix", "distribution_params": [{"value": 7200}]}
UNIFORM_1_TO_3_HOURS = {"distribution_name": "uniform", "distribution_params": [{"value": 3600}, {"value": 10800}]}


def _warehouse(tmp_path, definition, drying):
    """The warehouse's engine; an item arrives at 09:00 and one at 15:30. Dry has definition and takes drying."""
    elements = "".join(
        f'<bpmn:{kind} id="{element}">' + (f'<bpmn:incoming>F{index}</bpmn:incoming>' if index else "")
        + (f'<bpmn:outgoing>F{index + 1}</bpmn:outgoing>' if index < len(FLOW) - 1 else "")
        + element_definition.replace("DRYING", definition) + f'</bpmn:{kind}>'
        for index, (kind, element, element_definition) in enumerate(FLOW))
    flows = "".join(f'<bpmn:sequenceFlow id="F{index + 1}" sourceRef="{FLOW[index][1]}" '
                    f'targetRef="{FLOW[index + 1][1]}"/>' for index in range(len(FLOW) - 1))
    (tmp_path / "warehouse.bpmn").write_text(
        '<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">'
        f'<bpmn:process id="Warehouse">{elements}{flows}</bpmn:process></bpmn:definitions>')
    tasks = [("Paint", 3600), ("Pack", 1800)]
    (tmp_path / "warehouse.json").write_text(json.dumps({
        "resource_profiles": [{"id": "Workers", "name": "Workers", "resource_list": [
            {"id": "Worker", "name": "Worker", "cost_per_hour": 1, "amount": 5, "calendar": "Weekdays",
             "assigned_tasks": [task for task, _ in tasks]}]}],
        "arrival_time_distribution": {"distribution_name": "fix", "distribution_params": [{"value": 6.5 * 3600}]},
        "arrival_time_calendar": ALWAYS, "gateway_branching_probabilities": [],
        "task_resource_distribution": [{"task_id": task, "resources": [
            {"resource_id": "Worker", "distribution_name": "fix", "distribution_params": [{"value": seconds}]}]}
            for task, seconds in tasks],
        "resource_calendars": [{"id": "Weekdays", "name": "Weekdays", "time_periods": WEEKDAYS}],
        "event_distribution": [{"event_id": "Dry", **drying}],
        "messages": {"consume": [{"event_id": "Shelf", "type": "ItemQuote"}]}}))
    spec = ProcessSpec("Warehouse", str(tmp_path / "warehouse.bpmn"), str(tmp_path / "warehouse.json"), 2, "item")
    return ProsimosEngine(spec, MONDAY, seed=1)


def _slots(engine, now):
    """Steps the engine through everything due before now, then its slots for a one-line quote."""
    while engine.next_event_time() is not None and engine.next_event_time() < now:
        engine.step()
    asked = Message("ItemQuote", {"object_id": "order-9"}, "m9", "Sales", now,
                    Quote(QuoteKind.QUOTE, "ItemQuote", "order-9", [Line({"object_id": "order-9"}, "l1")],
                          timedelta(days=3), "QuoteAccepted", "QuoteDeclined"))
    return engine.estimate(asked, now)["l1"]


def _queued(engine):
    """(case id, element, time) of each event queued at the engine."""
    return sorted((event.p_case, event.task_id, event.enabled_datetime)
                  for event in engine._env.events_queue.enabled_events.entry_finder)


@pytest.mark.parametrize("definition, drying, dry_by", [
    ("<bpmn:timerEventDefinition/>", FIXED_2_HOURS, MONDAY + 3 * HOUR),
    ("", UNIFORM_1_TO_3_HOURS, MONDAY + timedelta(hours=3, minutes=32, seconds=41.748208))],  # drawn with seed 1
    ids=["a timer", "a drawn duration"])
def test_an_event_between_two_tasks_adds_its_average_duration_in_clock_time(tmp_path, definition, drying, dry_by):
    # item-0: painted by 10:00, dry by 12:00 on average, packed by 12:30. item-1: painted by 16:30, dry by 18:30
    # overnight, packed on Tuesday from 9:00
    slots = [MONDAY + 3.5 * HOUR, MONDAY + timedelta(days=1, minutes=30)]
    engine = _warehouse(tmp_path, definition, drying)
    assert _slots(engine, MONDAY) == slots

    # at 09:30 item-0's Paint runs: item-0 is queued at Dry at 10:00, its duration not drawn yet
    assert _slots(engine, MONDAY + 30 * MINUTE) == slots
    assert _queued(engine) == [(0, "Dry", MONDAY + HOUR), (1, "Paint", MONDAY + 6.5 * HOUR)]

    # at 10:50 item-0 dries: Prosimos drew its drying time at 10:00 and queued Pack at its end
    assert _slots(engine, MONDAY + 110 * MINUTE) == [dry_by + 30 * MINUTE, slots[1]]
    assert _queued(engine) == [(0, "Pack", dry_by), (1, "Paint", MONDAY + 6.5 * HOUR)]
