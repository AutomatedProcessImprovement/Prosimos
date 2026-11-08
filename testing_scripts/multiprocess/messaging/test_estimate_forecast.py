"""
The forecast for the tasks ahead (docs/messaging.md): an item queued at a task on a way to the catch event with only
tasks and throw events gets, per task in turn, the task's average wait for staff so far plus its average duration,
both in working time on the earliest calendar of its staff.
"""
import json
from datetime import datetime, timedelta

import pytest
import pytz
from pix_framework.discovery.resource_calendar_and_performance.crisp.resource_calendar import RCalendar

from prosimos.messaging.engine.estimate import forecast_end
from prosimos.orchestration.config import ProcessSpec
from prosimos.orchestration.prosimos_engine import ProsimosEngine
from prosimos.orchestration.protocol import Line, Message, Quote, QuoteKind

MINUTE, HOUR = timedelta(minutes=1), timedelta(hours=1)
MONDAY = pytz.utc.localize(datetime(2024, 1, 1, 9))
FRIDAY = MONDAY + timedelta(days=4)
WEEKDAYS = [{"from": "MONDAY", "to": "FRIDAY", "beginTime": "09:00:00", "endTime": "17:00:00"}]
ALWAYS = [{"from": "MONDAY", "to": "SUNDAY", "beginTime": "00:00:00", "endTime": "23:59:59"}]
# the warehouse polishes an item (1 hour, one Polisher) and packs it (30 minutes, one Packer), then shelves it
FLOW = ["Start", "Polish", "Pack", "Shelf", "End"]
KINDS = {"Start": "startEvent", "Polish": "task", "Pack": "task", "Shelf": "intermediateCatchEvent", "End": "endEvent"}


def _calendar(periods):
    calendar = RCalendar("staff")
    for period in periods:
        calendar.add_calendar_item(period["from"], period["to"], period["beginTime"], period["endTime"])
    return calendar


@pytest.mark.parametrize("reaches, ready", [
    (MONDAY + timedelta(days=1, hours=5), MONDAY + timedelta(days=1, hours=6, minutes=20)),  # Tuesday 14:00
    (FRIDAY + 7 * HOUR + 50 * MINUTE, MONDAY + timedelta(days=7, hours=1, minutes=10)),  # Friday 16:50
    (FRIDAY + 7 * HOUR + 30 * MINUTE, MONDAY + timedelta(days=7, minutes=50))])  # Friday 16:30
def test_a_wait_and_a_task_are_counted_in_working_time(reaches, ready):
    # Polish: a 20-minute wait and a 1-hour task, Monday to Friday 9 to 17
    assert forecast_end(reaches, [_calendar(WEEKDAYS)], (20 * MINUTE + HOUR).total_seconds()) == ready


def test_the_earliest_calendar_of_the_staff_counts():
    # on Friday at 16:30, a weekend worker ends the 80 minutes on Saturday; the weekday one on Monday
    weekends = _calendar([{"from": "SATURDAY", "to": "SUNDAY", "beginTime": "09:00:00", "endTime": "17:00:00"}])

    assert forecast_end(FRIDAY + 7 * HOUR + 30 * MINUTE, [_calendar(WEEKDAYS), weekends], 80 * 60) == \
        FRIDAY + timedelta(days=1, hours=1, minutes=20)


def _warehouse(tmp_path, start, items, calendar):
    """The warehouse's engine from start: an item arrives every 30 minutes; its staff work on calendar."""
    elements = "".join(
        f'<bpmn:{KINDS[element]} id="{element}">'
        + (f'<bpmn:incoming>F{index}</bpmn:incoming>' if index else "")
        + (f'<bpmn:outgoing>F{index + 1}</bpmn:outgoing>' if index < len(FLOW) - 1 else "")
        + ("<bpmn:messageEventDefinition/>" if element == "Shelf" else "") + f'</bpmn:{KINDS[element]}>'
        for index, element in enumerate(FLOW))
    flows = "".join(f'<bpmn:sequenceFlow id="F{index + 1}" sourceRef="{FLOW[index]}" targetRef="{FLOW[index + 1]}"/>'
                    for index in range(len(FLOW) - 1))
    (tmp_path / "warehouse.bpmn").write_text(
        '<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">'
        f'<bpmn:process id="Warehouse">{elements}{flows}</bpmn:process></bpmn:definitions>')
    (tmp_path / "warehouse.json").write_text(json.dumps({
        "resource_profiles": [{"id": task, "name": task, "resource_list": [
            {"id": f"{task}er", "name": f"{task}er", "cost_per_hour": 1, "amount": 1, "calendar": "Staff",
             "assigned_tasks": [task]}]} for task in ("Polish", "Pack")],
        "arrival_time_distribution": {"distribution_name": "fix", "distribution_params": [{"value": 1800}]},
        "arrival_time_calendar": ALWAYS, "gateway_branching_probabilities": [],
        "task_resource_distribution": [
            {"task_id": task, "resources": [{"resource_id": f"{task}er", "distribution_name": "fix",
                                             "distribution_params": [{"value": seconds}]}]}
            for task, seconds in (("Polish", 3600), ("Pack", 1800))],
        "resource_calendars": [{"id": "Staff", "name": "Staff", "time_periods": calendar}],
        "messages": {"consume": [{"event_id": "Shelf", "type": "ItemQuote"}]}}))
    spec = ProcessSpec("Warehouse", str(tmp_path / "warehouse.bpmn"), str(tmp_path / "warehouse.json"), items, "item")
    return ProsimosEngine(spec, start, seed=1)


def _slots(engine, now):
    """Steps the engine through everything due before now, then its slots for a one-line quote."""
    while engine.next_event_time() is not None and engine.next_event_time() < now:
        engine.step()
    asked = Message("ItemQuote", {"object_id": "order-9"}, "m9", "Sales", now,
                    Quote(QuoteKind.QUOTE, "ItemQuote", "order-9", [Line({"object_id": "order-9"}, "l1")],
                          timedelta(days=3), "QuoteAccepted", "QuoteDeclined"))
    return engine.estimate(asked, now)["l1"]


def test_two_tasks_ahead_with_no_wait_measured_yet(tmp_path):
    engine = _warehouse(tmp_path, MONDAY, 4, ALWAYS)

    # each item: Polish for an hour, then Pack for 30 minutes, from its arrival
    assert _slots(engine, MONDAY) == [MONDAY + 90 * MINUTE + index * 30 * MINUTE for index in range(4)]


def test_the_wait_so_far_is_added(tmp_path):
    # at 10:05, Polish has started for item-0 at once and for item-1 after 30 minutes: 15 on average; item-2 waits
    # until 11:00, which hasn't come yet. Pack started for item-0 at once
    engine = _warehouse(tmp_path, MONDAY, 4, ALWAYS)

    assert _slots(engine, MONDAY + HOUR + 5 * MINUTE) == [
        MONDAY + 90 * MINUTE,  # item-0, queued at the shelf
        MONDAY + 150 * MINUTE,  # item-1, queued at Pack at 11:00
        MONDAY + 3 * HOUR + 15 * MINUTE,  # item-3, queued at Polish at 10:30: a 15-minute wait, Polish and Pack
        MONDAY + 210 * MINUTE]  # item-2, queued at Pack at 12:00


def test_a_wait_over_the_weekend_counts_only_its_working_time(tmp_path):
    # item-0 is polished on Friday from 16:00 to 17:00, and packed on Monday from 9:00: a wait of no working time
    engine = _warehouse(tmp_path, FRIDAY + 7 * HOUR, 2, WEEKDAYS)

    monday = MONDAY + timedelta(days=7)
    # item-1 is polished on Monday from 9:00 to 10:00, then packed with no wait
    assert _slots(engine, monday + MINUTE) == [monday + 30 * MINUTE, monday + HOUR + 30 * MINUTE]
