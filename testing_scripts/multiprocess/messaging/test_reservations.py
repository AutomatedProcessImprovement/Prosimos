"""
Reservations (docs/messaging.md): reserve() opens a quote's lines at a Prosimos engine, in reservation order. A line
is matched by the consume entry for the quote type, like a message of that type with the requester as source: an
item already waiting takes a line at once, and an arriving item takes the oldest open line it matches.
"""
import json
from datetime import datetime, timedelta

import pytz

from prosimos.orchestration.config import ProcessSpec
from prosimos.orchestration.prosimos_engine import ProsimosEngine
from prosimos.orchestration.protocol import Line, Message, Quote, QuoteKind

START = pytz.utc.localize(datetime(2024, 1, 1, 9))
MINUTE, HOUR = timedelta(minutes=1), timedelta(hours=1)
# the warehouse makes an item every hour from 09:00 (10 minutes), which waits on the shelf until a line takes it
FLOW = [("startEvent", "Start", False), ("task", "Make_Item", False), ("intermediateCatchEvent", "Shelf", True),
        ("task", "Ship", False), ("endEvent", "End", False)]


def warehouse(tmp_path, product="iPhone", items=3, source=None):
    """The warehouse's engine; with source, its shelf takes only the quotes of that requester."""
    elements = []
    for index, (kind, element_id, message) in enumerate(FLOW):
        arrows = (f"<bpmn:incoming>Flow_{index}</bpmn:incoming>" if index else "") + \
                 (f"<bpmn:outgoing>Flow_{index + 1}</bpmn:outgoing>" if index < len(FLOW) - 1 else "")
        definition = "<bpmn:messageEventDefinition/>" if message else ""
        elements.append(f'<bpmn:{kind} id="{element_id}">{arrows}{definition}</bpmn:{kind}>')
    flows = "".join(f'<bpmn:sequenceFlow id="Flow_{index + 1}" sourceRef="{FLOW[index][1]}" '
                    f'targetRef="{FLOW[index + 1][1]}"/>' for index in range(len(FLOW) - 1))
    (tmp_path / "warehouse.bpmn").write_text(
        '<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">'
        f'<bpmn:process id="Warehouse">{"".join(elements)}{flows}</bpmn:process></bpmn:definitions>')
    always = [{"from": "MONDAY", "to": "SUNDAY", "beginTime": "00:00:00", "endTime": "23:59:59"}]
    task = {"resource_id": "Worker", "distribution_name": "fix", "distribution_params": [{"value": 600}]}
    (tmp_path / "warehouse.json").write_text(json.dumps({
        "resource_profiles": [{"id": "Workers", "name": "Workers", "resource_list": [
            {"id": "Worker", "name": "Worker", "cost_per_hour": 1, "amount": 5, "calendar": "Always",
             "assigned_tasks": ["Make_Item", "Ship"]}]}],
        "arrival_time_distribution": {"distribution_name": "fix", "distribution_params": [{"value": 3600}]},
        "arrival_time_calendar": always, "gateway_branching_probabilities": [],
        "task_resource_distribution": [{"task_id": task_id, "resources": [task]} for task_id in ("Make_Item", "Ship")],
        "resource_calendars": [{"id": "Always", "name": "Always", "time_periods": always}],
        "object_attributes": [{"name": "product", "type": "discrete", "values": [{"key": product, "value": 1}]}],
        "messages": {"consume": [{"event_id": "Shelf", "type": "ItemQuote", "condition": [
            [{"attribute": "product", "comparison": "=", "object_attribute": "product"}]
            + ([{"attribute": "source", "comparison": "=", "value": source}] if source else [])]}]}}))
    spec = ProcessSpec("Warehouse", str(tmp_path / "warehouse.bpmn"), str(tmp_path / "warehouse.json"), items, "item")
    return ProsimosEngine(spec, START, seed=1)


def until(engine, time):
    """Steps the engine through everything due before time."""
    while engine.next_event_time() is not None and engine.next_event_time() < time:
        engine.step()


def quote(message_id, *products, requester="Sales"):
    """A stamped quote message from requester, its lines named after the message: m1 asks m1.1, m1.2, ..."""
    lines = [Line({"object_id": "order-7", "product": product}, f"{message_id}.{number}")
             for number, product in enumerate(products, 1)]
    return Message("ItemQuote", {"object_id": "order-7"}, message_id, requester, START,
                   Quote(QuoteKind.QUOTE, "ItemQuote", "order-7", lines, timedelta(days=3), "QuoteAccepted",
                         "QuoteDeclined"))


def held(engine):
    """Item -> (the quote message, the line) it took."""
    env = engine._env
    return {env.object_id(p_case): (line["quote"], line["line"]) for (p_case, _), line in env._set_aside.items()}


def test_an_item_waiting_on_the_shelf_takes_a_line_at_once(tmp_path):
    engine = warehouse(tmp_path)
    until(engine, START + 30 * MINUTE)  # item-0 has been on the shelf since 09:10
    asked = quote("m1", "iPhone")

    engine.reserve(asked, ["m1.1"], START + 30 * MINUTE)

    assert held(engine) == {"item-0": ("m1", "m1.1")}
    assert engine._env._open_lines == []


def test_arriving_items_take_the_open_lines_of_the_older_quote_first(tmp_path):
    engine = warehouse(tmp_path)
    until(engine, START + 5 * MINUTE)  # nothing on the shelf yet
    engine.reserve(quote("m1", "iPhone"), ["m1.1"], START + 5 * MINUTE)
    engine.reserve(quote("m2", "iPhone"), ["m2.1"], START + 6 * MINUTE)

    until(engine, START + 10 * MINUTE + MINUTE)
    assert held(engine) == {"item-0": ("m1", "m1.1")}
    until(engine, START + HOUR + 11 * MINUTE)
    assert held(engine) == {"item-0": ("m1", "m1.1"), "item-1": ("m2", "m2.1")}


def test_an_item_takes_one_line_only(tmp_path):
    engine = warehouse(tmp_path)
    until(engine, START + 30 * MINUTE)

    engine.reserve(quote("m1", "iPhone", "iPhone"), ["m1.1", "m1.2"], START + 30 * MINUTE)

    assert held(engine) == {"item-0": ("m1", "m1.1")}
    until(engine, START + HOUR + 11 * MINUTE)
    assert held(engine) == {"item-0": ("m1", "m1.1"), "item-1": ("m1", "m1.2")}


def test_a_line_no_item_matches_stays_open(tmp_path):
    engine = warehouse(tmp_path)
    until(engine, START + 30 * MINUTE)

    engine.reserve(quote("m1", "iPad", "iPhone"), ["m1.1", "m1.2"], START + 30 * MINUTE)

    assert held(engine) == {"item-0": ("m1", "m1.2")}
    assert [line["line"] for line in engine._env._open_lines] == ["m1.1"]


def test_only_the_lines_reserved_here_are_opened(tmp_path):
    # the other lines of the quote went to another member
    engine = warehouse(tmp_path)
    until(engine, START + 30 * MINUTE)

    engine.reserve(quote("m1", "iPhone", "iPhone", "iPhone"), ["m1.2"], START + 30 * MINUTE)

    assert held(engine) == {"item-0": ("m1", "m1.2")}
    assert engine._env._open_lines == []


def test_a_line_is_matched_with_its_requester_as_source(tmp_path):
    engine = warehouse(tmp_path, source="Sales")
    until(engine, START + 30 * MINUTE)

    engine.reserve(quote("m1", "iPhone", requester="Outlet"), ["m1.1"], START + 30 * MINUTE)
    engine.reserve(quote("m2", "iPhone"), ["m2.1"], START + 30 * MINUTE)

    assert held(engine) == {"item-0": ("m2", "m2.1")}
    assert [line["line"] for line in engine._env._open_lines] == ["m1.1"]
