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
    """The warehouse's engine: an item that takes a quote's line copies the requester's object_id into order_id. Its
    shelf also takes ordinary ItemOrdered messages for its product; with source, it takes only the quotes of that
    requester."""
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
        "messages": {"consume": [{"event_id": "Shelf", "type": "ItemQuote", "copy": {"order_id": "object_id"},
                                  "condition": [
            [{"attribute": "product", "comparison": "=", "object_attribute": "product"}]
            + ([{"attribute": "source", "comparison": "=", "value": source}] if source else [])]},
            {"event_id": "Shelf", "type": "ItemOrdered", "condition": [
                [{"attribute": "product", "comparison": "=", "object_attribute": "product"}]]}]}}))
    spec = ProcessSpec("Warehouse", str(tmp_path / "warehouse.bpmn"), str(tmp_path / "warehouse.json"), items, "item")
    return ProsimosEngine(spec, START, seed=1)


def until(engine, time):
    """Steps the engine through everything due before time."""
    while engine.next_event_time() is not None and engine.next_event_time() < time:
        engine.step()


def quote(message_id, *products, requester="Sales", request="order-7", kind=QuoteKind.QUOTE):
    """A stamped quote (or modify) message of request from requester, its lines named after the message: m1 asks
    m1.1, m1.2, ..."""
    lines = [Line({"object_id": request, "product": product}, f"{message_id}.{number}")
             for number, product in enumerate(products, 1)]
    return Message("ItemQuote" if kind is QuoteKind.QUOTE else "QuoteModified", {"object_id": request}, message_id,
                   requester, START, Quote(kind, "ItemQuote", request, lines, timedelta(days=3), "QuoteAccepted",
                                           "QuoteDeclined"))


def held(engine):
    """Item -> (the quote message, the line) it took."""
    env = engine._env
    return {env.object_id(p_case): (line["quote"], line["line"]) for (p_case, _), (line, _) in env._set_aside.items()}


def test_an_item_waiting_on_the_shelf_takes_a_line_at_once(tmp_path):
    engine = warehouse(tmp_path)
    until(engine, START + 30 * MINUTE)  # item-0 has been on the shelf since 09:10
    # nobody has a MacBook, so the quote isn't complete and item-0 keeps waiting with its line
    engine.reserve(quote("m1", "iPhone", "MacBook"), ["m1.1", "m1.2"], START + 30 * MINUTE)

    assert held(engine) == {"item-0": ("m1", "m1.1")}
    assert [line["line"] for line in engine._env._open_lines] == ["m1.2"]


def test_arriving_items_take_the_open_lines_of_the_older_quote_first(tmp_path):
    engine = warehouse(tmp_path)
    until(engine, START + 5 * MINUTE)  # nothing on the shelf yet
    engine.reserve(quote("m1", "iPhone", "MacBook"), ["m1.1", "m1.2"], START + 5 * MINUTE)
    engine.reserve(quote("m2", "iPhone", "MacBook", request="order-8"), ["m2.1", "m2.2"], START + 6 * MINUTE)

    until(engine, START + 10 * MINUTE + MINUTE)
    assert held(engine) == {"item-0": ("m1", "m1.1")}
    until(engine, START + HOUR + 11 * MINUTE)
    assert held(engine) == {"item-0": ("m1", "m1.1"), "item-1": ("m2", "m2.1")}


def test_an_item_takes_one_line_only(tmp_path):
    engine = warehouse(tmp_path)
    until(engine, START + 30 * MINUTE)

    engine.reserve(quote("m1", "iPhone", "iPhone", "MacBook"), ["m1.1", "m1.2", "m1.3"], START + 30 * MINUTE)

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

    engine.reserve(quote("m1", "iPhone", "iPhone", "MacBook"), ["m1.2", "m1.3"], START + 30 * MINUTE)

    assert held(engine) == {"item-0": ("m1", "m1.2")}
    assert [line["line"] for line in engine._env._open_lines] == ["m1.3"]


def test_a_line_is_matched_with_its_requester_as_source(tmp_path):
    engine = warehouse(tmp_path, source="Sales")
    until(engine, START + 30 * MINUTE)

    engine.reserve(quote("m1", "iPhone", requester="Outlet"), ["m1.1"], START + 30 * MINUTE)
    engine.reserve(quote("m2", "iPhone", "MacBook"), ["m2.1", "m2.2"], START + 30 * MINUTE)

    assert held(engine) == {"item-0": ("m2", "m2.1")}
    assert [line["line"] for line in engine._env._open_lines] == ["m1.1", "m2.2"]


# reserved items stay away from ordinary claims

def ordered(engine, now):
    """Offers an ordinary order for an iPhone at now; the verdict, and the item that claimed it, if any."""
    verdict = engine.deliver(Message("ItemOrdered", {"product": "iPhone"}, "m9", "Shop", now), now)
    return verdict.value, [engine._env.object_id(p_case) for p_case, _, _ in engine._env.claimed_by]


def test_an_item_holding_a_line_is_never_claimed_by_an_ordinary_message(tmp_path):
    engine = warehouse(tmp_path)
    until(engine, START + 30 * MINUTE)
    engine.reserve(quote("m1", "iPhone", "MacBook"), ["m1.1", "m1.2"], START + 30 * MINUTE)

    # item-0 holds the line; the next items can still take the order, so it waits
    assert ordered(engine, START + 30 * MINUTE) == ("pending", [])


def test_an_arriving_item_takes_an_open_line_before_a_pending_ordinary_message(tmp_path):
    engine = warehouse(tmp_path)
    until(engine, START + 5 * MINUTE)
    engine.reserve(quote("m1", "iPhone", "MacBook"), ["m1.1", "m1.2"], START + 5 * MINUTE)
    assert ordered(engine, START + 5 * MINUTE) == ("pending", [])

    # item-0 reaches the shelf at 09:10 and takes the line within its step, before the order is offered again
    until(engine, START + 11 * MINUTE)
    assert held(engine) == {"item-0": ("m1", "m1.1")}
    assert ordered(engine, START + 11 * MINUTE) == ("pending", [])
    # item-1 finds no open line, so the order takes it
    until(engine, START + HOUR + 11 * MINUTE)
    assert ordered(engine, START + HOUR + 11 * MINUTE) == ("claimed", ["item-1"])


# a member's share moves on together

def shipped(engine):
    """Item -> when its Ship task was enabled."""
    env = engine._env
    return {env.object_id(p_case): enabled for p_case, element_id, enabled, _ in env.logged_rows
            if element_id == "Ship"}


def test_the_items_of_a_quote_wait_for_its_last_line_and_leave_together(tmp_path):
    engine = warehouse(tmp_path)
    until(engine, START + 5 * MINUTE)
    asked = quote("m1", "iPhone", "iPhone", "iPhone")
    engine.reserve(asked, ["m1.1", "m1.2", "m1.3"], START + 5 * MINUTE)

    # the items reach the shelf at 09:10, 10:10, and 11:10; all three leave at 11:10
    until(engine, START + timedelta(days=1))
    assert shipped(engine) == {f"item-{number}": START + 2 * HOUR + 10 * MINUTE for number in range(3)}
    values = engine._env.sim_setup.bpmn_graph.all_attributes
    assert [values[p_case]["order_id"] for p_case in range(3)] == ["order-7"] * 3


def test_moved_is_reported_from_the_moment_the_share_moves_on(tmp_path):
    engine = warehouse(tmp_path)
    until(engine, START + 5 * MINUTE)
    asked = quote("m1", "iPhone", "iPhone")
    engine.reserve(asked, ["m1.1", "m1.2"], START + 5 * MINUTE)

    until(engine, START + HOUR)  # item-0 holds a line, item-1 isn't on the shelf yet
    assert not engine.moved(asked, START + HOUR)
    until(engine, START + HOUR + 11 * MINUTE)  # item-1 took the last line at 10:10
    assert engine.moved(asked, START + HOUR + 11 * MINUTE)
    assert not engine.moved(quote("m2", "iPhone", request="order-8"), START + HOUR + 11 * MINUTE)


def test_a_quote_whose_lines_are_all_taken_at_once_moves_on_at_once(tmp_path):
    engine = warehouse(tmp_path)
    until(engine, START + 30 * MINUTE)
    asked = quote("m1", "iPhone")

    engine.reserve(asked, ["m1.1"], START + 30 * MINUTE)

    assert engine.moved(asked, START + 30 * MINUTE) and held(engine) == {}
    until(engine, START + timedelta(days=1))
    assert shipped(engine)["item-0"] == START + 30 * MINUTE


def test_only_the_quotes_own_items_move_on(tmp_path):
    # m1's item-0 waits for its MacBook; m2's line is taken by item-1, which leaves alone
    engine = warehouse(tmp_path)
    until(engine, START + 5 * MINUTE)
    engine.reserve(quote("m1", "iPhone", "MacBook"), ["m1.1", "m1.2"], START + 5 * MINUTE)
    engine.reserve(quote("m2", "iPhone", request="order-8"), ["m2.1"], START + 5 * MINUTE)

    until(engine, START + timedelta(days=1))
    assert held(engine) == {"item-0": ("m1", "m1.1")}
    # item-2 finds no line it matches and waits on the shelf
    assert shipped(engine) == {"item-1": START + HOUR + 10 * MINUTE}


def test_after_a_modify_the_shares_lines_of_both_messages_leave_together_at_the_last_one(tmp_path):
    # m1 asks two iPhones; at 09:30 its modify m3 adds a third at the same warehouse
    engine = warehouse(tmp_path)
    until(engine, START + 5 * MINUTE)
    asked = quote("m1", "iPhone", "iPhone")
    engine.reserve(asked, ["m1.1", "m1.2"], START + 5 * MINUTE)
    until(engine, START + 30 * MINUTE)
    modified = quote("m3", "iPhone", "iPhone", "iPhone", kind=QuoteKind.MODIFY)
    engine.reserve(modified, ["m3.3"], START + 30 * MINUTE)

    # at 10:10 m1's own lines are both taken, but the share still waits for m3's line
    until(engine, START + HOUR + 11 * MINUTE)
    assert held(engine) == {"item-0": ("m1", "m1.1"), "item-1": ("m1", "m1.2")}
    assert not engine.moved(asked, START + HOUR + 11 * MINUTE)
    until(engine, START + timedelta(days=1))
    assert shipped(engine) == {f"item-{number}": START + 2 * HOUR + 10 * MINUTE for number in range(3)}
    assert engine.moved(asked, START + timedelta(days=1)) and engine.moved(modified, START + timedelta(days=1))


def test_a_later_quote_of_the_same_order_isnt_seen_as_moved(tmp_path):
    engine = warehouse(tmp_path)
    until(engine, START + 30 * MINUTE)
    engine.reserve(quote("m1", "iPhone"), ["m1.1"], START + 30 * MINUTE)  # item-0 leaves at once

    until(engine, START + HOUR + 30 * MINUTE)  # item-1 has been on the shelf since 10:10
    again = quote("m5", "iPhone", "MacBook")
    engine.reserve(again, ["m5.1", "m5.2"], START + HOUR + 30 * MINUTE)

    assert held(engine) == {"item-1": ("m5", "m5.1")}
    assert not engine.moved(again, START + HOUR + 30 * MINUTE)


# release

def test_a_released_item_serves_the_next_quote(tmp_path):
    # item-0 holds order-7's iPhone; order-8's lines wait, as no other item is on the shelf
    engine = warehouse(tmp_path)
    until(engine, START + 30 * MINUTE)
    first = quote("m1", "iPhone", "MacBook")
    engine.reserve(first, ["m1.1", "m1.2"], START + 30 * MINUTE)
    engine.reserve(quote("m2", "iPhone", "MacBook", request="order-8"), ["m2.1", "m2.2"], START + 30 * MINUTE)

    engine.release(first, ["m1.1", "m1.2"], START + 40 * MINUTE)

    assert held(engine) == {"item-0": ("m2", "m2.1")}
    assert [line["line"] for line in engine._env._open_lines] == ["m2.2"]


def test_releasing_some_lines_keeps_the_others(tmp_path):
    engine = warehouse(tmp_path)
    until(engine, START + 5 * MINUTE)
    asked = quote("m1", "iPhone", "iPhone", "MacBook")
    engine.reserve(asked, ["m1.1", "m1.2", "m1.3"], START + 5 * MINUTE)
    until(engine, START + HOUR + 30 * MINUTE)
    assert held(engine) == {"item-0": ("m1", "m1.1"), "item-1": ("m1", "m1.2")}

    engine.release(asked, ["m1.2"], START + HOUR + 30 * MINUTE)

    assert held(engine) == {"item-0": ("m1", "m1.1")}
    assert [line["line"] for line in engine._env._open_lines] == ["m1.3"]


def test_a_released_item_is_ordinary_stock_again(tmp_path):
    engine = warehouse(tmp_path)
    until(engine, START + 30 * MINUTE)
    asked = quote("m1", "iPhone", "MacBook")
    engine.reserve(asked, ["m1.1", "m1.2"], START + 30 * MINUTE)
    assert ordered(engine, START + 30 * MINUTE) == ("pending", [])

    engine.release(asked, ["m1.1", "m1.2"], START + 40 * MINUTE)

    assert ordered(engine, START + 40 * MINUTE) == ("claimed", ["item-0"])


def test_releasing_a_shares_last_open_line_lets_its_items_move_on(tmp_path):
    # the modification dropped the MacBook nobody had: the iPhone's item leaves at once
    engine = warehouse(tmp_path)
    until(engine, START + 30 * MINUTE)
    asked = quote("m1", "iPhone", "MacBook")
    engine.reserve(asked, ["m1.1", "m1.2"], START + 30 * MINUTE)

    engine.release(asked, ["m1.2"], START + 40 * MINUTE)

    assert held(engine) == {} and engine._env._open_lines == []
    assert engine.moved(asked, START + 40 * MINUTE)
    until(engine, START + timedelta(days=1))
    assert shipped(engine)["item-0"] == START + 40 * MINUTE
