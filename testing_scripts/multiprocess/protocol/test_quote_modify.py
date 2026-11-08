"""
Modifying a quote (docs/orchestrator.md): lines in both lists, matched by their attributes, stay where they are; the
added ones are quoted, and within the original limit date the dropped ones are released, the added ones reserved,
and the reply is accepted; otherwise nothing changes. Too late once some line has moved on; after a decline, a fresh
quote of the new list; without a quote, declined with a warning.
"""
from datetime import datetime, timedelta

import pytest
import pytz

from prosimos.orchestration.orchestrator import orchestrate
from prosimos.orchestration.protocol import Line, Message, Quote, QuoteKind, Verdict
from testing_scripts.multiprocess.scripted_engine import ScriptedEngine

START = pytz.utc.localize(datetime(2024, 1, 1, 9))
LATER = START + timedelta(hours=1)
HOUR, DAY = timedelta(hours=1), timedelta(days=1)
# what each warehouse has, and when it would be ready
SHELVES = {"Tartu": {"iPhone": [START + HOUR], "MacBook": [START + 5 * HOUR]},
           "Tallinn": {"iPad": [START + 2 * HOUR, START + 3 * HOUR]}}


def _ask(kind, message_type, products, within):
    return lambda: Message(message_type, {"object_id": "order-7"}, quote=Quote(
        kind, "ItemQuote", "order-7", [Line({"product": product}) for product in products], within,
        "QuoteAccepted", "QuoteDeclined"))


def quote(*products, within=timedelta(days=3)):
    return _ask(QuoteKind.QUOTE, "ItemQuote", products, within)


def modify(*products, within=timedelta(days=3)):
    return _ask(QuoteKind.MODIFY, "QuoteModified", products, within)


def _run(script, shelves=SHELVES, moved=None):
    """Sales publishes what script gives, (time, message maker), and takes the replies; each warehouse answers a
    line from shelves, and Tartu with moved whether a share has moved on. The engines, the replies, and the
    warnings."""
    sales = ScriptedEngine("Sales")
    for reply in ("QuoteAccepted", "QuoteDeclined"):
        sales.consume(reply, lambda message, now: Verdict.CLAIMED)
    for time, make in script:
        sales.at(time, lambda now, make=make: [make()])
    engines = {"Sales": sales}
    for name, shelf in shelves.items():
        engines[name] = ScriptedEngine(name)
        engines[name].slots("ItemQuote", lambda line, now, shelf=shelf: shelf.get(line.attributes["product"], []))
    if moved is not None:
        engines["Tartu"].moved_when(moved)
    report = orchestrate(engines, {"Sales": ["Sales"], "Warehouses": list(shelves)}, 1)
    replies = [(message.time, message.type, {key: value for key, value in message.attributes.items()
                                             if key != "request"})
               for message in report.published if message.source == "orchestrator"]
    return engines, replies, report.warnings


def _ids(calls):
    return [(message.id, *rest) for message, *rest in calls]


def test_an_added_line_is_quoted_and_reserved_and_the_kept_lines_stay_where_they_are():
    # m1 asks l1 (iPhone) and l2 (iPad); m3 asks l3, l4, and l5 (MacBook) at 10:00
    engines, replies, _ = _run([(START, quote("iPhone", "iPad")), (LATER, modify("iPhone", "iPad", "MacBook"))])

    assert replies[1] == (LATER, "QuoteAccepted", {"ready_by": START + 5 * HOUR, "deliveries": 2})
    assert _ids(engines["Tartu"].reserved) == [("m1", ["l1"], START), ("m3", ["l5"], LATER)]
    assert _ids(engines["Tallinn"].reserved) == [("m1", ["l2"], START)]
    assert engines["Tartu"].released == [] and engines["Tallinn"].released == []


def test_a_dropped_line_is_released_with_the_message_that_reserved_it():
    engines, replies, _ = _run([(START, quote("iPhone", "iPad")), (LATER, modify("iPhone"))])

    assert replies[1] == (LATER, "QuoteAccepted", {"ready_by": START + HOUR, "deliveries": 1})
    assert _ids(engines["Tallinn"].released) == [("m1", ["l2"], LATER)]
    # nothing added, so nobody was asked for slots again
    assert [message.id for message, _ in engines["Tartu"].estimated] == ["m1"]


def test_of_identical_lines_the_first_ones_are_kept():
    engines, replies, _ = _run([(START, quote("iPad", "iPad")), (LATER, modify("iPad"))])

    assert replies[1] == (LATER, "QuoteAccepted", {"ready_by": START + 2 * HOUR, "deliveries": 1})
    assert _ids(engines["Tallinn"].released) == [("m1", ["l2"], LATER)]


def test_declined_leaves_everything_as_it_was():
    # the MacBook would only be ready after the limit, so the iPad isn't dropped either
    shelves = {**SHELVES, "Tartu": {"iPhone": [START + HOUR], "MacBook": [START + 5 * DAY]}}

    engines, replies, _ = _run([(START, quote("iPhone", "iPad")), (LATER, modify("iPhone", "MacBook"))], shelves)

    assert replies[1] == (LATER, "QuoteDeclined", {"earliest": START + 5 * DAY})
    assert _ids(engines["Tartu"].reserved) == [("m1", ["l1"], START)]
    assert engines["Tartu"].released == [] and engines["Tallinn"].released == []


def test_too_late_once_some_line_has_moved_on():
    engines, replies, _ = _run([(START, quote("iPhone", "iPad")), (LATER, modify("iPhone"))],
                               moved=lambda message, now: True)

    assert replies[1] == (LATER, "QuoteDeclined", {"earliest": None})
    assert engines["Tallinn"].released == []


def test_after_a_decline_a_fresh_quote_of_the_new_list_against_the_original_limit_date():
    # the MacBook misses the limit; then the iPad alone is quoted, ready 2 hours after the start
    shelves = {**SHELVES, "Tartu": {"MacBook": [START + 5 * DAY]}}

    engines, replies, _ = _run([(START, quote("MacBook")), (LATER, modify("iPad"))], shelves)

    assert replies == [(START, "QuoteDeclined", {"earliest": START + 5 * DAY}),
                       (LATER, "QuoteAccepted", {"ready_by": START + 2 * HOUR, "deliveries": 1})]
    assert _ids(engines["Tallinn"].reserved) == [("m3", ["l2"], LATER)]


def test_after_a_decline_the_limit_date_stays_the_original_quotes():
    # ready 3 days and 30 minutes after the quote: within 3 days of the modify, but not of the quote
    shelves = {"Tartu": {"MacBook": [START + 5 * DAY]}, "Tallinn": {"iPad": [START + 3 * DAY + HOUR / 2]}}

    _, replies, _ = _run([(START, quote("MacBook")), (LATER, modify("iPad"))], shelves)

    assert replies[1] == (LATER, "QuoteDeclined", {"earliest": START + 3 * DAY + HOUR / 2})


@pytest.mark.parametrize("within", [timedelta(days=3), None])
def test_after_a_decline_a_modify_without_lines_is_accepted_at_once_with_a_warning(within):
    # two MacBooks can't both be ready: Tartu has one
    engines, replies, warnings = _run([(START, quote("MacBook", "MacBook", within=within)),
                                       (LATER, modify(within=within))])

    assert replies == [(START, "QuoteDeclined", {"earliest": None}),
                       (LATER, "QuoteAccepted", {"ready_by": LATER, "deliveries": 0})]
    assert warnings == ["QuoteModified m3 from Sales has no lines"]
    assert [message.id for message, _ in engines["Tartu"].estimated] == ["m1"]


def test_without_a_quote_a_modify_is_declined_with_a_warning():
    engines, replies, warnings = _run([(START, modify("iPhone"))])

    assert replies == [(START, "QuoteDeclined", {"earliest": None})]
    assert warnings == ["order-7 has no ItemQuote to modify"]
    assert all(engine.estimated == [] and engine.reserved == [] for engine in engines.values())
