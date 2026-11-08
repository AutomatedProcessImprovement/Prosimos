"""
Canceling a quote (docs/orchestrator.md): every line is released at its member, with the message that reserved it,
unless some line has already moved on: then it is too late and nothing is released. A cancel of a declined, locked,
or unknown quote does nothing, and a cancel gets no reply.
"""
from datetime import datetime, timedelta

import pytz

from prosimos.orchestration.orchestrator import orchestrate
from prosimos.orchestration.protocol import Line, Message, Quote, QuoteKind, Verdict
from prosimos.orchestration.quotes import QuoteBook
from testing_scripts.multiprocess.scripted_engine import ScriptedEngine

START = pytz.utc.localize(datetime(2024, 1, 1, 9))
HOUR = timedelta(hours=1)


def _quote(*products):
    return Message("ItemQuote", {"object_id": "order-7"}, quote=Quote(
        QuoteKind.QUOTE, "ItemQuote", "order-7", [Line({"product": product}) for product in products],
        timedelta(days=3), "QuoteAccepted", "QuoteDeclined"))


def _cancel():
    return Message("QuoteCanceled", {"object_id": "order-7"}, quote=Quote(QuoteKind.CANCEL, "ItemQuote", "order-7"))


def _run(script, shelves=None, moved=None):
    """Sales publishes what script gives, (time, message maker); Tartu has the iPhone and Tallinn the iPad (or
    shelves), and moved says whether something has moved on at Tallinn. The engines and the replies."""
    sales = ScriptedEngine("Sales")
    for reply in ("QuoteAccepted", "QuoteDeclined"):
        sales.consume(reply, lambda message, now: Verdict.CLAIMED)
    for time, make in script:
        sales.at(time, lambda now, make=make: [make()])
    engines = {"Sales": sales}
    for name, product in (shelves or {"Tartu": "iPhone", "Tallinn": "iPad"}).items():
        engines[name] = ScriptedEngine(name)
        engines[name].slots("ItemQuote", lambda line, now, product=product:
                            [now] if line.attributes["product"] == product else [])
    if moved is not None:
        engines["Tallinn"].moved_when(moved)
    report = orchestrate(engines, {"Sales": ["Sales"], "Warehouses": ["Tartu", "Tallinn"]}, 1)
    replies = [(message.time, message.type) for message in report.published if message.source == "orchestrator"]
    return engines, replies


def _ids(calls):
    return [(message.id, *rest) for message, *rest in calls]


def test_a_cancel_releases_every_line_at_its_member_with_the_message_that_reserved_it():
    engines, replies = _run([(START, lambda: _quote("iPhone", "iPad")), (START + HOUR, _cancel)])

    assert _ids(engines["Tartu"].released) == [("m1", ["l1"], START + HOUR)]
    assert _ids(engines["Tallinn"].released) == [("m1", ["l2"], START + HOUR)]
    assert replies == [(START, "QuoteAccepted")]  # no reply to the cancel


def test_after_a_cancel_the_requester_may_quote_again():
    engines, replies = _run([(START, lambda: _quote("iPhone")), (START + HOUR, _cancel),
                             (START + 2 * HOUR, lambda: _quote("iPhone"))])

    assert replies == [(START, "QuoteAccepted"), (START + 2 * HOUR, "QuoteAccepted")]
    # asked only by the cancel: the canceled quote was gone from the book when the next one came
    assert _ids(engines["Tartu"].moved_checks) == [("m1", START + HOUR)]


def test_too_late_once_one_member_has_moved_nothing_is_released():
    engines, _ = _run([(START, lambda: _quote("iPhone", "iPad")), (START + HOUR, _cancel),
                       (START + 2 * HOUR, _cancel)], moved=lambda message, now: True)

    assert engines["Tartu"].released == [] and engines["Tallinn"].released == []
    # Tartu's iPhone hasn't moved, Tallinn's iPad has; the second cancel finds the quote locked and asks nobody
    assert _ids(engines["Tartu"].moved_checks) == [("m1", START + HOUR)]
    assert _ids(engines["Tallinn"].moved_checks) == [("m1", START + HOUR)]


def test_a_cancel_of_a_quote_without_lines_removes_it_with_nothing_to_release():
    # nothing could have moved, so it isn't too late: the quote leaves the book, not locked
    tartu, book = ScriptedEngine("Tartu"), QuoteBook()
    asked, canceled = _quote(), _cancel()
    for number, (message, time) in enumerate(((asked, START), (canceled, START + HOUR)), 1):
        message.id, message.source, message.time = f"m{number}", "Sales", time
    book.answer(asked, ["Tartu"], {"Tartu": tartu}.get, START, lambda warning: None)

    book.cancel(canceled, {"Tartu": tartu}.get, START + HOUR)

    assert book.quotes == {}
    assert tartu.released == [] and tartu.moved_checks == []


def test_a_cancel_of_a_declined_quote_does_nothing():
    engines, replies = _run([(START, lambda: _quote("MacBook")), (START + HOUR, _cancel)])

    assert replies == [(START, "QuoteDeclined")]
    assert all(engine.released == [] and engine.moved_checks == [] for engine in engines.values())


def test_a_cancel_of_an_unknown_quote_does_nothing():
    engines, replies = _run([(START, _cancel)])

    assert replies == []
    assert all(engine.released == [] and engine.moved_checks == [] for engine in engines.values())
