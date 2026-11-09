"""
The report's quotes section (docs/orchestrator.md, "Output"): per quote type, how many quotes were accepted and
declined, canceled and canceled too late, modified (accepted or declined), and declined as a second quote. A run
without quotes has no such section.
"""
from datetime import datetime, timedelta

import pytz

from prosimos.orchestration.orchestrator import orchestrate
from prosimos.orchestration.protocol import Line, Message, Quote, QuoteKind, Verdict
from testing_scripts.multiprocess.scripted_engine import ScriptedEngine

START = pytz.utc.localize(datetime(2024, 1, 1, 9))
HOUR = timedelta(hours=1)


def _asked(kind, request, *products):
    message_type = {QuoteKind.QUOTE: "ItemQuote", QuoteKind.MODIFY: "QuoteModified"}[kind]
    return Message(message_type, {"object_id": request}, quote=Quote(
        kind, "ItemQuote", request, [Line({"product": product}) for product in products], timedelta(days=3),
        "QuoteAccepted", "QuoteDeclined"))


def _cancel(request):
    return Message("QuoteCanceled", {"object_id": request}, quote=Quote(QuoteKind.CANCEL, "ItemQuote", request))


def _report(script, tartu_moved=lambda message, now: False):
    """The run of Sales publishing what script gives, (time, message maker), against Tartu with the iPhone and
    Tallinn with the iPad, whose portions of order-8, order-9, and order-10 have moved on once it is asked; tartu_moved
    says whether Tartu's has."""
    sales = ScriptedEngine("Sales")
    for reply in ("QuoteAccepted", "QuoteDeclined"):
        sales.consume(reply, lambda message, now: Verdict.CLAIMED)
    for time, make in script:
        sales.at(time, lambda now, make=make: [make()])
    engines = {"Sales": sales}
    for name, product in (("Tartu", "iPhone"), ("Tallinn", "iPad")):
        engines[name] = ScriptedEngine(name)
        engines[name].slots("ItemQuote", lambda line, now, product=product:
                            [now] if line.attributes["product"] == product else [])
    engines["Tallinn"].moved_when(lambda message, now: message.quote.request in ("order-8", "order-9", "order-10"))
    engines["Tartu"].moved_when(tartu_moved)
    return orchestrate(engines, {"Sales": ["Sales"], "Warehouses": ["Tartu", "Tallinn"]}, 1)


def test_each_outcome_is_counted_per_quote_type():
    report = _report([
        (START, lambda: _asked(QuoteKind.QUOTE, "order-7", "iPhone")),  # accepted
        (START + HOUR, lambda: _asked(QuoteKind.QUOTE, "order-7", "iPad")),  # declined as a second quote
        (START + 2 * HOUR, lambda: _asked(QuoteKind.MODIFY, "order-7", "iPhone", "iPad")),  # modified, accepted
        (START + 3 * HOUR, lambda: _asked(QuoteKind.MODIFY, "order-7", "MacBook")),  # modified, declined: no MacBook
        (START + 4 * HOUR, lambda: _cancel("order-7")),  # canceled
        (START + 5 * HOUR, lambda: _asked(QuoteKind.QUOTE, "order-7", "MacBook")),  # declined
        (START + 6 * HOUR, lambda: _asked(QuoteKind.QUOTE, "order-8", "iPhone", "iPad")),  # accepted
        (START + 7 * HOUR, lambda: _cancel("order-8")),  # too late: Tallinn's portion has moved on, Tartu's not
        (START + 8 * HOUR, lambda: _asked(QuoteKind.QUOTE, "order-9", "iPad")),  # accepted
        (START + 9 * HOUR, lambda: _cancel("order-9"))])  # not counted: the quote was delivered in full

    counts = {"accepted": 3, "declined": 1, "canceled": 1, "cancel_too_late": 1, "modified_accepted": 1,
              "modified_declined": 1, "declined_as_second": 1}
    assert report.quotes == {"ItemQuote": counts}
    assert report.to_dict()["quotes"] == [{"type": "ItemQuote", **counts}]


def test_a_cancel_of_a_quote_locked_and_then_delivered_in_full_is_not_too_late():
    # the modification is too late, as Tallinn's portion has moved on, and locks the quote; then Tartu delivers the
    # rest, and the case ends, canceling its quote
    report = _report([(START, lambda: _asked(QuoteKind.QUOTE, "order-10", "iPhone", "iPad")),
                      (START + HOUR, lambda: _asked(QuoteKind.MODIFY, "order-10", "iPhone")),
                      (START + 3 * HOUR, lambda: _cancel("order-10"))],
                     tartu_moved=lambda message, now: now >= START + 2 * HOUR)

    assert report.quotes == {"ItemQuote": {"accepted": 1, "declined": 0, "canceled": 0, "cancel_too_late": 0,
                                           "modified_accepted": 0, "modified_declined": 1, "declined_as_second": 0}}


def test_a_run_without_quotes_has_no_quotes_section():
    report = _report([])

    assert report.quotes == {} and "quotes" not in report.to_dict()
