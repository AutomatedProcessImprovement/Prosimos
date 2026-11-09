"""
Broken promises (docs/orchestrator.md, "Output"): a member's portion of a quote that moved on after the ready_by of
the quote's last accepted reply breaks its promise, counted per quote type with how late on average and at most. A
portion not moved on by the end of the run, past its ready_by, is broken too, still not delivered: counted on its own,
with the longest overdue. A quote a later one replaced keeps its promise; a portion promised after the end, or of a
canceled quote, isn't counted.
"""
from datetime import datetime, timedelta

import pytz

from prosimos.orchestration.orchestrator import orchestrate
from prosimos.orchestration.protocol import Line, Message, Quote, QuoteKind, Verdict
from testing_scripts.multiprocess.scripted_engine import ScriptedEngine

START = pytz.utc.localize(datetime(2024, 1, 1, 9))
MINUTE, HOUR = timedelta(minutes=1), timedelta(hours=1)
ON_TIME = {"count": 0, "average_late": None, "most_late": None, "undelivered": 0, "most_overdue": None}


def _asked(request, *products, kind=QuoteKind.QUOTE, quote_type="ItemQuote"):
    message_type = {QuoteKind.QUOTE: quote_type, QuoteKind.MODIFY: "QuoteModified"}[kind]
    return Message(message_type, {"object_id": request}, quote=Quote(
        kind, quote_type, request, [Line({"product": product}) for product in products], timedelta(days=3),
        "QuoteAccepted", "QuoteDeclined"))


def _cancel(request):
    return Message("QuoteCanceled", {"object_id": request}, quote=Quote(QuoteKind.CANCEL, "ItemQuote", request))


def _late(count, average, most):
    """The broken promises of portions that moved on late, none undelivered."""
    return {**ON_TIME, "count": count, "average_late": average, "most_late": most}


def _broken(script, moved_on, ready=None, ends_at=None):
    """The broken promises of a run of Sales publishing what script gives, (time, message maker), against Tartu
    with the iPhone and Tallinn with the iPad, each ready an hour after it is asked (ready: product -> how long
    after instead). moved_on: (member, message id) -> when the portion that message reserved there moved on; a
    portion counts as moved on for the book from then on. The run ends at Sales's last step, or at ends_at."""
    sales = ScriptedEngine("Sales")
    for reply in ("QuoteAccepted", "QuoteDeclined"):
        sales.consume(reply, lambda message, now: Verdict.CLAIMED)
    for time, make in script:
        sales.at(time, lambda now, make=make: [make()])
    if ends_at is not None:
        sales.at(ends_at, lambda now: [])
    engines = {"Sales": sales}
    for name, product in (("Tartu", "iPhone"), ("Tallinn", "iPad")):
        engine = engines[name] = ScriptedEngine(name)
        later = (ready or {}).get(product, HOUR)
        for quote_type in ("ItemQuote", "PartQuote"):
            engine.slots(quote_type, lambda line, now, product=product, later=later:
                         [now + later] if line.attributes["product"] == product else [])
        engine.portions_moved(lambda message, name=name: moved_on.get((name, message.id)))
        engine.moved_when(lambda message, now, name=name: moved_on.get((name, message.id), now + HOUR) <= now)
    return orchestrate(engines, {"Sales": ["Sales"], "Warehouses": ["Tartu", "Tallinn"]}, 1).broken_promises


def test_late_portions_count_with_their_average_and_most_lateness():
    # each order's iPhone is promised for 10:00 and 11:00; they move on an hour and three hours late
    broken = _broken([(START, lambda: _asked("order-1", "iPhone")),
                      (START + HOUR, lambda: _asked("order-2", "iPhone"))],
                     {("Tartu", "m1"): START + 2 * HOUR, ("Tartu", "m3"): START + 5 * HOUR})

    assert broken == {"ItemQuote": _late(2, 2 * HOUR, 3 * HOUR)}


def test_each_members_portion_of_a_split_quote_counts_on_its_own():
    # promised for 11:00: Tartu's portion moves on at 10:30, Tallinn's at 11:15
    broken = _broken([(START, lambda: _asked("order-1", "iPhone", "iPad"))],
                     {("Tartu", "m1"): START + 90 * MINUTE, ("Tallinn", "m1"): START + 135 * MINUTE},
                     ready={"iPad": 2 * HOUR})

    assert broken == {"ItemQuote": _late(1, 15 * MINUTE, 15 * MINUTE)}


def test_a_modify_that_moves_the_promise_later_keeps_a_portion_between_the_two_dates_on_time():
    # promised for 10:00; at 09:30 the modify adds an iPad, ready at 12:30: Tartu's portion moves on at 11:00
    broken = _broken([(START, lambda: _asked("order-1", "iPhone")),
                      (START + 30 * MINUTE, lambda: _asked("order-1", "iPhone", "iPad", kind=QuoteKind.MODIFY))],
                     {("Tartu", "m1"): START + 2 * HOUR}, ready={"iPad": 3 * HOUR})

    assert broken == {"ItemQuote": ON_TIME}


def test_a_portion_promised_before_the_end_and_not_moved_on_is_broken_still_not_delivered():
    # promised for 10:00, the run ends at 12:00 with nothing moved on: overdue by 2 hours at least
    broken = _broken([(START, lambda: _asked("order-1", "iPhone"))], {}, ends_at=START + 3 * HOUR)

    assert broken == {"ItemQuote": {**ON_TIME, "undelivered": 1, "most_overdue": 2 * HOUR}}


def test_a_portion_promised_after_the_end_is_not_counted():
    # promised for 10:00, the run ends at 09:00
    assert _broken([(START, lambda: _asked("order-1", "iPhone"))], {}) == {"ItemQuote": ON_TIME}


def test_a_canceled_quote_is_not_counted():
    # canceled at 09:30, nothing had moved on; the item moves on much later, as ordinary stock
    broken = _broken([(START, lambda: _asked("order-1", "iPhone")), (START + 30 * MINUTE, lambda: _cancel("order-1"))],
                     {("Tartu", "m1"): START + 5 * HOUR})

    assert broken == {"ItemQuote": ON_TIME}


def test_a_quote_replaced_by_a_later_one_keeps_its_promise():
    # a looping order: its first quote, promised for 10:00, moves on at 11:00; its second, at 12:00, promised for
    # 13:00, moves on at 14:30
    broken = _broken([(START, lambda: _asked("order-1", "iPhone")),
                      (START + 3 * HOUR, lambda: _asked("order-1", "iPhone"))],
                     {("Tartu", "m1"): START + 2 * HOUR, ("Tartu", "m3"): START + 330 * MINUTE})

    assert broken == {"ItemQuote": _late(2, 75 * MINUTE, 90 * MINUTE)}


def test_each_quote_type_counts_on_its_own():
    # the item is an hour late, the part half an hour
    broken = _broken([(START, lambda: _asked("order-1", "iPhone")),
                      (START, lambda: _asked("order-1", "iPhone", quote_type="PartQuote"))],
                     {("Tartu", "m1"): START + 2 * HOUR, ("Tartu", "m3"): START + 90 * MINUTE})

    assert broken == {"ItemQuote": _late(1, HOUR, HOUR),
                      "PartQuote": _late(1, 30 * MINUTE, 30 * MINUTE)}
