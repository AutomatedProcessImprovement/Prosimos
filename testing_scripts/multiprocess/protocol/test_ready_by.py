"""
A reply's ready_by is never before the quote's own time (docs/orchestrator.md): an item waiting on the shelf since
earlier is ready only now. The same holds for a modify, a modify after a decline, and a decline's earliest.
"""
from datetime import datetime, timedelta

import pytz

from prosimos.orchestration.orchestrator import orchestrate
from prosimos.orchestration.protocol import Line, Message, Quote, QuoteKind, Verdict
from testing_scripts.multiprocess.scripted_engine import ScriptedEngine

START = pytz.utc.localize(datetime(2024, 1, 1, 9))
HOUR, DAY = timedelta(hours=1), timedelta(days=1)


def _asked(kind, product, within):
    return Message("ItemQuote" if kind is QuoteKind.QUOTE else "QuoteModified", {"object_id": "order-7"},
                   quote=Quote(kind, "ItemQuote", "order-7", [Line({"product": product})], within, "QuoteAccepted",
                               "QuoteDeclined"))


def _run(script, ready, moved_at=None):
    """The replies of a run of Sales publishing what script gives, (time, message maker), against Tartu, which has
    product's item ready at ready(product, now), or none; moved_at(message) is when its portion moved on, or None.
    And the report."""
    sales = ScriptedEngine("Sales")
    for reply in ("QuoteAccepted", "QuoteDeclined"):
        sales.consume(reply, lambda message, now: Verdict.CLAIMED)
    for time, make in script:
        sales.at(time, lambda now, make=make: [make()])
    tartu = ScriptedEngine("Tartu")
    tartu.slots("ItemQuote", lambda line, now: [] if ready(line.attributes["product"], now) is None
                else [ready(line.attributes["product"], now)])
    tartu.portions_moved(moved_at or (lambda message: None))
    report = orchestrate({"Sales": sales, "Tartu": tartu}, {"Sales": ["Sales"], "Warehouses": ["Tartu"]}, 1)
    replies = [(message.time, message.type, message.attributes.get("ready_by", message.attributes.get("earliest")))
               for message in report.published if message.source == "orchestrator"]
    return replies, report


def test_an_item_already_waiting_is_promised_for_the_quotes_time_and_isnt_late_when_it_moves_on():
    # the iPhone has waited on the shelf since 08:00; it moves on at once, at 09:00
    replies, report = _run([(START, lambda: _asked(QuoteKind.QUOTE, "iPhone", DAY))],
                           ready=lambda product, now: START - HOUR, moved_at=lambda message: START)

    assert replies == [(START, "QuoteAccepted", START)]
    assert report.broken_promises["ItemQuote"]["count"] == 0


def test_a_modify_after_the_quotes_deadline_is_declined():
    # accepted at 09:00 for 10:00, within a day; the modify keeps the line but comes on day 3: too late for the limit
    replies, _ = _run([(START, lambda: _asked(QuoteKind.QUOTE, "iPhone", DAY)),
                       (START + 2 * DAY, lambda: _asked(QuoteKind.MODIFY, "iPhone", DAY))],
                      ready=lambda product, now: START + HOUR)

    assert replies == [(START, "QuoteAccepted", START + HOUR), (START + 2 * DAY, "QuoteDeclined", START + 2 * DAY)]


def test_a_declines_earliest_is_never_before_the_quotes_time():
    # the iPhone can't be ready within the hour: declined. At 11:00, past the original limit, the order changes to an
    # iPad waiting since 08:00: declined too, at the earliest 11:00
    replies, _ = _run([(START, lambda: _asked(QuoteKind.QUOTE, "iPhone", HOUR)),
                       (START + 2 * HOUR, lambda: _asked(QuoteKind.MODIFY, "iPad", HOUR))],
                      ready=lambda product, now: START + 3 * HOUR if product == "iPhone" else START - HOUR)

    assert replies == [(START, "QuoteDeclined", START + 3 * HOUR),
                       (START + 2 * HOUR, "QuoteDeclined", START + 2 * HOUR)]
