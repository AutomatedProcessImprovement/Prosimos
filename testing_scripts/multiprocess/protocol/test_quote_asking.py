"""
A quote goes to the members, not the pool (docs/orchestrator.md): a message with a Quote is stamped as usual, its
lines get ids, and each member of the one group that supplies the quote type is asked for its estimate at once,
never the requester. It is never offered with deliver().
"""
from datetime import datetime, timedelta

import pytz

from prosimos.orchestration.orchestrator import orchestrate
from prosimos.orchestration.protocol import Line, Message, Quote, QuoteKind, Verdict
from testing_scripts.multiprocess.scripted_engine import ScriptedEngine

START = pytz.utc.localize(datetime(2024, 1, 1, 9))
LATER = START + timedelta(hours=1)


def _quote(*products):
    return Message("ItemQuote", {"object_id": "order-7"}, quote=Quote(
        QuoteKind.QUOTE, "ItemQuote", "order-7", [Line({"product": product}) for product in products],
        timedelta(days=3)))


def _run():
    """Sales asks a quote for an iPhone and an iPad at 09:00, publishes OrderPlaced at 10:00 and asks a quote for a
    MacBook then; Tartu and Tallinn answer ItemQuote and would also claim any ItemQuote they were offered."""
    sales = ScriptedEngine("Sales")
    sales.at(START, lambda now: [_quote("iPhone", "iPad")])
    sales.at(LATER, lambda now: [Message("OrderPlaced", {"order_id": "order-8"}), _quote("MacBook")])
    engines = {"Sales": sales}
    for name in ("Tartu", "Tallinn"):
        engines[name] = ScriptedEngine(name)
        engines[name].slots("ItemQuote", lambda line, now: [now])
        engines[name].consume("ItemQuote", lambda message, now: Verdict.CLAIMED)
        engines[name].consume("OrderPlaced", lambda message, now: Verdict.CLAIMED)
    return orchestrate(engines, {"Sales": ["Sales"], "Warehouses": ["Tartu", "Tallinn"]}, 1), engines


def test_a_quote_never_reaches_deliver_and_each_member_is_asked_once_per_quote():
    report, engines = _run()

    for name in ("Tartu", "Tallinn"):
        assert [(message.id, now) for message, now in engines[name].estimated] == [("m1", START), ("m3", LATER)]
    # only the ordinary order is offered, to the warehouse that claims it
    offered = [message.id for name in ("Tartu", "Tallinn") for message, _ in engines[name].offered]
    assert offered == ["m2"]
    assert report.copies == [("m2", "Warehouses")]
    assert report.warnings == [] and report.unclaimed == []


def test_the_quote_is_stamped_and_its_lines_get_ids_across_quotes():
    report, engines = _run()

    first, second = (message for message, _ in engines["Tartu"].estimated)
    assert (first.id, first.source, first.time) == ("m1", "Sales", START)
    assert [line.id for line in first.quote.lines] == ["l1", "l2"]
    assert [line.id for line in second.quote.lines] == ["l3"]
    assert [message.id for message in report.published] == ["m1", "m2", "m3"]


def test_the_requester_and_members_not_subscribed_to_the_quote_type_are_never_asked():
    # Sales answers ItemQuote too, and Billing, in the same group, doesn't
    sales, tartu, billing = ScriptedEngine("Sales"), ScriptedEngine("Tartu"), ScriptedEngine("Billing")
    sales.at(START, lambda now: [_quote("iPhone")])
    for engine in (sales, tartu):
        engine.slots("ItemQuote", lambda line, now: [now])
    billing.consume("OrderPlaced", lambda message, now: Verdict.CLAIMED)

    orchestrate({"Sales": sales, "Tartu": tartu, "Billing": billing}, {"Shop": ["Sales", "Tartu", "Billing"]}, 1)

    assert [message.id for message, _ in tartu.estimated] == ["m1"]
    assert sales.estimated == [] and billing.estimated == []
