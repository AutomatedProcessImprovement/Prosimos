"""
Open quotes (docs/orchestrator.md): the orchestrator tracks each quote by its requester and quote type, with its
state, its limit date, and the member and reserving message of each line. A second quote of the same type while one
is open is declined at once, with a warning; after a decline, quoting again is fine.
"""
from datetime import datetime, timedelta

import pytz

from prosimos.orchestration.orchestrator import orchestrate
from prosimos.orchestration.protocol import Line, Message, Quote, QuoteKind, Verdict
from prosimos.orchestration.quotes import HeldLine, QuoteBook, QuoteState, TrackedQuote
from testing_scripts.multiprocess.scripted_engine import ScriptedEngine

START = pytz.utc.localize(datetime(2024, 1, 1, 9))
LATER = START + timedelta(hours=1)
HALF_HOUR = timedelta(minutes=30)
WITHIN = timedelta(days=3)


def _quote(request="order-7", *products):
    return Message("ItemQuote", {"object_id": request}, quote=Quote(
        QuoteKind.QUOTE, "ItemQuote", request, [Line({"product": product}) for product in products or ("iPhone",)],
        WITHIN, "QuoteAccepted", "QuoteDeclined"))


def _run(asks, shelf=lambda line, now: [now], requesters=("Sales",), moved=None):
    """Each requester asks the quotes asks gives, (time, object id), and takes the replies; Tartu answers with shelf,
    and with moved whether a quote's share has moved on. The replies and the warnings."""
    engines = {}
    for name in requesters:
        engines[name] = ScriptedEngine(name)
        for reply in ("QuoteAccepted", "QuoteDeclined"):
            engines[name].consume(reply, lambda message, now: Verdict.CLAIMED)
        for time, request in asks:
            engines[name].at(time, lambda now, request=request: [_quote(request)])
    engines["Tartu"] = ScriptedEngine("Tartu")
    engines["Tartu"].slots("ItemQuote", shelf)
    if moved is not None:
        engines["Tartu"].moved_when(moved)
    groups = {name: [name] for name in requesters} | {"Warehouses": ["Tartu"]}
    report = orchestrate(engines, groups, 1)
    replies = [(message.time, message.type, message.attributes["request"]) for message in report.published
               if message.source == "orchestrator"]
    return replies, report.warnings, engines


def test_a_second_quote_while_one_is_open_is_declined_at_once_with_a_warning():
    replies, warnings, engines = _run([(START, "order-7"), (LATER, "order-7")])

    assert replies == [(START, "QuoteAccepted", "order-7"), (LATER, "QuoteDeclined", "order-7")]
    assert warnings == ["order-7 already has an open ItemQuote; use modify"]
    # only whether the first quote's share has moved: the members aren't asked for slots again
    assert [(message.id, now) for message, now in engines["Tartu"].moved_checks] == [("m1", LATER)]
    assert [message.id for message, _ in engines["Tartu"].estimated] == ["m1"]
    assert [message.id for message, _, _ in engines["Tartu"].reserved] == ["m1"]


def test_a_second_quote_after_the_firsts_share_moved_is_answered_and_replaces_it():
    # the first order's iPhone left Tartu at 09:30; order-7 loops and quotes again at 10:00, then once more at 11:00
    replies, warnings, engines = _run([(START, "order-7"), (LATER, "order-7"), (LATER + timedelta(hours=1), "order-7")],
                                      moved=lambda message, now: message.id == "m1" and now >= START + HALF_HOUR)

    assert replies == [(START, "QuoteAccepted", "order-7"), (LATER, "QuoteAccepted", "order-7"),
                       (LATER + timedelta(hours=1), "QuoteDeclined", "order-7")]
    # at 11:00 the second quote, m3, is the one asked about, and nothing of it has moved
    assert [message.id for message, _ in engines["Tartu"].moved_checks] == ["m1", "m3"]
    assert warnings == ["order-7 already has an open ItemQuote; use modify"]


def test_a_quote_without_lines_never_blocks_the_next():
    sales, tartu = ScriptedEngine("Sales"), ScriptedEngine("Tartu")
    tartu.slots("ItemQuote", lambda line, now: [now])
    for time, products in ((START, ()), (LATER, ("iPhone",))):
        sales.at(time, lambda now, products=products: [Message("ItemQuote", {"object_id": "order-7"}, quote=Quote(
            QuoteKind.QUOTE, "ItemQuote", "order-7", [Line({"product": product}) for product in products], WITHIN,
            "QuoteAccepted", "QuoteDeclined"))])

    report = orchestrate({"Sales": sales, "Tartu": tartu}, None, 1)

    assert [message.type for message in report.published if message.source == "orchestrator"] == [
        "QuoteAccepted", "QuoteAccepted"]
    assert tartu.moved_checks == []


def test_after_a_decline_quoting_again_is_fine():
    # Tartu has nothing before 10:00
    replies, warnings, _ = _run([(START, "order-7"), (LATER, "order-7")],
                                shelf=lambda line, now: [now] if now >= LATER else [])

    assert replies == [(START, "QuoteDeclined", "order-7"), (LATER, "QuoteAccepted", "order-7")]
    assert warnings == []


def test_another_objects_quote_is_its_own():
    replies, warnings, _ = _run([(START, "order-7"), (LATER, "order-8")])

    assert replies == [(START, "QuoteAccepted", "order-7"), (LATER, "QuoteAccepted", "order-8")]
    assert warnings == []


def test_the_same_object_id_in_another_process_is_another_requester():
    # object ids are unique only within a process
    replies, warnings, _ = _run([(START, "order-7")], requesters=("Shop", "Outlet"))

    assert replies == [(START, "QuoteAccepted", "order-7"), (START, "QuoteAccepted", "order-7")]
    assert warnings == []


def _book_answers(*shelves):
    """A QuoteBook answering one stamped quote for an iPhone and an iPad, with Tartu and Tallinn answering from
    shelves (product -> times); the book and the quote."""
    engines = {}
    for name, shelf in zip(("Tartu", "Tallinn"), shelves):
        engines[name] = ScriptedEngine(name)
        engines[name].slots("ItemQuote", lambda line, now, shelf=shelf: shelf.get(line.attributes["product"], []))
    message = _quote("order-7", "iPhone", "iPad")
    message.id, message.source, message.time = "m1", "Sales", START
    for number, line in enumerate(message.quote.lines, 1):
        line.id = f"l{number}"
    book = QuoteBook()
    book.answer(message, list(engines), engines.get, START, lambda warning: None)
    return book, message


def test_an_accepted_quote_is_tracked_open_with_its_limit_date_and_each_lines_member_message_and_time():
    book, message = _book_answers({"iPhone": [LATER]}, {"iPad": [LATER]})
    iphone, ipad = message.quote.lines

    assert book.quotes == {("Sales", "order-7", "ItemQuote"): TrackedQuote(QuoteState.OPEN, START + WITHIN, {
        "l1": HeldLine(iphone, "Tartu", message, LATER), "l2": HeldLine(ipad, "Tallinn", message, LATER)})}


def test_a_declined_quote_is_tracked_declined_holding_nothing():
    book, _ = _book_answers({"iPhone": [LATER]}, {})

    assert book.quotes == {("Sales", "order-7", "ItemQuote"): TrackedQuote(QuoteState.DECLINED, START + WITHIN)}


def test_each_member_holding_lines_is_asked_with_the_message_that_reserved_them():
    book, first = _book_answers({"iPhone": [LATER]}, {"iPad": [LATER]})
    engines = {"Tartu": ScriptedEngine("Tartu"), "Tallinn": ScriptedEngine("Tallinn")}
    engines["Tallinn"].moved_when(lambda message, now: True)
    second = _quote("order-7", "iPhone")
    second.id, second.source, second.time = "m2", "Sales", LATER

    book.answer(second, [], engines.get, LATER, lambda warning: None)

    assert [(message.id, now) for message, now in engines["Tartu"].moved_checks] == [("m1", LATER)]
    assert [(message.id, now) for message, now in engines["Tallinn"].moved_checks] == [("m1", LATER)]
    # Tallinn's iPad has left, so the second quote replaced the first
    assert book.quotes[("Sales", "order-7", "ItemQuote")].lines == {}
