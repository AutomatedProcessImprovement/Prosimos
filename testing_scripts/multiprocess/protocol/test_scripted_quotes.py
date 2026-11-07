"""
Scripted engines answer quotes from a script (scripted_engine.py): slots give each line's slot times, a moved script
says whether some line has moved on, and every estimate, moved, reserve, and release call is recorded.
"""
from datetime import datetime, timedelta

import pytz

from prosimos.orchestration.protocol import Line, Message, Quote, QuoteKind
from testing_scripts.multiprocess.scripted_engine import ScriptedEngine

START = pytz.utc.localize(datetime(2024, 1, 1, 9))
SHELF = {"iPhone": [START + timedelta(hours=2)], "iPad": [START + timedelta(hours=1), START + timedelta(days=1)]}


def _quote_message(*products, quote_type="ItemQuote"):
    lines = [Line({"product": product}, f"l{number}") for number, product in enumerate(products, 1)]
    return Message(quote_type, {"object_id": "order-7"}, "m1", "Sales", START,
                   Quote(QuoteKind.QUOTE, quote_type, "order-7", lines, timedelta(days=3)))


def _tartu():
    tartu = ScriptedEngine("TartuWarehouse")
    tartu.slots("ItemQuote", lambda line, now: SHELF.get(line.attributes["product"], []))
    return tartu


def test_a_scripted_engine_answers_each_line_from_its_script():
    estimate = _tartu().estimate(_quote_message("iPhone", "iPad", "MacBook"), START)

    assert estimate == {"l1": SHELF["iPhone"], "l2": SHELF["iPad"], "l3": []}


def test_without_a_script_for_the_quote_type_it_answers_no_slots():
    assert _tartu().estimate(_quote_message("iPhone", quote_type="PartQuote"), START) == {}


def test_moved_answers_from_its_script_and_nothing_has_moved_without_one():
    tartu, asked = _tartu(), _quote_message("iPhone")
    assert not tartu.moved(asked, START)

    tartu.moved_when(lambda message, now: now >= START + timedelta(hours=1))

    assert [tartu.moved(asked, START), tartu.moved(asked, START + timedelta(hours=1))] == [False, True]
    assert tartu.moved_checks == [(asked, START), (asked, START), (asked, START + timedelta(hours=1))]


def test_a_script_subscribes_the_engine_to_the_quote_type():
    tartu = _tartu()
    tartu.consume("OrderPlaced", lambda message, now: None)

    assert tartu.subscriptions() == ["ItemQuote", "OrderPlaced"]


def test_every_call_is_recorded():
    tartu, asked = _tartu(), _quote_message("iPhone", "iPad")
    later = START + timedelta(hours=1)

    tartu.estimate(asked, START)
    tartu.reserve(asked, ["l1", "l2"], START)
    tartu.release(asked, ["l2"], later)

    assert tartu.estimated == [(asked, START)]
    assert tartu.reserved == [(asked, ["l1", "l2"], START)]
    assert tartu.released == [(asked, ["l2"], later)]
