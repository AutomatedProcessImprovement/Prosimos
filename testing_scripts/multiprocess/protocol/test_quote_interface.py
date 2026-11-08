"""
The quote calls of the engine interface (docs/orchestrator.md, "Engine interface"): estimate, moved, reserve, and
release, next to the five messaging and lifecycle calls.
"""
from datetime import datetime, timedelta

import pytz

from prosimos.orchestration.config import ProcessSpec
from prosimos.orchestration.prosimos_engine import ProsimosEngine
from prosimos.orchestration.protocol import Line, Message, Quote, QuoteKind, SimulationEngine
from testing_scripts.multiprocess.scripted_engine import ScriptedEngine

START = pytz.utc.localize(datetime(2024, 1, 1, 9))
WAREHOUSE = "testing_scripts/assets/order_routing/tartu_warehouse"
LINES = [Line({"product": "iPhone"}, "l1"), Line({"product": "iPad"}, "l2")]
QUOTE = Quote(QuoteKind.QUOTE, "ItemQuote", "order-7", LINES, timedelta(days=3))
# a quote message as the orchestrator stamps it
ASKED = Message("ItemQuote", {"object_id": "order-7"}, "m1", "Sales", START, QUOTE)


def test_the_interface_has_nine_methods():
    assert SimulationEngine.__abstractmethods__ == {"subscriptions", "next_event_time", "step", "deliver", "estimate",
                                                    "moved", "reserve", "release", "finish"}


def test_every_engine_implements_them():
    assert ProsimosEngine.__abstractmethods__ == ScriptedEngine.__abstractmethods__ == frozenset()


def test_a_message_carries_no_quote_unless_its_requester_sets_one():
    assert Message("OrderPlaced").quote is None
    assert Message("ItemQuote", {"object_id": "order-7"}, quote=QUOTE).quote.lines[1].attributes == {"product": "iPad"}


def test_a_line_has_no_id_until_the_orchestrator_stamps_it():
    assert Line({"product": "iPhone"}).id is None


def test_a_prosimos_engine_that_takes_no_such_quote_answers_no_slots_and_nothing_moved():
    # the Order Routing warehouse takes orders, not ItemQuote
    engine = ProsimosEngine(ProcessSpec("TartuWarehouse", f"{WAREHOUSE}.bpmn", f"{WAREHOUSE}.json"), START, seed=1)

    assert engine.estimate(ASKED, START) == {"l1": [], "l2": []}
    assert engine.moved(ASKED, START) is False
    assert engine.next_event_time() is None  # the warehouse waits for orders as before
