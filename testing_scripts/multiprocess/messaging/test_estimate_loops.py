"""
Loops in estimates (docs/messaging.md): a loop back with chance q repeats its elements 1 / (1 - q) times on average;
within the model's loop limit, which stops a case at a split it has passed that many times, the item leaves the loop
with chance 1 - q to the power of the passes left, after fewer rounds on average.
"""
from datetime import datetime, timedelta

import pytz

from prosimos.orchestration.config import ProcessSpec
from prosimos.orchestration.prosimos_engine import ProsimosEngine
from prosimos.orchestration.protocol import Line, Message, Quote, QuoteKind

START = pytz.utc.localize(datetime(2024, 1, 1, 9))
MINUTE, HOUR = timedelta(minutes=1), timedelta(hours=1)
ASSETS = "testing_scripts/assets/quotes"
# rework_loop: an item arrives every hour from 09:00, is made (10 minutes) and inspected (30 minutes); 10% are
# reworked (1 hour) and inspected again, the others packed (30 minutes) for the shelf. rework_loop_limited: 50% are
# reworked, and the loop limit is 2


def _slots(settings, items):
    """The slots at 09:00 of a one-line quote at the warehouse with the settings asset, for items."""
    spec = ProcessSpec("Warehouse", f"{ASSETS}/rework_loop.bpmn", f"{ASSETS}/{settings}.json", items, "item")
    engine = ProsimosEngine(spec, START, seed=1)
    asked = Message("ItemQuote", {"object_id": "order-9"}, "m9", "Sales", START,
                    Quote(QuoteKind.QUOTE, "ItemQuote", "order-9", [Line({"object_id": "order-9"}, "l1")],
                          timedelta(days=3), "QuoteAccepted", "QuoteDeclined"))
    return engine.estimate(asked, START)["l1"]


def _near(slots, expected):
    return len(slots) == len(expected) and all(abs(slot - at) < timedelta(milliseconds=1)
                                               for slot, at in zip(slots, expected))


def test_a_10_percent_rework_loop_repeats_its_elements_1_11_times():
    # made by 09:10; inspected 1.11 times (33.3 minutes) and reworked 0.11 times (6.7 minutes); packed by 10:20
    assert _near(_slots("rework_loop", 1), [START + 80 * MINUTE])


def test_the_loop_limit_cuts_the_rounds_and_the_chance_of_leaving():
    # with 2 passes of the check, an item leaves with chance 0.75, after 1/3 rework on average: inspected for 40
    # minutes and reworked for 20, packed by 1:40 after its arrival. The expected count reaches 1 with item-1,
    # 2 with item-2, and 3 with item-3
    assert _near(_slots("rework_loop_limited", 4), [START + index * HOUR + 100 * MINUTE for index in (1, 2, 3)])
