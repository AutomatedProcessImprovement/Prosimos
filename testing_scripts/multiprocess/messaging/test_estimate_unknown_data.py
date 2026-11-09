"""
Choices decided by data not known yet, in estimates (docs/messaging.md): a split whose condition reads data the item
doesn't have yet, set by a task still ahead, goes by the share of the items that took each branch there so far; with
no history yet, the branch on the way to the catch event passes. Data already known stays exact.
"""
from datetime import datetime, timedelta

import pytz

from prosimos.orchestration.config import ProcessSpec
from prosimos.orchestration.prosimos_engine import ProsimosEngine
from prosimos.orchestration.protocol import Line, Message, Quote, QuoteKind

START = pytz.utc.localize(datetime(2024, 1, 1, 9))
MINUTE, HOUR = timedelta(minutes=1), timedelta(hours=1)
ASSETS = "testing_scripts/assets/quotes"
# quality_check: an item arrives every hour from 09:00; it is made, inspected, and washed (10 minutes each).
# Inspecting sets its quality, good (75%) or bad; the check packs a good one (30 minutes) for the shelf and scraps a
# bad one. With seed 10, items 0 to 4 are bad, good, good, good, and bad


def _engine():
    spec = ProcessSpec("Warehouse", f"{ASSETS}/quality_check.bpmn", f"{ASSETS}/quality_check.json", 8, "item")
    return ProsimosEngine(spec, START, seed=10)


def _slots(engine, now):
    """Steps the engine through everything due before now, then its slots for a one-line quote."""
    while engine.next_event_time() is not None and engine.next_event_time() < now:
        engine.step()
    asked = Message("ItemQuote", {"object_id": "order-9"}, "m9", "Sales", now,
                    Quote(QuoteKind.QUOTE, "ItemQuote", "order-9", [Line({"object_id": "order-9"}, "l1")],
                          timedelta(days=3), "QuoteAccepted", "QuoteDeclined"))
    return engine.estimate(asked, now)["l1"]


def ready(index):
    """When item index is on the shelf if it passes: an hour after its arrival."""
    return START + (index + 1) * HOUR


def test_with_no_history_the_way_to_the_shelf_passes():
    # no item is inspected yet: every item passes
    assert _slots(_engine(), START) == [ready(index) for index in range(8)]


def test_known_data_stays_exact():
    # at 09:15 item-0 is inspected, bad, and washed next: it is scrapped. The others pass, with no history yet
    engine = _engine()

    assert _slots(engine, START + 15 * MINUTE) == [ready(index) for index in range(1, 8)]
    assert engine._env.sim_setup.bpmn_graph.all_attributes[0]["quality"] == "bad"


def test_data_not_known_yet_goes_by_the_history_of_the_split():
    # at 13:15 three of the four items checked so far were good; item-4 is inspected, bad. Items 5, 6, and 7 are
    # good with chance 0.75: after items 1, 2, and 3 on the shelf, the expected count reaches 4 with item-6 and 5
    # with item-7
    engine = _engine()

    assert _slots(engine, START + 4 * HOUR + 15 * MINUTE) == [ready(1), ready(2), ready(3), ready(6), ready(7)]
    assert engine._env._history("Check") == ({"Bad": 1, "Good": 3}, 4)


def test_data_a_task_ahead_sets_again_is_not_known_yet():
    # rework_loop with rework_loop_graded: inspecting sets the quality, and a bad item is reworked and inspected
    # again. With seed 12 items 0 and 1 are good and item-2 bad: at 11:45 it is queued at Inspect again, its quality
    # still bad from before. Inspect sets it anew: by the history, it loops back with chance 1/3, 0.5 times on
    # average: inspected by 13:10, half a rework and half an inspection more, packed by 14:25
    spec = ProcessSpec("Warehouse", f"{ASSETS}/rework_loop.bpmn", f"{ASSETS}/rework_loop_graded.json", 3, "item")
    engine = ProsimosEngine(spec, START, seed=12)

    assert _slots(engine, START + 2 * HOUR + 45 * MINUTE) == [START + 70 * MINUTE, START + 2 * HOUR + 10 * MINUTE,
                                                              START + 5 * HOUR + 25 * MINUTE]
    assert engine._env.sim_setup.bpmn_graph.all_attributes[2]["quality"] == "bad"
    assert engine._env._history("Check") == ({"Good": 2, "Bad": 1}, 3)
