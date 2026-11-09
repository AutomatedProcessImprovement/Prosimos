"""
Parallel splits in estimates (docs/messaging.md): an item meeting a parallel split goes on from its join when the
longest branch would get there; an item already inside a split goes on from the join when its latest branch gets
there, a branch whose task has run at that task's end. Nested splits go on from the inner join first.
"""
from datetime import datetime, timedelta

import pytz

from prosimos.orchestration.config import ProcessSpec
from prosimos.orchestration.prosimos_engine import ProsimosEngine
from prosimos.orchestration.protocol import Line, Message, Quote, QuoteKind

START = pytz.utc.localize(datetime(2024, 1, 1, 9))
MINUTE, HOUR = timedelta(minutes=1), timedelta(hours=1)
ASSETS = "testing_scripts/assets/quotes"
# polish_and_test: an item arrives every hour from 09:00 and is received (10 minutes); then it is polished (1 hour)
# while it is cleaned (10 minutes) and tested (30 minutes), and packed (30 minutes) for the shelf. In
# polish_and_test_split_by_receive, Receive's two outgoing flows are the split: Prosimos's loader puts a hidden parallel
# split after it. In polish_and_test_nested, the same split, then inspected (20 minutes), is inside an outer one:
# meanwhile the item is labeled (10 minutes) and printed on (40 minutes)


def _warehouse(model="polish_and_test", settings="polish_and_test"):
    """The warehouse's engine, for two items, from the model and settings assets; enough staff that no task waits."""
    spec = ProcessSpec("Warehouse", f"{ASSETS}/{model}.bpmn", f"{ASSETS}/{settings}.json", 2, "item")
    return ProsimosEngine(spec, START, seed=1)


def _slots(engine, now):
    """Steps the engine through everything due before now, then its slots for a one-line quote."""
    while engine.next_event_time() is not None and engine.next_event_time() < now:
        engine.step()
    asked = Message("ItemQuote", {"object_id": "order-9"}, "m9", "Sales", now,
                    Quote(QuoteKind.QUOTE, "ItemQuote", "order-9", [Line({"object_id": "order-9"}, "l1")],
                          timedelta(days=3), "QuoteAccepted", "QuoteDeclined"))
    return engine.estimate(asked, now)["l1"]


def test_two_branches_ahead_go_on_from_the_join_when_the_longest_gets_there():
    # received by 09:10, polished by 10:10 (cleaned and tested by 09:50), packed by 10:40
    assert _slots(_warehouse(), START) == [START + 100 * MINUTE, START + HOUR + 100 * MINUTE]


def test_a_task_with_two_outgoing_flows_splits_like_a_parallel_gateway():
    engine = _warehouse("polish_and_test_split_by_receive")
    assert "and_split_Receive" in engine._env.sim_setup.bpmn_graph.element_info

    assert _slots(engine, START) == [START + 100 * MINUTE, START + HOUR + 100 * MINUTE]


def test_an_item_inside_a_split_goes_on_when_its_latest_branch_gets_there():
    # at 09:15 item-0's Polish runs until 10:10, and its Test is queued at 09:20 (done by 09:50): the join at 10:10,
    # packed by 10:40. item-1 is still ahead
    assert _slots(_warehouse(), START + 15 * MINUTE) == [START + 100 * MINUTE, START + HOUR + 100 * MINUTE]


def test_nested_splits_ahead_go_on_from_the_inner_join_then_the_outer_one():
    # received by 09:10; polished by 10:10, inspected by 10:30, while labeled and printed on by 10:00; packed by 11:00
    engine = _warehouse("polish_and_test_nested", "polish_and_test_nested")

    assert _slots(engine, START) == [START + 2 * HOUR, START + 3 * HOUR]


def test_an_item_inside_both_splits_goes_on_from_the_inner_join_first():
    # at 09:15 item-0's Polish runs until 10:10 and its Test is queued at 09:20 (done by 09:50): the inner join at
    # 10:10, inspected by 10:30; Print is queued at 09:20 (done by 10:00): the outer join at 10:30, packed by 11:00
    engine = _warehouse("polish_and_test_nested", "polish_and_test_nested")

    assert _slots(engine, START + 15 * MINUTE) == [START + 2 * HOUR, START + 3 * HOUR]
