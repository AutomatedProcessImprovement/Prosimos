"""
Races in estimates (docs/messaging.md): a race on the way goes by the history of its winners in this run: each
branch that has won goes on with its share of the wins, after its winners' average wait. With no history yet, the
branch on the way to the catch event wins at once. For an item already racing, only the wins that took longer than
it has waited count, a timer's at the item's own known time; with none, the message wins now.
"""
import json
from datetime import datetime, timedelta

import pytz

from prosimos.orchestration.config import ProcessSpec
from prosimos.orchestration.prosimos_engine import ProsimosEngine
from prosimos.orchestration.protocol import Line, Message, Quote, QuoteKind

START = pytz.utc.localize(datetime(2024, 1, 1, 9))
MINUTE, HOUR = timedelta(minutes=1), timedelta(hours=1)
ASSETS = "testing_scripts/assets/quotes"
# race_approval_or_timeout: an item arrives every hour from 09:00 and is made (10 minutes); then quality's approval
# for it and a 2-hour timeout race: approved, it is packed (30 minutes) for the shelf; timed out, it is scrapped


def _warehouse(items, settings=f"{ASSETS}/race_approval_or_timeout.json"):
    spec = ProcessSpec("Warehouse", f"{ASSETS}/race_approval_or_timeout.bpmn", str(settings), items, "item")
    return ProsimosEngine(spec, START, seed=1)


def _until(engine, now):
    """Steps the engine through everything due before now."""
    while engine.next_event_time() is not None and engine.next_event_time() < now:
        engine.step()


def _approve(engine, item, now):
    _until(engine, now)
    engine.deliver(Message("Approved", {"item": item}, f"m-{item}", "Quality", now), now)


def _slots(engine, now):
    _until(engine, now)
    asked = Message("ItemQuote", {"object_id": "order-9"}, "m9", "Sales", now,
                    Quote(QuoteKind.QUOTE, "ItemQuote", "order-9", [Line({"object_id": "order-9"}, "l1")],
                          timedelta(days=3), "QuoteAccepted", "QuoteDeclined"))
    return engine.estimate(asked, now)["l1"]


def test_with_no_history_the_branch_on_the_way_to_the_shelf_wins_at_once():
    # each item: made by its arrival + 10 minutes, approved at once, packed by + 40
    assert _slots(_warehouse(4), START) == [START + index * HOUR + 40 * MINUTE for index in range(4)]


def test_a_race_goes_by_the_history_of_its_winners():
    # item-0 is approved after 30 minutes (09:40), item-2 too (11:40), and item-1 times out after 2 hours (12:10)
    engine = _warehouse(7)
    _approve(engine, "item-0", START + 40 * MINUTE)
    _approve(engine, "item-2", START + 2 * HOUR + 40 * MINUTE)

    slots = _slots(engine, START + 3 * HOUR + 15 * MINUTE)

    assert engine._env._race_wins == {"Race": [("Approve", 1800), ("Approve", 1800), ("Timeout", 7200)]}
    # item-0 and item-2 are on the shelf. item-3 has raced for 5 minutes: all three wins took longer. item-3 and
    # item-4, item-5, and item-6 ahead are each approved with chance 2/3 after 30 minutes: the expected count
    # reaches 3 with item-4 (packed by 14:10) and 4 with item-5
    assert slots == [START + HOUR + 10 * MINUTE, START + 3 * HOUR + 10 * MINUTE, START + 5 * HOUR + 10 * MINUTE,
                     START + 6 * HOUR + 10 * MINUTE]


def _race_ways(engine, p_case, now):
    """Where the case p_case, racing at now, goes on after its race: (element, time, chance) of each branch."""
    env = engine._env
    race = env._races[(p_case, "Race")]
    return env._race_ways(p_case, "Race", race.started, {"Shelf"}, now, {}, race)


def test_an_item_in_a_race_counts_only_the_wins_that_took_longer_than_its_wait():
    # item-1 is approved after 20 minutes (10:30), item-0 times out (11:10), item-2 is approved after 40 minutes
    # (11:50); at 13:00 item-3 has raced for 50 minutes: neither approval counts, only the timeout, certain at
    # item-3's own 2-hour mark
    engine = _warehouse(4)
    _approve(engine, "item-1", START + HOUR + 30 * MINUTE)
    _approve(engine, "item-2", START + 2 * HOUR + 50 * MINUTE)

    assert _slots(engine, START + 4 * HOUR) == [START + 2 * HOUR, START + 3 * HOUR + 20 * MINUTE]
    assert engine._env._race_wins == {"Race": [("Approve", 1200), ("Timeout", 7200), ("Approve", 2400)]}
    assert _race_ways(engine, 3, START + 4 * HOUR) == [("Scrap", START + 5 * HOUR + 10 * MINUTE, 1.0)]


def test_a_timer_counts_at_the_items_own_known_time(tmp_path):
    # the same with a timeout drawn from 1.5 to 2.5 hours: item-3's drawn time, not item-0's
    with open(f"{ASSETS}/race_approval_or_timeout.json") as file:
        settings = json.load(file)
    settings["event_distribution"][0].update(
        distribution_name="uniform", distribution_params=[{"value": 5400}, {"value": 9000}])
    (tmp_path / "warehouse.json").write_text(json.dumps(settings))
    engine = _warehouse(4, tmp_path / "warehouse.json")
    _approve(engine, "item-1", START + HOUR + 30 * MINUTE)
    _approve(engine, "item-2", START + 2 * HOUR + 50 * MINUTE)
    _until(engine, START + 4 * HOUR)

    [timeout] = [seconds for winner, seconds in engine._env._race_wins["Race"] if winner == "Timeout"]
    [due] = [event.armed_event.enabled_datetime + timedelta(seconds=event.duration_sec)
             for event in engine._env.events_queue.enabled_events.entry_finder if event.p_case == 3]
    assert due != START + 3 * HOUR + 10 * MINUTE + timedelta(seconds=timeout)
    assert _race_ways(engine, 3, START + 4 * HOUR) == [("Scrap", due, 1.0)]


def test_an_item_in_a_race_longer_than_every_win_is_approved_now():
    # item-0 is approved after 20 minutes, item-1 after 40; at 12:00 item-2 has raced for 50 minutes
    engine = _warehouse(3)
    _approve(engine, "item-0", START + 30 * MINUTE)
    _approve(engine, "item-1", START + HOUR + 50 * MINUTE)

    assert _slots(engine, START + 3 * HOUR) == [START + HOUR, START + 2 * HOUR + 20 * MINUTE,
                                                START + 3 * HOUR + 30 * MINUTE]
    assert _race_ways(engine, 2, START + 3 * HOUR) == [("Pack", START + 3 * HOUR, 1.0)]


def test_an_item_in_a_race_with_no_history_is_approved_now():
    # at 09:30 item-0 has raced for 20 minutes, and no race has been won yet
    engine = _warehouse(1)

    assert _slots(engine, START + 30 * MINUTE) == [START + HOUR]
