"""
Expected counts over exclusive gateways (docs/messaging.md): each item not yet at the catch event gets its chance to
reach it and its average time to get there, branches weighted by their probability; slot k is when the expected
count there reaches k. A split the item's data decides is exact. A split by probabilities that an item passes after now,
at the end of a task still running, counts as not passed yet, though Prosimos drew its branch when the task started.
"""
import json
from datetime import datetime, timedelta

import pytest
import pytz

from prosimos.orchestration.config import ProcessSpec
from prosimos.orchestration.prosimos_engine import ProsimosEngine
from prosimos.orchestration.protocol import Line, Message, Quote, QuoteKind

START = pytz.utc.localize(datetime(2024, 1, 1, 9))
MINUTE, HOUR = timedelta(minutes=1), timedelta(hours=1)
ASSETS = "testing_scripts/assets/quotes"
# the warehouse makes an item every hour from 09:00 (10 minutes) and checks it: 80% go to the shelf, 20% are fixed
# (1 hour) and then scrapped (check_then_scrap), shelved (check_then_shelve), or checked again (check_again)


def _warehouse(tmp_path, items=5, model="check_then_scrap", graded=False):
    """The warehouse's engine, from the model asset and the shared check settings. graded draws each item's grade,
    A or B, and shelves the A ones only."""
    json_path = f"{ASSETS}/check.json"
    if graded:
        with open(json_path) as file:
            settings = json.load(file)
        for probability, grade in zip(settings["gateway_branching_probabilities"][0]["probabilities"], "AB"):
            probability["condition_id"] = grade
        settings["object_attributes"] = [{"name": "grade", "type": "discrete", "values": [
            {"key": "A", "value": 0.5}, {"key": "B", "value": 0.5}]}]
        settings["branch_rules"] = [
            {"id": grade, "rules": [[{"attribute": "grade", "comparison": "=", "value": grade}]]} for grade in "AB"]
        json_path = tmp_path / "warehouse.json"
        json_path.write_text(json.dumps(settings))
    spec = ProcessSpec("Warehouse", f"{ASSETS}/{model}.bpmn", str(json_path), items, "item")
    return ProsimosEngine(spec, START, seed=1)


def _quote(message_id, request, lines=1):
    """A stamped quote of request from Sales with lines identical lines: m1 asks m1.1, m1.2, ..."""
    return Message("ItemQuote", {"object_id": request}, message_id, "Sales", START, Quote(
        QuoteKind.QUOTE, "ItemQuote", request, [Line({"object_id": request}, f"{message_id}.{number}")
                                                for number in range(1, lines + 1)],
        timedelta(days=3), "QuoteAccepted", "QuoteDeclined"))


def _slots(engine):
    return engine.estimate(_quote("m9", "order-9"), START)["m9.1"]


def made(index):
    """When item index is made: Make_Item from its arrival, for 10 minutes."""
    return START + index * HOUR + 10 * MINUTE


def test_a_20_percent_scrap_gives_the_expected_slots(tmp_path):
    # five items, each 80% sure: the expected count reaches 1 with item-1 (1.6), 2 with item-2 (2.4), and so on
    assert _slots(_warehouse(tmp_path)) == [made(1), made(2), made(3), made(4)]


def test_open_lines_take_the_earliest_expected_items_first(tmp_path):
    # five open lines take an expected count of 5: items 0 to 5 (4.8) and 0.2 of item-6. What is left: 0.6 with
    # item-6, 1.4 with item-7, 2.2 with item-8, 3.0 with item-9
    engine = _warehouse(tmp_path, items=10)
    engine.reserve(_quote("m1", "order-7", lines=5), [f"m1.{number}" for number in range(1, 6)], START)

    assert _slots(engine) == [made(7), made(8), made(9)]


def test_a_branch_on_the_way_is_weighted_by_its_probability(tmp_path):
    # every item reaches the shelf; 20% after an hour of fixing: 12 minutes later on average
    assert _slots(_warehouse(tmp_path, model="check_then_shelve")) == [made(index) + 12 * MINUTE for index in range(5)]


def test_a_loop_back_adds_its_expected_number_of_rounds(tmp_path):
    # a fixed item is checked again: on average 0.25 fixes, 15 minutes
    assert _slots(_warehouse(tmp_path, items=1, model="check_again")) == [made(0) + 15 * MINUTE]


def test_a_split_the_items_data_decides_is_exact(tmp_path):
    engine = _warehouse(tmp_path, graded=True)
    grades = [engine._env.sim_setup.bpmn_graph.all_attributes[p_case]["grade"] for p_case in range(5)]
    assert set(grades) == {"A", "B"}

    assert _slots(engine) == [made(index) for index, grade in enumerate(grades) if grade == "A"]


def _making(tmp_path, seed, until, now=None):
    """The warehouse of three items, 10% of them fixed and scrapped, run with the seed up to until, and its slots at
    now (until by default): Make_Item of item-0 runs from 09:00 to 09:10, and Prosimos drew its check at 09:00."""
    with open(f"{ASSETS}/check.json") as file:
        settings = json.load(file)
    settings["gateway_branching_probabilities"][0]["probabilities"] = [{"path_id": "Good", "value": 0.9},
                                                                       {"path_id": "Bad", "value": 0.1}]
    (tmp_path / "check.json").write_text(json.dumps(settings))
    engine = ProsimosEngine(ProcessSpec("Warehouse", f"{ASSETS}/check_then_scrap.bpmn", str(tmp_path / "check.json"),
                                        3, "item"), START, seed=seed)
    while engine.next_event_time() <= until:
        engine.step()
    queued = {event.task_id for event in engine._env.events_queue.enabled_events.entry_finder if event.p_case == 0}
    return queued, engine.estimate(_quote("m9", "order-9"), now or until)["m9.1"]


@pytest.mark.parametrize("seed, heading_for", [(1, "Shelf"), (19, "Fix")])
def test_an_item_being_made_counts_its_chance_until_its_check(tmp_path, seed, heading_for):
    # at 09:05 item-0 is queued for its drawn branch, but counts 0.9 like the others: 1.8 with item-1, 2.7 with item-2
    queued, slots = _making(tmp_path, seed, START + 5 * MINUTE)

    assert queued == {heading_for}
    assert slots == [made(1), made(2)]


@pytest.mark.parametrize("seed, slots", [(1, [made(0), made(2)]), (19, [made(2)])])
def test_after_its_check_an_item_counts_exactly(tmp_path, seed, slots):
    # at 09:30 item-0 is on the shelf, 1 (1.9 with item-1, 2.8 with item-2), or being fixed for scrap, 0
    assert _making(tmp_path, seed, START + 30 * MINUTE)[1] == slots


def test_a_check_due_now_counts_as_made(tmp_path):
    # asked at 09:10, the moment of item-0's check, before the engine's step at 09:10: item-0 is on its way to the
    # shelf, so it counts 1
    queued, slots = _making(tmp_path, 1, START + 5 * MINUTE, now=made(0))

    assert queued == {"Shelf"} and slots == [made(0), made(2)]
