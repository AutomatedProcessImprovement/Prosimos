"""
Expected counts over exclusive gateways (docs/messaging.md): each item not yet at the catch event gets its chance to
reach it and its average time to get there, branches weighted by their probability; slot k is when the expected
count there reaches k. A split the item's data decides is exact.
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


def test_a_loop_back_is_followed_until_its_chance_is_negligible(tmp_path):
    # a fixed item is checked again: on average 0.25 fixes, 15 minutes; the ways cut off shift it by microseconds
    [slot] = _slots(_warehouse(tmp_path, items=1, model="check_again"))

    assert abs(slot - (made(0) + 15 * MINUTE)) < timedelta(milliseconds=1)


def test_a_split_the_items_data_decides_is_exact(tmp_path):
    engine = _warehouse(tmp_path, graded=True)
    grades = [engine._env.sim_setup.bpmn_graph.all_attributes[p_case]["grade"] for p_case in range(5)]
    assert set(grades) == {"A", "B"}

    assert _slots(engine) == [made(index) for index, grade in enumerate(grades) if grade == "A"]
