"""
Fan-out (docs/messaging.md): a count on a publish entry publishes N messages at once, one per object, each
with its index 1..N.
"""
import json
from datetime import datetime

import pytest
import pytz

from prosimos.exceptions import InvalidSimScenarioException
from prosimos.orchestrator import ProcessSpec, ProsimosEngine, run_engines
from prosimos.simulation_setup import SimDiffSetup

ASSETS = "testing_scripts/assets/messaging"
# an order arrives every hour from 09:00; after Take order (10 minutes) it publishes one ItemOrdered per item
# (items = 3), so order 0 publishes at 09:10
SALES = f"{ASSETS}/orders_with_items.bpmn"
SALES_JSON = f"{ASSETS}/orders_with_items.json"
# every ItemOrdered starts a picking case, which copies case_id into order_id and index into item
PICKING = f"{ASSETS}/picking.bpmn"
PICKING_JSON = f"{ASSETS}/picking.json"
START = pytz.utc.localize(datetime(2024, 1, 1, 9))
TEN_PAST_NINE = START.replace(minute=10)


class _Log:
    def __init__(self):
        self.rows = []

    def writerow(self, header):
        pass

    def writerows(self, rows):
        self.rows.extend(rows)


def _settings(tmp_path, change, base=SALES_JSON):
    with open(base) as file:
        settings = json.load(file)
    change(settings)
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(settings))
    return str(path)


def _items(value):
    def change(settings):
        settings["case_attributes"][0]["values"]["distribution_params"] = [{"value": value}]
    return change


def _run(sales_json=SALES_JSON, orders=1):
    sales = ProsimosEngine(ProcessSpec("Sales", SALES, sales_json, orders), START, _Log(), seed=1)
    picking_log = _Log()
    picking = ProsimosEngine(ProcessSpec("Picking", PICKING, PICKING_JSON, None), START, picking_log, seed=1)
    report = run_engines({"Sales": sales, "Picking": picking}, None, 1)
    return report, picking, picking_log


def _published(report):
    return [(message.type, message.attributes, message.time) for message in report.published]


def test_an_order_with_3_items_publishes_three_item_ordered_with_index_1_2_3_at_the_same_time():
    report, _, _ = _run()

    assert _published(report) == [("ItemOrdered", {"case_id": "Sales-0", "index": index}, TEN_PAST_NINE)
                                  for index in (1, 2, 3)]


def test_a_message_started_picking_process_starts_three_cases_each_with_its_own_index():
    report, picking, picking_log = _run()

    assert [(row[0], row[1]) for row in picking_log.rows] == [(0, "Pick item"), (1, "Pick item"), (2, "Pick item")]
    case_values = picking._env.sim_setup.bpmn_graph.all_attributes
    assert [(case_values[case]["order_id"], case_values[case]["item"]) for case in (0, 1, 2)] == [
        ("Sales-0", 1), ("Sales-0", 2), ("Sales-0", 3)]
    assert report.stalled == []


def test_items_0_publishes_none(tmp_path):
    report, _, picking_log = _run(_settings(tmp_path, _items(0)))

    assert _published(report) == []
    assert picking_log.rows == []


def test_a_fixed_count_publishes_that_many(tmp_path):
    def fixed_2(settings):
        settings["messages"]["publish"][0]["count"] = {"value": 2}

    report, _, _ = _run(_settings(tmp_path, fixed_2))

    assert [attributes["index"] for _, attributes, _ in _published(report)] == [1, 2]


def test_a_count_that_is_not_a_whole_number_publishes_one_message_with_one_warning(tmp_path):
    report, _, _ = _run(_settings(tmp_path, _items(2.5)), orders=2)

    assert _published(report) == [
        ("ItemOrdered", {"case_id": "Sales-0", "index": 1}, TEN_PAST_NINE),
        ("ItemOrdered", {"case_id": "Sales-1", "index": 1}, START.replace(hour=10, minute=10))]
    assert report.engine_warnings == [  # once per event and attribute, though both orders had 2.5
        ("Sales", "case 0 passes Throw_Items with no valid items (got 2.5); it publishes one ItemOrdered message")]


def _load(json_path, bpmn_path=SALES):
    return SimDiffSetup(bpmn_path, json_path, False, 1 if bpmn_path == SALES else None, START)


def test_index_on_an_entry_without_count_is_rejected(tmp_path):
    def no_count(settings):
        del settings["messages"]["publish"][0]["count"]

    with pytest.raises(InvalidSimScenarioException,
                       match=r"messages.publish\[0\]: 'index' is the message's number among the ones count publishes, "
                             r"so it needs a 'count' on the same entry"):
        _load(_settings(tmp_path, no_count))


@pytest.mark.parametrize("bpmn_path, base, where", [
    (PICKING, PICKING_JSON, "the start event"),
    (f"{ASSETS}/orders_and_items.bpmn", f"{ASSETS}/orders_and_items.json", "a catch event"),
])
def test_count_on_a_start_or_catch_event_is_rejected(tmp_path, bpmn_path, base, where):
    def counted(settings):
        settings["messages"]["consume"][0]["count"] = {"value": 2}

    with pytest.raises(InvalidSimScenarioException,
                       match=r"messages.consume\[0\]: count is only for publish entries, on throw and end events: "
                             r"only they publish, and a start event starts exactly one case per message"):
        SimDiffSetup(bpmn_path, _settings(tmp_path, counted, base=base), False,
                     None if bpmn_path == PICKING else 1, START)


@pytest.mark.parametrize("count, message", [
    ({"value": -1}, r"'count' value must be a whole number of at least 0, got -1"),
    ({"value": 1.5}, r"'count' value must be a whole number of at least 0, got 1.5"),
    ({"case_attribute": "weight"}, r"'count' case_attribute 'weight' is not a declared case, global or event attribute"),
    ({"value": 2, "case_attribute": "items"}, r"'count' must be either"),
])
def test_an_invalid_count_is_rejected(tmp_path, count, message):
    def invalid(settings):
        settings["messages"]["publish"][0]["count"] = count

    with pytest.raises(InvalidSimScenarioException, match=r"messages.publish\[0\]: " + message):
        _load(_settings(tmp_path, invalid))
