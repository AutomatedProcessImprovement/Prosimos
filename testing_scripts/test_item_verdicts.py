"""
Returning items without a new feature (docs/messaging.md, "One verdict per item"): each item is its own
case waiting for its order's verdict on it, and the order sends one verdict per item with count.
"""
import json
from collections import Counter
from datetime import datetime, timedelta

import pytz

from prosimos.orchestrator import ProcessSpec, ProsimosEngine, run_engines

ASSETS = "testing_scripts/assets/messaging"
# Sales: an order (one a day from 09:00, Place order 10 minutes) publishes one ItemOrdered per item (3), then
# races collecting an ItemReady per item against 2 days; it then publishes Shipped or Canceled per item
SALES = f"{ASSETS}/orders_with_verdicts.bpmn"
SALES_JSON = f"{ASSETS}/orders_with_verdicts.json"
# Picking: every ItemOrdered starts an item case, picked by one picker (1 hour each), which publishes
# ItemReady and races Shipped against Canceled for its own order and index; Canceled returns it to stock
PICKING = f"{ASSETS}/items_awaiting_verdict.bpmn"
PICKING_JSON = f"{ASSETS}/items_awaiting_verdict.json"
START = pytz.utc.localize(datetime(2024, 1, 1, 9))
# order 0 reaches its race at 09:10 on day 1; a race timer's branch continues one microsecond after its time
DEADLINE = START.replace(day=3, minute=10) + timedelta(microseconds=1)


class _Log:
    def __init__(self):
        self.rows = []

    def writerow(self, header):
        pass

    def writerows(self, rows):
        self.rows.extend(rows)


def _time(logged):
    return logged if isinstance(logged, datetime) else datetime.fromisoformat(logged)


def _picking_in(tmp_path, hours):
    """The picking settings with each item taking the given hours to pick."""
    with open(PICKING_JSON) as file:
        settings = json.load(file)
    settings["task_resource_distribution"][0]["resources"][0]["distribution_params"] = [{"value": hours * 3600}]
    path = tmp_path / "picking.json"
    path.write_text(json.dumps(settings))
    return str(path)


def _run(orders=1, picking_json=PICKING_JSON):
    sales = ProsimosEngine(ProcessSpec("Sales", SALES, SALES_JSON, orders), START, _Log(), seed=1)
    picking_log = _Log()
    picking = ProsimosEngine(ProcessSpec("Picking", PICKING, picking_json, None), START, picking_log, seed=1)
    report = run_engines({"Sales": sales, "Picking": picking}, None, 1)
    return report, picking, picking_log


def _item(picking, case):
    """(order, index) of a picking case."""
    values = picking._env.sim_setup.bpmn_graph.all_attributes[case]
    return values["order_id"], values["index"]


def _times(picking, log, activity, column):
    """(order, index) -> the activity's enable (column 2) or end (column 4) time."""
    return {_item(picking, row[0]): _time(row[column]) for row in log.rows if row[1] == activity}


def _verdicts_claimed(report):
    """(verdict type, order, index) -> how many times it was claimed."""
    messages = {message.id: message for message in report.published}
    return Counter((messages[message_id].type, messages[message_id].attributes["case_id"],
                    messages[message_id].attributes["index"])
                   for message_id, process, _ in report.claims if messages[message_id].type in ("Shipped", "Cancelled"))


def test_an_order_cancelled_while_2_of_3_items_are_ready_returns_those_2_at_once_and_the_third_once_ready(tmp_path):
    # one picker taking 20 hours an item: items 1 and 2 are ready before the 2-day deadline, item 3 only after it
    report, picking, log = _run(picking_json=_picking_in(tmp_path, 20))

    ready = _times(picking, log, "Pick item", 4)
    assert ready[("Sales-0", 2)] < DEADLINE < ready[("Sales-0", 3)]
    assert [message.time for message in report.published if message.type == "Cancelled"] == [DEADLINE] * 3
    # items 1 and 2 waited in their race and are returned at once; item 3's Canceled stayed pending in the
    # pool until it reached its race
    assert _times(picking, log, "Return to stock", 2) == {
        ("Sales-0", 1): DEADLINE, ("Sales-0", 2): DEADLINE, ("Sales-0", 3): ready[("Sales-0", 3)]}
    assert _verdicts_claimed(report) == Counter({("Cancelled", "Sales-0", index): 1 for index in (1, 2, 3)})
    assert report.stalled == []
    assert report.unclaimed == []  # no verdict is left in the pool
    # item 3's ItemReady, sent after the order ended, finds no waiting order: discarded, as expected
    messages = {message.id: message for message in report.published}
    assert [(messages[message_id].type, time) for message_id, _, time in report.discards] == [
        ("ItemReady", ready[("Sales-0", 3)])]


def test_an_order_cancelled_after_collecting_1_of_3_items_returns_it_at_once_and_the_2_late_ones_once_ready(tmp_path):
    # one picker taking 30 hours an item: only item 1 is ready before the 2-day deadline, items 2 and 3 after it
    report, picking, log = _run(picking_json=_picking_in(tmp_path, 30))

    ready = _times(picking, log, "Pick item", 4)
    assert ready[("Sales-0", 1)] < DEADLINE < ready[("Sales-0", 2)] < ready[("Sales-0", 3)]
    messages = {message.id: message for message in report.published}
    # the order collected only item 1's ItemReady before its timer won
    assert [messages[message_id].type for message_id, process, _ in report.claims if process == "Sales"] == [
        "ItemReady"]
    assert [message.time for message in report.published if message.type == "Cancelled"] == [DEADLINE] * 3
    # item 1 waited in its race and is returned at once; items 2 and 3 each find their Canceled pending in the
    # pool and claim it as soon as they reach their race
    assert _times(picking, log, "Return to stock", 2) == {
        ("Sales-0", 1): DEADLINE, ("Sales-0", 2): ready[("Sales-0", 2)], ("Sales-0", 3): ready[("Sales-0", 3)]}
    assert _verdicts_claimed(report) == Counter({("Cancelled", "Sales-0", index): 1 for index in (1, 2, 3)})
    assert report.stalled == []
    assert report.unclaimed == []
    # the ItemReady of items 2 and 3, sent after the order ended, are discarded, as expected
    assert [(messages[message_id].type, time) for message_id, _, time in report.discards] == [
        ("ItemReady", ready[("Sales-0", 2)]), ("ItemReady", ready[("Sales-0", 3)])]


def test_a_shipped_orders_items_all_end_as_done():
    report, picking, log = _run(orders=2)

    assert _times(picking, log, "Return to stock", 2) == {}
    assert _verdicts_claimed(report) == Counter(
        {("Shipped", order, index): 1 for order in ("Sales-0", "Sales-1") for index in (1, 2, 3)})
    assert all(not any(state.tokens.values()) for state in picking._env.all_process_states.values())
    assert report.stalled == []
    assert report.unclaimed == []
    assert report.discards == []
