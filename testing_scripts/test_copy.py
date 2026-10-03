"""
Copying message values into the case (docs/messaging.md): when a message is claimed, the attributes
listed in the consume entry's copy are written into the case it starts or resumes.
"""
from datetime import datetime

import pytz

from prosimos.orchestrator import Message, ProcessSpec, ProsimosEngine, Verdict

ASSETS = "testing_scripts/assets/messaging"
START = pytz.utc.localize(datetime(2024, 1, 1, 9))
NOON = START.replace(hour=12)


class _Log:
    def __init__(self):
        self.rows = []

    def writerow(self, header):
        self.header = header

    def writerows(self, rows):
        self.rows.extend(rows)


def _engine(name, bpmn, json_path, cases=None):
    log = _Log()
    return ProsimosEngine(ProcessSpec(name, f"{ASSETS}/{bpmn}", f"{ASSETS}/{json_path}", cases), START, log), log


def _run_until_idle(engine):
    published = []
    while engine.next_event_time() is not None:
        published.extend(engine.step())
    return published


def _case_values(engine, case):
    return engine._env.sim_setup.bpmn_graph.all_attributes[case]


def test_a_case_created_by_a_start_message_has_the_copied_attribute():
    # the Tartu warehouse copies the order's case_id into its new case's order_id
    warehouse, _ = _engine("TartuWarehouse", "tartu_warehouse.bpmn", "tartu_warehouse.json")

    warehouse.deliver(Message("OrderPlaced", {"case_id": "Sales-3", "city": "Tartu"}, source="Sales"), START)

    assert _case_values(warehouse, 0)["order_id"] == "Sales-3"
    assert [(m.type, m.attributes) for m in _run_until_idle(warehouse)] == [("Shipment", {"order_id": "Sales-3"})]
    assert warehouse.finish().warnings == []


def test_a_waiting_case_gets_the_copied_value_at_the_claim_time_and_publishes_it_later():
    # Sales waits for Shipment{order_id, tracking_no}, copies tracking_no (drawn as "unknown"),
    # closes the order and publishes OrderClosed{case_id, tracking_no}
    sales, log = _engine("Sales", "sales_closing.bpmn", "sales_closing.json", cases=2)
    _run_until_idle(sales)
    assert [_case_values(sales, case)["tracking_no"] for case in (0, 1)] == ["unknown", "unknown"]

    shipment = Message("Shipment", {"order_id": "Sales-0", "tracking_no": "TRK-1"}, source="Warehouse")
    assert sales.deliver(shipment, NOON) is Verdict.CLAIMED

    # replaced right at the claim, before the case moves on, and only in the claiming case
    assert [_case_values(sales, case)["tracking_no"] for case in (0, 1)] == ["TRK-1", "unknown"]
    published = _run_until_idle(sales)
    assert [(m.type, m.attributes) for m in published] == [
        ("OrderClosed", {"case_id": "Sales-0", "tracking_no": "TRK-1"})]
    # the log's case attribute column shows the value from the claim on
    assert [(row[1], row[-1]) for row in log.rows if row[0] == 0] == [("Take order", "unknown"),
                                                                     ("Close order", "TRK-1")]


def test_a_missing_message_attribute_leaves_the_value_unchanged_with_one_warning():
    sales, _ = _engine("Sales", "sales_closing.bpmn", "sales_closing.json", cases=2)
    _run_until_idle(sales)

    for case_id in ("Sales-0", "Sales-1"):  # neither shipment has a tracking_no
        assert sales.deliver(Message("Shipment", {"order_id": case_id}, source="Warehouse"), NOON) is Verdict.CLAIMED
    published = _run_until_idle(sales)

    assert [_case_values(sales, case)["tracking_no"] for case in (0, 1)] == ["unknown", "unknown"]
    assert [m.attributes["tracking_no"] for m in published] == ["unknown", "unknown"]
    assert sales.finish().warnings == [
        "Shipment message accepted at Catch_Shipment has no tracking_no to copy into tracking_no; "
        "tracking_no is left unchanged"]
