"""
Message records (docs/orchestrator.md): each engine records which of its cases and elements published and took
each message, and hands the records over at the end; the orchestrator joins them on the message id.
"""
from datetime import datetime

import pytz

from prosimos.orchestrator import ProcessSpec, ProsimosEngine, SimulationConfig, Verdict, run_engines, run_orchestrator
from testing_scripts.scripted_engine import ScriptedEngine

ASSETS = "testing_scripts/assets/messaging"
START = pytz.utc.localize(datetime(2024, 1, 1, 9))


def _engine(name, model, settings, cases=None):
    return ProsimosEngine(ProcessSpec(name, f"{ASSETS}/{model}.bpmn", settings, cases), START, None, seed=1)


def _records(report, message_type):
    return [record for record in report.message_records if record.type == message_type]


def _where(element):
    """(process, case id, element id) of a publisher or claimer, or None."""
    return None if element is None else (element.process, element.case_id, element.element_id)


def _publishers(records):
    return [_where(record.publisher) for record in records]


def _claimers(records):
    return [[_where(claimer) for claimer in record.claimers] for record in records]


def _verdicts_run():
    """One order with 3 items: Sales publishes ItemOrdered per item (count), each starts an item case in Picking,
    which publishes ItemReady; Sales collects all 3, then sends each item its Shipped, taken at a race branch."""
    sales = _engine("Sales", "orders_with_verdicts", f"{ASSETS}/orders_with_verdicts.json", 1)
    picking = _engine("Picking", "items_awaiting_verdict", f"{ASSETS}/items_awaiting_verdict.json")
    return run_engines({"Sales": sales, "Picking": picking}, None, 1), picking


def test_a_message_start_records_the_new_case_as_claimer():
    report, picking = _verdicts_run()

    records = _records(report, "ItemOrdered")
    assert _publishers(records) == [("Sales", "Sales-0", "Throw_Items")] * 3
    assert _claimers(records) == [[("Picking", f"Picking-{case}", "Start_Item")] for case in (0, 1, 2)]
    # each new case is the item its message announced
    messages = {message.id: message for message in report.published}
    case_values = picking._env.sim_setup.bpmn_graph.all_attributes
    for record in records:
        case = int(record.claimers[0].case_id.split("-")[1])
        assert case_values[case]["index"] == messages[record.message_id].attributes["index"]


def test_collect_3_records_three_messages_with_the_same_claimer():
    report, _ = _verdicts_run()

    records = _records(report, "ItemReady")
    assert _publishers(records) == [("Picking", f"Picking-{case}", "Throw_Ready") for case in (0, 1, 2)]
    assert _claimers(records) == [[("Sales", "Sales-0", "Catch_Items")]] * 3


def test_a_race_branch_records_the_waiting_case_at_its_catch_event():
    report, _ = _verdicts_run()

    records = _records(report, "Shipped")
    assert _publishers(records) == [("Sales", "Sales-0", "Throw_Shipped")] * 3
    messages = {message.id: message for message in report.published}
    # Shipped for item i goes to the item case started by ItemOrdered i, at its race branch
    started_by_index = {messages[r.message_id].attributes["index"]: r.claimers[0].case_id
                        for r in _records(report, "ItemOrdered")}
    assert _claimers(records) == [
        [("Picking", started_by_index[messages[record.message_id].attributes["index"]], "Catch_Shipped")]
        for record in records]


def test_a_catch_event_records_the_waiting_case():
    report = run_orchestrator(SimulationConfig.from_json("testing_scripts/assets/running_example/simulation.json"), None)

    records = _records(report, "Shipment")
    assert records  # with this seed, warehouses ship orders
    for record in records:
        messages = {message.id: message for message in report.published}
        order = messages[record.message_id].attributes["order_id"]
        assert record.publisher.process in ("TartuWarehouse", "TallinnWarehouse")
        assert record.publisher.element_id == "End_Shipped"
        assert _claimers([record]) == [[("Sales", order, "Catch_Shipment")]]


def test_capacity_2_records_both_claimers():
    # an order arrives every 10 minutes from 09:00 and waits for a truck after 1 minute of packing
    orders = _engine("Orders", "orders_and_trucks", f"{ASSETS}/orders_and_trucks.json", 3)
    carrier = ScriptedEngine("Carrier")
    carrier.publish_at(START.replace(minute=30), "Truck", dock="Tartu", case_id="Carrier-0")

    report = run_engines({"Orders": orders, "Carrier": carrier}, None, 1)

    [record] = _records(report, "Truck")
    assert record.publisher is None  # a scripted engine keeps no records
    assert _claimers([record]) == [[("Orders", "Orders-0", "Catch_Truck"), ("Orders", "Orders-1", "Catch_Truck")]]


def test_a_discarded_and_an_unclaimed_message_have_no_claimer():
    # the running example's Sales: each order publishes OrderPlaced; a scripted archive discards order 0's and
    # keeps answering PENDING to order 1's, which stays in the pool until the end
    sales = ProsimosEngine(ProcessSpec("Sales", "testing_scripts/assets/running_example/sales.bpmn",
                                       "testing_scripts/assets/running_example/sales.json", 2), START, None, seed=1)
    archive = ScriptedEngine("Archive")
    archive.consume("OrderPlaced", lambda message, now:
                    Verdict.DISCARDED if message.attributes["case_id"] == "Sales-0" else Verdict.PENDING)

    report = run_engines({"Sales": sales, "Archive": archive}, None, 1)

    discarded, unclaimed = _records(report, "OrderPlaced")
    assert [message_id for message_id, _, _ in report.discards] == [discarded.message_id]
    assert [message.id for _, message in report.unclaimed] == [unclaimed.message_id]
    assert _publishers([discarded, unclaimed]) == [("Sales", "Sales-0", "Throw_OrderPlaced"),
                                                   ("Sales", "Sales-1", "Throw_OrderPlaced")]
    assert discarded.claimers == [] and unclaimed.claimers == []


def test_the_records_are_in_the_report_as_json():
    report, _ = _verdicts_run()

    first = report.to_dict()["message_records"][0]
    # the links attach to row 0 of each log: Sales-0's place order and Picking-0's pick item
    assert first == {"message": report.published[0].id, "type": "ItemOrdered",
                     "publisher": {"process": "Sales", "case_id": "Sales-0", "element_id": "Throw_Items",
                                   "qualifier": "ItemOrdered", "task_row": 0},
                     "claimers": [{"process": "Picking", "case_id": "Picking-0", "element_id": "Start_Item",
                                   "qualifier": "ItemOrdered", "task_row": 0}]}
