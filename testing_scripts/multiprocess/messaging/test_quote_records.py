"""
Quotes in the message records and the OCEL output (docs/orchestrator.md, "Output"): a reply has the orchestrator as
publisher and the case that caught it as claimer; an item that took a line claims the quote (or modify) message
that reserved it, so the OCEL links it with the order like a regular message; and the automatic cancel of an ended
case attaches where the case ended, not at the cancel's own event, which the case never passed.
"""
import json
from datetime import datetime, timedelta

import pytest
import pytz

from prosimos.orchestration.config import Organization, ProcessSpec, SimulationConfig
from prosimos.orchestration.protocol import ORCHESTRATOR, CaseElement
from prosimos.orchestration.simulation import run_multiprocess_simulation

START = pytz.utc.localize(datetime(2024, 1, 1, 9))
ASSETS = "testing_scripts/assets/quotes"
# sales_quote: an order every hour from 09:00 is placed (10 minutes) and quotes one item, catches the reply, and then
# races a Shipped message against giving up after 2 hours; shipped, it is received (10 minutes) and cancels the quote.
# shelf_and_ship: an item every 4 hours waits on the shelf for a line, and is shipped (10 minutes). order-0 takes
# item-0 at 09:10; order-1 gives up at 12:10, before item-1 arrives
SALES_MODEL = f"{ASSETS}/sales_quote.bpmn"


def _run(tmp_path, sales_model=SALES_MODEL):
    """The report of a run of two orders and two items, and its OCEL events as (activity, object, linked objects)."""
    sales = ProcessSpec("Sales", str(sales_model), f"{ASSETS}/sales_quote.json", 2, "order")
    warehouse = ProcessSpec("Warehouse", f"{ASSETS}/shelf_and_ship.bpmn", f"{ASSETS}/shelf_and_ship.json", 2, "item")
    config = SimulationConfig([Organization("Shop", [sales, warehouse])], START, 1)
    report = run_multiprocess_simulation(config, str(tmp_path / "log.csv"), ocel_out_path=str(tmp_path / "ocel.json"))
    with open(tmp_path / "ocel.json") as file:
        events = [(event["type"], event["relationships"][0]["objectId"],
                   [(link["objectId"], link["qualifier"]) for link in event["relationships"][1:]])
                  for event in json.load(file)["events"]]
    return report, events


def _records(report, message_type):
    return [record for record in report.message_records if record.type == message_type]


def test_a_reply_is_published_by_the_orchestrator_and_claimed_by_the_case_that_caught_it(tmp_path):
    report, _ = _run(tmp_path)

    assert [(record.publishers, [(claimer.object_id, claimer.element_id) for claimer in record.claimers])
            for record in _records(report, "QuoteAccepted")] == [
        ([CaseElement(ORCHESTRATOR, "", "")], [("order-0", "Catch_Reply")]),
        ([CaseElement(ORCHESTRATOR, "", "")], [("order-1", "Catch_Reply")])]


def test_an_item_that_took_a_line_claims_the_quote_and_is_linked_with_the_order(tmp_path):
    report, events = _run(tmp_path)

    first_quote = _records(report, "ItemQuote")[0]
    assert [(claimer.process, claimer.object_id, claimer.element_id) for claimer in first_quote.claimers] == [
        ("Warehouse", "item-0", "Shelf")]
    # the quote's own event links the item, and the item's shelf links the order (these elements have no names)
    assert ("Throw_Quote", "order-0", [("item-0", "requests")]) in events
    assert ("Shelf", "item-0", [("order-0", "reserved for")]) in events


@pytest.mark.parametrize("terminate", [False, True])
def test_the_automatic_cancel_attaches_where_the_case_ended(tmp_path, terminate):
    # order-1 ends at End_Gave_Up, where its cancel's event is written; the cancel's own throw event comes after
    # Receive, which order-1 never did
    sales_model = SALES_MODEL
    if terminate:
        with open(SALES_MODEL) as file:
            model = file.read()
        sales_model = tmp_path / "sales.bpmn"
        sales_model.write_text(model.replace('<bpmn:incoming>F8</bpmn:incoming></bpmn:endEvent>',
                                             '<bpmn:incoming>F8</bpmn:incoming><bpmn:terminateEventDefinition/>'
                                             '</bpmn:endEvent>'))
    report, _ = _run(tmp_path, sales_model)

    [_, automatic] = _records(report, "QuoteCanceled")
    [publisher] = automatic.publishers
    assert (publisher.object_id, publisher.element_id, publisher.qualifier) == ("order-1", "End_Gave_Up", "cancels")
    gave_up = START + timedelta(hours=3, minutes=10, microseconds=1)  # a race timer's branch goes on 1 µs late
    assert (publisher.activity, publisher.at, publisher.enabled) == ("End_Gave_Up", gave_up, gave_up)
    assert report.engine_warnings == []
