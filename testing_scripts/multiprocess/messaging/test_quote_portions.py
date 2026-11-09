"""
Portions (docs/messaging.md): when a supplier's portion of a quote moves on, each of its items claims its line with
portion_size, how many items the portion holds there, which a fan-in can wait for to send one message per portion;
the requester learns how many portions to expect from the accepted reply's portions.
"""
from datetime import datetime, timedelta

import pytz

from prosimos.orchestration.config import Organization, ProcessSpec, SimulationConfig
from prosimos.orchestration.protocol import Line, Message, Quote, QuoteKind, Verdict
from prosimos.orchestration.simulation import run_multiprocess_simulation
from testing_scripts.multiprocess.scripted_engine import ScriptedEngine

START = pytz.utc.localize(datetime(2024, 1, 1, 9))
MINUTE = timedelta(minutes=1)
ASSETS = "testing_scripts/assets/quotes"
# sales_portions: an order is placed (10 minutes) and quotes 3 items, copies portions from the accepted reply, and
# collects that many Shipments. portion_warehouse: an item waits on the shelf for a line, copying order_id and
# portion_size, and is shipped (10 minutes); a fan-in of portion_size items with the same order_id sends one
# Shipment. Tartu (tartu.json) has an item at 09:00 and one at 09:40, Tallinn (tallinn.json) one at 09:00
GROUPS = {"Sales": ["Sales"], "Warehouses": ["Tartu", "Tallinn"]}


def _warehouse(name, items):
    return ProcessSpec(name, f"{ASSETS}/portion_warehouse.bpmn", f"{ASSETS}/{name.lower()}.json", items, "item")


def _portion_sizes(report, process):
    """Item -> the portion_size it claimed its line with."""
    return {case.object_id: value for name, case in report.objects if name == process
            for attribute, value, _ in case.attributes if attribute == "portion_size"}


def test_one_shipment_per_portion_and_the_order_collects_as_many_as_its_portions():
    # lines at 09:00 Tartu, 09:00 Tallinn, and 09:40 Tartu: Tallinn's portion moves on at once, Tartu's at 09:40
    sales = ProcessSpec("Sales", f"{ASSETS}/sales_portions.bpmn", f"{ASSETS}/sales_portions.json", 1, "order")
    report = run_multiprocess_simulation(SimulationConfig(
        [Organization("Shop", [sales, _warehouse("Tartu", 2), _warehouse("Tallinn", 1)])], START, 1, GROUPS))

    [accepted] = [message for message in report.published if message.type == "QuoteAccepted"]
    assert accepted.attributes["portions"] == 2
    assert _portion_sizes(report, "Tartu") == {"item-0": 2, "item-1": 2}
    assert _portion_sizes(report, "Tallinn") == {"item-0": 1}
    shipments = [message for message in report.published if message.type == "Shipment"]
    assert [(message.source, message.time, message.attributes["item_ids"]) for message in shipments] == [
        ("Tallinn", START + 20 * MINUTE, ["item-0"]), ("Tartu", START + 50 * MINUTE, ["item-0", "item-1"])]
    # the order collects both and goes on to cancel its quote, delivered in full
    assert [message_id for message_id, process, _ in report.claims if process == "Sales"] == [
        accepted.id, *(message.id for message in shipments)]
    assert [message.time for message in report.published if message.type == "QuoteCanceled"] == [
        START + 50 * MINUTE]


def test_a_modify_that_adds_a_line_at_the_same_supplier_gives_the_new_size():
    # Tartu alone, with items at 09:00, 09:40, and 10:20: the quote at 09:10 takes the first two; at 09:20 a modify
    # asks a third line, which item-2 takes at 10:20, and the three move on with portion_size 3
    def asked(kind, lines):
        return Message("ItemQuote" if kind is QuoteKind.QUOTE else "QuoteModified", {"object_id": "order-7"},
                       quote=Quote(kind, "ItemQuote", "order-7", [Line({"object_id": "order-7"}) for _ in range(lines)],
                                   timedelta(days=3), "QuoteAccepted", "QuoteDeclined"))

    sales = ScriptedEngine("Sales")
    for reply in ("QuoteAccepted", "QuoteDeclined", "Shipment"):
        sales.consume(reply, lambda message, now: Verdict.CLAIMED)
    sales.at(START + 10 * MINUTE, lambda now: [asked(QuoteKind.QUOTE, 2)])
    sales.at(START + 20 * MINUTE, lambda now: [asked(QuoteKind.MODIFY, 3)])
    report = run_multiprocess_simulation(SimulationConfig(
        [Organization("Shop", [_warehouse("Tartu", 3)])], START, 1, {"Sales": ["Sales"], "Warehouses": ["Tartu"]},
        extra_processes=("Sales",)), extra_engines={"Sales": sales})

    assert [message.type for message in report.published if message.source == "orchestrator"] == [
        "QuoteAccepted", "QuoteAccepted"]
    assert _portion_sizes(report, "Tartu") == {"item-0": 3, "item-1": 3, "item-2": 3}
    assert [(message.time, message.attributes["item_ids"]) for message in report.published
            if message.type == "Shipment"] == [(START + 90 * MINUTE, ["item-0", "item-1", "item-2"])]
