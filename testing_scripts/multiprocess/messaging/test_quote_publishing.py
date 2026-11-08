"""
A Prosimos requester publishes quotes (docs/messaging.md): a case passing a throw event with a quote entry publishes
a message with a Quote, its lines read from the case and its limit; the modify and cancel thrown at their own events
carry theirs. A scripted warehouse answers, and the requester catches the replies.
"""
import json
from datetime import datetime, timedelta

import pytest
import pytz

from prosimos.orchestration.catalogue.model import CatalogueObject, Reference
from prosimos.orchestration.config import Organization, ProcessSpec, SimulationConfig
from prosimos.orchestration.protocol import QuoteKind
from prosimos.orchestration.simulation import run_multiprocess_simulation
from testing_scripts.multiprocess.scripted_engine import ScriptedEngine

START = pytz.utc.localize(datetime(2024, 1, 1, 9))
# Sales: place the order, ask the quote, wait for its acceptance, modify it, wait again, cancel it
FLOW = [("startEvent", "Start", None), ("task", "Place_Order", None), ("intermediateThrowEvent", "Throw_Quote", "m"),
        ("intermediateCatchEvent", "Catch_Accepted", "m"), ("intermediateThrowEvent", "Throw_Modify", "m"),
        ("intermediateCatchEvent", "Catch_Modified", "m"), ("intermediateThrowEvent", "Throw_Cancel", "m"),
        ("endEvent", "End", None)]
BY_REQUEST = [[{"attribute": "request", "comparison": "=", "object_attribute": "object_id"}]]
QUOTE = {"quantity": {"list": "products", "as": "product"}, "within": {"days": 3}, "accepted": "QuoteAccepted",
         "declined": "QuoteDeclined", "cancel": "QuoteCanceled", "modify": "QuoteModified"}


def _model(tmp_path):
    elements = []
    for index, (kind, element_id, definition) in enumerate(FLOW):
        arrows = (f'<bpmn:incoming>Flow_{index}</bpmn:incoming>' if index else "") + \
                 (f'<bpmn:outgoing>Flow_{index + 1}</bpmn:outgoing>' if index < len(FLOW) - 1 else "")
        message = "<bpmn:messageEventDefinition/>" if definition else ""
        elements.append(f'<bpmn:{kind} id="{element_id}">{arrows}{message}</bpmn:{kind}>')
    flows = "".join(f'<bpmn:sequenceFlow id="Flow_{index + 1}" sourceRef="{FLOW[index][1]}" '
                    f'targetRef="{FLOW[index + 1][1]}"/>' for index in range(len(FLOW) - 1))
    path = tmp_path / "sales.bpmn"
    path.write_text('<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">'
                    f'<bpmn:process id="Sales">{"".join(elements)}{flows}</bpmn:process></bpmn:definitions>')
    return str(path)


def _settings(tmp_path, quote, drawn=()):
    always = [{"from": "MONDAY", "to": "SUNDAY", "beginTime": "00:00:00", "endTime": "23:59:59"}]
    settings = {
        "resource_profiles": [{"id": "Clerks", "name": "Clerks", "resource_list": [
            {"id": "Clerk", "name": "Clerk", "cost_per_hour": 1, "amount": 1, "calendar": "Always",
             "assigned_tasks": ["Place_Order"]}]}],
        "arrival_time_distribution": {"distribution_name": "fix", "distribution_params": [{"value": 3600}]},
        "arrival_time_calendar": always, "gateway_branching_probabilities": [],
        "task_resource_distribution": [{"task_id": "Place_Order", "resources": [
            {"resource_id": "Clerk", "distribution_name": "fix", "distribution_params": [{"value": 600}]}]}],
        "resource_calendars": [{"id": "Always", "name": "Always", "time_periods": always}],
        "object_attributes": [{"name": name, "type": "continuous", "values": {
            "distribution_name": "fix", "distribution_params": [{"value": value}]}} for name, value in drawn],
        "messages": {
            "publish": [{"event_id": "Throw_Quote", "type": "ItemQuote", "attributes": ["object_id"], "quote": quote},
                        {"event_id": "Throw_Modify", "type": "QuoteModified", "attributes": ["object_id"]},
                        {"event_id": "Throw_Cancel", "type": "QuoteCanceled", "attributes": ["object_id"]}],
            "consume": [{"event_id": event, "type": "QuoteAccepted", "condition": BY_REQUEST}
                        for event in ("Catch_Accepted", "Catch_Modified")]}}
    path = tmp_path / "sales.json"
    path.write_text(json.dumps(settings))
    return str(path)


def _run(tmp_path, quote=QUOTE, drawn=()):
    """One order, whose two products are picked from an iPhone and an iPad, against a warehouse that has three of
    each, one, two, and three hours after it is asked. The report and the warehouse."""
    products = Reference("products", "product", pick="uniform", quantity=2)
    sales = ProcessSpec("Sales", _model(tmp_path), _settings(tmp_path, quote, drawn), 1, "order",
                        references=(products,))
    shop = Organization("Shop", [sales], {"product": [CatalogueObject("iPhone"), CatalogueObject("iPad")]})
    warehouse = ScriptedEngine("Warehouse")
    warehouse.slots("ItemQuote", lambda line, now: [now + timedelta(hours=hours) for hours in (1, 2, 3)])
    config = SimulationConfig([shop], START, 1, extra_processes=["Warehouse"])
    return run_multiprocess_simulation(config, None, {"Warehouse": warehouse}), warehouse


def _picked(report):
    order = next(case for process, case in report.objects if process == "Sales")
    return next(value for name, value, _ in order.attributes if name == "products")


def test_a_quote_carries_one_line_per_list_entry_and_its_limit(tmp_path):
    report, warehouse = _run(tmp_path)

    [(asked, now)] = warehouse.estimated
    assert (asked.type, asked.source, asked.time) == ("ItemQuote", "Sales", START + timedelta(minutes=10))
    quote = asked.quote
    assert (quote.kind, quote.quote_type, quote.request, quote.limit) == (QuoteKind.QUOTE, "ItemQuote", "order-0",
                                                                           timedelta(days=3))
    assert (quote.accepted, quote.declined) == ("QuoteAccepted", "QuoteDeclined")
    assert [line.attributes for line in quote.lines] == [{"object_id": "order-0", "product": product}
                                                         for product in _picked(report)]


def test_the_requester_catches_its_replies_and_modifies_and_cancels(tmp_path):
    report, warehouse = _run(tmp_path)

    by_type = {message.type: message for message in report.published}
    # the modify reads the lines again from the case; the cancel names the quote only
    modify, cancel = by_type["QuoteModified"].quote, by_type["QuoteCanceled"].quote
    assert (modify.kind, modify.quote_type, modify.request) == (QuoteKind.MODIFY, "ItemQuote", "order-0")
    assert [line.attributes for line in modify.lines] == [line.attributes for line in by_type["ItemQuote"].quote.lines]
    assert (cancel.kind, cancel.quote_type, cancel.request, cancel.lines) == (QuoteKind.CANCEL, "ItemQuote",
                                                                              "order-0", [])
    replies = [message.id for message in report.published if message.source == "orchestrator"]
    assert [message_id for message_id, process, _ in report.claims if process == "Sales"] == replies
    assert report.stalled == []
    # the cancel released both lines, at the warehouse that held them
    assert [(message.id, line_ids) for message, line_ids, _ in warehouse.released] == [("m1", ["l1", "l2"])]


def test_a_fixed_or_drawn_number_of_lines_carries_the_entrys_attributes(tmp_path):
    report, warehouse = _run(tmp_path, {**QUOTE, "quantity": {"object_attribute": "items"}}, drawn=[("items", 3)])

    [(asked, _)] = warehouse.estimated
    assert [line.attributes for line in asked.quote.lines] == [{"object_id": "order-0"}] * 3


@pytest.mark.parametrize("patience, limit, warnings", [
    (2, timedelta(days=2), []),
    (0, None, ["case 0 passes Throw_Quote with no valid patience (got 0.0); its ItemQuote has no limit"])])
def test_a_limit_read_from_the_case_is_a_number_of_its_unit(tmp_path, patience, limit, warnings):
    report, warehouse = _run(tmp_path, {**QUOTE, "within": {"days": {"object_attribute": "patience"}}},
                             drawn=[("patience", patience)])

    [(asked, _)] = warehouse.estimated
    assert asked.quote.limit == limit
    assert [warning for _, warning in report.engine_warnings] == warnings


def test_a_drawn_number_that_isnt_whole_asks_one_line_with_a_warning(tmp_path):
    report, warehouse = _run(tmp_path, {**QUOTE, "quantity": {"object_attribute": "items"}}, drawn=[("items", 2.5)])

    [(asked, _)] = warehouse.estimated
    assert [line.attributes for line in asked.quote.lines] == [{"object_id": "order-0"}]
    # once per event: the modify reads the lines again
    assert [warning for _, warning in report.engine_warnings] == [
        f"case 0 passes {event} with no valid items (got 2.5); its ItemQuote asks one line"
        for event in ("Throw_Quote", "Throw_Modify")]
