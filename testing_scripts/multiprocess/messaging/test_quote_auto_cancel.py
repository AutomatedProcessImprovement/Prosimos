"""
The automatic cancel (docs/messaging.md): a case that ends with an open quote publishes the quote's cancel at its
end, as if the model had thrown it, so the lines it reserved are freed.
"""
import json
from datetime import datetime, timedelta

import pytest
import pytz

from prosimos.orchestration.config import Organization, ProcessSpec, SimulationConfig
from prosimos.orchestration.protocol import QuoteKind
from prosimos.orchestration.simulation import run_multiprocess_simulation
from testing_scripts.multiprocess.scripted_engine import ScriptedEngine

START = pytz.utc.localize(datetime(2024, 1, 1, 9))
MINUTE, HOUR = timedelta(minutes=1), timedelta(hours=1)
# Sales places the order (10 minutes) and quotes an iPhone; then it waits for the shipment, and after it cancels the
# quote itself, or gives up after 2 hours and ends with the quote still open
SALES = """<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL"><bpmn:process id="Sales">
<bpmn:startEvent id="Start"><bpmn:outgoing>F1</bpmn:outgoing></bpmn:startEvent>
<bpmn:task id="Place_Order"><bpmn:incoming>F1</bpmn:incoming><bpmn:outgoing>F2</bpmn:outgoing></bpmn:task>
<bpmn:intermediateThrowEvent id="Throw_Quote"><bpmn:incoming>F2</bpmn:incoming><bpmn:outgoing>F3</bpmn:outgoing>
  <bpmn:messageEventDefinition/></bpmn:intermediateThrowEvent>
<bpmn:eventBasedGateway id="Race"><bpmn:incoming>F3</bpmn:incoming><bpmn:outgoing>F4</bpmn:outgoing>
  <bpmn:outgoing>F5</bpmn:outgoing></bpmn:eventBasedGateway>
<bpmn:intermediateCatchEvent id="Catch_Shipped"><bpmn:incoming>F4</bpmn:incoming><bpmn:outgoing>F6</bpmn:outgoing>
  <bpmn:messageEventDefinition/></bpmn:intermediateCatchEvent>
<bpmn:intermediateCatchEvent id="Give_Up"><bpmn:incoming>F5</bpmn:incoming><bpmn:outgoing>F7</bpmn:outgoing>
  <bpmn:timerEventDefinition/></bpmn:intermediateCatchEvent>
<bpmn:intermediateThrowEvent id="Throw_Cancel"><bpmn:incoming>F6</bpmn:incoming><bpmn:outgoing>F8</bpmn:outgoing>
  <bpmn:messageEventDefinition/></bpmn:intermediateThrowEvent>
<bpmn:endEvent id="End_Shipped"><bpmn:incoming>F8</bpmn:incoming></bpmn:endEvent>
<bpmn:endEvent id="End_Gave_Up"><bpmn:incoming>F7</bpmn:incoming></bpmn:endEvent>
<bpmn:sequenceFlow id="F1" sourceRef="Start" targetRef="Place_Order"/>
<bpmn:sequenceFlow id="F2" sourceRef="Place_Order" targetRef="Throw_Quote"/>
<bpmn:sequenceFlow id="F3" sourceRef="Throw_Quote" targetRef="Race"/>
<bpmn:sequenceFlow id="F4" sourceRef="Race" targetRef="Catch_Shipped"/>
<bpmn:sequenceFlow id="F5" sourceRef="Race" targetRef="Give_Up"/>
<bpmn:sequenceFlow id="F6" sourceRef="Catch_Shipped" targetRef="Throw_Cancel"/>
<bpmn:sequenceFlow id="F7" sourceRef="Give_Up" targetRef="End_Gave_Up"/>
<bpmn:sequenceFlow id="F8" sourceRef="Throw_Cancel" targetRef="End_Shipped"/>
</bpmn:process></bpmn:definitions>"""


def _run(tmp_path, shipped_at=None, terminate=False):
    """One order against a scripted warehouse that has every line 3 hours after it is asked, and ships the order
    at shipped_at, if given; with terminate, giving up ends at a terminate end event. The report and the warehouse."""
    model = SALES.replace('<bpmn:endEvent id="End_Gave_Up"><bpmn:incoming>F7</bpmn:incoming></bpmn:endEvent>',
                          '<bpmn:endEvent id="End_Gave_Up"><bpmn:incoming>F7</bpmn:incoming>'
                          '<bpmn:terminateEventDefinition/></bpmn:endEvent>') if terminate else SALES
    (tmp_path / "sales.bpmn").write_text(model)
    always = [{"from": "MONDAY", "to": "SUNDAY", "beginTime": "00:00:00", "endTime": "23:59:59"}]
    (tmp_path / "sales.json").write_text(json.dumps({
        "resource_profiles": [{"id": "Clerks", "name": "Clerks", "resource_list": [
            {"id": "Clerk", "name": "Clerk", "cost_per_hour": 1, "amount": 1, "calendar": "Always",
             "assigned_tasks": ["Place_Order"]}]}],
        "arrival_time_distribution": {"distribution_name": "fix", "distribution_params": [{"value": 3600}]},
        "arrival_time_calendar": always, "gateway_branching_probabilities": [],
        "task_resource_distribution": [{"task_id": "Place_Order", "resources": [
            {"resource_id": "Clerk", "distribution_name": "fix", "distribution_params": [{"value": 600}]}]}],
        "resource_calendars": [{"id": "Always", "name": "Always", "time_periods": always}],
        "event_distribution": [{"event_id": "Give_Up", "distribution_name": "fix",
                                "distribution_params": [{"value": 7200}]}],
        "messages": {
            "publish": [{"event_id": "Throw_Quote", "type": "ItemQuote", "attributes": ["object_id"], "quote": {
                "quantity": 1, "within": {"days": 3}, "accepted": "QuoteAccepted", "declined": "QuoteDeclined",
                "cancel": "QuoteCanceled"}},
                        {"event_id": "Throw_Cancel", "type": "QuoteCanceled", "attributes": ["object_id"]}],
            "consume": [{"event_id": "Catch_Shipped", "type": "Shipped", "condition": [
                [{"attribute": "order_id", "comparison": "=", "object_attribute": "object_id"}]]}]}}))
    sales = ProcessSpec("Sales", str(tmp_path / "sales.bpmn"), str(tmp_path / "sales.json"), 1, "order")
    warehouse = ScriptedEngine("Warehouse")
    warehouse.slots("ItemQuote", lambda line, now: [now + 3 * HOUR])
    if shipped_at is not None:
        warehouse.publish_at(shipped_at, "Shipped", order_id="order-0")
    config = SimulationConfig([Organization("Shop", [sales])], START, 1, extra_processes=["Warehouse"])
    return run_multiprocess_simulation(config, None, {"Warehouse": warehouse}), warehouse


def _cancels(report):
    return [(message.time, message.source, message.quote.kind) for message in report.published
            if message.type == "QuoteCanceled"]


@pytest.mark.parametrize("terminate", [False, True])
def test_an_order_ended_by_a_timer_cancels_its_quote_and_frees_its_lines(tmp_path, terminate):
    # the quote at 09:10 is accepted; the order gives up at 11:10 and ends with it open, at a plain or terminate end
    report, warehouse = _run(tmp_path, terminate=terminate)

    # after a race timer wins, its branch goes on a microsecond later: the order ends then
    gave_up = START + 2 * HOUR + 10 * MINUTE + timedelta(microseconds=1)
    assert _cancels(report) == [(gave_up, "Sales", QuoteKind.CANCEL)]
    assert [(message.id, line_ids, now) for message, line_ids, now in warehouse.released] == [("m1", ["l1"], gave_up)]


def test_an_order_that_canceled_its_quote_itself_isnt_canceled_again(tmp_path):
    # shipped at 10:00, the order throws its own cancel and ends
    report, warehouse = _run(tmp_path, shipped_at=START + HOUR)

    assert _cancels(report) == [(START + HOUR, "Sales", QuoteKind.CANCEL)]
    assert len(warehouse.released) == 1
