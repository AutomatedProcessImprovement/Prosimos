"""The 'messages' section of a process's JSON settings (docs/messaging-model.md): one test per rule."""
import json
from datetime import datetime

import pytest
import pytz

from prosimos.exceptions import InvalidSimScenarioException
from prosimos.messaging_parser import ConditionTerm, ConsumePoint, MessagingModel, PublishPoint, parse_messages
from prosimos.simulation_setup import SimDiffSetup

EVENTS = {
    "Start": '<bpmn:startEvent id="Start"/>',
    "Start_Message": '<bpmn:startEvent id="Start_Message"><bpmn:messageEventDefinition/></bpmn:startEvent>',
    "Place_Order": '<bpmn:task id="Place_Order"/>',
    "Throw_OrderPlaced": '<bpmn:intermediateThrowEvent id="Throw_OrderPlaced"><bpmn:messageEventDefinition/></bpmn:intermediateThrowEvent>',
    "Throw_Plain": '<bpmn:intermediateThrowEvent id="Throw_Plain"/>',
    "Catch_Shipment": '<bpmn:intermediateCatchEvent id="Catch_Shipment"><bpmn:messageEventDefinition/></bpmn:intermediateCatchEvent>',
    "Catch_Reminder": '<bpmn:intermediateCatchEvent id="Catch_Reminder"><bpmn:messageEventDefinition/></bpmn:intermediateCatchEvent>',
    "Timer_Cancel": '<bpmn:intermediateCatchEvent id="Timer_Cancel"><bpmn:timerEventDefinition/></bpmn:intermediateCatchEvent>',
    "End_Closed": '<bpmn:endEvent id="End_Closed"><bpmn:messageEventDefinition/></bpmn:endEvent>',
}
SHIPMENT_FOR_THIS_CASE = [[{"attribute": "order_id", "comparison": "=", "case_attribute": "case_id"}]]


@pytest.fixture
def bpmn(tmp_path):
    # the elements only, no flows: the section is checked against the kinds of events, not the control flow
    path = tmp_path / "sales.bpmn"
    path.write_text('<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">'
                    f'<bpmn:process id="Sales">{"".join(EVENTS.values())}</bpmn:process></bpmn:definitions>')
    return path


def publish(event_id="Throw_OrderPlaced", type="OrderPlaced", **extra):
    return {"publish": [{"event_id": event_id, "type": type, **extra}]}


def consume(event_id="Catch_Shipment", type="Shipment", **extra):
    return {"consume": [{"event_id": event_id, "type": type, **extra}]}


def rejected(section, bpmn):
    with pytest.raises(InvalidSimScenarioException) as error:
        parse_messages(section, bpmn)
    return str(error.value)


def test_a_valid_section_is_parsed(bpmn):
    section = {
        "publish": [
            {"event_id": "Throw_OrderPlaced", "type": "OrderPlaced", "attributes": ["case_id", "city"]},
            {"event_id": "End_Closed", "type": "OrderClosed"},
        ],
        "consume": [
            {"event_id": "Catch_Shipment", "type": "Shipment", "condition": SHIPMENT_FOR_THIS_CASE},
            {"event_id": "Catch_Reminder", "type": "Shipment",
             "condition": [[{"attribute": "source", "comparison": "=", "value": "TartuWarehouse"}],
                           [{"attribute": "weight", "comparison": "in", "value": [0, 10]}]]},
        ],
    }

    model = parse_messages(section, bpmn)

    assert model.publish == (PublishPoint("Throw_OrderPlaced", "OrderPlaced", ("case_id", "city")),
                             PublishPoint("End_Closed", "OrderClosed", ()))
    assert model.consume[0] == ConsumePoint("Catch_Shipment", "Shipment",
                                            ((ConditionTerm("order_id", "=", case_attribute="case_id"),),))
    assert model.consume[1].condition[1] == (ConditionTerm("weight", "in", value=[0, 10]),)
    assert model.subscriptions() == ["Shipment"]


def test_a_consuming_point_without_a_condition_accepts_every_message(bpmn):
    assert parse_messages(consume(), bpmn).consume[0].condition is None


def test_no_section_gives_an_empty_model(bpmn):
    assert parse_messages(None, bpmn) == MessagingModel()
    assert MessagingModel().subscriptions() == []


def test_section_must_only_have_publish_and_consume(bpmn):
    assert "only 'publish' and/or 'consume'" in rejected({"publish": [], "subscribe": []}, bpmn)


def test_event_id_must_exist_in_the_bpmn(bpmn):
    assert "'Throw_Typo' is not an element of the BPMN model" in rejected(publish("Throw_Typo"), bpmn)


@pytest.mark.parametrize("event_id, found", [
    ("Catch_Shipment", "a message intermediateCatchEvent"),
    ("Place_Order", "a task"),
    ("Throw_Plain", "an intermediateThrowEvent"),  # a throw event without a message definition
])
def test_publishing_needs_a_message_throw_or_end_event(bpmn, event_id, found):
    reason = rejected(publish(event_id), bpmn)

    assert f"{event_id!r} is {found}, expected an intermediate message throw event or a message end event" in reason


@pytest.mark.parametrize("event_id, found", [
    ("Throw_OrderPlaced", "a message intermediateThrowEvent"),
    ("End_Closed", "a message endEvent"),
    ("Timer_Cancel", "an intermediateCatchEvent"),  # a timer, not a message
])
def test_consuming_needs_a_message_catch_event(bpmn, event_id, found):
    reason = rejected(consume(event_id), bpmn)

    assert f"{event_id!r} is {found}, expected an intermediate message catch event" in reason


def test_message_start_events_are_not_supported_yet(bpmn):
    assert "'Start_Message' is a message start event, which isn't supported yet" in rejected(consume("Start_Message"), bpmn)


@pytest.mark.parametrize("section", [publish(type=""), consume(type="   "), {"publish": [{"event_id": "Throw_OrderPlaced"}]}])
def test_type_must_be_non_empty(bpmn, section):
    assert "'type' must be a non-empty string" in rejected(section, bpmn)


def test_published_attributes_must_be_names(bpmn):
    assert "'attributes' must be a list of attribute names" in rejected(publish(attributes="city"), bpmn)


@pytest.mark.parametrize("condition", [
    {"attribute": "order_id", "comparison": "=", "value": 1},  # a term, not a list of alternatives
    [{"attribute": "order_id", "comparison": "=", "value": 1}],  # one alternative, not a list of alternatives
    [],
    [[]],
])
def test_condition_must_be_a_list_of_alternatives_of_terms(bpmn, condition):
    assert "condition" in rejected(consume(condition=condition), bpmn)


@pytest.mark.parametrize("term, reason", [
    ({"comparison": "=", "value": 1}, "'attribute' must be a message attribute name or 'source'"),
    ({"attribute": "order_id", "comparison": "==", "value": 1}, "'comparison' must be one of =, !=, <, <=, >, >=, in"),
    ({"attribute": "order_id", "comparison": "="}, "give exactly one of 'value' and 'case_attribute'"),
    ({"attribute": "order_id", "comparison": "=", "value": 1, "case_attribute": "case_id"},
     "give exactly one of 'value' and 'case_attribute'"),
    ({"attribute": "weight", "comparison": "in", "case_attribute": "max_weight"}, "'in' needs a fixed 'value' [low, high]"),
    ({"attribute": "order_id", "comparison": "=", "value": 1, "attr": "x"}, "unknown keys ['attr']"),
])
def test_condition_terms_use_the_branch_rule_format(bpmn, term, reason):
    assert reason in rejected(consume(condition=[[term]]), bpmn)


def test_the_engine_rejects_an_invalid_section_when_it_loads_a_model(tmp_path):
    # timer_with_task has a timer catch event, which can't consume messages
    with open("testing_scripts/assets/timer_with_task.json") as file:
        settings = json.load(file)
    settings["messages"] = consume("Event_056pdi5")
    json_path = tmp_path / "timer_with_task.json"
    json_path.write_text(json.dumps(settings))

    with pytest.raises(InvalidSimScenarioException, match="Event_056pdi5"):
        SimDiffSetup("testing_scripts/assets/timer_with_task.bpmn", json_path, False, 1,
                     pytz.utc.localize(datetime(2024, 1, 1)))
