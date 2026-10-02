"""The 'messages' section of a process's JSON settings (docs/messaging.md): one test per rule."""
import json
from datetime import datetime

import pytest
import pytz

from prosimos.exceptions import InvalidSimScenarioException
from prosimos.messaging_parser import (ConditionTerm, ConsumePoint, MessagingModel, PublishPoint, declared_attributes,
                                       parse_messages)
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
    "Catch_Merged": '<bpmn:intermediateCatchEvent id="Catch_Merged"><bpmn:messageEventDefinition/></bpmn:intermediateCatchEvent>',
    "End_Merged": '<bpmn:endEvent id="End_Merged"><bpmn:messageEventDefinition/></bpmn:endEvent>',
}
# one arrow into each catch and end event, except two into the *_Merged ones
ARROWS = [("Place_Order", target) for target in ("Catch_Shipment", "Catch_Reminder", "Timer_Cancel", "End_Closed",
                                                 "Catch_Merged", "Catch_Merged", "End_Merged", "End_Merged")]
DECLARED = {"city"}  # attributes declared in the process's JSON settings
SHIPMENT_FOR_THIS_CASE = [[{"attribute": "order_id", "comparison": "=", "case_attribute": "case_id"}]]


@pytest.fixture
def bpmn(tmp_path):
    # not a runnable model: the section is checked against the kinds of events and their incoming arrows
    flows = "".join(f'<bpmn:sequenceFlow id="Flow_{index}" sourceRef="{source}" targetRef="{target}"/>'
                    for index, (source, target) in enumerate(ARROWS))
    path = tmp_path / "sales.bpmn"
    path.write_text('<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">'
                    f'<bpmn:process id="Sales">{"".join(EVENTS.values())}{flows}</bpmn:process></bpmn:definitions>')
    return path


def publish(event_id="Throw_OrderPlaced", type="OrderPlaced", **extra):
    return {"publish": [{"event_id": event_id, "type": type, **extra}]}


def consume(event_id="Catch_Shipment", type="Shipment", **extra):
    return {"consume": [{"event_id": event_id, "type": type, **extra}]}


def rejected(section, bpmn):
    with pytest.raises(InvalidSimScenarioException) as error:
        parse_messages(section, bpmn, DECLARED)
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

    model = parse_messages(section, bpmn, DECLARED)

    assert model.publish == (PublishPoint("Throw_OrderPlaced", "OrderPlaced", ("case_id", "city")),
                             PublishPoint("End_Closed", "OrderClosed", ()))
    assert model.consume[0] == ConsumePoint("Catch_Shipment", "Shipment",
                                            ((ConditionTerm("order_id", "=", case_attribute="case_id"),),))
    assert model.consume[1].condition[1] == (ConditionTerm("weight", "in", value=[0, 10]),)
    assert model.subscriptions() == ["Shipment"]


def test_a_consuming_point_without_a_condition_accepts_every_message(bpmn):
    assert parse_messages(consume(), bpmn, DECLARED).consume[0].condition is None


def test_no_section_gives_an_empty_model(bpmn):
    assert parse_messages(None, bpmn, DECLARED) == MessagingModel()
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


def test_a_process_started_by_messages_has_exactly_one_start_event(bpmn):
    # the test model has a plain start event next to the message start event
    reason = rejected(consume("Start_Message"), bpmn)

    assert "a process started by messages must have exactly one start event; the model has 2 (Start, Start_Message)" in reason


def test_a_start_event_without_a_message_definition_cannot_start_cases(bpmn):
    assert "'Start' is a startEvent, expected an intermediate message catch event or a message start event" \
        in rejected(consume("Start"), bpmn)


@pytest.mark.parametrize("section", [publish(type=""), consume(type="   "), {"publish": [{"event_id": "Throw_OrderPlaced"}]}])
def test_type_must_be_non_empty(bpmn, section):
    assert "'type' must be a non-empty string" in rejected(section, bpmn)


def test_published_attributes_must_be_declared(bpmn):
    reason = rejected(publish(attributes=["case_id", "city", "cty", "zip"]), bpmn)

    assert reason.endswith("messages.publish[0]: 'cty', 'zip' not declared as a case, global or event attribute")


@pytest.mark.parametrize("section", [publish("End_Merged"), consume("Catch_Merged")])
def test_end_and_catch_message_events_need_exactly_one_incoming_arrow(bpmn, section):
    event_id = (section.get("publish") or section["consume"])[0]["event_id"]

    assert f"{event_id!r} has 2 incoming arrows, expected exactly one; draw an explicit gateway before {event_id}" \
        in rejected(section, bpmn)


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


def _load(tmp_path, bpmn_path, messages, **extra_settings):
    """Loads a model as the simulator does, with the test JSON settings plus the given sections."""
    with open("testing_scripts/assets/throw_events/two_tasks.json") as file:
        settings = json.load(file)
    settings.update(messages=messages, **extra_settings)
    json_path = tmp_path / "settings.json"
    json_path.write_text(json.dumps(settings))
    return SimDiffSetup(bpmn_path, json_path, False, 1, pytz.utc.localize(datetime(2024, 1, 1)))


CITY = {"name": "city", "type": "discrete",
        "values": [{"key": "Tartu", "value": 0.5}, {"key": "Tallinn", "value": 0.5}]}


def test_a_model_publishing_an_undeclared_attribute_is_rejected(tmp_path):
    model = "testing_scripts/assets/throw_events/with_message_throw.bpmn"
    published = publish("Throw", attributes=["case_id", "city"])

    assert _load(tmp_path, model, published, case_attributes=[CITY]).messaging.publish[0].attributes == ("case_id", "city")
    with pytest.raises(InvalidSimScenarioException, match="'city' not declared as a case, global or event attribute"):
        _load(tmp_path, model, published)


def test_declared_attributes_are_case_global_and_event_attributes():
    settings = {"case_attributes": [{"name": "city"}], "global_attributes": [{"name": "stock"}],
                "event_attributes": [{"event_id": "Task_A", "attributes": [{"name": "weight"}]}]}

    assert declared_attributes(settings) == {"city", "stock", "weight"}
    assert declared_attributes({}) == set()


@pytest.mark.parametrize("model, messages", [
    ("two_arrows_into_end.bpmn", publish("End")),
    ("two_arrows_into_catch.bpmn", consume("Catch")),
])
def test_a_model_with_two_arrows_into_a_message_end_or_catch_event_is_rejected(tmp_path, model, messages):
    event_id = (messages.get("publish") or messages["consume"])[0]["event_id"]

    with pytest.raises(InvalidSimScenarioException, match=f"draw an explicit gateway before {event_id}$"):
        _load(tmp_path, f"testing_scripts/assets/messaging/{model}", messages)
