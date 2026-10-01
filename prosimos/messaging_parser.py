"""
The optional 'messages' section of a process's JSON settings (docs/messaging-model.md). The BPMN
model says where a process publishes or waits (message events); this section says what it
publishes or accepts. Parsing only: nothing here publishes or consumes a message yet.
"""
import json
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from prosimos.exceptions import InvalidSimScenarioException

BPMN_NS = "{http://www.omg.org/spec/BPMN/20100524/MODEL}"
MESSAGE_DEFINITION = BPMN_NS + "messageEventDefinition"
PUBLISHING_EVENTS = {"intermediateThrowEvent": "an intermediate message throw event", "endEvent": "a message end event"}
CONSUMING_EVENTS = {"intermediateCatchEvent": "an intermediate message catch event"}
COMPARISONS = ("=", "!=", "<", "<=", ">", ">=", "in")  # the ones branch rules understand
TERM_KEYS = {"attribute", "comparison", "value", "case_attribute"}


@dataclass(frozen=True)
class ConditionTerm:
    """Compares a message attribute (or 'source') with a fixed value or with a case attribute."""

    attribute: str
    comparison: str
    value: Any = None
    case_attribute: Optional[str] = None


@dataclass(frozen=True)
class PublishPoint:
    event_id: str
    type: str
    attributes: Tuple[str, ...]  # copied from the case's current values; 'case_id' is the case identifier


@dataclass(frozen=True)
class ConsumePoint:
    event_id: str
    type: str
    condition: Optional[Tuple[Tuple[ConditionTerm, ...], ...]]  # any alternative whose terms all hold; None accepts every message


@dataclass(frozen=True)
class MessagingModel:
    publish: Tuple[PublishPoint, ...] = ()
    consume: Tuple[ConsumePoint, ...] = ()

    def subscriptions(self) -> List[str]:
        return sorted({point.type for point in self.consume})


def parse_messages_file(json_path, bpmn_path) -> MessagingModel:
    with open(json_path) as json_file:
        settings = json.load(json_file)
    return parse_messages(settings.get("messages"), bpmn_path, declared_attributes(settings))


def declared_attributes(settings):
    """Names of the case, global and event attributes declared in a process's JSON settings."""
    names = {attribute["name"] for section in ("case_attributes", "global_attributes")
             for attribute in settings.get(section, [])}
    names.update(attribute["name"] for event in settings.get("event_attributes", [])
                 for attribute in event["attributes"])
    return names


def parse_messages(messages_json, bpmn_path, declared_attributes) -> MessagingModel:
    """Validates the 'messages' section against the BPMN model and the declared attribute names.
    None (no section) gives an empty model, so processes without messages behave as before."""
    if messages_json is None:
        return MessagingModel()
    if not isinstance(messages_json, dict) or set(messages_json) - {"publish", "consume"}:
        _fail("'messages' must be an object with only 'publish' and/or 'consume'")

    events = _bpmn_events(bpmn_path)
    publish = tuple(
        PublishPoint(entry["event_id"], entry["type"], _attributes(entry, where, declared_attributes))
        for where, entry in _entries(messages_json, "publish", events, PUBLISHING_EVENTS)
    )
    consume = tuple(
        ConsumePoint(entry["event_id"], entry["type"], _condition(entry, where))
        for where, entry in _entries(messages_json, "consume", events, CONSUMING_EVENTS)
    )
    return MessagingModel(publish, consume)


def _entries(messages_json, section, events, allowed_kinds):
    entries = messages_json.get(section, [])
    if not isinstance(entries, list):
        _fail(f"messages.{section} must be a list")
    for index, entry in enumerate(entries):
        where = f"messages.{section}[{index}]"
        if not isinstance(entry, dict):
            _fail(f"{where} must be an object")
        _check_event(where, entry.get("event_id"), events, allowed_kinds)
        if not isinstance(entry.get("type"), str) or not entry["type"].strip():
            _fail(f"{where}: 'type' must be a non-empty string")
        yield where, entry


def _check_event(where, event_id, events, allowed_kinds):
    if not isinstance(event_id, str) or event_id not in events:
        _fail(f"{where}: event_id {event_id!r} is not an element of the BPMN model")
    kind, is_message, incoming, after_event_gateway = events[event_id]
    if kind == "startEvent" and is_message:
        _fail(f"{where}: event_id {event_id!r} is a message start event, which isn't supported yet")
    if kind not in allowed_kinds or not is_message:
        expected = " or ".join(allowed_kinds.values())
        found = f"a message {kind}" if is_message else f"{'an' if kind[0] in 'aeiou' else 'a'} {kind}"
        _fail(f"{where}: event_id {event_id!r} is {found}, expected {expected}")
    # several arrows into an end or catch event mean "once per arriving token" in BPMN, but Prosimos
    # joins them (OR for end events, AND for catch events); requiring an explicit gateway avoids the
    # question. Throw events need no rule: they get a hidden XOR join, as the standard says
    if kind in ("endEvent", "intermediateCatchEvent") and incoming != 1:
        _fail(f"{where}: event_id {event_id!r} has {incoming} incoming arrows, expected exactly one; "
              f"draw an explicit gateway before {event_id}")
    # an event-based gateway races the events after it by drawing a duration for each; a case
    # waiting for a message can't take part in such a race yet (future work)
    if kind == "intermediateCatchEvent" and after_event_gateway:
        _fail(f"{where}: event_id {event_id!r} follows an event-based gateway, where waiting for a "
              f"message isn't supported yet")


def _attributes(entry, where, declared):
    attributes = entry.get("attributes", [])
    if not isinstance(attributes, list) or not all(isinstance(name, str) and name for name in attributes):
        _fail(f"{where}: 'attributes' must be a list of attribute names")
    unknown = [name for name in attributes if name != "case_id" and name not in declared]
    if unknown:
        _fail(f"{where}: {', '.join(map(repr, unknown))} not declared as a case, global or event attribute")
    return tuple(attributes)


def _condition(entry, where):
    if "condition" not in entry:
        return None
    condition = entry["condition"]
    if not isinstance(condition, list) or not condition:
        _fail(f"{where}: 'condition' must be a non-empty list of alternatives, each a list of terms that must all hold")
    alternatives = []
    for alternative_index, alternative in enumerate(condition):
        if not isinstance(alternative, list) or not alternative:
            _fail(f"{where}.condition[{alternative_index}] must be a non-empty list of terms")
        alternatives.append(tuple(_term(term, f"{where}.condition[{alternative_index}][{term_index}]")
                                  for term_index, term in enumerate(alternative)))
    return tuple(alternatives)


def _term(term, where):
    if not isinstance(term, dict):
        _fail(f"{where} must be an object")
    if set(term) - TERM_KEYS:
        _fail(f"{where} has unknown keys {sorted(set(term) - TERM_KEYS)}")
    if not isinstance(term.get("attribute"), str) or not term["attribute"]:
        _fail(f"{where}: 'attribute' must be a message attribute name or 'source'")
    if term.get("comparison") not in COMPARISONS:
        _fail(f"{where}: 'comparison' must be one of {', '.join(COMPARISONS)}, got {term.get('comparison')!r}")
    if ("value" in term) == ("case_attribute" in term):
        _fail(f"{where}: give exactly one of 'value' and 'case_attribute'")
    if "case_attribute" in term and (not isinstance(term["case_attribute"], str) or not term["case_attribute"]):
        _fail(f"{where}: 'case_attribute' must be a case attribute name")
    if term["comparison"] == "in" and not (isinstance(term.get("value"), list) and len(term["value"]) == 2):
        _fail(f"{where}: 'in' needs a fixed 'value' [low, high]")
    return ConditionTerm(term["attribute"], term["comparison"], term.get("value"), term.get("case_attribute"))


def _bpmn_events(bpmn_path):
    """event id -> (element kind, whether it has a message event definition, incoming arrows,
    whether one of them comes from an event-based gateway)."""
    elements = list(ET.parse(bpmn_path).getroot().iter())
    flows = [element.attrib for element in elements if element.tag == BPMN_NS + "sequenceFlow"]
    incoming = Counter(flow.get("targetRef") for flow in flows)
    event_gateways = {element.attrib.get("id") for element in elements if element.tag == BPMN_NS + "eventBasedGateway"}
    after_event_gateway = {flow.get("targetRef") for flow in flows if flow.get("sourceRef") in event_gateways}
    events = {}
    for element in elements:
        if element.tag.startswith(BPMN_NS) and "id" in element.attrib:
            element_id = element.attrib["id"]
            kind = element.tag[len(BPMN_NS):]
            events[element_id] = (kind, element.find(MESSAGE_DEFINITION) is not None, incoming[element_id],
                                  element_id in after_event_gateway)
    return events


def _fail(reason):
    raise InvalidSimScenarioException(f"Invalid 'messages' section: {reason}")
