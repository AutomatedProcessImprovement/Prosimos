"""
The optional 'messages' section of a process's JSON settings (docs/messaging.md). The BPMN
model says where a process publishes, waits or is started (message events); this section says what
it publishes or accepts. Parsing and validation only; the engine acts on the result.
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
CONSUMING_EVENTS = {"intermediateCatchEvent": "an intermediate message catch event", "startEvent": "a message start event"}
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
    copy: Tuple[Tuple[str, str], ...] = ()  # (case attribute, message attribute): copied into the case
    starts_case: bool = False  # at the start event: an accepted message starts a new case
    capacity: int = 1  # how many waiting cases one message resumes, unless read from the message
    capacity_attribute: Optional[str] = None  # the message attribute holding the capacity, if any
    collect: int = 1  # how many messages a case must claim here before it continues, unless read from the case
    collect_attribute: Optional[str] = None  # the case attribute holding that number, if any


@dataclass(frozen=True)
class MessagingModel:
    publish: Tuple[PublishPoint, ...] = ()
    consume: Tuple[ConsumePoint, ...] = ()

    def subscriptions(self) -> List[str]:
        return sorted({point.type for point in self.consume})

    @property
    def started_by_messages(self) -> bool:
        """Whether the process's cases are started by messages instead of an arrival schedule."""
        return any(point.starts_case for point in self.consume)


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
    # attributes a consume entry copies from a message into the case can be published too
    messages = settings.get("messages")
    consume = messages.get("consume", []) if isinstance(messages, dict) else []
    names.update(target for entry in consume if isinstance(entry, dict) and isinstance(entry.get("copy"), dict)
                 for target in entry["copy"] if isinstance(target, str))
    return names


def parse_messages(messages_json, bpmn_path, declared_attributes) -> MessagingModel:
    """Validates the 'messages' section against the BPMN model and the declared attribute names.
    None (no section) gives an empty model, so processes without messages behave as before."""
    if messages_json is None:
        return MessagingModel()
    if not isinstance(messages_json, dict) or set(messages_json) - {"publish", "consume"}:
        _fail("'messages' must be an object with only 'publish' and/or 'consume'")

    events, gateway_branches = _bpmn_events(bpmn_path)
    # consume first, so that a malformed copy (whose targets publish entries may use) is reported
    # as such rather than as an undeclared published attribute
    consume, collecting = [], []
    for where, entry in _entries(messages_json, "consume", events, CONSUMING_EVENTS):
        starts_case = events[entry["event_id"]][0] == "startEvent"
        point = ConsumePoint(entry["event_id"], entry["type"], _condition(entry, where), _copy(entry, where),
                             starts_case, *_capacity(entry, where, starts_case), *_collect(entry, where, starts_case))
        if point.starts_case:
            _check_start(where, point, events)
        consume.append(point)
        if "collect" in entry:
            collecting.append((where, point.event_id))
    _check_collect_has_its_own_event(collecting, consume)
    _check_races(consume, events, gateway_branches)
    publish = tuple(
        PublishPoint(entry["event_id"], entry["type"], _attributes(entry, where, declared_attributes))
        for where, entry in _entries(messages_json, "publish", events, PUBLISHING_EVENTS)
    )
    return MessagingModel(publish, tuple(consume))


def _check_start(where, point, events):
    # Prosimos runs one start event per process, so a process is started either by its arrival
    # schedule or by messages, and there is no case yet for a condition to compare with
    starts = sorted(event_id for event_id, (kind, *_) in events.items() if kind == "startEvent")
    if len(starts) > 1:
        _fail(f"{where}: a process started by messages must have exactly one start event; "
              f"the model has {len(starts)} ({', '.join(starts)})")
    on_case = [term.attribute for alternative in point.condition or () for term in alternative if term.case_attribute]
    if on_case:
        _fail(f"{where}: the condition of a start event may only use fixed values and source, not "
              f"case_attribute (there is no case yet), but its terms on {', '.join(map(repr, on_case))} do")


def _capacity(entry, where, starts_case):
    """(fixed capacity, message attribute holding it): {"value": 2} or {"attribute": "capacity"}."""
    if "capacity" not in entry:
        return 1, None
    if starts_case:
        _fail(f"{where}: a start event takes no capacity: a start message starts exactly one case")
    capacity = entry["capacity"]
    if not isinstance(capacity, dict) or len(capacity) != 1 or not set(capacity) <= {"value", "attribute"}:
        _fail(f"{where}: 'capacity' must be either {{\"value\": <number>}} or {{\"attribute\": <message attribute>}}")
    if "attribute" in capacity:
        if not isinstance(capacity["attribute"], str) or not capacity["attribute"]:
            _fail(f"{where}: 'capacity' attribute must be a message attribute name")
        return 1, capacity["attribute"]
    if not is_whole_number_of_at_least_one(capacity["value"]):
        _fail(f"{where}: 'capacity' value must be a whole number of at least 1, got {capacity['value']!r}")
    return int(capacity["value"]), None


def _collect(entry, where, starts_case):
    """(fixed number to collect, case attribute holding it): {"value": 3} or {"case_attribute": "items"}."""
    if "collect" not in entry:
        return 1, None
    if starts_case:
        _fail(f"{where}: a start event can't collect: it turns one message into a new case, and before that case "
              f"exists nothing holds the earlier messages or tells which ones belong together. Start on the first "
              f"message, then collect the rest at a catch event right after the start (e.g. a picking batch: "
              f"start on 1 order, then collect {{\"value\": 4}})")
    if "capacity" in entry:
        _fail(f"{where}: collect and capacity can't be combined on one entry: that would be several messages "
              f"completing several cases at once. Use two steps through an intermediate case instead: it collects "
              f"the messages, then publishes one message per target (e.g. a truck collects packages, then sends "
              f"one Shipment per package)")
    collect = entry["collect"]
    if not isinstance(collect, dict) or len(collect) != 1 or not set(collect) <= {"value", "case_attribute"}:
        _fail(f"{where}: 'collect' must be either {{\"value\": <number>}} or {{\"case_attribute\": <case attribute>}}")
    if "case_attribute" in collect:
        if not isinstance(collect["case_attribute"], str) or not collect["case_attribute"]:
            _fail(f"{where}: 'collect' case_attribute must be a case attribute name")
        return 1, collect["case_attribute"]
    if not is_whole_number(collect["value"], minimum=0):
        _fail(f"{where}: 'collect' value must be a whole number of at least 0, got {collect['value']!r}")
    return int(collect["value"]), None


def _check_collect_has_its_own_event(collecting, consume):
    """collect is allowed only on a catch event with exactly one consume entry: a waiting case keeps one
    count there, so it must be clear which messages it counts."""
    for where, event_id in collecting:
        entries = sum(point.event_id == event_id for point in consume)
        if entries > 1:
            _fail(f"{where}: collect is only allowed on a catch event with exactly one consume entry, but "
                  f"{event_id} has {entries}: a case waiting there keeps one count. To accept several variants of "
                  f"one type, use one entry with alternatives in its condition; to wait for several kinds, use one "
                  f"catch event per type")


def is_whole_number(value, minimum):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value >= minimum and float(value).is_integer()


def is_whole_number_of_at_least_one(value):
    return is_whole_number(value, minimum=1)


def _copy(entry, where):
    copy = entry.get("copy", {})
    if not isinstance(copy, dict) or not all(isinstance(target, str) and target and isinstance(source, str) and source
                                             for target, source in copy.items()):
        _fail(f"{where}: 'copy' must map case attribute names to message attribute names")
    if "case_id" in copy:
        _fail(f"{where}: 'copy' can't set case_id, which is reserved for the case's own identifier")
    return tuple(copy.items())


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
    kind, is_message, incoming, _ = events[event_id]
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


def _check_races(consume, events, gateway_branches):
    """An event-based gateway with a branch that waits for a message is a race: every branch must be a
    message catch event or a timer, and only one of them may wait for a message (for now)."""
    waiting = {point.event_id for point in consume if not point.starts_case}
    for gateway_id in sorted({events[event_id][3] for event_id in waiting if events[event_id][3]}):
        branches = gateway_branches[gateway_id]
        for target_id, kind, definition in branches:
            if kind != "intermediateCatchEvent" or definition not in ("message", "timer"):
                found = f"a {definition} {kind}" if definition else f"{'an' if kind[0] in 'aeiou' else 'a'} {kind}"
                _fail(f"after the event-based gateway {gateway_id}, which races a branch waiting for a message, "
                      f"every branch must be a message catch event or a timer, but {target_id} is {found}")
        message_branches = sorted(target_id for target_id, _, _ in branches if target_id in waiting)
        if len(message_branches) > 1:
            _fail(f"the event-based gateway {gateway_id} has several branches waiting for a message "
                  f"({', '.join(message_branches)}); a race with more than one isn't supported yet")


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
    """element id -> (element kind, whether it has a message event definition, incoming arrows, the
    event-based gateway it follows or None), and event-based gateway id -> its branches, each
    (target id, target kind, its event definition: "message", "timer", ... or None)."""
    elements = list(ET.parse(bpmn_path).getroot().iter())
    flows = [element.attrib for element in elements if element.tag == BPMN_NS + "sequenceFlow"]
    incoming = Counter(flow.get("targetRef") for flow in flows)
    event_gateways = {element.attrib.get("id") for element in elements if element.tag == BPMN_NS + "eventBasedGateway"}
    gateway_before = {flow.get("targetRef"): flow.get("sourceRef") for flow in flows if flow.get("sourceRef") in event_gateways}
    events, definitions = {}, {}
    for element in elements:
        if element.tag.startswith(BPMN_NS) and "id" in element.attrib:
            element_id = element.attrib["id"]
            kind = element.tag[len(BPMN_NS):]
            events[element_id] = (kind, element.find(MESSAGE_DEFINITION) is not None, incoming[element_id],
                                  gateway_before.get(element_id))
            definitions[element_id] = next((child.tag[len(BPMN_NS):].removesuffix("EventDefinition") for child in element
                                            if child.tag.endswith("EventDefinition")), None)
    gateway_branches = {gateway_id: [] for gateway_id in event_gateways}
    for flow in flows:
        if flow.get("sourceRef") in event_gateways:
            target_id = flow.get("targetRef")
            gateway_branches[flow.get("sourceRef")].append((target_id, events[target_id][0], definitions[target_id]))
    return events, gateway_branches


def _fail(reason):
    raise InvalidSimScenarioException(f"Invalid 'messages' section: {reason}")
