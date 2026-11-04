"""Reads a process's 'messages' section and checks it against its BPMN model (docs/messaging.md)."""
import json
import xml.etree.ElementTree as ET
from collections import Counter

from prosimos.control_flow_manager import TASK_TAGS
from prosimos.messaging.model import ConsumePoint, MessagingModel, PublishPoint
from prosimos.messaging.parser.common import check_keys, fail
from prosimos.messaging.parser.consume import (check_aggregated_only, check_appended_only,
                                               check_collect_has_its_own_event, check_races, check_start,
                                               read_aggregate, read_append, read_capacity, read_collect, read_condition,
                                               read_copy, read_same)
from prosimos.messaging.parser.publish import read_attributes, read_carry, read_fan_in, read_fan_out

BPMN_NS = "{http://www.omg.org/spec/BPMN/20100524/MODEL}"
MESSAGE_DEFINITION = BPMN_NS + "messageEventDefinition"
PUBLISHING_EVENTS = {"intermediateThrowEvent": "an intermediate message throw event", "endEvent": "a message end event"}
CONSUMING_EVENTS = {"intermediateCatchEvent": "an intermediate message catch event", "startEvent": "a message start event"}
PUBLISH_KEYS = {"event_id", "type", "attributes", "fan_out", "fan_in", "same", "deadline", "qualifier", "o2o",
                "carry"}
CONSUME_KEYS = {"event_id", "type", "condition", "copy", "append", "aggregate", "capacity", "collect", "same",
                "qualifier", "o2o", "carry"}


def parse_messages_file(json_path, bpmn_path, also_declared=()) -> MessagingModel:
    """Reads the 'messages' section of json_path, in Prosimos's names (case_attributes), as load_process hands
    them over. also_declared: attributes given from outside the settings, e.g. picked, which messages may use too."""
    with open(json_path) as json_file:
        settings = json.load(json_file)
    return parse_messages(settings.get("messages"), bpmn_path, declared_attributes(settings) | set(also_declared),
                          {attribute["name"] for attribute in settings.get("case_attributes", [])})


def declared_attributes(settings):
    """Names of the case, global and event attributes declared in a process's JSON settings."""
    names = {attribute["name"] for section in ("case_attributes", "global_attributes")
             for attribute in settings.get(section, [])}
    names.update(attribute["name"] for event in settings.get("event_attributes", [])
                 for attribute in event["attributes"])
    # attributes a consume entry copies, appends or aggregates into can be published too
    messages = settings.get("messages")
    consume = messages.get("consume", []) if isinstance(messages, dict) else []
    names.update(target for entry in consume if isinstance(entry, dict) for key in ("copy", "append", "aggregate")
                 if isinstance(entry.get(key), dict) for target in entry[key] if isinstance(target, str))
    return names


def parse_messages(messages_json, bpmn_path, declared_attributes, case_attributes=()) -> MessagingModel:
    """Checks the 'messages' section against the BPMN model and the declared attributes; None gives an empty
    model. case_attributes: the attributes the process draws, which append can't add to."""
    if messages_json is None:
        return MessagingModel()
    if not isinstance(messages_json, dict):
        fail("'messages' must be an object with only 'publish' and/or 'consume'")
    check_keys("messages", messages_json, {"publish", "consume"})

    events, gateway_branches = _bpmn_events(bpmn_path)
    tasks = _tasks(bpmn_path)
    # consume first, so a malformed copy is reported as such, not as an undeclared published attribute
    consume, collecting = [], []
    # fan_out and fan_in get their own reasons below
    for where, entry in _entries(messages_json, "consume", CONSUME_KEYS, events, CONSUMING_EVENTS,
                                 ("fan_out", "fan_in")):
        starts_case = events[entry["event_id"]][0] == "startEvent"
        low, low_attribute, high, high_attribute = read_collect(entry, where, starts_case)
        fewest, fewest_attribute, most, most_attribute = read_capacity(entry, where, starts_case)
        point = ConsumePoint(entry["event_id"], entry["type"], read_condition(entry, where), read_copy(entry, where),
                             starts_case, most, most_attribute, low, low_attribute,
                             _name(entry, "qualifier", where), _name(entry, "o2o", where),
                             read_append(entry, where, case_attributes), read_aggregate(entry, where, case_attributes),
                             high, high_attribute, read_same(entry, where), fewest, fewest_attribute,
                             read_carry(entry, where, tasks))
        if "fan_out" in entry:
            fail(f"{where}: fan_out is only for publish entries, on throw and end events: only they publish, and a "
                  f"start event starts exactly one case per message")
        if "fan_in" in entry:
            fail(f"{where}: fan_in is only for publish entries, where several cases make one message; to take "
                  f"several messages at a catch event, use collect")
        if point.starts_case:
            check_start(where, point)
        consume.append(point)
        if "collect" in entry:
            collecting.append((where, point.event_id))
    check_collect_has_its_own_event(collecting, consume)
    check_appended_only(consume)
    check_aggregated_only(consume)
    check_races(consume, events, gateway_branches)
    publish = []
    for where, entry in _entries(messages_json, "publish", PUBLISH_KEYS, events, PUBLISHING_EVENTS):
        links = _name(entry, "qualifier", where), _name(entry, "o2o", where), read_carry(entry, where, tasks)
        if "fan_in" in entry:
            publish.append(read_fan_in(entry, where, declared_attributes, links))
            continue
        for key in ("same", "deadline"):
            if key in entry:
                fail(f"{where}: '{key}' on a publish entry needs a fan_in: it is about a group of cases' parts")
        publish.append(PublishPoint(entry["event_id"], entry["type"],
                                    read_attributes(entry, where, declared_attributes),
                                    *read_fan_out(entry, where, declared_attributes), *links))
    return MessagingModel(tuple(publish), tuple(consume))


def _entries(messages_json, section, allowed_keys, events, allowed_kinds, own_reasons=()):
    """Each entry of the section, with where it is; own_reasons: keys the caller rejects with its own reason."""
    entries = messages_json.get(section, [])
    if not isinstance(entries, list):
        fail(f"messages.{section} must be a list")
    for index, entry in enumerate(entries):
        where = f"messages.{section}[{index}]"
        if not isinstance(entry, dict):
            fail(f"{where} must be an object")
        check_keys(where, {key: value for key, value in entry.items() if key not in own_reasons}, allowed_keys)
        _check_event(where, entry.get("event_id"), events, allowed_kinds)
        if not isinstance(entry.get("type"), str) or not entry["type"].strip():
            fail(f"{where}: 'type' must be a non-empty string")
        yield where, entry


def _check_event(where, event_id, events, allowed_kinds):
    if not isinstance(event_id, str) or event_id not in events:
        fail(f"{where}: event_id {event_id!r} is not an element of the BPMN model")
    kind, is_message, incoming, _ = events[event_id]
    if kind not in allowed_kinds or not is_message:
        expected = " or ".join(allowed_kinds.values())
        found = f"a message {kind}" if is_message else f"{'an' if kind[0] in 'aeiou' else 'a'} {kind}"
        fail(f"{where}: event_id {event_id!r} is {found}, expected {expected}")
    if kind == "intermediateCatchEvent" and incoming != 1:
        fail(f"{where}: event_id {event_id!r} has {incoming} incoming arrows; a catch event that waits for a message "
              f"takes exactly one, since a case waits there once (waiting and collect are counted per case and "
              f"event): draw one catch event per path, or a gateway before {event_id}")


def _name(entry, key, where):
    """The entry's optional OCEL qualifier, 'qualifier' or 'o2o'; None without the key."""
    if key not in entry:
        return None
    if not isinstance(entry[key], str) or not entry[key].strip():
        fail(f"{where}: '{key}' must be a non-empty string")
    return entry[key]


def _tasks(bpmn_path):
    """The ids and names of the model's tasks."""
    tags = {BPMN_NS + tag for tag in TASK_TAGS}
    return {key for element in ET.parse(bpmn_path).getroot().iter() if element.tag in tags
            for key in (element.attrib.get("id"), element.attrib.get("name")) if key}


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
