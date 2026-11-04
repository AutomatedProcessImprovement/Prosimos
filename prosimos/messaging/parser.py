"""Reads a process's 'messages' section and checks it against its BPMN model (docs/messaging.md)."""
import json
import math
import numbers
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import timedelta

from prosimos.control_flow_manager import TASK_TAGS
from prosimos.exceptions import InvalidSimScenarioException
from prosimos.messaging.model import (AGGREGATE_FUNCTIONS, COMPARISONS, ConditionTerm, ConsumePoint, MessagingModel,
                                      PublishPoint)

BPMN_NS = "{http://www.omg.org/spec/BPMN/20100524/MODEL}"
MESSAGE_DEFINITION = BPMN_NS + "messageEventDefinition"
PUBLISHING_EVENTS = {"intermediateThrowEvent": "an intermediate message throw event", "endEvent": "a message end event"}
CONSUMING_EVENTS = {"intermediateCatchEvent": "an intermediate message catch event", "startEvent": "a message start event"}
TERM_KEYS = {"attribute", "comparison", "value", "object_attribute"}
PUBLISH_KEYS = {"event_id", "type", "attributes", "fan_out", "fan_in", "same", "deadline", "qualifier", "o2o",
                "carry"}
PERIODS = ("weeks", "days", "hours")  # of a deadline, and of a catalogue change
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
        _fail("'messages' must be an object with only 'publish' and/or 'consume'")
    _check_keys("messages", messages_json, {"publish", "consume"})

    events, gateway_branches = _bpmn_events(bpmn_path)
    tasks = _tasks(bpmn_path)
    # consume first, so a malformed copy is reported as such, not as an undeclared published attribute
    consume, collecting = [], []
    # fan_out and fan_in get their own reasons below
    for where, entry in _entries(messages_json, "consume", CONSUME_KEYS, events, CONSUMING_EVENTS,
                                 ("fan_out", "fan_in")):
        starts_case = events[entry["event_id"]][0] == "startEvent"
        low, low_attribute, high, high_attribute = _collect(entry, where, starts_case)
        fewest, fewest_attribute, most, most_attribute = _capacity(entry, where, starts_case)
        point = ConsumePoint(entry["event_id"], entry["type"], _condition(entry, where), _copy(entry, where),
                             starts_case, most, most_attribute, low, low_attribute,
                             _name(entry, "qualifier", where), _name(entry, "o2o", where),
                             _append(entry, where, case_attributes), _aggregate(entry, where, case_attributes),
                             high, high_attribute, _same(entry, where), fewest, fewest_attribute,
                             _carry(entry, where, tasks))
        if "fan_out" in entry:
            _fail(f"{where}: fan_out is only for publish entries, on throw and end events: only they publish, and a "
                  f"start event starts exactly one case per message")
        if "fan_in" in entry:
            _fail(f"{where}: fan_in is only for publish entries, where several cases make one message; to take "
                  f"several messages at a catch event, use collect")
        if point.starts_case:
            _check_start(where, point)
        consume.append(point)
        if "collect" in entry:
            collecting.append((where, point.event_id))
    _check_collect_has_its_own_event(collecting, consume)
    _check_appended_only(consume)
    _check_aggregated_only(consume)
    _check_races(consume, events, gateway_branches)
    publish = []
    for where, entry in _entries(messages_json, "publish", PUBLISH_KEYS, events, PUBLISHING_EVENTS):
        links = _name(entry, "qualifier", where), _name(entry, "o2o", where), _carry(entry, where, tasks)
        if "fan_in" in entry:
            publish.append(_fan_in_point(entry, where, declared_attributes, links))
            continue
        for key in ("same", "deadline"):
            if key in entry:
                _fail(f"{where}: '{key}' on a publish entry needs a fan_in: it is about a group of cases' parts")
        publish.append(PublishPoint(entry["event_id"], entry["type"], _attributes(entry, where, declared_attributes),
                                    *_fan_out(entry, where, declared_attributes), *links))
    return MessagingModel(tuple(publish), tuple(consume))


def _check_start(where, point):
    # at the start there is no case to compare with yet
    on_case = [term.attribute for alternative in point.condition or () for term in alternative if term.object_attribute]
    if on_case:
        _fail(f"{where}: the condition of a start event may only use fixed values and source, not "
              f"object_attribute (there is no object yet), but its terms on {', '.join(map(repr, on_case))} do")


def _capacity(entry, where, starts_case):
    """(min, its message attribute, max, its message attribute): e.g. {"min": 2, "max": {"attribute": "capacity"}},
    2 for (1, 2), {"min": 2} for (2, no limit)."""
    if "capacity" not in entry:
        return 1, None, 1, None
    if starts_case:
        _fail(f"{where}: a start event takes no capacity: a start message starts exactly one case")
    capacity = entry["capacity"]
    if is_whole_number_of_at_least_one(capacity):
        return 1, None, int(capacity), None
    if isinstance(capacity, dict) and "min" in capacity:
        _check_keys(f"{where}.capacity", capacity, {"min", "max"})
        fewest, fewest_attribute = _capacity_bound(capacity, "min", where)
        most, most_attribute = _capacity_bound(capacity, "max", where) if "max" in capacity else (math.inf, None)
        if fewest_attribute is None and most_attribute is None and fewest > most:
            _fail(f"{where}: 'capacity' min {fewest} is above max {most}")
        return fewest, fewest_attribute, most, most_attribute
    if isinstance(capacity, dict):
        _check_keys(f"{where}.capacity", capacity, {"attribute", "min", "max"})
        if set(capacity) == {"max"}:
            _fail(f"{where}: 'capacity' max needs a min, e.g. {{\"min\": 1, \"max\": {capacity['max']!r}}}")
    if not isinstance(capacity, dict) or len(capacity) != 1:
        _fail(f"{where}: 'capacity' must be either a whole number of at least 1, {{\"attribute\": <message "
              f"attribute>}} or {{\"min\": <bound>, \"max\": <bound>}} with each bound a number or {{\"attribute\": "
              f"<message attribute>}}")
    if not isinstance(capacity["attribute"], str) or not capacity["attribute"]:
        _fail(f"{where}: 'capacity' attribute must be a message attribute name")
    return 1, None, 1, capacity["attribute"]


def _capacity_bound(capacity, name, where):
    bound = capacity[name]
    if isinstance(bound, dict):
        _check_keys(f"{where}.capacity.{name}", bound, {"attribute"})
        if isinstance(bound.get("attribute"), str) and bound["attribute"]:
            return 1, bound["attribute"]
    elif is_whole_number_of_at_least_one(bound):
        return int(bound), None
    _fail(f"{where}: 'capacity' {name} must be a whole number of at least 1 or {{\"attribute\": <message "
          f"attribute>}}, got {bound!r}")


def _collect(entry, where, starts_case):
    """(min, its case attribute, max, its case attribute): e.g. {"min": 2, "max": 5}, 3 for (3, 3), "all" for
    (1, no limit). A max of None means the same as min."""
    if "collect" not in entry:
        return 1, None, None, None
    if starts_case:
        _fail(f"{where}: a start event can't collect: it turns one message into a new case, and before that case "
              f"exists nothing holds the earlier messages or tells which ones belong together. Start on the first "
              f"message, then collect the rest at a catch event right after the start (e.g. a picking batch: "
              f"start on 1 order, then collect 4)")
    if "capacity" in entry:
        _fail(f"{where}: collect and capacity can't be combined on one entry: that would be several messages "
              f"completing several cases at once. Use two steps through an intermediate case instead: it collects "
              f"the messages, then publishes one message per target (e.g. a truck collects packages, then sends "
              f"one Shipment per package)")
    collect = entry["collect"]
    if collect == "all":
        return 1, None, math.inf, None
    if isinstance(collect, dict) and "min" in collect:
        _check_keys(f"{where}.collect", collect, {"min", "max"})
        low, low_attribute = _bound(collect, "min", where)
        high, high_attribute = _bound(collect, "max", where) if "max" in collect else (math.inf, None)
        if high_attribute is None and low_attribute is None and low > high:
            _fail(f"{where}: 'collect' min {low} is above max {high}")
        if high == 0 and low_attribute is not None:
            _fail(f"{where}: 'collect' max is 0 but min is read from {low_attribute}: a case waiting for a message "
                  f"couldn't take any")
        return low, low_attribute, high, high_attribute
    if isinstance(collect, dict):
        _check_keys(f"{where}.collect", collect, {"object_attribute", "min", "max"})
    if not isinstance(collect, dict) or len(collect) != 1:
        if is_whole_number(collect, minimum=0):
            return int(collect), None, None, None
        _fail(f"{where}: 'collect' must be either a whole number of at least 0, \"all\", {{\"object_attribute\": <object "
              f"attribute>}} or {{\"min\": <bound>, \"max\": <bound>}} with each bound a number or "
              f"{{\"object_attribute\": <object attribute>}}")
    if "max" in collect:
        _fail(f"{where}: 'collect' max needs a min, e.g. {{\"min\": 1, \"max\": {collect['max']!r}}}")
    if not isinstance(collect["object_attribute"], str) or not collect["object_attribute"]:
        _fail(f"{where}: 'collect' object_attribute must be an object attribute name")
    return 1, collect["object_attribute"], None, None


def _same(entry, where, kind="message"):
    """The attributes whose values all messages a collecting case claims, or all members of a fan-in group,
    share: ["customer"]."""
    if "same" not in entry:
        return ()
    same = entry["same"]
    if not isinstance(same, list) or not same or not all(isinstance(name, str) and name for name in same):
        _fail(f"{where}: 'same' must be a non-empty list of {kind} attribute names, got {same!r}")
    repeated = sorted(name for name, times in Counter(same).items() if times > 1)
    if repeated:
        _fail(f"{where}: 'same' lists {', '.join(repeated)} more than once")
    if "collect" not in entry and kind == "message":
        _fail(f"{where}: 'same' needs a collect: without one, a case takes a single message, so there is nothing "
              f"to compare it with")
    return tuple(same)


def _bound(collect, name, where):
    bound = collect[name]
    if isinstance(bound, dict):
        _check_keys(f"{where}.collect.{name}", bound, {"object_attribute"})
        if isinstance(bound.get("object_attribute"), str) and bound["object_attribute"]:
            return 1, bound["object_attribute"]
    elif is_whole_number(bound, minimum=0):
        return int(bound), None
    _fail(f"{where}: 'collect' {name} must be a whole number of at least 0 or {{\"object_attribute\": <object "
          f"attribute>}}, got {bound!r}")


def _check_collect_has_its_own_event(collecting, consume):
    """collect needs a catch event with exactly one consume entry: a waiting case keeps one count there."""
    for where, event_id in collecting:
        entries = sum(point.event_id == event_id for point in consume)
        if entries > 1:
            _fail(f"{where}: collect is only allowed on a catch event with exactly one consume entry, but "
                  f"{event_id} has {entries}: a case waiting there keeps one count. To accept several variants of "
                  f"one type, use one entry with alternatives in its condition; to wait for several kinds, use one "
                  f"catch event per type")


def is_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def is_whole_number(value, minimum):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value >= minimum and float(value).is_integer()


def is_whole_number_of_at_least_one(value):
    return is_whole_number(value, minimum=1)


def _copy(entry, where, key="copy"):
    copy = entry.get(key, {})
    if not isinstance(copy, dict) or not all(isinstance(target, str) and target and isinstance(source, str) and source
                                             for target, source in copy.items()):
        _fail(f"{where}: '{key}' must map object attribute names to message attribute names")
    if "object_id" in copy:
        _fail(f"{where}: '{key}' can't set object_id, which is reserved for the object's own id")
    return tuple(copy.items())


def _append(entry, where, case_attributes):
    append = _copy(entry, where, "append")
    both = [target for target, _ in append if target in entry.get("copy", {})]
    if both:
        _fail(f"{where}: {', '.join(map(repr, both))} is in both copy and append: a case attribute is either "
              f"overwritten by each message or a list that each message adds to")
    drawn = [target for target, _ in append if target in case_attributes]
    if drawn:
        _fail(f"{where}: append adds to {', '.join(map(repr, drawn))}, which object_attributes declares: an appended "
              f"attribute is a list that starts empty, not a drawn value")
    return append


def _aggregate(entry, where, case_attributes):
    aggregate = entry.get("aggregate", {})
    if not isinstance(aggregate, dict) or not all(isinstance(target, str) and target and isinstance(spec, dict)
                                                  for target, spec in aggregate.items()):
        _fail(f"{where}: 'aggregate' must map object attribute names to {{<function>: <message attribute>}}")
    if "object_id" in aggregate:
        _fail(f"{where}: 'aggregate' can't set object_id, which is reserved for the object's own id")
    result = []
    for target, spec in aggregate.items():
        _check_keys(f"{where}.aggregate.{target}", spec, AGGREGATE_FUNCTIONS | {"initial"})
        functions = {key: source for key, source in spec.items() if key != "initial"}
        if len(functions) != 1 or not all(isinstance(source, str) and source for source in functions.values()):
            _fail(f"{where}: aggregate {target!r} must be one function and a message attribute, e.g. "
                  f"{{\"sum\": \"weight\"}}")
        initial = spec.get("initial")
        if "initial" in spec and not is_number(initial):
            _fail(f"{where}: aggregate {target!r} initial must be a number, got {initial!r}")
        (function, source), = functions.items()
        result.append((target, function, source, initial))
    for key in ("copy", "append"):
        both = [target for target, *_ in result if isinstance(entry.get(key), dict) and target in entry[key]]
        if both:
            _fail(f"{where}: {', '.join(map(repr, both))} is in both {key} and aggregate: an aggregated attribute is "
                  f"computed over the messages claimed, which {key} would overwrite")
    drawn = [target for target, *_ in result if target in case_attributes]
    if drawn:
        _fail(f"{where}: aggregate writes {', '.join(map(repr, drawn))}, which object_attributes declares: an "
              f"aggregated attribute is computed over the messages claimed, not a drawn value")
    return tuple(result)


def _check_aggregated_only(consume):
    """An attribute aggregated into is neither copied nor appended into, and has one function and one initial."""
    written_at = {}
    for index, point in enumerate(consume):
        for key, pairs in (("copies", point.copy), ("appends to", point.append)):
            for target, _ in pairs:
                written_at.setdefault(target, (index, key))
    function_at, initial_at = {}, {}
    for index, point in enumerate(consume):
        for target, function, _, initial in point.aggregate:
            if target in written_at:
                other, verb = written_at[target]
                _fail(f"messages.consume[{index}]: aggregate writes {target!r}, which messages.consume[{other}] "
                      f"{verb}: an aggregated attribute is computed over the messages claimed, which that would "
                      f"overwrite")
            first, first_function = function_at.setdefault(target, (index, function))
            if function != first_function:
                _fail(f"messages.consume[{index}]: aggregate writes {target!r} with {function}, but "
                      f"messages.consume[{first}] with {first_function}: an attribute is one function over all the "
                      f"messages claimed into it")
            first, first_initial = initial_at.setdefault(target, (index, initial))
            if initial != first_initial:
                _fail(f"messages.consume[{index}]: aggregate writes {target!r} with initial {initial!r}, but "
                      f"messages.consume[{first}] with initial {first_initial!r}: an attribute starts from one "
                      f"value")


def _check_appended_only(consume):
    """An attribute some entry appends to is a list, so no other entry may overwrite it with copy."""
    copied_at = {}
    for index, point in enumerate(consume):
        for target, _ in point.copy:
            copied_at.setdefault(target, index)
    for index, point in enumerate(consume):
        for target, _ in point.append:
            if target in copied_at:
                _fail(f"messages.consume[{index}]: append adds to {target!r}, which messages.consume"
                      f"[{copied_at[target]}] copies into: an appended attribute is a list, which copy would "
                      f"overwrite")


def _entries(messages_json, section, allowed_keys, events, allowed_kinds, own_reasons=()):
    """Each entry of the section, with where it is; own_reasons: keys the caller rejects with its own reason."""
    entries = messages_json.get(section, [])
    if not isinstance(entries, list):
        _fail(f"messages.{section} must be a list")
    for index, entry in enumerate(entries):
        where = f"messages.{section}[{index}]"
        if not isinstance(entry, dict):
            _fail(f"{where} must be an object")
        _check_keys(where, {key: value for key, value in entry.items() if key not in own_reasons}, allowed_keys)
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
    if kind == "intermediateCatchEvent" and incoming != 1:
        _fail(f"{where}: event_id {event_id!r} has {incoming} incoming arrows; a catch event that waits for a message "
              f"takes exactly one, since a case waits there once (waiting and collect are counted per case and "
              f"event): draw one catch event per path, or a gateway before {event_id}")


def _check_races(consume, events, gateway_branches):
    """A race (an event-based gateway with a message branch) has only message and timer branches, none with a
    fixed collect of 0."""
    waiting = {point.event_id for point in consume if not point.starts_case}
    for gateway_id in sorted({events[event_id][3] for event_id in waiting if events[event_id][3]}):
        branches = gateway_branches[gateway_id]
        for target_id, kind, definition in branches:
            if kind != "intermediateCatchEvent" or definition not in ("message", "timer"):
                found = f"a {definition} {kind}" if definition else f"{'an' if kind[0] in 'aeiou' else 'a'} {kind}"
                _fail(f"after the event-based gateway {gateway_id}, which races a branch waiting for a message, "
                      f"every branch must be a message catch event or a timer, but {target_id} is {found}")
        for point in consume:
            if events[point.event_id][3] == gateway_id and point.collect == 0 and point.collect_attribute is None:
                _fail(f"{point.event_id}, after the event-based gateway {gateway_id}: a race branch with a fixed "
                      f"collect of 0 always wins its race; remove the race or the branch")


def _name(entry, key, where):
    """The entry's optional OCEL qualifier, 'qualifier' or 'o2o'; None without the key."""
    if key not in entry:
        return None
    if not isinstance(entry[key], str) or not entry[key].strip():
        _fail(f"{where}: '{key}' must be a non-empty string")
    return entry[key]


def _carry(entry, where, tasks):
    """The entry's optional 'carry': true, or a list of activities; it needs a qualifier on the entry."""
    if "carry" not in entry:
        return False
    carry = entry["carry"]
    if not (isinstance(carry, bool)
            or isinstance(carry, list) and all(isinstance(name, str) and name.strip() for name in carry)):
        _fail(f"{where}: 'carry' must be true or a list of activities (task names or ids), got {carry!r}")
    unknown = [name for name in carry if name not in tasks] if isinstance(carry, list) else []
    if unknown:
        _fail(f"{where}: carry names {unknown}, which are no tasks of the process")
    if carry and "qualifier" not in entry:
        _fail(f"{where}: carry needs a qualifier on the same entry; a carried link needs its role word")
    return tuple(carry) if isinstance(carry, list) else carry


def _tasks(bpmn_path):
    """The ids and names of the model's tasks."""
    tags = {BPMN_NS + tag for tag in TASK_TAGS}
    return {key for element in ET.parse(bpmn_path).getroot().iter() if element.tag in tags
            for key in (element.attrib.get("id"), element.attrib.get("name")) if key}


def _fan_out(entry, where, declared):
    """(number, its case attribute, a list attribute, the name each message carries its entry under): e.g. 3,
    {"object_attribute": "items"} or {"list": "item_ids", "as": "item_id"}; all None without fan_out."""
    if "fan_out" not in entry:
        return None, None, None, None
    fan_out = entry["fan_out"]
    if is_whole_number(fan_out, minimum=0):
        return int(fan_out), None, None, None
    if isinstance(fan_out, dict):
        _check_keys(f"{where}.fan_out", fan_out, {"object_attribute", "list", "as"})
    if isinstance(fan_out, dict) and "list" in fan_out:
        return (None, None) + _fan_out_over_list(fan_out, where, declared)
    if not isinstance(fan_out, dict) or set(fan_out) != {"object_attribute"}:
        _fail(f"{where}: 'fan_out' must be either a whole number of at least 0, {{\"object_attribute\": <object "
              f"attribute>}} or {{\"list\": <object attribute>, \"as\": <attribute name>}}, got {fan_out!r}")
    if not isinstance(fan_out["object_attribute"], str) or fan_out["object_attribute"] not in declared:
        _fail(f"{where}: 'fan_out' object_attribute {fan_out['object_attribute']!r} is not a declared object, "
              f"global or event attribute")
    return None, fan_out["object_attribute"], None, None


def _fan_out_over_list(fan_out, where, declared):
    if not set(fan_out) <= {"list", "as"}:
        _fail(f"{where}: 'fan_out' over a list has only 'list' and 'as', got {', '.join(map(repr, sorted(fan_out)))}")
    if not isinstance(fan_out["list"], str) or fan_out["list"] not in declared:
        _fail(f"{where}: 'fan_out' list {fan_out['list']!r} is not a declared object, global or event attribute")
    if "as" not in fan_out:
        _fail(f"{where}: 'fan_out' over a list needs 'as', the attribute name under which each message carries its "
              f"entry")
    if not isinstance(fan_out["as"], str) or not fan_out["as"].strip():
        _fail(f"{where}: 'fan_out' as must be a non-empty string")
    if fan_out["as"] in ("object_id", "index"):
        _fail(f"{where}: 'fan_out' as can't be {fan_out['as']}, which is reserved: object_id is the publishing "
              f"object's id and index the message's number")
    return fan_out["list"], fan_out["as"]


def _attributes(entry, where, declared):
    attributes = entry.get("attributes", [])
    if isinstance(attributes, list) and any(isinstance(name, dict) for name in attributes):
        _fail(f"{where}: a list or a function in 'attributes' needs a fan_in: without one, the message carries a "
              f"single case's values, by their names")
    if not isinstance(attributes, list) or not all(isinstance(name, str) and name for name in attributes):
        _fail(f"{where}: 'attributes' must be a list of attribute names")
    if "index" in attributes and "fan_out" not in entry:
        _fail(f"{where}: 'index' is the message's number among the ones fan_out publishes, so it needs a 'fan_out' "
              f"on the same entry")
    fan_out = entry.get("fan_out")
    # (a reserved as is reported by _fan_out)
    if isinstance(fan_out, dict) and fan_out.get("as") in attributes and fan_out["as"] not in ("object_id", "index"):
        _fail(f"{where}: {fan_out['as']!r} is both in 'attributes' and fan_out's 'as': each message carries its own "
              f"entry of {fan_out.get('list')} under that name, so it can't also carry the case's attribute")
    unknown = [name for name in attributes if name not in ("object_id", "index") and name not in declared]
    if unknown:
        _fail(f"{where}: {', '.join(map(repr, unknown))} not declared as an object, global or event attribute")
    return tuple(attributes)


def _fan_in_point(entry, where, declared, links):
    """A publish entry with fan_in: e.g. {"fan_in": {"object_attribute": "items"}, "same": ["order_id"],
    "attributes": ["order_id", {"item_ids": "object_id"}, {"total_weight": {"sum": "weight"}}]}."""
    if "fan_out" in entry:
        _fail(f"{where}: fan_in and fan_out can't be combined on one entry: one turns several cases into one "
              f"message, the other one case into several messages")
    size = entry["fan_in"]
    if isinstance(size, dict):
        _check_keys(f"{where}.fan_in", size, {"object_attribute"})
    if is_whole_number_of_at_least_one(size):
        number, attribute = int(size), None
    elif isinstance(size, dict) and isinstance(size.get("object_attribute"), str):
        number, attribute = None, size["object_attribute"]
        if attribute not in declared:
            _fail(f"{where}: 'fan_in' object_attribute {attribute!r} is not a declared object, global or event "
                  f"attribute")
    else:
        _fail(f"{where}: 'fan_in' must be either a whole number of at least 1 or {{\"object_attribute\": <object "
              f"attribute>}}, got {size!r}")
    same = _same(entry, where, kind="object")
    undeclared = [name for name in same if name not in declared]
    if undeclared:
        _fail(f"{where}: 'same' names {', '.join(map(repr, undeclared))}, not declared as an object, global or event "
              f"attribute")
    plain, gathered, computed = _fan_in_attributes(entry, where, declared, same)
    qualifier, o2o, carry = links
    return PublishPoint(entry["event_id"], entry["type"], plain, qualifier=qualifier, o2o=o2o, carry=carry,
                        fan_in=number, fan_in_attribute=attribute, same=same, gathered=gathered, computed=computed,
                        deadline=_deadline(entry, where))


def _deadline(entry, where):
    if "deadline" not in entry:
        return None
    deadline = entry["deadline"]
    if isinstance(deadline, dict):
        _check_keys(f"{where}.deadline", deadline, set(PERIODS))
    if not (isinstance(deadline, dict) and len(deadline) == 1 and is_number(next(iter(deadline.values())))
            and next(iter(deadline.values())) > 0):
        _fail(f"{where}: 'deadline' must be {{\"weeks\" | \"days\" | \"hours\": n}}, n a number above 0, got "
              f"{deadline!r}")
    [(unit, amount)] = deadline.items()
    return timedelta(**{unit: amount})


def _fan_in_attributes(entry, where, declared, same):
    """(plain names, (name, source) lists, (name, function, source, initial) functions) of a fan-in entry."""
    attributes = entry.get("attributes", [])
    if not isinstance(attributes, list):
        attributes = [None]
    plain, gathered, computed = [], [], []
    for item in attributes:
        if isinstance(item, str) and item:
            plain.append(item)
            continue
        if not isinstance(item, dict) or not item:
            _fail(f"{where}: 'attributes' of a fan-in entry must list attribute names, {{<message attribute>: "
                  f"<object attribute>}} or {{<message attribute>: {{<function>: <object attribute>}}}}")
        for name, source in item.items():
            if isinstance(source, str) and source:
                gathered.append((name, source))
                continue
            if not isinstance(source, dict):
                _fail(f"{where}: attribute {name!r} must be an object attribute, whose values go into a list, or "
                      f"{{<function>: <object attribute>}}, got {source!r}")
            _check_keys(f"{where}.attributes.{name}", source, AGGREGATE_FUNCTIONS | {"initial"})
            functions = {key: value for key, value in source.items() if key != "initial"}
            if len(functions) != 1 or not all(isinstance(value, str) and value for value in functions.values()):
                _fail(f"{where}: attribute {name!r} must be one function and an object attribute, e.g. "
                      f"{{\"sum\": \"weight\"}}")
            if "initial" in source and not is_number(source["initial"]):
                _fail(f"{where}: attribute {name!r} initial must be a number, got {source['initial']!r}")
            (function, of), = functions.items()
            computed.append((name, function, of, source.get("initial")))
    names = plain + [name for name, _ in gathered] + [name for name, *_ in computed]
    sources = [source for _, source in gathered] + [source for _, _, source, _ in computed]
    if "index" in names + sources:
        _fail(f"{where}: 'index' numbers the messages of a fan_out, but a fan-in entry sends one message per group")
    if "object_id" in names:
        _fail(f"{where}: a fan-in message can't carry object_id: it has several publishers. Send their ids as a "
              f"list, e.g. {{\"item_ids\": \"object_id\"}}")
    repeated = sorted(name for name, times in Counter(names).items() if times > 1)
    if repeated:
        _fail(f"{where}: 'attributes' sends {', '.join(map(repr, repeated))} more than once")
    differing = [name for name in plain if name not in same]
    if differing:
        reason = "isn't in 'same', so the members of a group may differ in it" if same else (
            "is sent by its plain name, but without 'same' the members of a group may differ in every attribute")
        _fail(f"{where}: {', '.join(map(repr, differing))} {reason}: send it as a list, "
              f"{{<name>: \"{differing[0]}\"}}, or a function over the members, {{<name>: {{\"sum\": "
              f"\"{differing[0]}\"}}}}")
    undeclared = [source for source in sources if source != "object_id" and source not in declared]
    if undeclared:
        _fail(f"{where}: {', '.join(map(repr, undeclared))} not declared as an object, global or event attribute")
    return tuple(plain), tuple(gathered), tuple(computed)


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
    _check_keys(where, term, TERM_KEYS)
    if not isinstance(term.get("attribute"), str) or not term["attribute"]:
        _fail(f"{where}: 'attribute' must be a message attribute name or 'source'")
    if term.get("comparison") not in COMPARISONS:
        _fail(f"{where}: 'comparison' must be one of {', '.join(COMPARISONS)}, got {term.get('comparison')!r}")
    if ("value" in term) == ("object_attribute" in term):
        _fail(f"{where}: give exactly one of 'value' and 'object_attribute'")
    if "object_attribute" in term and (not isinstance(term["object_attribute"], str) or not term["object_attribute"]):
        _fail(f"{where}: 'object_attribute' must be an object attribute name")
    if term["comparison"] == "in" and not (isinstance(term.get("value"), list) and len(term["value"]) == 2):
        _fail(f"{where}: 'in' needs a fixed 'value' [low, high]")
    return ConditionTerm(term["attribute"], term["comparison"], term.get("value"), term.get("object_attribute"))


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


def unknown_keys(where, mapping, allowed):
    """The load error naming the keys of mapping that aren't allowed; None if there are none. A misspelled key
    would otherwise change the run silently."""
    unknown = [key for key in mapping if key not in allowed]
    if not unknown:
        return None
    return (f"{where}: unknown key{'s' if len(unknown) > 1 else ''} {', '.join(map(repr, unknown))}; "
            f"allowed: {', '.join(sorted(allowed))}")


def _check_keys(where, mapping, allowed):
    problem = unknown_keys(where, mapping, allowed)
    if problem:
        _fail(problem)


def _fail(reason):
    raise InvalidSimScenarioException(f"Invalid 'messages' section: {reason}")
