"""
The optional 'messages' section of a process's JSON settings (docs/messaging.md). The BPMN
model says where a process publishes, waits or is started (message events); this section says what
it publishes or accepts. Parsing and validation only; the engine acts on the result.
"""
import json
import math
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
PUBLISH_KEYS = {"event_id", "type", "attributes", "count", "qualifier", "o2o"}
CONSUME_KEYS = {"event_id", "type", "condition", "copy", "append", "aggregate", "capacity", "collect", "qualifier",
                "o2o"}
AGGREGATE_FUNCTIONS = {"sum", "count", "min", "max"}


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
    attributes: Tuple[str, ...]  # copied from the case's current values; 'case_id' is the case identifier,
    # 'index' the message's number among the ones count publishes
    count: Optional[int] = None  # how many messages are published at once, unless read from the case
    count_attribute: Optional[str] = None  # the case attribute holding that number, if any
    count_list: Optional[str] = None  # the case attribute holding a list: one message per entry, if any
    count_as: Optional[str] = None  # the attribute under which each of those messages carries its entry
    qualifier: Optional[str] = None  # of the OCEL links its messages give; None means the message type
    o2o: Optional[str] = None  # OCEL object link from the publishing case's object to each claimer's; None: none

    @property
    def link_qualifier(self):
        return self.qualifier or self.type

    @property
    def counted(self):
        """Whether the entry has a count (else it publishes one message, without an index)."""
        return self.count is not None or self.count_attribute is not None or self.count_list is not None


@dataclass(frozen=True)
class ConsumePoint:
    event_id: str
    type: str
    condition: Optional[Tuple[Tuple[ConditionTerm, ...], ...]]  # any alternative whose terms all hold; None accepts every message
    copy: Tuple[Tuple[str, str], ...] = ()  # (case attribute, message attribute): copied into the case
    starts_case: bool = False  # at the start event: an accepted message starts a new case
    capacity: int = 1  # how many waiting cases one message resumes, unless read from the message
    capacity_attribute: Optional[str] = None  # the message attribute holding the capacity, if any
    collect: int = 1  # min: how many messages a case must claim here before it continues, unless read from the case
    collect_attribute: Optional[str] = None  # the case attribute holding that number, if any
    qualifier: Optional[str] = None  # of the OCEL links the messages it takes give; None means the message type
    o2o: Optional[str] = None  # OCEL object link from the claiming case's object to the publisher's; None: none
    # (case attribute, message attribute): the message attribute's value is added to a list in the case, which
    # starts empty
    append: Tuple[Tuple[str, str], ...] = ()
    # (case attribute, function, message attribute): the function over the message attribute's values of every
    # message claimed, at this entry and any other aggregating into the same case attribute
    aggregate: Tuple[Tuple[str, str, str], ...] = ()
    # max: how many messages a case may claim here in all; once it has min, it also takes every other matching
    # message offered at that instant, up to max. None means the same as min, math.inf no limit; unless read from
    # the case attribute collect_max_attribute
    collect_max: Optional[float] = None
    collect_max_attribute: Optional[str] = None

    @property
    def link_qualifier(self):
        return self.qualifier or self.type


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


def parse_messages_file(json_path, bpmn_path, also_declared=()) -> MessagingModel:
    """also_declared: names of case attributes given from outside the JSON settings (e.g. picked by the
    orchestrator's references), which messages may use too."""
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
    # attributes a consume entry copies or appends from a message into the case can be published too
    messages = settings.get("messages")
    consume = messages.get("consume", []) if isinstance(messages, dict) else []
    names.update(target for entry in consume if isinstance(entry, dict) for key in ("copy", "append", "aggregate")
                 if isinstance(entry.get(key), dict) for target in entry[key] if isinstance(target, str))
    return names


def parse_messages(messages_json, bpmn_path, declared_attributes, case_attributes=()) -> MessagingModel:
    """Validates the 'messages' section against the BPMN model and the declared attribute names; case_attributes
    are the names drawn by the process's case_attributes, which append can't add to. None (no section) gives an
    empty model, so processes without messages behave as before."""
    if messages_json is None:
        return MessagingModel()
    if not isinstance(messages_json, dict):
        _fail("'messages' must be an object with only 'publish' and/or 'consume'")
    _check_keys("messages", messages_json, {"publish", "consume"})

    events, gateway_branches = _bpmn_events(bpmn_path)
    # consume first, so that a malformed copy (whose targets publish entries may use) is reported
    # as such rather than as an undeclared published attribute
    consume, collecting = [], []
    # count gets its own reason below
    for where, entry in _entries(messages_json, "consume", CONSUME_KEYS, events, CONSUMING_EVENTS, ("count",)):
        starts_case = events[entry["event_id"]][0] == "startEvent"
        low, low_attribute, high, high_attribute = _collect(entry, where, starts_case)
        point = ConsumePoint(entry["event_id"], entry["type"], _condition(entry, where), _copy(entry, where),
                             starts_case, *_capacity(entry, where, starts_case), low, low_attribute,
                             _name(entry, "qualifier", where), _name(entry, "o2o", where),
                             _append(entry, where, case_attributes), _aggregate(entry, where, case_attributes),
                             high, high_attribute)
        if "count" in entry:
            _fail(f"{where}: count is only for publish entries, on throw and end events: only they publish, and a "
                  f"start event starts exactly one case per message")
        if point.starts_case:
            _check_start(where, point, events)
        consume.append(point)
        if "collect" in entry:
            collecting.append((where, point.event_id))
    _check_collect_has_its_own_event(collecting, consume)
    _check_appended_only(consume)
    _check_aggregated_only(consume)
    _check_races(consume, events, gateway_branches)
    publish = tuple(
        PublishPoint(entry["event_id"], entry["type"], _attributes(entry, where, declared_attributes),
                     *_count(entry, where, declared_attributes), _name(entry, "qualifier", where),
                     _name(entry, "o2o", where))
        for where, entry in _entries(messages_json, "publish", PUBLISH_KEYS, events, PUBLISHING_EVENTS)
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
    if isinstance(capacity, dict):
        _check_keys(f"{where}.capacity", capacity, {"value", "attribute"})
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
    """(min, case attribute holding it, max, case attribute holding it), each bound fixed or read from the case:
    {"min": 2, "max": 5}, {"min": {"case_attribute": "items"}}, 3 or {"value": 3} for (3, 3), {"case_attribute":
    "items"} for both from items, "all" for (1, no limit), 0 for (0, 0). A max of None means the same as min, and
    math.inf no limit."""
    if "collect" not in entry:
        return 1, None, None, None
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
        _check_keys(f"{where}.collect", collect, {"value", "case_attribute", "min", "max"})
    if not isinstance(collect, dict) or len(collect) != 1:
        if is_whole_number(collect, minimum=0):
            return int(collect), None, None, None
        _fail(f"{where}: 'collect' must be either a whole number of at least 0, \"all\", {{\"min\": <bound>, \"max\": "
              f"<bound>}} with each bound a number or {{\"case_attribute\": <case attribute>}}, {{\"value\": <number>}} or "
              f"{{\"case_attribute\": <case attribute>}}")
    if "case_attribute" in collect:
        if not isinstance(collect["case_attribute"], str) or not collect["case_attribute"]:
            _fail(f"{where}: 'collect' case_attribute must be a case attribute name")
        return 1, collect["case_attribute"], None, None
    if "max" in collect:
        _fail(f"{where}: 'collect' max needs a min, e.g. {{\"min\": 1, \"max\": {collect['max']!r}}}")
    if not is_whole_number(collect["value"], minimum=0):
        _fail(f"{where}: 'collect' value must be a whole number of at least 0, got {collect['value']!r}")
    return int(collect["value"]), None, None, None


def _bound(collect, name, where):
    """(fixed number, case attribute holding it) of collect's bound name (min or max): 2 or
    {"case_attribute": "items"}."""
    bound = collect[name]
    if isinstance(bound, dict):
        _check_keys(f"{where}.collect.{name}", bound, {"case_attribute"})
        if isinstance(bound.get("case_attribute"), str) and bound["case_attribute"]:
            return 1, bound["case_attribute"]
    elif is_whole_number(bound, minimum=0):
        return int(bound), None
    _fail(f"{where}: 'collect' {name} must be a whole number of at least 0 or {{\"case_attribute\": <case "
          f"attribute>}}, got {bound!r}")


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


def _copy(entry, where, key="copy"):
    copy = entry.get(key, {})
    if not isinstance(copy, dict) or not all(isinstance(target, str) and target and isinstance(source, str) and source
                                             for target, source in copy.items()):
        _fail(f"{where}: '{key}' must map case attribute names to message attribute names")
    if "case_id" in copy:
        _fail(f"{where}: '{key}' can't set case_id, which is reserved for the case's own identifier")
    return tuple(copy.items())


def _append(entry, where, case_attributes):
    append = _copy(entry, where, "append")
    both = [target for target, _ in append if target in entry.get("copy", {})]
    if both:
        _fail(f"{where}: {', '.join(map(repr, both))} is in both copy and append: a case attribute is either "
              f"overwritten by each message or a list that each message adds to")
    drawn = [target for target, _ in append if target in case_attributes]
    if drawn:
        _fail(f"{where}: append adds to {', '.join(map(repr, drawn))}, which case_attributes declares: an appended "
              f"attribute is a list that starts empty, not a drawn value")
    return append


def _aggregate(entry, where, case_attributes):
    aggregate = entry.get("aggregate", {})
    if not isinstance(aggregate, dict) or not all(isinstance(target, str) and target and isinstance(spec, dict)
                                                  for target, spec in aggregate.items()):
        _fail(f"{where}: 'aggregate' must map case attribute names to {{<function>: <message attribute>}}")
    if "case_id" in aggregate:
        _fail(f"{where}: 'aggregate' can't set case_id, which is reserved for the case's own identifier")
    result = []
    for target, spec in aggregate.items():
        _check_keys(f"{where}.aggregate.{target}", spec, AGGREGATE_FUNCTIONS)
        if len(spec) != 1 or not all(isinstance(source, str) and source for source in spec.values()):
            _fail(f"{where}: aggregate {target!r} must be one function and a message attribute, e.g. "
                  f"{{\"sum\": \"weight\"}}")
        (function, source), = spec.items()
        result.append((target, function, source))
    for key in ("copy", "append"):
        both = [target for target, _, _ in result if isinstance(entry.get(key), dict) and target in entry[key]]
        if both:
            _fail(f"{where}: {', '.join(map(repr, both))} is in both {key} and aggregate: an aggregated attribute is "
                  f"computed over the messages claimed, which {key} would overwrite")
    drawn = [target for target, _, _ in result if target in case_attributes]
    if drawn:
        _fail(f"{where}: aggregate writes {', '.join(map(repr, drawn))}, which case_attributes declares: an "
              f"aggregated attribute is computed over the messages claimed, not a drawn value")
    return tuple(result)


def _check_aggregated_only(consume):
    """An attribute some entry aggregates into is computed over the messages claimed, so no other entry may copy or
    append into it, and every entry aggregating into it uses the same function."""
    written_at = {}
    for index, point in enumerate(consume):
        for key, pairs in (("copies", point.copy), ("appends to", point.append)):
            for target, _ in pairs:
                written_at.setdefault(target, (index, key))
    function_at = {}
    for index, point in enumerate(consume):
        for target, function, _ in point.aggregate:
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
    """Each entry of the section, with where it is; own_reasons are keys the caller rejects with a reason of their
    own rather than as unknown."""
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
    # several arrows into an end or catch event mean "once per arriving token" in BPMN, but Prosimos
    # joins them (OR for end events, AND for catch events); requiring an explicit gateway avoids the
    # question. Throw events need no rule: they get a hidden XOR join, as the standard says
    if kind in ("endEvent", "intermediateCatchEvent") and incoming != 1:
        _fail(f"{where}: event_id {event_id!r} has {incoming} incoming arrows, expected exactly one; "
              f"draw an explicit gateway before {event_id}")


def _check_races(consume, events, gateway_branches):
    """An event-based gateway with a branch that waits for a message is a race: every branch must be a
    message catch event or a timer, and no branch may have a fixed collect of 0."""
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
    """An optional OCEL qualifier on the entry: 'qualifier' (of its messages' event-to-object links) or 'o2o'
    (of the object-to-object link they give); None without the key."""
    if key not in entry:
        return None
    if not isinstance(entry[key], str) or not entry[key].strip():
        _fail(f"{where}: '{key}' must be a non-empty string")
    return entry[key]


def _count(entry, where, declared):
    """(fixed number of messages, case attribute holding it, case attribute holding a list, the name each
    message carries its entry under): {"value": 3}, {"case_attribute": "items"} or
    {"list": "item_ids", "as": "item_id"}; all None without count."""
    if "count" not in entry:
        return None, None, None, None
    count = entry["count"]
    if isinstance(count, dict):
        _check_keys(f"{where}.count", count, {"value", "case_attribute", "list", "as"})
    if isinstance(count, dict) and "list" in count:
        return (None, None) + _count_over_list(count, where, declared)
    if not isinstance(count, dict) or len(count) != 1 or not set(count) <= {"value", "case_attribute"}:
        _fail(f"{where}: 'count' must be either {{\"value\": <number>}}, {{\"case_attribute\": <case attribute>}} or "
              f"{{\"list\": <case attribute>, \"as\": <attribute name>}}")
    if "case_attribute" in count:
        if not isinstance(count["case_attribute"], str) or count["case_attribute"] not in declared:
            _fail(f"{where}: 'count' case_attribute {count['case_attribute']!r} is not a declared case, global or "
                  f"event attribute")
        return None, count["case_attribute"], None, None
    if not is_whole_number(count["value"], minimum=0):
        _fail(f"{where}: 'count' value must be a whole number of at least 0, got {count['value']!r}")
    return int(count["value"]), None, None, None


def _count_over_list(count, where, declared):
    if not set(count) <= {"list", "as"}:
        _fail(f"{where}: 'count' over a list has only 'list' and 'as', got {', '.join(map(repr, sorted(count)))}")
    if not isinstance(count["list"], str) or count["list"] not in declared:
        _fail(f"{where}: 'count' list {count['list']!r} is not a declared case, global or event attribute")
    if "as" not in count:
        _fail(f"{where}: 'count' over a list needs 'as', the attribute name under which each message carries its "
              f"entry")
    if not isinstance(count["as"], str) or not count["as"].strip():
        _fail(f"{where}: 'count' as must be a non-empty string")
    if count["as"] in ("case_id", "index"):
        _fail(f"{where}: 'count' as can't be {count['as']}, which is reserved: case_id is the publishing case's "
              f"identifier and index the message's number")
    return count["list"], count["as"]


def _attributes(entry, where, declared):
    attributes = entry.get("attributes", [])
    if not isinstance(attributes, list) or not all(isinstance(name, str) and name for name in attributes):
        _fail(f"{where}: 'attributes' must be a list of attribute names")
    if "index" in attributes and "count" not in entry:
        _fail(f"{where}: 'index' is the message's number among the ones count publishes, so it needs a 'count' "
              f"on the same entry")
    count = entry.get("count")
    # (a reserved as is reported by _count)
    if isinstance(count, dict) and count.get("as") in attributes and count["as"] not in ("case_id", "index"):
        _fail(f"{where}: {count['as']!r} is both in 'attributes' and count's 'as': each message carries its own "
              f"entry of {count.get('list')} under that name, so it can't also carry the case's attribute")
    unknown = [name for name in attributes if name not in ("case_id", "index") and name not in declared]
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
    _check_keys(where, term, TERM_KEYS)
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


def unknown_keys(where, mapping, allowed):
    """The load error for the keys of mapping that aren't allowed, naming them, where they are and the allowed
    keys; None if there are none. Ignored, a misspelled key would change the run without any message."""
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
