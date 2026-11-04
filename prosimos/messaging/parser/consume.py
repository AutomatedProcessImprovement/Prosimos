"""The rules of 'consume' entries: start, capacity, collect, same, copy, append, aggregate, races, conditions."""
import math
from collections import Counter

from prosimos.messaging.model import AGGREGATE_FUNCTIONS, COMPARISONS, ConditionTerm
from prosimos.messaging.parser.common import (check_keys, fail, is_number, is_whole_number,
                                              is_whole_number_of_at_least_one)

TERM_KEYS = {"attribute", "comparison", "value", "object_attribute"}


def check_start(where, point):
    # at the start there is no case to compare with yet
    on_case = [term.attribute for alternative in point.condition or () for term in alternative if term.object_attribute]
    if on_case:
        fail(f"{where}: the condition of a start event may only use fixed values and source, not "
              f"object_attribute (there is no object yet), but its terms on {', '.join(map(repr, on_case))} do")


def read_capacity(entry, where, starts_case):
    """(min, its message attribute, max, its message attribute): e.g. {"min": 2, "max": {"attribute": "capacity"}},
    2 for (1, 2), {"min": 2} for (2, no limit)."""
    if "capacity" not in entry:
        return 1, None, 1, None
    if starts_case:
        fail(f"{where}: a start event takes no capacity: a start message starts exactly one case")
    capacity = entry["capacity"]
    if is_whole_number_of_at_least_one(capacity):
        return 1, None, int(capacity), None
    if isinstance(capacity, dict) and "min" in capacity:
        check_keys(f"{where}.capacity", capacity, {"min", "max"})
        fewest, fewest_attribute = _capacity_bound(capacity, "min", where)
        most, most_attribute = _capacity_bound(capacity, "max", where) if "max" in capacity else (math.inf, None)
        if fewest_attribute is None and most_attribute is None and fewest > most:
            fail(f"{where}: 'capacity' min {fewest} is above max {most}")
        return fewest, fewest_attribute, most, most_attribute
    if isinstance(capacity, dict):
        check_keys(f"{where}.capacity", capacity, {"attribute", "min", "max"})
        if set(capacity) == {"max"}:
            fail(f"{where}: 'capacity' max needs a min, e.g. {{\"min\": 1, \"max\": {capacity['max']!r}}}")
    if not isinstance(capacity, dict) or len(capacity) != 1:
        fail(f"{where}: 'capacity' must be either a whole number of at least 1, {{\"attribute\": <message "
              f"attribute>}} or {{\"min\": <bound>, \"max\": <bound>}} with each bound a number or {{\"attribute\": "
              f"<message attribute>}}")
    if not isinstance(capacity["attribute"], str) or not capacity["attribute"]:
        fail(f"{where}: 'capacity' attribute must be a message attribute name")
    return 1, None, 1, capacity["attribute"]


def _capacity_bound(capacity, name, where):
    bound = capacity[name]
    if isinstance(bound, dict):
        check_keys(f"{where}.capacity.{name}", bound, {"attribute"})
        if isinstance(bound.get("attribute"), str) and bound["attribute"]:
            return 1, bound["attribute"]
    elif is_whole_number_of_at_least_one(bound):
        return int(bound), None
    fail(f"{where}: 'capacity' {name} must be a whole number of at least 1 or {{\"attribute\": <message "
          f"attribute>}}, got {bound!r}")


def read_collect(entry, where, starts_case):
    """(min, its case attribute, max, its case attribute): e.g. {"min": 2, "max": 5}, 3 for (3, 3), "all" for
    (1, no limit). A max of None means the same as min."""
    if "collect" not in entry:
        return 1, None, None, None
    if starts_case:
        fail(f"{where}: a start event can't collect: it turns one message into a new case, and before that case "
              f"exists nothing holds the earlier messages or tells which ones belong together. Start on the first "
              f"message, then collect the rest at a catch event right after the start (e.g. a picking batch: "
              f"start on 1 order, then collect 4)")
    if "capacity" in entry:
        fail(f"{where}: collect and capacity can't be combined on one entry: that would be several messages "
              f"completing several cases at once. Use two steps through an intermediate case instead: it collects "
              f"the messages, then publishes one message per target (e.g. a truck collects packages, then sends "
              f"one Shipment per package)")
    collect = entry["collect"]
    if collect == "all":
        return 1, None, math.inf, None
    if isinstance(collect, dict) and "min" in collect:
        check_keys(f"{where}.collect", collect, {"min", "max"})
        low, low_attribute = _bound(collect, "min", where)
        high, high_attribute = _bound(collect, "max", where) if "max" in collect else (math.inf, None)
        if high_attribute is None and low_attribute is None and low > high:
            fail(f"{where}: 'collect' min {low} is above max {high}")
        if high == 0 and low_attribute is not None:
            fail(f"{where}: 'collect' max is 0 but min is read from {low_attribute}: a case waiting for a message "
                  f"couldn't take any")
        return low, low_attribute, high, high_attribute
    if isinstance(collect, dict):
        check_keys(f"{where}.collect", collect, {"object_attribute", "min", "max"})
    if not isinstance(collect, dict) or len(collect) != 1:
        if is_whole_number(collect, minimum=0):
            return int(collect), None, None, None
        fail(f"{where}: 'collect' must be either a whole number of at least 0, \"all\", {{\"object_attribute\": <object "
              f"attribute>}} or {{\"min\": <bound>, \"max\": <bound>}} with each bound a number or "
              f"{{\"object_attribute\": <object attribute>}}")
    if "max" in collect:
        fail(f"{where}: 'collect' max needs a min, e.g. {{\"min\": 1, \"max\": {collect['max']!r}}}")
    if not isinstance(collect["object_attribute"], str) or not collect["object_attribute"]:
        fail(f"{where}: 'collect' object_attribute must be an object attribute name")
    return 1, collect["object_attribute"], None, None


def read_same(entry, where, kind="message"):
    """The attributes whose values all messages a collecting case claims, or all members of a fan-in group,
    share: ["customer"]."""
    if "same" not in entry:
        return ()
    same = entry["same"]
    if not isinstance(same, list) or not same or not all(isinstance(name, str) and name for name in same):
        fail(f"{where}: 'same' must be a non-empty list of {kind} attribute names, got {same!r}")
    repeated = sorted(name for name, times in Counter(same).items() if times > 1)
    if repeated:
        fail(f"{where}: 'same' lists {', '.join(repeated)} more than once")
    if "collect" not in entry and kind == "message":
        fail(f"{where}: 'same' needs a collect: without one, a case takes a single message, so there is nothing "
              f"to compare it with")
    return tuple(same)


def _bound(collect, name, where):
    bound = collect[name]
    if isinstance(bound, dict):
        check_keys(f"{where}.collect.{name}", bound, {"object_attribute"})
        if isinstance(bound.get("object_attribute"), str) and bound["object_attribute"]:
            return 1, bound["object_attribute"]
    elif is_whole_number(bound, minimum=0):
        return int(bound), None
    fail(f"{where}: 'collect' {name} must be a whole number of at least 0 or {{\"object_attribute\": <object "
          f"attribute>}}, got {bound!r}")


def check_collect_has_its_own_event(collecting, consume):
    """collect needs a catch event with exactly one consume entry: a waiting case keeps one count there."""
    for where, event_id in collecting:
        entries = sum(point.event_id == event_id for point in consume)
        if entries > 1:
            fail(f"{where}: collect is only allowed on a catch event with exactly one consume entry, but "
                  f"{event_id} has {entries}: a case waiting there keeps one count. To accept several variants of "
                  f"one type, use one entry with alternatives in its condition; to wait for several kinds, use one "
                  f"catch event per type")


def read_copy(entry, where, key="copy"):
    copy = entry.get(key, {})
    if not isinstance(copy, dict) or not all(isinstance(target, str) and target and isinstance(source, str) and source
                                             for target, source in copy.items()):
        fail(f"{where}: '{key}' must map object attribute names to message attribute names")
    if "object_id" in copy:
        fail(f"{where}: '{key}' can't set object_id, which is reserved for the object's own id")
    return tuple(copy.items())


def read_append(entry, where, case_attributes):
    append = read_copy(entry, where, "append")
    both = [target for target, _ in append if target in entry.get("copy", {})]
    if both:
        fail(f"{where}: {', '.join(map(repr, both))} is in both copy and append: a case attribute is either "
              f"overwritten by each message or a list that each message adds to")
    drawn = [target for target, _ in append if target in case_attributes]
    if drawn:
        fail(f"{where}: append adds to {', '.join(map(repr, drawn))}, which object_attributes declares: an appended "
              f"attribute is a list that starts empty, not a drawn value")
    return append


def read_aggregate(entry, where, case_attributes):
    aggregate = entry.get("aggregate", {})
    if not isinstance(aggregate, dict) or not all(isinstance(target, str) and target and isinstance(spec, dict)
                                                  for target, spec in aggregate.items()):
        fail(f"{where}: 'aggregate' must map object attribute names to {{<function>: <message attribute>}}")
    if "object_id" in aggregate:
        fail(f"{where}: 'aggregate' can't set object_id, which is reserved for the object's own id")
    result = []
    for target, spec in aggregate.items():
        check_keys(f"{where}.aggregate.{target}", spec, AGGREGATE_FUNCTIONS | {"initial"})
        functions = {key: source for key, source in spec.items() if key != "initial"}
        if len(functions) != 1 or not all(isinstance(source, str) and source for source in functions.values()):
            fail(f"{where}: aggregate {target!r} must be one function and a message attribute, e.g. "
                  f"{{\"sum\": \"weight\"}}")
        initial = spec.get("initial")
        if "initial" in spec and not is_number(initial):
            fail(f"{where}: aggregate {target!r} initial must be a number, got {initial!r}")
        (function, source), = functions.items()
        result.append((target, function, source, initial))
    for key in ("copy", "append"):
        both = [target for target, *_ in result if isinstance(entry.get(key), dict) and target in entry[key]]
        if both:
            fail(f"{where}: {', '.join(map(repr, both))} is in both {key} and aggregate: an aggregated attribute is "
                  f"computed over the messages claimed, which {key} would overwrite")
    drawn = [target for target, *_ in result if target in case_attributes]
    if drawn:
        fail(f"{where}: aggregate writes {', '.join(map(repr, drawn))}, which object_attributes declares: an "
              f"aggregated attribute is computed over the messages claimed, not a drawn value")
    return tuple(result)


def check_aggregated_only(consume):
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
                fail(f"messages.consume[{index}]: aggregate writes {target!r}, which messages.consume[{other}] "
                      f"{verb}: an aggregated attribute is computed over the messages claimed, which that would "
                      f"overwrite")
            first, first_function = function_at.setdefault(target, (index, function))
            if function != first_function:
                fail(f"messages.consume[{index}]: aggregate writes {target!r} with {function}, but "
                      f"messages.consume[{first}] with {first_function}: an attribute is one function over all the "
                      f"messages claimed into it")
            first, first_initial = initial_at.setdefault(target, (index, initial))
            if initial != first_initial:
                fail(f"messages.consume[{index}]: aggregate writes {target!r} with initial {initial!r}, but "
                      f"messages.consume[{first}] with initial {first_initial!r}: an attribute starts from one "
                      f"value")


def check_appended_only(consume):
    """An attribute some entry appends to is a list, so no other entry may overwrite it with copy."""
    copied_at = {}
    for index, point in enumerate(consume):
        for target, _ in point.copy:
            copied_at.setdefault(target, index)
    for index, point in enumerate(consume):
        for target, _ in point.append:
            if target in copied_at:
                fail(f"messages.consume[{index}]: append adds to {target!r}, which messages.consume"
                      f"[{copied_at[target]}] copies into: an appended attribute is a list, which copy would "
                      f"overwrite")


def check_races(consume, events, gateway_branches):
    """A race (an event-based gateway with a message branch) has only message and timer branches, none with a
    fixed collect of 0."""
    waiting = {point.event_id for point in consume if not point.starts_case}
    for gateway_id in sorted({events[event_id][3] for event_id in waiting if events[event_id][3]}):
        branches = gateway_branches[gateway_id]
        for target_id, kind, definition in branches:
            if kind != "intermediateCatchEvent" or definition not in ("message", "timer"):
                found = f"a {definition} {kind}" if definition else f"{'an' if kind[0] in 'aeiou' else 'a'} {kind}"
                fail(f"after the event-based gateway {gateway_id}, which races a branch waiting for a message, "
                      f"every branch must be a message catch event or a timer, but {target_id} is {found}")
        for point in consume:
            if events[point.event_id][3] == gateway_id and point.collect == 0 and point.collect_attribute is None:
                fail(f"{point.event_id}, after the event-based gateway {gateway_id}: a race branch with a fixed "
                      f"collect of 0 always wins its race; remove the race or the branch")


def read_condition(entry, where):
    if "condition" not in entry:
        return None
    condition = entry["condition"]
    if not isinstance(condition, list) or not condition:
        fail(f"{where}: 'condition' must be a non-empty list of alternatives, each a list of terms that must all hold")
    alternatives = []
    for alternative_index, alternative in enumerate(condition):
        if not isinstance(alternative, list) or not alternative:
            fail(f"{where}.condition[{alternative_index}] must be a non-empty list of terms")
        alternatives.append(tuple(_term(term, f"{where}.condition[{alternative_index}][{term_index}]")
                                  for term_index, term in enumerate(alternative)))
    return tuple(alternatives)


def _term(term, where):
    if not isinstance(term, dict):
        fail(f"{where} must be an object")
    check_keys(where, term, TERM_KEYS)
    if not isinstance(term.get("attribute"), str) or not term["attribute"]:
        fail(f"{where}: 'attribute' must be a message attribute name or 'source'")
    if term.get("comparison") not in COMPARISONS:
        fail(f"{where}: 'comparison' must be one of {', '.join(COMPARISONS)}, got {term.get('comparison')!r}")
    if ("value" in term) == ("object_attribute" in term):
        fail(f"{where}: give exactly one of 'value' and 'object_attribute'")
    if "object_attribute" in term and (not isinstance(term["object_attribute"], str) or not term["object_attribute"]):
        fail(f"{where}: 'object_attribute' must be an object attribute name")
    if term["comparison"] == "in" and not (isinstance(term.get("value"), list) and len(term["value"]) == 2):
        fail(f"{where}: 'in' needs a fixed 'value' [low, high]")
    return ConditionTerm(term["attribute"], term["comparison"], term.get("value"), term.get("object_attribute"))
