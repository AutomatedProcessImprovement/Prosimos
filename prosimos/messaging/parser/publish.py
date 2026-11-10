"""The rules of 'publish' entries: fan-out, attributes, fan-in, deadline, and carry."""
from collections import Counter

from prosimos.messaging.model import AGGREGATE_FUNCTIONS, PublishPoint
from prosimos.messaging.parser.common import (check_keys, fail, is_number, is_whole_number_of_at_least_one,
                                              read_period)
from prosimos.messaging.parser.consume import read_same

ONE_SIZE = "a whole number of at least 1 or {\"object_attribute\": <object attribute>}"
MIN_MAX = "{\"min\": <whole number>, \"max\": <whole number or object_attribute>}"


def read_carry(entry, where, tasks, writing_events):
    """The entry's optional 'carry': true, or a list of activities, tasks or message events that write OCEL events
    (writing_events: by id and name -> whether they do); it needs a qualifier on the entry."""
    if "carry" not in entry:
        return False
    carry = entry["carry"]
    if not (isinstance(carry, bool)
            or isinstance(carry, list) and all(isinstance(name, str) and name.strip() for name in carry)):
        fail(f"{where}: 'carry' must be true or a list of activities (names or ids of tasks or message events), got "
             f"{carry!r}")
    named = [name for name in carry if name not in tasks] if isinstance(carry, list) else []
    unknown = [name for name in named if name not in writing_events]
    if unknown:
        fail(f"{where}: carry names {unknown}, which are no tasks or message events of the process")
    silent = [name for name in named if not writing_events[name]]
    if silent:
        fail(f"{where}: carry names {silent[0]}, which writes no event: its entry has no qualifier")
    if carry and "qualifier" not in entry:
        fail(f"{where}: carry needs a qualifier on the same entry; a carried link needs its role word")
    return tuple(carry) if isinstance(carry, list) else carry


def read_attributes(entry, where, declared):
    attributes = entry.get("attributes", [])
    if isinstance(attributes, list) and any(isinstance(name, dict) for name in attributes):
        fail(f"{where}: a list or a function in 'attributes' needs a fan_in: without one, the message carries a "
              f"single case's values, by their names")
    if not isinstance(attributes, list) or not all(isinstance(name, str) and name for name in attributes):
        fail(f"{where}: 'attributes' must be a list of attribute names")
    if "index" in attributes and "fan_out" not in entry:
        fail(f"{where}: 'index' is the message's number among the ones fan_out publishes, so it needs a 'fan_out' "
              f"on the same entry")
    fan_out = entry.get("fan_out")
    # (a reserved as is reported by read_multiplicity)
    if isinstance(fan_out, dict) and fan_out.get("as") in attributes and fan_out["as"] not in ("object_id", "index"):
        fail(f"{where}: {fan_out['as']!r} is both in 'attributes' and fan_out's 'as': each message carries its own "
              f"entry of {fan_out.get('list')} under that name, so it can't also carry the case's attribute")
    unknown = [name for name in attributes if name not in ("object_id", "index") and name not in declared]
    if unknown:
        fail(f"{where}: {', '.join(map(repr, unknown))} not declared as an object, global or event attribute")
    return tuple(attributes)


def read_fan_in(entry, where, declared, links):
    """A publish entry with fan_in: e.g. {"fan_in": {"object_attribute": "items"}, "same": ["order_id"],
    "attributes": ["order_id", {"item_ids": "object_id"}, {"total_weight": {"sum": "weight"}}]}."""
    if "fan_out" in entry:
        fail(f"{where}: fan_in and fan_out can't be combined on one entry: one turns several cases into one "
              f"message, the other one case into several messages")
    size, fewest, fewest_attribute = entry["fan_in"], None, None
    if isinstance(size, dict) and ("min" in size or "max" in size):
        check_keys(f"{where}.fan_in", size, {"min", "max"})
        if set(size) != {"min", "max"}:
            fail(f"{where}: 'fan_in' needs both min and max, e.g. {{\"min\": 2, \"max\": 5}}, got {size!r}")
        for bound in ("min", "max"):
            if isinstance(size[bound], dict):
                check_keys(f"{where}.fan_in.{bound}", size[bound], {"object_attribute"})
        fewest, fewest_attribute = _fan_in_size(size["min"], "'fan_in' min", ONE_SIZE, where, declared)
        number, attribute = _fan_in_size(size["max"], "'fan_in' max", ONE_SIZE, where, declared)
        # from the case, a max below min counts as min, with a warning
        if number is not None and fewest is not None and fewest > number:
            fail(f"{where}: 'fan_in' min {fewest} is above max {number}")
        if "deadline" not in entry:
            fail(f"{where}: 'fan_in' min needs a deadline: without one, a group is sent only once it has max "
                  f"parts, so min would never count")
    else:
        if isinstance(size, dict):
            check_keys(f"{where}.fan_in", size, {"object_attribute", "min", "max"})
        number, attribute = _fan_in_size(size, "'fan_in'", f"{ONE_SIZE}, or {MIN_MAX}", where, declared)
    same = read_same(entry, where, kind="object")
    undeclared = [name for name in same if name not in declared]
    if undeclared:
        fail(f"{where}: 'same' names {', '.join(map(repr, undeclared))}, not declared as an object, global or event "
              f"attribute")
    plain, gathered, computed = _fan_in_attributes(entry, where, declared, same)
    qualifier, o2o, carry = links
    return PublishPoint(entry["event_id"], entry["type"], plain, qualifier=qualifier, o2o=o2o, carry=carry,
                        fan_in=number, fan_in_attribute=attribute, same=same, gathered=gathered, computed=computed,
                        deadline=read_period(entry, where, "deadline"), fan_in_min=fewest,
                        fan_in_min_attribute=fewest_attribute)


def _fan_in_size(size, what, forms, where, declared):
    """(a fixed size, the case attribute holding it) of a fan_in, or of its max: 3 or {"object_attribute": "items"}."""
    if is_whole_number_of_at_least_one(size):
        return int(size), None
    if isinstance(size, dict) and isinstance(size.get("object_attribute"), str):
        if size["object_attribute"] not in declared:
            fail(f"{where}: {what} object_attribute {size['object_attribute']!r} is not a declared object, global or "
                  f"event attribute")
        return None, size["object_attribute"]
    fail(f"{where}: {what} must be {forms}, got {size!r}")


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
            fail(f"{where}: 'attributes' of a fan-in entry must list attribute names, {{<message attribute>: "
                  f"<object attribute>}} or {{<message attribute>: {{<function>: <object attribute>}}}}")
        for name, source in item.items():
            if isinstance(source, str) and source:
                gathered.append((name, source))
                continue
            if not isinstance(source, dict):
                fail(f"{where}: attribute {name!r} must be an object attribute, whose values go into a list, or "
                      f"{{<function>: <object attribute>}}, got {source!r}")
            check_keys(f"{where}.attributes.{name}", source, AGGREGATE_FUNCTIONS | {"initial"})
            functions = {key: value for key, value in source.items() if key != "initial"}
            if len(functions) != 1 or not all(isinstance(value, str) and value for value in functions.values()):
                fail(f"{where}: attribute {name!r} must be one function and an object attribute, e.g. "
                      f"{{\"sum\": \"weight\"}}")
            if "initial" in source and not is_number(source["initial"]):
                fail(f"{where}: attribute {name!r} initial must be a number, got {source['initial']!r}")
            (function, of), = functions.items()
            computed.append((name, function, of, source.get("initial")))
    names = plain + [name for name, _ in gathered] + [name for name, *_ in computed]
    sources = [source for _, source in gathered] + [source for _, _, source, _ in computed]
    if "index" in names + sources:
        fail(f"{where}: 'index' numbers the messages of a fan_out, but a fan-in entry sends one message per group")
    if "object_id" in names:
        fail(f"{where}: a fan-in message can't carry object_id: it has several publishers. Send their ids as a "
              f"list, e.g. {{\"item_ids\": \"object_id\"}}")
    repeated = sorted(name for name, times in Counter(names).items() if times > 1)
    if repeated:
        fail(f"{where}: 'attributes' sends {', '.join(map(repr, repeated))} more than once")
    differing = [name for name in plain if name not in same]
    if differing:
        reason = "isn't in 'same', so the members of a group may differ in it" if same else (
            "is sent by its plain name, but without 'same' the members of a group may differ in every attribute")
        fail(f"{where}: {', '.join(map(repr, differing))} {reason}: send it as a list, "
              f"{{<name>: \"{differing[0]}\"}}, or a function over the members, {{<name>: {{\"sum\": "
              f"\"{differing[0]}\"}}}}")
    undeclared = [source for source in sources if source != "object_id" and source not in declared]
    if undeclared:
        fail(f"{where}: {', '.join(map(repr, undeclared))} not declared as an object, global or event attribute")
    return tuple(plain), tuple(gathered), tuple(computed)
