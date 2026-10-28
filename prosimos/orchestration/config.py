"""
Reading and checking the simulation configuration (docs/orchestrator.md, "Configuration"): the processes, their
catalogues and references, and the consumer groups.
"""
import json

from prosimos.messaging_parser import AGGREGATE_FUNCTIONS, is_number, unknown_keys
from prosimos.orchestration.catalogue import (CatalogueObject, CataloguePool, Computed, CountFrom, GrowingCatalogue,
                                            NewShare, Reference, _brought_names)


CONFIG_KEYS = {"processes", "start_time", "seed", "consumer_groups"}
PROCESS_KEYS = {"name", "bpmn_path", "json_path", "arrivals", "object_type", "qualifier", "qualifier_by_activity",
                "carry_links", "ocel_attributes", "catalogues", "references"}
REFERENCE_KEYS = {"object_attribute", "catalogue", "o2o", "reverse", "link_events", "pick", "count", "with",
                  "tag_along"}


def _checked_ocel_attributes(processes):
    """Each name in a process's ocel_attributes is an attribute its objects have: drawn in its object_attributes,
    picked or brought (with) by one of its references, or copied, appended or aggregated into by one of its consume
    entries."""
    problems = []
    for index, spec in enumerate(processes):
        if not spec.ocel_attributes:
            continue
        with open(spec.json_path) as file:
            settings = json.load(file)
        has = {entry.get("name") for entry in settings.get("object_attributes", [])}
        has.update(reference.object_attribute for reference in spec.references if reference.pick is not None)
        has.update(name for reference in spec.references for name in _brought_names(reference))
        messages = settings.get("messages")
        for entry in messages.get("consume", []) if isinstance(messages, dict) else []:
            for key in ("copy", "append", "aggregate"):
                if isinstance(entry, dict) and isinstance(entry.get(key), dict):
                    has.update(entry[key])
        unknown = [name for name in spec.ocel_attributes if name not in has]
        if unknown:
            problems.append(f"processes[{index}].ocel_attributes names {unknown}, which the objects of process "
                            f"{spec.name} never have (they have {sorted(has)})")
    if problems:
        raise ValueError("Invalid ocel_attributes: " + "; ".join(problems))


def _checked_consumer_groups(names, consumer_groups):
    """Every process in exactly one group, and groups only naming known processes. Without
    groups, every process is its own group."""
    if consumer_groups is None:
        return {name: [name] for name in names}

    groups_of = {}
    for group, members in consumer_groups.items():
        for member in members:
            groups_of.setdefault(member, []).append(group)

    problems = []
    unknown = sorted(member for member in groups_of if member not in names)
    if unknown:
        problems.append(f"groups name unknown processes {unknown}")
    missing = sorted(name for name in names if name not in groups_of)
    if missing:
        problems.append(f"processes {missing} are in no group")
    repeated = {member: groups for member, groups in sorted(groups_of.items()) if len(groups) > 1}
    if repeated:
        problems.append(f"processes are listed more than once: {repeated}")
    if problems:
        raise ValueError("Invalid consumer_groups: " + "; ".join(problems))
    return consumer_groups


def _ocel_attributes(path, where, entry):
    names = entry.get("ocel_attributes", [])
    if not isinstance(names, list) or not all(isinstance(name, str) and name for name in names):
        raise ValueError(f"{path}: {where}: 'ocel_attributes' must be a list of attribute names, got {names!r}")
    return tuple(names)


def _object_type(path, where, entry):
    object_type = entry.get("object_type")
    if "object_type" in entry and (not isinstance(object_type, str) or not object_type):
        raise ValueError(f"{path}: {where}: 'object_type' must be a non-empty string, got {object_type!r}")
    return object_type


def _catalogue(path, where, entries):
    if isinstance(entries, dict):  # a pool, or growing without count
        _check_keys(path, where, entries, {"count", "attributes"})
    if isinstance(entries, dict) and "count" not in entries:
        return GrowingCatalogue(tuple(entries.get("attributes", [])))
    if isinstance(entries, dict):
        return CataloguePool(entries.get("count"), tuple(entries.get("attributes", [])))
    return [_catalogue_object(path, f"{where}[{number}]", entry) for number, entry in enumerate(entries)]


def _catalogue_object(path, where, entry):
    _check_keys(path, where, entry, {"id", "attributes"})
    if not isinstance(entry.get("id"), str) or not entry["id"]:
        raise ValueError(f"{path}: {where}: 'id' must be a non-empty string")
    return CatalogueObject(entry["id"], tuple(entry.get("attributes", {}).items()))


def _reference(path, where, entry):
    _check_keys(path, where, entry, REFERENCE_KEYS)
    missing = [key for key in ("object_attribute", "catalogue") if key not in entry]
    if missing:
        raise ValueError(f"{path}: {where} is missing {missing}")
    link_events, pick = entry.get("link_events"), entry.get("pick")
    if isinstance(pick, dict):
        _check_keys(path, f"{where}.pick", pick, {"weights", "new_share"})
    # anything else but "uniform" and "new" is rejected when the configuration is checked
    if isinstance(pick, dict) and list(pick) == ["weights"] and isinstance(pick["weights"], dict):
        pick = tuple(pick["weights"].items())
    elif isinstance(pick, dict) and list(pick) == ["new_share"]:
        pick = NewShare(pick["new_share"])
    count = entry.get("count")
    if isinstance(count, dict):
        _check_keys(path, f"{where}.count", count, {"object_attribute"})
        count = CountFrom(count["object_attribute"]) if isinstance(count.get("object_attribute"), str) else count
    with_attributes = entry.get("with", ())
    if isinstance(with_attributes, dict):  # anything else is rejected when the configuration is checked
        with_attributes = tuple((name, _computed(source)) for name, source in with_attributes.items())
    return Reference(entry["object_attribute"], entry["catalogue"], entry.get("o2o"),
                     entry.get("reverse", False), None if link_events is None else tuple(link_events), pick,
                     with_attributes,
                     entry.get("tag_along", False), count)


def _computed(source):
    """A with source as JSON gives it: {function: attribute}, with initial if given, as Computed; anything else as
    it is, a plain name or a form _with_problems rejects."""
    if not isinstance(source, dict):
        return source
    functions = [key for key in source if key != "initial"]
    if len(functions) != 1 or functions[0] not in AGGREGATE_FUNCTIONS or "initial" in source and not is_number(
            source["initial"]):
        return source
    return Computed(functions[0], source[functions[0]], source.get("initial"))


def _check_keys(path, where, mapping, allowed):
    problem = unknown_keys(where, mapping, allowed)
    if problem:
        raise ValueError(f"{path}: {problem}")
