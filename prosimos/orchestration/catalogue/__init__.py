"""
Catalogues and picks (docs/catalogue.md): objects with no events of their own, a process's references to them, and
the picks that assign them to its cases.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

from prosimos.case_attributes import CaseAttribute
from prosimos.control_flow_manager import BPMN
from prosimos.messaging.model import AGGREGATE_FUNCTIONS
from prosimos.messaging.parser import is_number, is_whole_number
from prosimos.orchestration.catalogue.model import (CataloguePool, Computed, CountFrom, GrowingCatalogue, NewShare,
                                                    catalogue_ids)
from prosimos.simulation_properties_parser import parse_simulation_model


def checked_catalogue(organizations):
    """Of the organizations and their processes: catalogue names unique across the organizations and unlike the
    processes' object types, since both are OCEL object types; ids unique across all catalogues and unlike the
    processes' object ids (<object type>-<number>), since both are OCEL object ids; pools with a count of at least 1
    and attributes in the format of object_attributes; and references only naming catalogues of their own
    organization and, in link_events, tasks of their process (by name or id). A
    pick is "uniform" or weights on known ids of a listed catalogue, "uniform" on a pool, or "new" or new_share on
    a growing catalogue, for an attribute that the process's object_attributes don't have and its consume entries
    don't copy, append or aggregate into, and at most one per process and attribute. A growing catalogue's objects
    are created by exactly one reference of its organization, and no other id looks like them
    (<catalogue name>-<number>). with gives attributes that the process doesn't otherwise pick, draw, copy,
    append or aggregate into, and not object_id, from attributes that every object of the picked listed catalogue
    or pool has, as numbers for a sum, min or max; a pick with count only computes them. A catalogue change names a
    catalogue of its own organization and an attribute that its objects have, as a number to multiply or add to,
    and only ids of the catalogue in objects."""
    # (where it is in the configuration, process)
    placed = [(f"organizations[{number}].processes[{index}]", spec)
              for number, organization in enumerate(organizations) for index, spec in enumerate(organization.processes)]
    processes = [spec for _, spec in placed]
    problems = []
    owners = defaultdict(list)  # catalogue name -> the organizations declaring it
    for organization in organizations:
        for name in organization.catalogues:
            owners[name].append(organization.name)
    for name, declaring in owners.items():
        if len(declaring) > 1:
            problems.append(f"catalogue {name!r} is declared by more than one organization: {', '.join(declaring)}")
    catalogue = {name: entry for organization in organizations for name, entry in organization.catalogues.items()}
    # process name -> the catalogues of its organization, the ones its references may name
    catalogues_of = {spec.name: organization.catalogues for organization in organizations
                     for spec in organization.processes}
    # (where it is in the configuration, reference)
    references = [(f"{where}.references[{number}]", reference)
                  for where, spec in placed for number, reference in enumerate(spec.references)]
    for name, entry in catalogue.items():
        if not isinstance(entry, (CataloguePool, GrowingCatalogue)):
            continue
        if isinstance(entry, CataloguePool) and (not isinstance(entry.count, int) or isinstance(entry.count, bool)
                                                 or entry.count < 1):
            problems.append(f"catalogue {name!r}: count must be a whole number of at least 1, got {entry.count!r}")
        for number, attribute in enumerate(entry.attributes):
            try:
                CaseAttribute(attribute["name"], attribute["type"], attribute["values"])
            except Exception as error:  # Prosimos's own messages, as for object_attributes
                problems.append(f"catalogue {name!r}: attributes[{number}]: {type(error).__name__}: {error}")
    ids = [object_id for name, entry in catalogue.items() for object_id in catalogue_ids(name, entry)]
    repeated = sorted(object_id for object_id, times in Counter(ids).items() if times > 1)
    if repeated:
        problems.append(f"catalogue ids are repeated: {repeated}")
    object_types = {spec.object_type: spec.name for spec in processes if spec.object_type is not None}
    object_id = re.compile("|".join(rf"{re.escape(object_type)}-\d+" for object_type in object_types))
    like_object_ids = sorted({obj for obj in ids if object_types and object_id.fullmatch(obj)})
    if like_object_ids:
        problems.append(f"catalogue ids look like object ids (<object type>-<number>): {like_object_ids}")
    growing = [name for name, entry in catalogue.items() if isinstance(entry, GrowingCatalogue)]
    for name in growing:
        created_id = re.compile(rf"{re.escape(name)}-\d+")
        like_created = sorted({object_id for object_id in ids if created_id.fullmatch(object_id)})
        if like_created:
            problems.append(f"catalogue ids look like the ids growing catalogue {name!r} creates "
                            f"({name}-<number>): {like_created}")
        creators = [where for where, reference in references if reference.catalogue == name and reference.creates]
        if not creators:
            problems.append(f"growing catalogue {name!r} needs exactly one reference that creates its objects "
                            f'("new" or new_share), got 0')
        elif len(creators) > 1:
            # in an organization, one process creates the objects; the others learn their ids from messages
            problems.append(f"growing catalogue {name!r} has more than one reference that creates its objects "
                            f'("new" or new_share): {", ".join(creators)}; one creates them, the others learn their '
                            f"ids from messages")
    for name in catalogue:
        if name in object_types:
            problems.append(f"catalogue {name!r} is the object type of process {object_types[name]}")
    specs = {spec.name: spec for spec in processes}
    tasks_of = {}
    for where, reference in references:
        if reference.catalogue not in catalogues_of[reference.process]:
            # what another organization knows reaches this one only in messages
            problems.append(f"{where} names catalogue {reference.catalogue!r} of organization "
                            f"{', '.join(owners[reference.catalogue])}; a reference names a catalogue of its own "
                            f"organization"
                            if reference.catalogue in owners else f"{where} names unknown catalogue {reference.catalogue!r}")
            continue
        if reference.pick is None and reference.object_attribute in _consumed_into(reference.process, specs,
                                                                                   "aggregate"):
            # a number computed over the messages, not catalogue ids: nothing to link or to look up
            problems.append(f"{where} {reference.object_attribute} is computed by aggregate; a reference needs an "
                            f"attribute holding catalogue ids (copy, append, drawn or picked)")
            continue
        if reference.link_events:
            if reference.process not in tasks_of:
                graph = parse_simulation_model(specs[reference.process].bpmn_path)
                tasks_of[reference.process] = {key for info in graph.element_info.values() if info.type == BPMN.TASK
                                               for key in (info.id, info.name)}
            unknown = [task for task in reference.link_events if task not in tasks_of[reference.process]]
            if unknown:
                problems.append(f"{where} link_events names {unknown}, which are no tasks of process {reference.process}")
        if reference.pick is not None:
            problems.extend(f"{where} {problem}"
                            for problem in _pick_problems(reference, catalogue[reference.catalogue], specs))
        if reference.count is not None:
            problems.extend(f"{where} {problem}" for problem in _count_problems(reference, specs))
    picked = [(reference.process, reference.object_attribute) for _, reference in references if reference.pick is not None]
    for process, attribute in sorted({pair for pair in picked if picked.count(pair) > 1}):
        problems.append(f"more than one reference picks {attribute} of process {process}")
    brought = [(reference.process, name) for _, reference in references for name in brought_names(reference)]
    for where, reference in references:
        if reference.with_attributes and reference.catalogue in catalogues_of[reference.process]:
            problems.extend(f"{where} {problem}" for problem in _with_problems(
                reference, catalogue[reference.catalogue], specs, set(picked), brought))
    for number, organization in enumerate(organizations):
        for index, change in enumerate(organization.catalogue_changes):
            problems.extend(f"organizations[{number}].catalogue_changes[{index}] {problem}"
                            for problem in _change_problems(change, organization.catalogues, owners))
    if problems:
        raise ValueError("Invalid catalogue: " + "; ".join(problems))


def _change_problems(change, catalogues, owners):
    """catalogues: those of the change's organization; owners: catalogue name -> the organizations declaring it."""
    name = change.catalogue
    if name not in catalogues:
        # an organization's data is its own; another learns of it only in messages
        return [f"changes catalogue {name!r} of organization {', '.join(owners[name])}; an organization changes "
                f"its own catalogues" if name in owners else f"changes unknown catalogue {name!r}"]
    entry = catalogues[name]
    arithmetic = change.operation in ("multiply", "add")
    if change.objects is not None:
        if isinstance(entry, GrowingCatalogue):
            return [f"names objects {list(change.objects)} of growing catalogue {name!r}, which has none when the run "
                    f"starts; without objects, the change applies to every object it creates"]
        unknown = [object_id for object_id in change.objects if object_id not in catalogue_ids(name, entry)]
        if unknown:
            return [f"names objects {unknown}, which are no ids of catalogue {name!r}"]
    if isinstance(entry, list):
        changed = [obj for obj in entry if change.objects is None or obj.id in change.objects]
        missing = [obj.id for obj in changed if change.attribute not in dict(obj.attributes)]
        if missing:
            return [f"changes {change.attribute!r}, which objects {missing} of catalogue {name!r} don't have"]
        not_numbers = [obj.id for obj in changed if not is_number(dict(obj.attributes)[change.attribute])]
        if arithmetic and not_numbers:
            return [f"{change.operation} needs a number, and {change.attribute!r} of objects {not_numbers} of catalogue "
                    f"{name!r} isn't one"]
        return []
    specs_of = {attribute.get("name"): attribute for attribute in entry.attributes}
    if change.attribute not in specs_of:
        return [f"changes {change.attribute!r}, which catalogue {name!r} doesn't draw in its attributes"]
    if arithmetic and not _drawn_as_numbers(specs_of[change.attribute]):
        return [f"{change.operation} needs a number, and catalogue {name!r} doesn't always draw {change.attribute!r} "
                f"as one"]
    return []


def _pick_problems(reference, entry, specs):
    if isinstance(entry, GrowingCatalogue) and not reference.creates:
        return [f'picks from growing catalogue {reference.catalogue!r}, which only takes "new" or {{"new_share": ...}}']
    if reference.creates:
        if entry is not None and not isinstance(entry, GrowingCatalogue):
            return [f'picks "new" or new_share from catalogue {reference.catalogue!r}, which is no growing catalogue']
        share = reference.pick.share if isinstance(reference.pick, NewShare) else 1
        if not isinstance(share, (int, float)) or isinstance(share, bool) or not 0 < share <= 1:
            return [f"new_share must be a number above 0 and at most 1, got {share!r}"]
    elif reference.pick != "uniform":
        if not isinstance(reference.pick, tuple):
            return ['pick must be "uniform", {"weights": {id: weight, ...}}, "new" or {"new_share": share}']
        if isinstance(entry, CataloguePool):
            return [f"has weights on pool {reference.catalogue!r}; a pool can only be picked from uniformly"]
        if entry is not None:
            unknown = [object_id for object_id, _ in reference.pick if object_id not in {obj.id for obj in entry}]
            if unknown:
                return [f"weights name {unknown}, which are no ids of catalogue {reference.catalogue!r}"]
        weights = [weight for _, weight in reference.pick]
        if not all(isinstance(weight, (int, float)) and not isinstance(weight, bool) and weight >= 0
                   for weight in weights) or sum(weights) <= 0:
            return [f"weights must be numbers of at least 0, not all 0, got {dict(reference.pick)}"]
    return _settings_problems("picks", reference.object_attribute, reference.process, specs)


def _count_problems(reference, specs):
    count = reference.count
    if reference.pick is None:
        return ["has count but no pick; count says how many objects the pick chooses"]
    if reference.creates:
        return ['has count with "new" or new_share; count works with "uniform" or weights, which choose among the '
                'objects there are']
    if isinstance(count, CountFrom):
        with open(specs[reference.process].json_path) as file:
            declared = {entry.get("name") for entry in json.load(file).get("object_attributes", [])}
        if count.object_attribute not in declared:
            return [f"count reads {count.object_attribute!r}, which process {reference.process} doesn't draw in its "
                    f"object_attributes; the count is read when the case is created"]
        return []
    if not is_whole_number(count, minimum=0):
        return [f'count must be a whole number of at least 0 or {{"object_attribute": <object attribute>}}, '
                f"got {count!r}"]
    return []


def _with_json(pairs):
    """with as the configuration wrote it, for messages."""
    if not isinstance(pairs, tuple):
        return pairs
    def written(source):
        if not isinstance(source, Computed):
            return source
        return {source.function: source.source, **({} if source.initial is None else {"initial": source.initial})}

    return {pair[0]: written(pair[1]) for pair in pairs if isinstance(pair, tuple) and len(pair) == 2}


def _drawn_as_numbers(attribute):
    """Whether a pool attribute, in the format of object_attributes, always draws a number: a continuous one, or a
    discrete one whose keys are all numbers."""
    if attribute.get("type") == "continuous":
        return True
    values = attribute.get("values")
    return isinstance(values, list) and all(isinstance(value, dict) and is_number(value.get("key")) for value in values)


def _settings_problems(verb, attribute, process, specs):
    """An attribute that the orchestrator gives a process's cases when they are created, which its JSON settings
    also set: drawn in object_attributes, or set from a message."""
    with open(specs[process].json_path) as file:
        settings = json.load(file)
    declared = {entry.get("name") for entry in settings.get("object_attributes", [])}
    if attribute in declared:
        return [f"{verb} {attribute}, which process {process} also draws in its object_attributes"]
    # a copied value would replace the given one (with "new", the replaced object is never written), an
    # appended attribute is a list, and an aggregated one is computed over the messages claimed
    messages = settings.get("messages")
    consume = messages.get("consume", []) if isinstance(messages, dict) else []
    for key, also in (("copy", "copies"), ("append", "appends to"), ("aggregate", "aggregates into")):
        found_at = [entry.get("event_id") for entry in consume if isinstance(entry, dict)
                    and isinstance(entry.get(key), dict) and attribute in entry[key]]
        if found_at:
            return [f"{verb} {attribute}, which process {process} also {also} from a message (consume entries at "
                    f"{found_at})"]
    return []


def _consumed_into(process, specs, key):
    """The attributes process's consume entries write with key: "append" (lists of the values its messages bring) or
    "aggregate" (numbers computed over them)."""
    with open(specs[process].json_path) as file:
        messages = json.load(file).get("messages")
    consume = messages.get("consume", []) if isinstance(messages, dict) else []
    return {target for entry in consume if isinstance(entry, dict) and isinstance(entry.get(key), dict)
            for target in entry[key]}


def brought_names(reference):
    """The attributes a reference's with gives its cases (none if with is malformed, which _with_problems reports)."""
    pairs = reference.with_attributes
    return [pair[0] for pair in pairs if isinstance(pair, tuple) and len(pair) == 2] if isinstance(pairs, tuple) else []


def _with_problems(reference, entry, specs, picked, brought):
    """picked: (process, attribute) of every pick; brought: (process, attribute) of every attribute a with gives."""
    pairs = reference.with_attributes
    if isinstance(pairs, tuple):
        for name, source in (pair for pair in pairs if isinstance(pair, tuple) and len(pair) == 2):
            if isinstance(source, dict) and set(source) == {"initial"}:
                return [f"with {name!r} has initial but no function; initial is the first value a sum, count, min "
                        f"or max works over, and a plain copy has none"]
            initial = source.get("initial") if isinstance(source, dict) else getattr(source, "initial", None)
            if (isinstance(source, Computed) and initial is not None or isinstance(source, dict) and "initial" in source) \
                    and not is_number(initial):
                return [f"with {name!r} initial must be a number, got {initial!r}"]
    if not (isinstance(pairs, tuple) and all(
            isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[0], str) and pair[0]
            and (isinstance(pair[1], str) and pair[1]
                 or isinstance(pair[1], Computed) and pair[1].function in AGGREGATE_FUNCTIONS
                 and isinstance(pair[1].source, str) and pair[1].source) for pair in pairs)):
        return ['with must map attribute names to an attribute of the picked object or to one function of one, '
                f'{{"sum" | "count" | "min" | "max": <attribute>}}, got {_with_json(pairs)!r}']
    if isinstance(entry, GrowingCatalogue):
        return [f"has with on growing catalogue {reference.catalogue!r}; with doesn't read a growing catalogue yet, "
                f"a known limitation"]
    copied = [name for name, source in pairs if isinstance(source, str)]
    if reference.count is not None and copied:
        return [f"copies {copied} on a pick with count; a list pick needs a function over the picked objects, "
                f'e.g. {{"sum": "price"}}']
    if reference.pick is None and copied and reference.object_attribute in _consumed_into(reference.process, specs,
                                                                                          "append"):
        return [f"copies {copied} from {reference.object_attribute}, a list process {reference.process} appends to "
                f'from messages; a list needs a function over its objects, e.g. {{"sum": "price"}}, recomputed at '
                f"every claim"]
    sources = list(dict.fromkeys(source.source if isinstance(source, Computed) else source for _, source in pairs))
    numeric = list(dict.fromkeys(source.source for _, source in pairs
                                 if isinstance(source, Computed) and source.function != "count"))
    if entry is not None:
        if isinstance(entry, CataloguePool):
            specs_of = {attribute.get("name"): attribute for attribute in entry.attributes}
            missing = [name for name in sources if name not in specs_of]
            not_numbers = [name for name in numeric if name in specs_of and not _drawn_as_numbers(specs_of[name])]
        else:
            missing = [name for name in sources if not all(name in dict(obj.attributes) for obj in entry)]
            not_numbers = [name for name in numeric if name not in missing
                           and not all(is_number(dict(obj.attributes)[name]) for obj in entry)]
        if missing:
            return [f"with names {missing}, which not every object of catalogue {reference.catalogue!r} has"]
        if not_numbers:
            return [f"with sums, or takes the min or max of, {not_numbers}, which not every object of catalogue "
                    f"{reference.catalogue!r} has as a number"]
    problems = []
    for name, _ in pairs:
        if name == "object_id":
            problems.append("brings object_id, which is reserved for the object's own id")
        elif (reference.process, name) in picked:
            problems.append(f"brings {name}, which process {reference.process} also picks")
        elif brought.count((reference.process, name)) > 1:
            problems.append(f"brings {name}, which another with also brings to process {reference.process}")
        else:
            problems.extend(_settings_problems("brings", name, reference.process, specs))
    return problems
