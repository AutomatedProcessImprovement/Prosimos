import copy
import dataclasses
import csv
import hashlib
import itertools
import json
import random
import re
import tempfile
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pytz

from prosimos.case_attributes import CaseAttribute
from prosimos.control_flow_manager import BPMN
from prosimos.messaging_parser import (AGGREGATE_FUNCTIONS, ConsumePoint, PublishPoint, is_number, is_whole_number,
                                       unknown_keys)
from prosimos.ocel_writer import OcelProcess, write_ocel
from prosimos.simulation_engine import SimBPMEnv, _aggregated, _is_number
from prosimos.simulation_properties_parser import parse_datetime, parse_simulation_model
from prosimos.simulation_setup import SimDiffSetup
from prosimos.warning_logger import warning_logger


@dataclass(frozen=True)
class ProcessSpec:
    name: str
    bpmn_path: str
    json_path: str
    arrivals: Optional[int] = None  # how many cases arrive; None for a process started by messages
    # the OCEL object type of its cases, one kind of thing, in the singular; without one, the process writes no
    # objects and no events in the OCEL output
    object_type: Optional[str] = None
    # the qualifier of each event's link to its own case's object: None means the object type; per task id,
    # qualifier_by_activity overrides it, e.g. (("Create_Package", "creates"),)
    qualifier: Optional[str] = None
    qualifier_by_activity: Tuple[Tuple[str, str], ...] = ()
    # every later event of a case also links the objects the case is linked to through its messages
    carry_links: bool = False
    # the attributes written to its objects in the OCEL output; the others, e.g. those only routing messages, aren't
    ocel_attributes: Tuple[str, ...] = ()
    # its catalogues, name -> objects with no events of their own (see SimulationConfig), and its references, each
    # naming one of its own catalogues
    catalogues: Dict[str, Union[List["CatalogueObject"], "CataloguePool", "GrowingCatalogue"]] = field(
        default_factory=dict, hash=False)
    references: Tuple["Reference", ...] = ()

    def __post_init__(self):
        # each reference knows the process whose entry lists it
        object.__setattr__(self, "references", tuple(dataclasses.replace(reference, process=self.name)
                                                     for reference in self.references))

    @property
    def id_prefix(self) -> str:
        """Its objects' ids are <id prefix>-<number>: the object type, or the name of a process without one."""
        return self.object_type or self.name


@dataclass(frozen=True)
class CatalogueObject:
    """An object with no events of its own (a customer, a product), declared in a catalogue of a process."""

    id: str
    attributes: Tuple[Tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class CataloguePool:
    """A generated catalogue, for sets too large to list: count objects with ids <catalogue name>-1 to
    <catalogue name>-<count>, each with attributes drawn once at the start of the run, given in the format
    of object_attributes (discrete or continuous)."""

    count: int
    attributes: Tuple[Dict[str, Any], ...] = ()


@dataclass(frozen=True)
class GrowingCatalogue:
    """A catalogue with no list and no count, for populations that aren't fixed: it starts empty, and a reference
    that picks "new" or new_share creates its objects as cases are created, with ids <catalogue name>-1,
    <catalogue name>-2, ... in creation order, each with attributes in the format of object_attributes."""

    attributes: Tuple[Dict[str, Any], ...] = ()


@dataclass(frozen=True)
class NewShare:
    """A pick on a growing catalogue: share of the cases create a new object, the rest reuse one created so far,
    uniformly; the first case always creates one."""

    share: float


@dataclass(frozen=True)
class Computed:
    """A with value computed over the objects a reference picked, as aggregate computes over messages: function
    ("sum", "count", "min" or "max") of their attribute source, and of initial, if not None, a number already
    among them."""

    function: str
    source: str
    initial: Optional[float] = None


@dataclass(frozen=True)
class CountFrom:
    """A pick's count read from a case attribute when the case is created."""

    object_attribute: str


@dataclass(frozen=True)
class Reference:
    """Says that a case attribute's value of its process is the id of an object of one of its catalogues, or a list
    of such ids (each one linked as a single value is), and which OCEL links to write for it: from the case's object
    to it with o2o as qualifier (reverse turns it around), and from the case's events named in link_events (task
    names or ids; None for every event, () for none), and, with tag_along, from every event of another case that
    links the case's object.
    With pick, the reference also assigns the value, when the case is created: "uniform" (every object
    equally likely) or, for a listed catalogue, weights as ((id, weight), ...) (ids left out are never picked);
    for a growing catalogue, "new" (every case creates its own object) or NewShare. Without it, the value comes
    from object_attributes or a message. with_attributes (JSON "with") gives the case starting values from the
    picked objects, as ((attribute, source), ...): source is an attribute of the picked object, copied, or Computed
    over the picked objects (a single pick is a list of one). count, with "uniform" or weights, makes the pick
    choose that many objects, each independently, into a list: a whole number, or CountFrom, an attribute the case
    draws in its object_attributes. process is the process whose entry lists it, set by its ProcessSpec."""

    object_attribute: str
    catalogue: str
    o2o: Optional[str] = None
    reverse: bool = False
    link_events: Optional[Tuple[str, ...]] = None
    pick: Union[None, str, Tuple[Tuple[str, float], ...], NewShare] = None
    with_attributes: Tuple[Tuple[str, Union[str, Computed]], ...] = ()
    tag_along: bool = False
    count: Union[None, int, CountFrom] = None
    process: str = ""

    @property
    def creates(self) -> bool:
        return self.pick == "new" or isinstance(self.pick, NewShare)


CONFIG_KEYS = {"processes", "start_time", "seed", "consumer_groups"}
PROCESS_KEYS = {"name", "bpmn_path", "json_path", "arrivals", "object_type", "qualifier", "qualifier_by_activity",
                "carry_links", "ocel_attributes", "catalogues", "references"}
REFERENCE_KEYS = {"object_attribute", "catalogue", "o2o", "reverse", "link_events", "pick", "count", "with",
                  "tag_along"}


def _catalogue_ids(name, entry) -> List[str]:
    """The ids known when the run starts: none yet for a growing catalogue."""
    if isinstance(entry, GrowingCatalogue):
        return []
    if isinstance(entry, CataloguePool):
        return [f"{name}-{number}" for number in range(1, entry.count + 1)] if isinstance(entry.count, int) else []
    return [obj.id for obj in entry]


@dataclass
class Message:
    """One object passed between processes. The publishing process sets type and attributes;
    the orchestrator sets id, source and time when it stamps the message."""

    type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    source: Optional[str] = None
    time: Optional[datetime] = None


def _checked_catalogue(processes):
    """Catalogue names unique across the processes and unlike their object types, since both are OCEL object types;
    ids unique across all catalogues and unlike the processes' object ids (<object type>-<number>), since both are
    OCEL object ids; pools with a count of at least 1 and attributes in the format of object_attributes; and
    references only naming catalogues of their own process and, in link_events, tasks of it (by name or id). A
    pick is "uniform" or weights on known ids of a listed catalogue, "uniform" on a pool, or "new" or new_share on
    a growing catalogue, for an attribute that the process's object_attributes don't have and its consume entries
    don't copy, append or aggregate into, and at most one per process and attribute. A growing catalogue's objects are created by exactly one reference, and no other id looks like
    them (<catalogue name>-<number>). with gives attributes that the process doesn't otherwise pick, draw, copy,
    append or aggregate into, and not object_id, from attributes that every object of the picked listed catalogue
    or pool has, as numbers for a sum, min or max; a pick with count only computes them."""
    problems = []
    owners = defaultdict(list)  # catalogue name -> the processes declaring it
    for spec in processes:
        for name in spec.catalogues:
            owners[name].append(spec.name)
    for name, declaring in owners.items():
        if len(declaring) > 1:
            problems.append(f"catalogue {name!r} is declared by more than one process: {', '.join(declaring)}")
    catalogue = {name: entry for spec in processes for name, entry in spec.catalogues.items()}
    # (where it is in the configuration, reference)
    references = [(f"processes[{index}].references[{number}]", reference)
                  for index, spec in enumerate(processes) for number, reference in enumerate(spec.references)]
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
    ids = [object_id for name, entry in catalogue.items() for object_id in _catalogue_ids(name, entry)]
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
        creators = [reference for _, reference in references if reference.catalogue == name and reference.creates]
        if len(creators) != 1:
            problems.append(f"growing catalogue {name!r} needs exactly one reference that creates its objects "
                            f'("new" or new_share), got {len(creators)}')
    for name in catalogue:
        if name in object_types:
            problems.append(f"catalogue {name!r} is the object type of process {object_types[name]}")
    specs = {spec.name: spec for spec in processes}
    tasks_of = {}
    for where, reference in references:
        if reference.catalogue not in specs[reference.process].catalogues:
            # what another process knows reaches this one only in messages
            problems.append(f"{where} names catalogue {reference.catalogue!r} of process "
                            f"{', '.join(owners[reference.catalogue])}; a reference names a catalogue of its own process"
                            if reference.catalogue in owners else f"{where} names unknown catalogue {reference.catalogue!r}")
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
    brought = [(reference.process, name) for _, reference in references for name in _brought_names(reference)]
    for where, reference in references:
        if reference.with_attributes and reference.catalogue in specs[reference.process].catalogues:
            problems.extend(f"{where} {problem}" for problem in _with_problems(
                reference, catalogue[reference.catalogue], specs, set(picked), brought))
    if problems:
        raise ValueError("Invalid catalogue: " + "; ".join(problems))


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
    return isinstance(values, list) and all(isinstance(value, dict) and _is_number(value.get("key")) for value in values)


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


def _brought_names(reference):
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
    if reference.pick is None:
        return ["has with but no pick; a catalogue object brings its attributes only when the reference picks it"]
    if isinstance(entry, GrowingCatalogue):
        return [f"has with on growing catalogue {reference.catalogue!r}, whose objects don't exist yet when a "
                f"case is created"]
    copied = [name for name, source in pairs if isinstance(source, str)]
    if reference.count is not None and copied:
        return [f"copies {copied} on a pick with count; a list pick needs a function over the picked objects, "
                f'e.g. {{"sum": "price"}}']
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
                           and not all(_is_number(dict(obj.attributes)[name]) for obj in entry)]
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


@dataclass(frozen=True)
class SimulationConfig:
    """
    Everything needed for one multi-process run. Each consumer group maps a group name to the
    processes in it; without consumer_groups, every process is its own group. Every process must
    be in exactly one group, and groups may only name known processes. extra_processes names the
    processes that run_orchestrator receives as ready-made engines instead of building them from a
    ProcessSpec (e.g. scripted test engines), so that groups can name them too. Each process's catalogues map a
    name (the OCEL object type) to the objects with no events of their own, a list of CatalogueObject, a
    CataloguePool or a GrowingCatalogue, and its references say which of its attributes hold their ids, and may
    pick them (see _checked_catalogue for what is rejected).
    """

    processes: List[ProcessSpec]
    start_datetime: datetime
    seed: Optional[int] = None
    consumer_groups: Optional[Dict[str, List[str]]] = None
    extra_processes: Tuple[str, ...] = ()

    @property
    def catalogue(self) -> Dict[str, Union[List[CatalogueObject], CataloguePool, GrowingCatalogue]]:
        """Every process's catalogues, name -> objects; the names are unique across the processes. Only for the
        OCEL output at the end of the run, which writes and links the objects of all of them: building a process
        uses its own entry (ProcessSpec.catalogues)."""
        return {name: entry for spec in self.processes for name, entry in spec.catalogues.items()}

    @property
    def references(self) -> Tuple[Reference, ...]:
        """Every process's references, in the order of the processes. Only for the OCEL output at the end of the
        run, which joins the links of all of them: building a process uses its own entry (ProcessSpec.references)."""
        return tuple(reference for spec in self.processes for reference in spec.references)

    def __post_init__(self):
        object.__setattr__(self, "extra_processes", tuple(self.extra_processes))
        names = [spec.name for spec in self.processes] + list(self.extra_processes)
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Process names must be unique, repeated: {duplicates}")

        object.__setattr__(self, "consumer_groups", _checked_consumer_groups(names, self.consumer_groups))
        _checked_catalogue(self.processes)
        _checked_ocel_attributes(self.processes)

    @classmethod
    def from_json(cls, path) -> "SimulationConfig":
        """
        Load a configuration file such as
            {"processes": [{"name": "Sales", "bpmn_path": "sales.bpmn",
                            "json_path": "sales.json", "arrivals": 100}, ...],
             "seed": 42, "start_time": "2024-01-01T09:00:00+00:00",
             "consumer_groups": {"Sales": ["Sales"], ...}}
        BPMN and JSON paths are relative to the configuration file's folder. seed and
        consumer_groups are optional; a start time without a time zone is taken as UTC. A process
        started by messages has no arrivals. A process's optional object_type names the OCEL object
        type of its cases; without one, it writes no objects and no events. Its
        optional qualifier (by default the object type) and qualifier_by_activity ({task id: qualifier})
        qualify each event's link to its own case's object, and carry_links (false by default) makes every
        later event of a case also link the objects the case is linked to through its messages. Its optional
        ocel_attributes list the attributes written to its objects; without it, none.
        A process's optional catalogues ({name: [{"id": ..., "attributes": {...}}]}, {name: {"count": ...,
        "attributes": [as in object_attributes]}} for a generated pool, or {name: {"attributes": [...]}} for a
        growing one) declare objects with no events of their own, and its references ([{"object_attribute",
        "catalogue", "o2o", "reverse", "link_events", "pick", "with", "tag_along"}], the last six optional) say
        which of its attributes hold the ids of its catalogues' objects, and may pick them and bring their
        attributes in.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        _check_keys(path, "top level", data, CONFIG_KEYS)
        missing = [key for key in ("processes", "start_time") if key not in data]
        if missing:
            raise ValueError(f"{path}: missing {missing}")

        folder = path.parent
        processes = []
        for number, entry in enumerate(data["processes"]):
            _check_keys(path, f"processes[{number}]", entry, PROCESS_KEYS)
            missing = [key for key in ("name", "bpmn_path", "json_path") if key not in entry]
            if missing:
                raise ValueError(f"{path}: process {entry.get('name', '?')} is missing {missing}")
            where = f"processes[{number}]"
            catalogues = {name: _catalogue(path, f"{where}.catalogues.{name}", entries)
                          for name, entries in entry.get("catalogues", {}).items()}
            references = [_reference(path, f"{where}.references[{index}]", reference)
                          for index, reference in enumerate(entry.get("references", []))]
            processes.append(ProcessSpec(
                entry["name"], str(folder / entry["bpmn_path"]), str(folder / entry["json_path"]), entry.get("arrivals"),
                _object_type(path, f"processes[{number}]", entry), entry.get("qualifier"),
                tuple(sorted(entry.get("qualifier_by_activity", {}).items())), entry.get("carry_links", False),
                _ocel_attributes(path, where, entry), catalogues, tuple(references)
            ))

        start = parse_datetime(data["start_time"], True)
        if start.tzinfo is None:
            start = pytz.utc.localize(start)

        return cls(processes, start, data.get("seed"), data.get("consumer_groups"))


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


class Verdict(Enum):
    """An engine's answer to one offered message."""

    CLAIMED = "claimed"
    DISCARDED = "discarded"
    PENDING = "pending"


@dataclass
class StalledCase:
    """A case still waiting for a message when the run ended."""

    object_id: str
    event_id: str
    message_types: List[str]  # the types it waits for: any one of them resumes it (several in a race)
    waiting_since: datetime
    collected: int = 0  # messages it had claimed at that event...
    needed: int = 1  # ...of the ones it needed (more than 1 with collect)


@dataclass(frozen=True)
class PublishRecord:
    """An engine's record that one of its cases published a message, at one of its elements. task_rows are the
    rows of its log (0 for the first) that the message's OCEL links attach to: the last task before the
    element, and through a parallel or inclusive join the last task of every branch that came in; empty if
    there is none. qualifier qualifies those links."""

    message_id: str
    object_id: str
    element_id: str
    qualifier: str = ""
    task_rows: Tuple[int, ...] = ()
    o2o: Optional[str] = None  # qualifier of the OCEL link from its case's object to the other end's, if any


@dataclass(frozen=True)
class ClaimRecord:
    """An engine's record that one of its cases took a message, at one of its elements. task_rows are the rows
    of its log that the message's OCEL links attach to: the next task after the element, and after a
    parallel or inclusive split the first run of every task on the branches taken; empty if there is none. qualifier
    qualifies those links."""

    message_id: str
    object_id: str
    element_id: str
    qualifier: str = ""
    task_rows: Tuple[int, ...] = ()
    o2o: Optional[str] = None  # qualifier of the OCEL link from its case's object to the other end's, if any


@dataclass
class CaseObject:
    """One case of an engine, as an object of the OCEL output: its id in the engine, when it was created, and its
    case attributes over time, as (attribute, value, time): the values it had when it was created, then every
    later change. All of them are kept here; the OCEL output leaves some out (see ocel_writer._written)."""

    object_id: str
    created: datetime
    attributes: List[Tuple[str, Any, datetime]] = field(default_factory=list)
    # its sums and counts whose value at creation is only their starting 0, no message having written them yet
    only_started: Tuple[str, ...] = ()


@dataclass
class EngineReport:
    """What one engine has to report when the run is over."""

    stalled: List[StalledCase] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)  # raised inside this engine during the run
    # which of its cases and elements published and took each message, kept by the engine during the run
    published: List[PublishRecord] = field(default_factory=list)
    claimed: List[ClaimRecord] = field(default_factory=list)
    objects: List[CaseObject] = field(default_factory=list)  # every case it created
    logged_elements: List[str] = field(default_factory=list)  # the element id of each row of its log, in order


@dataclass(frozen=True)
class CaseElement:
    """One element of one case of one process; for a message, also the qualifier of its OCEL links and the
    rows of the process's log they attach to (none if there is no such task)."""

    process: str
    object_id: str
    element_id: str
    qualifier: str = ""
    task_rows: Tuple[int, ...] = ()
    o2o: Optional[str] = None  # qualifier of the OCEL link from its case's object to the other end's, if any


@dataclass
class MessageRecord:
    """Who published a message and who took it, joined from the engines' records after the run. No
    claimers for a message that was discarded or never claimed; several with capacity; no publisher
    if the publishing engine keeps no records."""

    message_id: str
    type: str
    publisher: Optional[CaseElement]
    claimers: List[CaseElement] = field(default_factory=list)


class SimulationEngine(ABC):
    """
    The complete set of methods the orchestrator may call on an engine, as defined by the
    Orchestrator Protocol; nothing else may cross that boundary, and no process may read another
    process's data by any other route. Engines never call the orchestrator. See
    docs/orchestrator.md.
    """

    @abstractmethod
    def subscriptions(self) -> List[str]:
        """The message types this process consumes, from its model configuration. Called once, at setup."""

    @abstractmethod
    def next_event_time(self) -> Optional[datetime]:
        """When is your next event due? None when there is nothing to do right now (cases waiting
        for a message don't count)."""

    @abstractmethod
    def step(self) -> List[Message]:
        """Perform exactly one event and return the messages it published; the only way to publish."""

    @abstractmethod
    def deliver(self, message: Message, now: datetime) -> Verdict:
        """
        Offer the engine one pending message at time now.
        CLAIMED is a commitment: the message is already bound to one case, taking effect at now,
        and the orchestrator never offers this copy to anyone again (other groups keep their own).
        DISCARDED is permanent: the engine will never want this message, even after its state
        changes, so the orchestrator never offers it to this engine again.
        PENDING: not now; the message may be offered again later. An engine that isn't sure
        must answer PENDING.
        """

    @abstractmethod
    def finish(self) -> EngineReport:
        """Called once, after the loop stops: the cases still waiting for a message, the warnings this
        engine raised, and its records of which cases and elements published and took each message. A
        lifecycle call, not a messaging one: nothing is published or delivered."""


# the top-level sections of a process's JSON settings: Prosimos's own, with object_attributes in place of its
# case_attributes, and messages
PROCESS_SETTINGS_KEYS = {"resource_profiles", "resource_calendars", "task_resource_distribution",
                         "arrival_time_distribution", "arrival_time_calendar", "gateway_branching_probabilities",
                         "branch_rules", "event_distribution", "batch_processing", "prioritisation_rules",
                         "object_attributes", "event_attributes", "global_attributes", "gateway_execution_limit",
                         "multitask", "model_type", "granule_size", "messages"}


def load_process(bpmn_path, json_path, arrivals, start_datetime, picks=()) -> SimDiffSetup:
    """Prosimos's setup of one process, from its JSON settings in the orchestrator's names: Prosimos reads
    object_attributes as its case_attributes and arrivals as its total_cases, so the settings are handed to it
    translated, in a copy that is deleted once it has read them. A key that isn't one of the sections is
    rejected, as a misspelled one would be ignored. picks are attributes given from outside the settings (see
    _Pick)."""
    with open(json_path) as file:
        settings = json.load(file)
    problem = unknown_keys("top level", settings, PROCESS_SETTINGS_KEYS)
    if problem:
        raise ValueError(f"{json_path}: {problem}")
    if "object_attributes" in settings:
        settings["case_attributes"] = settings.pop("object_attributes")
    with tempfile.TemporaryDirectory() as folder:
        translated = Path(folder) / Path(json_path).name
        translated.write_text(json.dumps(settings))
        return SimDiffSetup(bpmn_path, str(translated), False, arrivals, start_datetime, picks, "arrivals")


class _GlobalGenerators:
    """Whose states the global Python and NumPy generators hold: the caller's (holder None) or a ProsimosEngine's.
    Swapping them copies both states, which costs more than many engine calls themselves, so while keep is set
    an engine leaves its states there after a call, and its next call finds them in place: they are swapped only
    when another engine is called. Each engine still draws from its own states only, so nothing it draws
    changes. keep is only for code that draws nothing from the global generators between engine calls, like
    the orchestrator loop."""

    def __init__(self):
        self.holder, self.keep, self._callers = None, False, None

    def take(self, engine):
        if self.holder is engine:
            return
        states = random.getstate(), np.random.get_state()
        if self.holder is None:
            self._callers = states
        else:
            self.holder._python_state, self.holder._numpy_state = states
        random.setstate(engine._python_state)
        np.random.set_state(engine._numpy_state)
        self.holder = engine

    def give_back(self):
        """Puts the caller's states back, keeping the holder's for its next call."""
        if self.holder is None:
            return
        self.holder._python_state, self.holder._numpy_state = random.getstate(), np.random.get_state()
        random.setstate(self._callers[0])
        np.random.set_state(self._callers[1])
        self.holder, self._callers = None, None


_generators = _GlobalGenerators()


class ProsimosEngine(SimulationEngine):
    """A Prosimos simulation (SimBPMEnv) seen through the SimulationEngine interface. It publishes
    the messages its model lists under 'publish', each at the time the case passes the event, and
    subscribes to the types under 'consume': a case reaching such a catch event waits there until a
    delivered message matches (docs/messaging.md).

    Prosimos writes its warnings to one list shared by every engine (warning_logger) and draws its
    random numbers from the global Python and NumPy generators. Each method therefore hands Prosimos
    this engine's own warning list and generator states for the duration of the call, so two engines'
    warnings never mix and one engine's draws don't depend on which other engines run beside it (inside
    the orchestrator loop, the generator states stay until another engine is called: _GlobalGenerators).

    With a seed, the engine's generators are seeded from the seed and the process name, so engines of
    the same model under different names draw different values. Without one, they are seeded from the
    global generators: if the caller seeded those first (random.seed, np.random.seed), the run is
    repeatable; otherwise every run differs."""

    def __init__(self, spec: ProcessSpec, start_datetime: datetime, log_writer=None, seed: Optional[int] = None,
                 picks: Tuple["_Pick", ...] = ()):
        self._spec = spec
        self._warnings: List[str] = []
        # (message as returned by step(), case id, element id, publish entry, when); the orchestrator stamps
        # the id on the message, which is read at the end
        self._published: List[Tuple[Message, int, str, PublishPoint, datetime]] = []
        # (message id, case id, element id, consume entry, when)
        self._claimed: List[Tuple[str, int, str, ConsumePoint, datetime]] = []
        self._python_state, self._numpy_state = _engine_random_states(seed, spec.name)
        with self._own_globals():
            # picked attributes join the declared ones: every case gets one when it is created, the log has a
            # column for it, and messages may carry it
            sim_setup = load_process(spec.bpmn_path, spec.json_path, spec.arrivals, start_datetime, picks)
            for pick in picks:
                if isinstance(pick, _Pick) and isinstance(pick.count, CountFrom):
                    pick.read_count_from(next(attribute for attribute in sim_setup.case_attributes.attributes
                                              if attribute.name == pick.count.object_attribute))
            self._env = SimBPMEnv(sim_setup, None, log_writer, id_prefix=spec.id_prefix)
        tasks = {element_id for element_id, element in sim_setup.bpmn_graph.element_info.items()
                 if element.type is BPMN.TASK}
        unknown = sorted(task_id for task_id, _ in spec.qualifier_by_activity if task_id not in tasks)
        if unknown:
            raise ValueError(f"{spec.name}: qualifier_by_activity names {', '.join(unknown)}, which "
                             f"{'is not a task' if len(unknown) == 1 else 'are not tasks'} of its model")

    @contextmanager
    def _own_globals(self):
        shared_warnings = warning_logger.warnings_queue
        warning_logger.warnings_queue = self._warnings
        _generators.take(self)
        try:
            yield
        finally:
            warning_logger.warnings_queue = shared_warnings
            if not _generators.keep:
                _generators.give_back()

    def subscriptions(self) -> List[str]:
        with self._own_globals():
            return self._env.sim_setup.messaging.subscriptions()

    def next_event_time(self) -> Optional[datetime]:
        # None while only waiting cases are left: nothing to do now, but not finished
        with self._own_globals():
            return self._env.next_event_time()

    def step(self) -> List[Message]:
        with self._own_globals():
            now = self._env.next_event_time()  # the time the orchestrator stamps on what this step publishes
            self._env.step()
            # the engine buffers its log rows; hand them over now so nobody outside the engine
            # has to reach into it to flush them at the end
            self._env.log_writer.force_write()
            released = []
            for message_type, attributes, p_case, element_id, point in self._env.outbox:
                message = Message(message_type, attributes)
                # the orchestrator stamps the id on this same object, so it can be read at the end
                self._published.append((message, p_case, element_id, point, now))
                released.append(message)
            self._env.outbox.clear()
            return released

    def deliver(self, message: Message, now: datetime) -> Verdict:
        with self._own_globals():
            verdict = Verdict(self._env.deliver(message.type, message.attributes, message.source, now))
            if verdict is Verdict.CLAIMED:
                self._claimed.extend((message.id, p_case, element_id, point, now)
                                     for p_case, element_id, point in self._env.claimed_by)
            return verdict

    def finish(self) -> EngineReport:
        with self._own_globals():
            stalled = [StalledCase(self._env.object_id(parked_event.p_case), place, message_types,
                                   parked_event.enabled_datetime, *self._env.collected(parked_event))
                       for parked_event, place, message_types in self._env.stalled_waits()]
            created, only_started = self._env.case_creation_times(), self._env.only_started_values()
            objects = [CaseObject(self._env.object_id(p_case), created[p_case], history, only_started[p_case])
                       for p_case, history in sorted(self._env.case_attribute_history().items())]
            published, claimed = self._attached_records()
            logged_elements = [element_id for _, element_id, _, _ in self._env.logged_rows]
        return EngineReport(stalled, list(self._warnings), published, claimed, objects, logged_elements)

    def _attached_records(self):
        """The publish and claim records, each attached to the rows of the log its OCEL links go to: for a
        message published at a throw or end event, the last task before it in the same case, and through a
        parallel or inclusive join the last task of every branch that came in; for one taken at a start or
        catch event, the next task after it, and after a parallel or inclusive split the first run of every
        task on the branches taken. Without such a task, the links are dropped, with one warning per element. A process left
        out of the OCEL output, or an entry without a qualifier, attaches nothing: it writes no links."""
        attach = self._spec.object_type is not None
        rows_of_case = defaultdict(list)
        for row, (p_case, element_id, enabled, completed) in enumerate(self._env.logged_rows):
            rows_of_case[p_case].append((row, element_id, enabled, completed))
        warned = set()

        def task_rows(p_case, element_id, time, forward, qualifier):
            if not attach or qualifier is None:
                return ()
            tasks = self._env.tasks_next_to(element_id, forward)
            if forward:  # the first of them enabled when the case took the message, or later
                rows = sorted((enabled, row, element) for row, element, enabled, _ in rows_of_case[p_case]
                              if element in tasks and enabled >= time)
                # ... but before its next round there: a claim there after it moved on (with collect, a case
                # claims several messages there before it moves on)
                left = rows[0][0] if rows else None
                until = min((when for _, case, element, _, when in self._claimed
                             if case == p_case and element == element_id and left is not None and when > left),
                            default=None)
                # after a parallel or inclusive split, the first run of every task on the branches taken
                first_of_task = {}
                for enabled, row, element in rows:
                    if tasks[element] and (until is None or enabled < until):
                        first_of_task.setdefault(element, row)
                found = sorted({*first_of_task.values(), *[row for _, row, _ in rows[:1]]})
            else:  # the last of them completed when the case published the message, or earlier
                # ... but after its previous publish there, so a loop doesn't bring back an earlier round
                since = max((when for _, case, element, _, when in self._published
                             if case == p_case and element == element_id and when < time), default=None)
                rows = sorted((completed, row, element) for row, element, _, completed in rows_of_case[p_case]
                              if element in tasks and completed <= time and (since is None or completed > since))
                # through a parallel or inclusive join, the last task of every branch that came in
                last_of_task = {element: row for _, row, element in rows if tasks[element]}
                found = sorted({*last_of_task.values(), *[row for _, row, _ in rows[-1:]]})
            if not found and (element_id, forward) not in warned:
                warned.add((element_id, forward))
                side, case = "after" if forward else "before", self._env.object_id(p_case)
                if not tasks:
                    reason = f"{element_id} has no task {side} it in the model"
                elif forward:  # e.g. the case was still waiting further on when the run ended
                    reason = f"case {case} reached no task after {element_id} before the run ended"
                else:
                    reason = f"case {case} passed no task before {element_id}"
                warning_logger.add_warning(f"{reason}; the OCEL links of its messages there are dropped")
            return tuple(found)

        published = [PublishRecord(message.id, self._env.object_id(p_case), element_id, point.qualifier or "",
                                   task_rows(p_case, element_id, when, False, point.qualifier), point.o2o)
                     for message, p_case, element_id, point, when in self._published]
        claimed = [ClaimRecord(message_id, self._env.object_id(p_case), element_id, point.qualifier or "",
                               task_rows(p_case, element_id, when, True, point.qualifier), point.o2o)
                   for message_id, p_case, element_id, point, when in self._claimed]
        return published, claimed


class _Pick:
    """A case attribute that a reference picks (pick) from its catalogue's ids. It has what the engine uses of
    Prosimos's CaseAttribute, name and get_next_value(), so it is drawn with the case's other attributes, when
    the case is created; its generator is its own, seeded from the seed, the process and the attribute, so it
    doesn't change the engine's other draws. With count, it picks that many ids, each independently, into a
    list."""

    def __init__(self, name: str, ids: List[str], weights: Optional[List[float]], seed: Optional[int], process: str,
                 count: Union[None, int, CountFrom] = None):
        self.name, self._ids, self._weights, self.count = name, ids, weights, count
        self.last = None  # the id picked for the latest case
        self._counted = None  # the value of count's attribute drawn for the latest case
        self._warned = False
        self._generator = random.Random()
        self._generator.setstate(_engine_random_states(seed, f"pick/{process}/{name}")[0])

    def read_count_from(self, attribute):
        """Has the case attribute that holds the count, one the engine draws before its picks, tell this pick the
        value it draws for each case. The attribute keeps its own draws."""
        draw = attribute.get_next_value

        def drawn():
            self._counted = draw()
            return self._counted

        attribute.get_next_value = drawn

    def get_next_value(self):
        if self.count is None:
            self.last = self._generator.choices(self._ids, self._weights)[0]
        else:
            self.last = self._generator.choices(self._ids, self._weights, k=self._how_many())
        return self.last

    def _how_many(self):
        """count, or the value its attribute drew for this case; one that isn't a whole number of at least 0 picks
        one, with one warning."""
        if not isinstance(self.count, CountFrom):
            return int(self.count)
        if is_whole_number(self._counted, minimum=0):
            return int(self._counted)
        if not self._warned:
            self._warned = True
            warning_logger.add_warning(f"{self.count.object_attribute} is {self._counted!r}, no whole number of at "
                                       f"least 0; the pick of {self.name} picks one")
        return 1


class _Brought:
    """A value a _Pick's objects give the case it has just picked them for: an attribute of the picked object,
    copied, or, with function, the function over that attribute of every object it picked, each as often as it
    was picked (a single pick is a list of one), and over initial, if given, as aggregate computes over messages.
    It comes right after its pick in the case's attributes, which are drawn in order, so it reads the pick's last
    value; it draws nothing. Over no objects, a sum and a count are 0 and a min or max has no value (None), unless
    initial gives them one."""

    def __init__(self, name: str, pick: _Pick, values: Dict[str, Any], function: Optional[str] = None,
                 initial: Optional[float] = None):
        self.name, self._pick, self._values, self._function, self._initial = name, pick, values, function, initial

    def get_next_value(self):
        if self._function is None:
            return self._values[self._pick.last]
        picked = self._pick.last if isinstance(self._pick.last, list) else [self._pick.last]
        if not picked and self._function in ("min", "max") and self._initial is None:
            return None
        return _aggregated(self._function, [self._values[object_id] for object_id in picked], self._initial)


class _NewPick:
    """A case attribute that a reference picks from a growing catalogue: a new id, <catalogue name>-<number> in
    creation order, for every case ("new") or for share of them, the rest reusing one created so far, uniformly.
    Like _Pick, it is drawn with the case's other attributes, from a generator of its own."""

    def __init__(self, name: str, catalogue: str, share: float, seed: Optional[int], process: str):
        self.name, self.catalogue, self._share = name, catalogue, share
        self.created = 0
        self._generator = random.Random()
        self._generator.setstate(_engine_random_states(seed, f"pick/{process}/{name}")[0])

    def get_next_value(self):
        if self.created == 0 or self._share == 1 or self._generator.random() < self._share:
            self.created += 1
            return f"{self.catalogue}-{self.created}"
        return f"{self.catalogue}-{self._generator.randint(1, self.created)}"


def _picks(spec: ProcessSpec, seed: Optional[int], catalogues: Dict[str, List[CatalogueObject]]) -> List[_Pick]:
    """The attributes a process's references pick, from its own catalogues as the run starts (see
    _drawn_catalogues)."""
    picks = []
    for reference in spec.references:
        if reference.creates:
            share = reference.pick.share if isinstance(reference.pick, NewShare) else 1
            picks.append(_NewPick(reference.object_attribute, reference.catalogue, share, seed, spec.name))
        elif reference.pick is not None:
            ids = [obj.id for obj in catalogues[reference.catalogue]]
            weights = None if reference.pick == "uniform" else [dict(reference.pick).get(object_id, 0) for object_id in ids]
            pick = _Pick(reference.object_attribute, ids, weights, seed, spec.name, reference.count)
            picks.append(pick)
            for name, source in reference.with_attributes:
                function, attribute, initial = ((source.function, source.source, source.initial)
                                                if isinstance(source, Computed) else (None, source, None))
                values = {obj.id: dict(obj.attributes)[attribute] for obj in catalogues[reference.catalogue]}
                picks.append(_Brought(name, pick, values, function, initial))
    return picks


def _drawn_catalogues(spec: ProcessSpec, seed: Optional[int]) -> Dict[str, List[CatalogueObject]]:
    """A process's catalogues, name -> their objects when the run starts: a listed catalogue's as given, a pool's
    with their attributes drawn (see _drawn_objects), none yet for a growing catalogue."""
    return {name: list(entry) if isinstance(entry, list) else _drawn_objects(seed, name, entry.attributes,
                                                                            _catalogue_ids(name, entry))
            for name, entry in spec.catalogues.items()}


def _grown_catalogues(spec: ProcessSpec, seed: Optional[int], catalogues: Dict[str, List[CatalogueObject]],
                      picks: List[_Pick]) -> Dict[str, List[CatalogueObject]]:
    """A process's catalogues after the run: each growing catalogue with the objects its pick created, <name>-1 to
    <name>-<created>, their attributes drawn in creation order (see _drawn_objects). Drawing them now rather
    than when each was created gives the same values, since nothing else draws from their generators."""
    grown = dict(catalogues)
    for pick in picks:
        if isinstance(pick, _NewPick):
            grown[pick.catalogue] = _drawn_objects(seed, pick.catalogue, spec.catalogues[pick.catalogue].attributes,
                                                   [f"{pick.catalogue}-{number}" for number in range(1, pick.created + 1)])
    return grown


def _drawn_objects(seed, name, attribute_specs, ids) -> List[CatalogueObject]:
    """The objects with these ids, their attributes drawn object after object by Prosimos's case attribute code,
    from generators seeded from the seed and the catalogue's name."""
    attributes = [CaseAttribute(attribute["name"], attribute["type"], attribute["values"]) for attribute in attribute_specs]
    python_state, numpy_state = _engine_random_states(seed, f"catalogue/{name}")
    shared_python, shared_numpy = random.getstate(), np.random.get_state()
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    try:
        return [CatalogueObject(object_id, tuple((attribute.name, _plain_value(attribute.get_next_value()))
                                                 for attribute in attributes))
                for object_id in ids]
    finally:
        random.setstate(shared_python)
        np.random.set_state(shared_numpy)


def _plain_value(value):
    return value.item() if hasattr(value, "item") else value  # a NumPy number, from a continuous attribute


def _engine_random_states(seed, process_name):
    """Initial Python and NumPy generator states for one engine. A stable hash of the seed and the
    process name, not Python's hash(), which differs between runs."""
    if seed is None:
        python_seed, numpy_seed = random.getrandbits(64), random.getrandbits(32)
    else:
        digest = hashlib.sha256(f"{seed}/{process_name}".encode("utf-8")).digest()
        python_seed, numpy_seed = int.from_bytes(digest[:8], "big"), int.from_bytes(digest[8:12], "big")
    return random.Random(python_seed).getstate(), np.random.RandomState(numpy_seed).get_state()


class _MergedLog:
    """Collects the rows of every process and writes them to one CSV, sorted by start time,
    with the process name as the first column. Models can add different extra columns
    (batch_id, case attributes, ...), so the header is the union of them all and a process
    leaves blank any column it doesn't produce."""

    def __init__(self):
        self._process_columns = {}
        self._rows = []

    def writer_for(self, process_name):
        return _ProcessLogWriter(self, process_name)

    def _register(self, process_name, header):
        self._process_columns[process_name] = list(header)

    @property
    def rows(self):
        """(process, {column: value}) for every row handed over, in the order they came."""
        return list(self._rows)

    def _add(self, process_name, rows):
        process_columns = self._process_columns[process_name]
        self._rows.extend((process_name, dict(zip(process_columns, row))) for row in rows)

    def write(self, path):
        columns = []
        for process_columns in self._process_columns.values():
            columns.extend(c for c in process_columns if c not in columns)

        # rows arrive in the order events are executed, which is not start-time order: a task
        # can wait for a busy resource while another process carries on. The sort is stable,
        # so rows with the same start time and process keep the order they were executed in.
        rows = sorted(self._rows, key=lambda row: (_as_datetime(row[1]["start_time"]), row[0]))

        with open(path, mode="w", newline="", encoding="utf-8") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(["process", *columns])
            for process_name, values in rows:
                writer.writerow([process_name, *(values.get(c, "") for c in columns)])


def _as_datetime(value):
    # the engine hands over timestamps as datetimes, or as preformatted strings when they
    # have no fractional seconds (see verify_miliseconds)
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)


class _ProcessLogWriter:
    """Handed to an engine in place of a csv.writer, routing its rows into the merged log."""

    def __init__(self, merged_log, process_name):
        self._merged_log = merged_log
        self._process_name = process_name
        self._header_seen = False

    def writerow(self, row):
        # the engine's FileManager writes its own header row first, as soon as it is created
        if not self._header_seen:
            self._header_seen = True
            self._merged_log._register(self._process_name, row)
        else:
            self._merged_log._add(self._process_name, [row])

    def writerows(self, rows):
        self._merged_log._add(self._process_name, rows)


@dataclass
class RunReport:
    """What happened during a run, and what was left over at the end."""

    executed: List[Tuple[datetime, str]] = field(default_factory=list)  # (event time, process) per step
    published: List[Message] = field(default_factory=list)  # every message, as stamped
    copies: List[Tuple[str, str]] = field(default_factory=list)  # (message id, group) per copy made
    claims: List[Tuple[str, str, datetime]] = field(default_factory=list)  # (message id, process, time)
    discards: List[Tuple[str, str, datetime]] = field(default_factory=list)  # (message id, process, time)
    unclaimed: List[Tuple[str, Message]] = field(default_factory=list)  # (group, message) left in the pool
    warnings: List[str] = field(default_factory=list)  # the orchestrator's own: routing, and catalogue values
    stalled: List[Tuple[str, StalledCase]] = field(default_factory=list)  # (process, case) from finish()
    engine_warnings: List[Tuple[str, str]] = field(default_factory=list)  # (process, warning) from finish()
    message_records: List[MessageRecord] = field(default_factory=list)  # per message, joined after finish()
    # (process, object) for every case, from finish(); written to the OCEL output, not to to_dict()
    objects: List[Tuple[str, CaseObject]] = field(default_factory=list)
    # process -> the element id of each row of its log, in order, from finish(); for the OCEL output only
    logged_elements: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def discarded_counts(self) -> Dict[Tuple[str, str], int]:
        """Discards per (message type, process)."""
        type_of = {message.id: message.type for message in self.published}
        return dict(Counter((type_of[message_id], process) for message_id, process, _ in self.discards))

    def to_dict(self) -> Dict[str, Any]:
        """The whole report as plain lists and dicts, with times as ISO strings, e.g. to save as JSON.
        Messages are referred to by id, except in published."""
        def when(time):
            return time.isoformat()

        return {
            "executed": [{"time": when(time), "process": process} for time, process in self.executed],
            "published": [{"id": m.id, "type": m.type, "source": m.source, "time": when(m.time),
                           "attributes": m.attributes} for m in self.published],
            "copies": [{"message": message_id, "group": group} for message_id, group in self.copies],
            "claims": [{"message": message_id, "process": process, "time": when(time)}
                       for message_id, process, time in self.claims],
            "discards": [{"message": message_id, "process": process, "time": when(time)}
                         for message_id, process, time in self.discards],
            "unclaimed": [{"group": group, "message": message.id} for group, message in self.unclaimed],
            "discarded_counts": [{"type": message_type, "process": process, "count": count}
                                 for (message_type, process), count in self.discarded_counts.items()],
            "stalled": [{"process": process, "object_id": case.object_id, "event_id": case.event_id,
                         "message_types": case.message_types, "waiting_since": when(case.waiting_since),
                         "collected": case.collected, "needed": case.needed}
                        for process, case in self.stalled],
            "warnings": list(self.warnings),
            "engine_warnings": [{"process": process, "warning": warning} for process, warning in self.engine_warnings],
            "message_records": [{"message": record.message_id, "type": record.type,
                                 "publisher": _plain(record.publisher) if record.publisher else None,
                                 "claimers": [_plain(claimer) for claimer in record.claimers]}
                                for record in self.message_records],
        }


def _plain(element: CaseElement) -> Dict[str, Any]:
    return {**asdict(element), "task_rows": list(element.task_rows)}


@dataclass(eq=False)
class _PooledCopy:
    """A group's copy of a message, waiting to be claimed."""

    message: Message
    group: str
    discarded_by: Set[str] = field(default_factory=set)


def run_engines(
    engines: Dict[str, SimulationEngine],
    consumer_groups: Optional[Dict[str, List[str]]] = None,
    seed: Optional[int] = None,
) -> RunReport:
    """
    The orchestrator loop of the Orchestrator Protocol, over already-built engines keyed by
    process name. After each step, every published message is stamped (id, source, time), one
    copy per subscribed group goes into the pool, and the new copies plus the pooled copies of the
    stepping engine's group are offered, oldest first, to their group's members in random order until one
    claims them. A message is never offered to the process that published it. A copy is removed when
    claimed, or when every member subscribed to its type other than the publisher has discarded it; nobody
    is offered a copy they already discarded. Member order comes from the orchestrator's own generator,
    never the global random module the engines use.
    """
    names = sorted(engines)
    consumer_groups = _checked_consumer_groups(names, consumer_groups)
    group_of = {member: group for group, members in consumer_groups.items() for member in members}
    subscriptions = {name: set(engines[name].subscriptions()) for name in names}
    routing = {}
    for group in sorted(consumer_groups):
        for message_type in sorted({t for member in consumer_groups[group] for t in subscriptions[member]}):
            routing.setdefault(message_type, []).append(group)

    rng = random.Random(f"orchestrator-{seed}") if seed is not None else random.Random()
    report = RunReport()
    pool: List[_PooledCopy] = []
    copies_left: Dict[str, int] = {}
    claimed_ids: Set[str] = set()
    next_id = itertools.count(1)

    def warn(text):
        report.warnings.append(text)

    def recipients_of(group, message):
        return [m for m in consumer_groups[group] if message.type in subscriptions[m] and m != message.source]

    def resolve(pooled):
        pool.remove(pooled)
        copies_left[pooled.message.id] -= 1
        if copies_left[pooled.message.id] == 0 and pooled.message.id not in claimed_ids:
            message = pooled.message
            warn(f"{message.type} {message.id} from {message.source} was discarded by every recipient")

    def engine(name):
        """engines[name], with the caller's generator states back in the global generators first for an engine
        that may draw from them itself: any but a ProsimosEngine."""
        if not isinstance(engines[name], ProsimosEngine):
            _generators.give_back()
        return engines[name]

    # nothing here draws from the global generators, so the engines may leave their states in them
    _generators.keep = True
    try:
        while True:
            due = [(t, name) for name in engines if (t := engine(name).next_event_time()) is not None]
            if not due:
                break
            now, name = min(due)
            published = engine(name).step()
            report.executed.append((now, name))

            new_copies = []
            for message in published:
                message.id, message.source, message.time = f"m{next(next_id)}", name, now
                report.published.append(message)
                # a message never comes back to the process that published it, so a group whose only member
                # subscribed to the type is the publisher gets no copy
                groups = [group for group in routing.get(message.type, []) if recipients_of(group, message)]
                if not groups:
                    if message.type in subscriptions[name]:
                        warn(f"{message.type} {message.id} from {name}: no subscriber other than its publisher")
                    else:
                        warn(f"{message.type} {message.id} from {name} has no subscribers")
                copies_left[message.id] = len(groups)
                for group in groups:
                    pooled = _PooledCopy(copy.deepcopy(message), group)
                    pool.append(pooled)
                    new_copies.append(pooled)
                    report.copies.append((message.id, group))

            # oldest first: the pool keeps copies in the order they were published, the new ones last
            own_group = group_of[name]
            new_ids = {id(pooled) for pooled in new_copies}
            to_offer = [pooled for pooled in pool if id(pooled) in new_ids or pooled.group == own_group]
            for pooled in to_offer:
                message = pooled.message
                recipients = recipients_of(pooled.group, message)
                candidates = [m for m in recipients if m not in pooled.discarded_by]
                rng.shuffle(candidates)
                for member in candidates:
                    verdict = engine(member).deliver(message, now)
                    if not isinstance(verdict, Verdict):
                        raise TypeError(f"{member}.deliver() must return a Verdict, got {verdict!r}")
                    if verdict is Verdict.CLAIMED:
                        report.claims.append((message.id, member, now))
                        claimed_ids.add(message.id)
                        resolve(pooled)
                        break
                    if verdict is Verdict.DISCARDED:
                        pooled.discarded_by.add(member)
                        report.discards.append((message.id, member, now))
                else:
                    if all(m in pooled.discarded_by for m in recipients):
                        resolve(pooled)
    finally:
        _generators.keep = False
        _generators.give_back()

    report.unclaimed = [(pooled.group, pooled.message) for pooled in pool]
    # the engines' records are only joined here, once the run is over: nothing during the run uses them
    publishers, claimers = {}, defaultdict(list)
    for name in names:
        engine_report = engines[name].finish()
        if not isinstance(engine_report, EngineReport):
            raise TypeError(f"{name}.finish() must return an EngineReport, got {engine_report!r}")
        report.stalled.extend((name, case) for case in engine_report.stalled)
        report.engine_warnings.extend((name, warning) for warning in engine_report.warnings)
        report.objects.extend((name, case_object) for case_object in engine_report.objects)
        report.logged_elements[name] = engine_report.logged_elements
        for record in engine_report.published:
            publishers[record.message_id] = CaseElement(name, record.object_id, record.element_id, record.qualifier,
                                                        record.task_rows, record.o2o)
        for record in engine_report.claimed:
            claimers[record.message_id].append(CaseElement(name, record.object_id, record.element_id, record.qualifier,
                                                           record.task_rows, record.o2o))
    report.message_records = [MessageRecord(message.id, message.type, publishers.get(message.id), claimers[message.id])
                              for message in report.published]
    return report


def run_orchestrator(config: SimulationConfig, log_out_path: Optional[str] = None,
                     extra_engines: Optional[Dict[str, SimulationEngine]] = None,
                     ocel_out_path: Optional[str] = None) -> RunReport:
    """
    Simulate the configured processes side by side on one shared clock (see run_engines),
    stepping the engine whose next_event_time() is earliest, ties broken by process name, so runs
    given the same seed are repeatable; without a seed each run draws different random values.
    When log_out_path is given, every process's events are written to that one CSV, sorted
    by start time, with the process name as the first column. When ocel_out_path is given, the run
    is also written as an OCEL 2.0 JSON file: one object per case, typed by its process's object_type,
    and one event per task, linked to its case's object and to the objects on the other end of its
    messages; objects are linked to each other where a message entry has an o2o key, and the catalogue's
    objects are written and linked where references say a case attribute holds their ids (see
    prosimos/ocel_writer.py).
    extra_engines are ready-made engines, keyed by process name, that run alongside the ones built
    from config.processes; their names must be exactly config.extra_processes. They get no log
    writer, so the merged log holds only the configured processes.
    """
    extra_engines = dict(extra_engines or {})
    if sorted(extra_engines) != sorted(config.extra_processes):
        raise ValueError(f"extra_engines must be exactly the configuration's extra_processes "
                         f"{sorted(config.extra_processes)}, got {sorted(extra_engines)}")

    merged_log = _MergedLog() if log_out_path is not None or ocel_out_path is not None else None

    # every engine has its own random generators, seeded from the seed and its process name; without a
    # seed they are seeded from the global generators, so build them in name order rather than input
    # order to keep the result independent of how the list was written
    # each process's picks come from its own entry only: its references and its catalogues. All catalogues are
    # drawn before any pick, and the picks before any engine, as without a seed each draws from the global generators
    drawn = {spec.name: _drawn_catalogues(spec, config.seed) for spec in config.processes}
    picks = {spec.name: _picks(spec, config.seed, drawn[spec.name]) for spec in config.processes}
    engines: Dict[str, SimulationEngine] = {}
    for spec in sorted(config.processes, key=lambda p: p.name):
        log_writer = merged_log.writer_for(spec.name) if merged_log is not None else None
        engines[spec.name] = ProsimosEngine(spec, config.start_datetime, log_writer, config.seed,
                                            tuple(picks[spec.name]))
    engines.update(extra_engines)

    report = run_engines(engines, config.consumer_groups, config.seed)
    # the OCEL output needs every process's catalogue objects, the growing ones as the run left them
    catalogue = {name: objects for spec in config.processes
                 for name, objects in _grown_catalogues(spec, config.seed, drawn[spec.name], picks[spec.name]).items()}

    if log_out_path is not None:
        merged_log.write(log_out_path)
    if ocel_out_path is not None:
        processes = {spec.name: OcelProcess(spec.object_type, spec.qualifier, dict(spec.qualifier_by_activity),
                                            spec.carry_links, spec.ocel_attributes) for spec in config.processes}
        static_objects, reference_object_links, reference_event_links = _reference_links(
            config, catalogue, processes, report.objects, merged_log.rows, report.logged_elements,
            report.warnings.append)
        object_links = _object_links(report.message_records, processes)
        for source, links in reference_object_links.items():
            object_links.setdefault(source, []).extend(links)
        write_ocel(ocel_out_path, processes, report.objects, merged_log.rows, report.logged_elements,
                   _message_links(report.message_records, processes), object_links, static_objects,
                   reference_event_links, _tag_along(config, catalogue, report.objects))

    return report


def _message_links(records: List[MessageRecord], processes: Dict[str, OcelProcess]):
    """(process, row of that process) -> [((process, object id), qualifier)]: the tasks a publisher's records
    attach to link the object of every case that claimed the message, and the tasks a claimer's records attach to
    link the publisher's object, each with its own end's qualifier. Ends without a qualifier or a task, or whose other
    end's process writes no objects, link nothing."""
    def has_object(element):
        return element.process in processes and processes[element.process].object_type is not None

    links = defaultdict(list)
    for record in records:
        if record.publisher is None:
            continue
        publisher = record.publisher
        for claimer in record.claimers:
            for row in publisher.task_rows if has_object(claimer) else ():
                links[(publisher.process, row)].append(((claimer.process, claimer.object_id), publisher.qualifier))
            for row in claimer.task_rows if has_object(publisher) else ():
                links[(claimer.process, row)].append(((publisher.process, publisher.object_id), claimer.qualifier))
    return dict(links)


def _object_links(records: List[MessageRecord], processes: Dict[str, OcelProcess]):
    """(process, object id) -> [((process, object id), qualifier)]: the OCEL links between objects that the
    messages' o2o keys give.
    A publish entry's o2o links the publisher's object to the object of every case that claimed the
    message, a consume entry's links the claimer's object to the publisher's. A link is written once, however
    many messages give it; ends whose process writes no objects link nothing."""
    def has_object(element):
        return element.process in processes and processes[element.process].object_type is not None

    links = defaultdict(list)

    def link(source, target, qualifier):
        if not (qualifier and has_object(source) and has_object(target)):
            return
        end = (target.process, target.object_id)
        if (end, qualifier) not in links[(source.process, source.object_id)]:
            links[(source.process, source.object_id)].append((end, qualifier))

    for record in records:
        for claimer in record.claimers if record.publisher is not None else ():
            link(record.publisher, claimer, record.publisher.o2o)
            link(claimer, record.publisher, claimer.o2o)
    return dict(links)


def _reference_links(config: SimulationConfig, catalogue: Dict[str, List[CatalogueObject]],
                     processes: Dict[str, OcelProcess], objects, rows, logged_elements, warn):
    """
    The catalogue's objects (catalogue: every process's, as _grown_catalogues gives them), as (catalogue name, id, [(attribute,
    value, time)]): every object of a listed catalogue, and the objects of a pool or a growing catalogue that some
    case's referenced attribute names; their attributes are timed at the start of the run, a growing catalogue's
    when the first case took its id; and the links its references give, joined with the case objects and log rows
    after the run:
    - object -> [(object, qualifier)]: a reference with o2o links the object of every case of its process, as
      (process, object id), to the catalogue object of each value its attribute took, as its id (reverse: from
      the catalogue object);
    - (process, row of that process) -> [(object id, qualifier)]: each event of those cases named in
      link_events (every event without the key) links the catalogue object of the value the attribute had when
      the event completed, with the attribute's name as qualifier.
    A value that is a list (appended, or picked) stands for each of its entries, and an entry it repeats is linked
    once. A value that is not in its catalogue gets no link, and a warning, once per attribute and value.
    Processes that write no objects link nothing.
    """
    ids_of = {name: {obj.id for obj in catalogue_objects} for name, catalogue_objects in catalogue.items()}
    history = {(process, case_object.object_id): case_object.attributes for process, case_object in objects}
    object_links, event_links, warned, referred = defaultdict(list), defaultdict(list), set(), set()
    first_taken = {}  # (catalogue name, id) -> when the first case took it

    def in_catalogue(reference, value):
        if str(value) in ids_of[reference.catalogue]:
            return True
        if (reference.object_attribute, value) not in warned:
            warned.add((reference.object_attribute, value))
            warn(f"object attribute {reference.object_attribute} has the value {value!r}, which is not in catalogue "
                 f"{reference.catalogue}; no OCEL link is written for it")
        return False

    def values(case, attribute, until=None):
        """The ids the attribute of case, (process, object id), named, in order and each once: all it ever named, or
        only those of the value it had at until."""
        taken = [value for name, value, time in history.get(case, [])
                 if name == attribute and value is not None and (until is None or time <= until)]
        return list(dict.fromkeys(entry for value in (taken[-1:] if until is not None else taken)
                                  for entry in _entries(value)))

    def link(source, target, qualifier):
        if (target, qualifier) not in object_links[source]:
            object_links[source].append((target, qualifier))

    for reference in config.references:
        has_objects = reference.process in processes and processes[reference.process].object_type is not None
        for process, case_object in objects:
            if process != reference.process:
                continue
            case = (process, case_object.object_id)
            for value in values(case, reference.object_attribute):
                if in_catalogue(reference, value):
                    referred.add((reference.catalogue, str(value)))
                    taken = min(time for name, taken_value, time in history[case]
                                if name == reference.object_attribute and value in _entries(taken_value))
                    key = (reference.catalogue, str(value))
                    first_taken[key] = min(taken, first_taken.get(key, taken))
                    if has_objects and reference.o2o:
                        ends = (str(value), case) if reference.reverse else (case, str(value))
                        link(*ends, reference.o2o)
        if not has_objects:
            continue
        rows_so_far = defaultdict(lambda: -1)
        for process, row in rows:
            rows_so_far[process] += 1
            number = rows_so_far[process]  # the row's place in its process's log
            if process != reference.process:
                continue
            element_ids = logged_elements.get(process, [])
            element_id = element_ids[number] if number < len(element_ids) else None
            if reference.link_events is not None and not {row["activity"], element_id} & set(reference.link_events):
                continue
            case = (process, f"{processes[process].object_type}-{row['case_id']}")
            for value in values(case, reference.object_attribute, _as_datetime(row["end_time"])):
                if in_catalogue(reference, value):
                    event_links[(process, number)].append((str(value), reference.object_attribute))

    def time_of(name, object_id):
        growing = isinstance(config.catalogue[name], GrowingCatalogue)
        return first_taken[(name, object_id)] if growing else config.start_datetime

    static_objects = [(name, obj.id, [(attribute, value, time_of(name, obj.id)) for attribute, value in obj.attributes])
                      for name, catalogue_objects in catalogue.items() for obj in catalogue_objects
                      if isinstance(config.catalogue[name], list) or (name, obj.id) in referred]
    return static_objects, dict(object_links), dict(event_links)


def _tag_along(config: SimulationConfig, catalogue: Dict[str, List[CatalogueObject]], objects):
    """(process, object id) -> [(qualifier, [(time, [catalogue object id, ...])])]: for each reference with
    tag_along, the catalogue objects its attribute named on the case's object, from when, with the attribute's name
    as qualifier: one for a single value, each entry once for a list. The OCEL output links every event of another
    case that links the object to those named when the event completed (see write_ocel). Ids not in the catalogue
    are left out, and a value with none at all; _reference_links warns about them."""
    ids_of = {name: {obj.id for obj in catalogue_objects} for name, catalogue_objects in catalogue.items()}
    tagged = defaultdict(list)
    for reference in config.references:
        if not reference.tag_along:
            continue
        for process, case_object in objects:
            if process == reference.process:
                timeline = []
                for name, value, time in case_object.attributes:
                    if name == reference.object_attribute and value is not None:
                        named = list(dict.fromkeys(str(entry) for entry in _entries(value)
                                                   if str(entry) in ids_of[reference.catalogue]))
                        if named:
                            timeline.append((time, named))
                tagged[(process, case_object.object_id)].append((reference.object_attribute, timeline))
    return dict(tagged)


def _entries(value):
    """The catalogue ids a referenced attribute's value names: a list's entries (an appended or picked list), or the
    value itself."""
    return [entry for entry in value if entry is not None] if isinstance(value, (list, tuple)) else [value]
