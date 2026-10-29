"""
Reading and checking the simulation configuration (docs/orchestrator.md, "Configuration"): the processes, their
catalogues and references, and the consumer groups.
"""
import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pytz

from prosimos.messaging.model import AGGREGATE_FUNCTIONS
from prosimos.messaging_parser import is_number, unknown_keys
from prosimos.orchestration.catalogue import (CatalogueObject, CataloguePool, Computed, CountFrom, GrowingCatalogue,
                                            NewShare, Reference, _brought_names, _checked_catalogue)
from prosimos.simulation_properties_parser import parse_datetime


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


@dataclass(frozen=True)
class SimulationConfig:
    """
    Everything needed for one multi-process run. Each consumer group maps a group name to the
    processes in it; without consumer_groups, every process is its own group. Every process must
    be in exactly one group, and groups may only name known processes. extra_processes names the
    processes that run_multiprocess_simulation receives as ready-made engines instead of building them from a
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
