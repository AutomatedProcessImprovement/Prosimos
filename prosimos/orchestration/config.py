"""Reads and checks the simulation configuration (docs/orchestrator.md, "Configuration")."""
import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pytz

from prosimos.messaging.model import AGGREGATE_FUNCTIONS
from prosimos.messaging.parser import is_number, is_whole_number, unknown_keys
from prosimos.orchestration.catalogue.checks import brought_names, checked_catalogue
from prosimos.orchestration.catalogue.model import (CatalogueChange, CatalogueObject, CataloguePool, Computed,
                                                    GrowingCatalogue, NewShare, QuantityFrom, Reference)
from prosimos.simulation_properties_parser import parse_datetime


@dataclass(frozen=True)
class ProcessSpec:
    """One process of a configuration: its model, its settings and how it is written to the OCEL file."""

    name: str
    bpmn_path: str
    json_path: str
    arrivals: Optional[int] = None  # how many cases arrive; None for a process started by messages
    # the OCEL object type of its cases, in the singular; without one, the process writes nothing to the OCEL file
    object_type: Optional[str] = None
    # of each event's link to its case's object (None: the object type); qualifier_by_activity overrides it per task
    qualifier: Optional[str] = None
    qualifier_by_activity: Tuple[Tuple[str, str], ...] = ()
    # the attributes written to its OCEL objects; the others stay in the log
    ocel_attributes: Tuple[str, ...] = ()
    references: Tuple["Reference", ...] = ()

    def __post_init__(self):
        # each reference knows the process whose entry lists it
        object.__setattr__(self, "references", tuple(dataclasses.replace(reference, process=self.name)
                                                     for reference in self.references))

    @property
    def id_prefix(self) -> str:
        """Its objects' ids are <id prefix>-<number>: the object type, or the name of a process without one."""
        return self.object_type or self.name


CONFIG_KEYS = {"organizations", "start_time", "seed", "consumer_groups"}
ORGANIZATION_KEYS = {"name", "catalogues", "catalogue_changes", "processes"}
PROCESS_KEYS = {"name", "bpmn_path", "json_path", "arrivals", "object_type", "qualifier", "qualifier_by_activity",
                "ocel_attributes", "references"}
REFERENCE_KEYS = {"object_attribute", "catalogue", "o2o", "reverse", "link_events", "pick", "quantity", "with",
                  "tag_along"}
CHANGE_KEYS = {"catalogue", "attribute", "every", "at", "multiply", "add", "set", "round", "objects"}
PERIODS = ("weeks", "days", "hours")
OPERATIONS = ("multiply", "add", "set")


def _checked_ocel_attributes(organizations):
    """Each name in a process's ocel_attributes must be an attribute its objects have."""
    problems = []
    placed = [(f"organizations[{number}].processes[{index}]", spec)
              for number, organization in enumerate(organizations) for index, spec in enumerate(organization.processes)]
    for where, spec in placed:
        if not spec.ocel_attributes:
            continue
        with open(spec.json_path) as file:
            settings = json.load(file)
        has = {entry.get("name") for entry in settings.get("object_attributes", [])}
        has.update(reference.object_attribute for reference in spec.references if reference.pick is not None)
        has.update(name for reference in spec.references for name in brought_names(reference))
        messages = settings.get("messages")
        for entry in messages.get("consume", []) if isinstance(messages, dict) else []:
            for key in ("copy", "append", "aggregate"):
                if isinstance(entry, dict) and isinstance(entry.get(key), dict):
                    has.update(entry[key])
        unknown = [name for name in spec.ocel_attributes if name not in has]
        if unknown:
            problems.append(f"{where}.ocel_attributes names {unknown}, which the objects of process "
                            f"{spec.name} never have (they have {sorted(has)})")
    if problems:
        raise ValueError("Invalid ocel_attributes: " + "; ".join(problems))


def checked_consumer_groups(names, consumer_groups):
    """Checks the consumer groups: each process in exactly one; without groups, every process is its own group."""
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
class Organization:
    """A business: a box around its processes and catalogues. Engines never learn their organization."""

    name: str
    processes: List[ProcessSpec] = field(default_factory=list)
    # name -> objects with no events of their own, shared by its processes
    catalogues: Dict[str, Union[List[CatalogueObject], CataloguePool, GrowingCatalogue]] = field(default_factory=dict)
    # how its catalogues' attributes change over the run, in order (see Timetable)
    catalogue_changes: Tuple[CatalogueChange, ...] = ()


@dataclass(frozen=True)
class SimulationConfig:
    """Everything one multi-process run needs, checked when it is built (docs/orchestrator.md, "Configuration").
    extra_processes names ready-made engines, e.g. scripted test engines, so that consumer groups can name them."""

    organizations: List[Organization]
    start_datetime: datetime
    seed: Optional[int] = None
    consumer_groups: Optional[Dict[str, List[str]]] = None
    extra_processes: Tuple[str, ...] = ()

    @property
    def processes(self) -> List[ProcessSpec]:
        """Every organization's processes, in the order of the organizations."""
        return [spec for organization in self.organizations for spec in organization.processes]

    @property
    def catalogue(self) -> Dict[str, Union[List[CatalogueObject], CataloguePool, GrowingCatalogue]]:
        """Every organization's catalogues, name -> objects, for the OCEL output; a process uses its organization's."""
        return {name: entry for organization in self.organizations for name, entry in organization.catalogues.items()}

    @property
    def references(self) -> Tuple[Reference, ...]:
        """Every process's references, for the OCEL output; a process uses its own."""
        return tuple(reference for spec in self.processes for reference in spec.references)

    def __post_init__(self):
        object.__setattr__(self, "extra_processes", tuple(self.extra_processes))
        organizations = [organization.name for organization in self.organizations]
        repeated = sorted({name for name in organizations if organizations.count(name) > 1})
        if repeated:
            raise ValueError(f"Organization names must be unique, repeated: {repeated}")
        names = [spec.name for spec in self.processes] + list(self.extra_processes)
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Process names must be unique, repeated: {duplicates}")

        object.__setattr__(self, "consumer_groups", checked_consumer_groups(names, self.consumer_groups))
        checked_catalogue(self.organizations)
        _checked_ocel_attributes(self.organizations)

    @classmethod
    def from_json(cls, path) -> "SimulationConfig":
        """Loads a configuration file (docs/orchestrator.md, "Configuration"). Its paths are relative to its folder,
        and a start time without a time zone is taken as UTC."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        _check_keys(path, "top level", data, CONFIG_KEYS)
        missing = [key for key in ("organizations", "start_time") if key not in data]
        if missing:
            raise ValueError(f"{path}: missing {missing}")

        folder = path.parent
        organizations = []
        for organization_number, organization in enumerate(data["organizations"]):
            in_organization = f"organizations[{organization_number}]"
            _check_keys(path, in_organization, organization, ORGANIZATION_KEYS)
            missing = [key for key in ("name", "processes") if key not in organization]
            if missing:
                raise ValueError(f"{path}: {in_organization} is missing {missing}")
            processes = []
            for number, entry in enumerate(organization["processes"]):
                where = f"{in_organization}.processes[{number}]"
                _check_keys(path, where, entry, PROCESS_KEYS)
                missing = [key for key in ("name", "bpmn_path", "json_path") if key not in entry]
                if missing:
                    raise ValueError(f"{path}: process {entry.get('name', '?')} is missing {missing}")
                references = [_reference(path, f"{where}.references[{index}]", reference)
                              for index, reference in enumerate(entry.get("references", []))]
                processes.append(ProcessSpec(
                    entry["name"], str(folder / entry["bpmn_path"]), str(folder / entry["json_path"]),
                    entry.get("arrivals"), _object_type(path, where, entry), entry.get("qualifier"),
                    tuple(sorted(entry.get("qualifier_by_activity", {}).items())),
                    _ocel_attributes(path, where, entry), tuple(references)
                ))
            catalogues = {name: _catalogue(path, f"{in_organization}.catalogues.{name}", entries)
                          for name, entries in organization.get("catalogues", {}).items()}
            changes = tuple(_catalogue_change(path, f"{in_organization}.catalogue_changes[{index}]", change)
                            for index, change in enumerate(organization.get("catalogue_changes", [])))
            organizations.append(Organization(organization["name"], processes, catalogues, changes))

        start = _utc_if_naive(parse_datetime(data["start_time"], True))

        return cls(organizations, start, data.get("seed"), data.get("consumer_groups"))


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
    if isinstance(entries, dict):  # a pool, or growing without size
        _check_keys(path, where, entries, {"size", "attributes"})
    if isinstance(entries, dict) and "size" not in entries:
        return GrowingCatalogue(tuple(entries.get("attributes", [])))
    if isinstance(entries, dict):
        return CataloguePool(entries.get("size"), tuple(entries.get("attributes", [])))
    return [_catalogue_object(path, f"{where}[{number}]", entry) for number, entry in enumerate(entries)]


def _catalogue_object(path, where, entry):
    _check_keys(path, where, entry, {"id", "attributes"})
    if not isinstance(entry.get("id"), str) or not entry["id"]:
        raise ValueError(f"{path}: {where}: 'id' must be a non-empty string")
    return CatalogueObject(entry["id"], tuple(entry.get("attributes", {}).items()))


def _utc_if_naive(moment):
    """A time without a time zone, taken as UTC."""
    return pytz.utc.localize(moment) if moment.tzinfo is None else moment


def _catalogue_change(path, where, entry):
    """A catalogue change as JSON gives it, checked for everything but what it names (see checked_catalogue)."""
    _check_keys(path, where, entry, CHANGE_KEYS)
    missing = [key for key in ("catalogue", "attribute") if key not in entry]
    if missing:
        raise ValueError(f"{path}: {where} is missing {missing}")
    for key in ("catalogue", "attribute"):
        if not isinstance(entry[key], str) or not entry[key]:
            raise ValueError(f"{path}: {where}: '{key}' must be a non-empty string, got {entry[key]!r}")
    when = [key for key in ("every", "at") if key in entry]
    if len(when) != 1:
        raise ValueError(f"{path}: {where} needs exactly one of every (repeated) and at (once), got "
                         f"{' and '.join(when) or 'neither'}")
    operations = [key for key in OPERATIONS if key in entry]
    if len(operations) != 1:
        raise ValueError(f"{path}: {where} needs exactly one of multiply, add and set, got "
                         f"{', '.join(operations) or 'none'}")
    [operation] = operations
    value = entry[operation]
    if operation != "set" and not is_number(value):
        raise ValueError(f"{path}: {where}: {operation} must be a number, got {value!r}")
    rounding = entry.get("round")
    if "round" in entry and not is_whole_number(rounding, minimum=0):
        raise ValueError(f"{path}: {where}: round must be a whole number of at least 0 (decimal places), got "
                         f"{rounding!r}")
    if "round" in entry and not is_number(value):
        raise ValueError(f"{path}: {where}: round needs a number to round, and set gives {value!r}")
    objects = entry.get("objects")
    if "objects" in entry and not (isinstance(objects, list) and objects
                                   and all(isinstance(object_id, str) and object_id for object_id in objects)):
        raise ValueError(f"{path}: {where}: objects must be a non-empty list of object ids, got {objects!r}")
    every, at = None, None
    if "every" in entry:
        every = _period(path, f"{where}.every", entry["every"])
    else:
        try:
            at = _utc_if_naive(parse_datetime(entry["at"], True))
        except (ValueError, TypeError, OverflowError):
            raise ValueError(f"{path}: {where}: at must be a date and time, e.g. \"2024-12-01T00:00:00+00:00\", got "
                             f"{entry['at']!r}") from None
    return CatalogueChange(entry["catalogue"], entry["attribute"], operation, value, every, at,
                           None if rounding is None else int(rounding), None if objects is None else tuple(objects))


def _period(path, where, period):
    if isinstance(period, dict):
        _check_keys(path, where, period, set(PERIODS))
    if not (isinstance(period, dict) and len(period) == 1 and is_number(next(iter(period.values())))
            and next(iter(period.values())) > 0):
        raise ValueError(f'{path}: {where} must be {{"weeks" | "days" | "hours": n}}, n a number above 0, got '
                         f"{period!r}")
    [(unit, amount)] = period.items()
    return timedelta(**{unit: amount})


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
    quantity = entry.get("quantity")
    if isinstance(quantity, dict):
        _check_keys(path, f"{where}.quantity", quantity, {"object_attribute"})
        if isinstance(quantity.get("object_attribute"), str):
            quantity = QuantityFrom(quantity["object_attribute"])
    with_attributes = entry.get("with", ())
    if isinstance(with_attributes, dict):  # anything else is rejected when the configuration is checked
        with_attributes = tuple((name, _computed(source)) for name, source in with_attributes.items())
    return Reference(entry["object_attribute"], entry["catalogue"], entry.get("o2o"),
                     entry.get("reverse", False), None if link_events is None else tuple(link_events), pick,
                     with_attributes,
                     entry.get("tag_along", False), quantity)


def _computed(source):
    """A 'with' source as JSON gives it: {function: attribute} as Computed, anything else as it is."""
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
