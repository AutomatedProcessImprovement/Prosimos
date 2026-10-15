import copy
import csv
import hashlib
import itertools
import json
import random
import re
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
from prosimos.messaging_parser import ConsumePoint, PublishPoint
from prosimos.ocel_writer import OcelProcess, write_ocel
from prosimos.simulation_engine import SimBPMEnv
from prosimos.simulation_properties_parser import parse_datetime, parse_simulation_model
from prosimos.simulation_setup import SimDiffSetup
from prosimos.warning_logger import warning_logger


@dataclass(frozen=True)
class ProcessSpec:
    name: str
    bpmn_path: str
    json_path: str
    total_cases: Optional[int] = None  # None for a process started by messages
    # the OCEL object type of its cases: "" (the default) means the process name; None writes no objects
    # and no events for this process in the OCEL output
    object_type: Optional[str] = ""
    # the qualifier of each event's link to its own case's object: None means the object type; per task id,
    # qualifier_by_activity overrides it, e.g. (("Create_Package", "creates"),)
    qualifier: Optional[str] = None
    qualifier_by_activity: Tuple[Tuple[str, str], ...] = ()
    # every later event of a case also links the objects the case is linked to through its messages
    carry_links: bool = False

    @property
    def ocel_object_type(self) -> Optional[str]:
        return self.name if self.object_type == "" else self.object_type


@dataclass(frozen=True)
class CatalogueObject:
    """An object with no events of its own (a customer, a product), declared in the configuration's catalogue."""

    id: str
    attributes: Tuple[Tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class CataloguePool:
    """A generated catalogue, for sets too large to list: count objects with ids <catalogue name>-1 to
    <catalogue name>-<count>, each with attributes drawn once at the start of the run, given in the format
    of case_attributes (discrete or continuous)."""

    count: int
    attributes: Tuple[Dict[str, Any], ...] = ()


@dataclass(frozen=True)
class GrowingCatalogue:
    """A catalogue with no list and no count, for populations that aren't fixed: it starts empty, and a reference
    that picks "new" or new_share creates its objects as cases are created, with ids <catalogue name>-1,
    <catalogue name>-2, ... in creation order, each with attributes in the format of case_attributes."""

    attributes: Tuple[Dict[str, Any], ...] = ()


@dataclass(frozen=True)
class NewShare:
    """A pick on a growing catalogue: share of the cases create a new object, the rest reuse one created so far,
    uniformly; the first case always creates one."""

    share: float


@dataclass(frozen=True)
class Reference:
    """Says that a case attribute's value of one process is the id of an object of a catalogue, and which OCEL
    links to write for it: from the case's object to it with o2o as qualifier (reverse turns it around), and
    from the case's events named in link_events (task names or ids; None for every event, () for none).
    With pick, the reference also assigns the value, when the case is created: "uniform" (every object
    equally likely) or, for a listed catalogue, weights as ((id, weight), ...) (ids left out are never picked);
    for a growing catalogue, "new" (every case creates its own object) or NewShare. Without it, the value comes
    from case_attributes or a message."""

    process: str
    case_attribute: str
    catalogue: str
    o2o: Optional[str] = None
    reverse: bool = False
    link_events: Optional[Tuple[str, ...]] = None
    pick: Union[None, str, Tuple[Tuple[str, float], ...], NewShare] = None

    @property
    def creates(self) -> bool:
        return self.pick == "new" or isinstance(self.pick, NewShare)


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


def _checked_catalogue(processes, names, catalogue, references):
    """Ids unique across the whole catalogue and unlike case ids (<process name>-<number>), since both are OCEL
    object ids; catalogue names unlike the processes' object types; pools with a count of at least 1 and
    attributes in the format of case_attributes; and references only naming known catalogues, processes and,
    in link_events, tasks of their process (by name or id). A pick is "uniform" or weights on known ids of a
    listed catalogue, "uniform" on a pool, or "new" or new_share on a growing catalogue, for a process built
    from a ProcessSpec whose case_attributes don't have the attribute and whose consume entries don't copy or
    append into it, and at most one per process and attribute. A growing catalogue's objects are created by exactly one reference, and no other id looks like
    them (<catalogue name>-<number>)."""
    problems = []
    for name, entry in catalogue.items():
        if not isinstance(entry, (CataloguePool, GrowingCatalogue)):
            continue
        if isinstance(entry, CataloguePool) and (not isinstance(entry.count, int) or isinstance(entry.count, bool)
                                                 or entry.count < 1):
            problems.append(f"catalogue {name!r}: count must be a whole number of at least 1, got {entry.count!r}")
        for number, attribute in enumerate(entry.attributes):
            try:
                CaseAttribute(attribute["name"], attribute["type"], attribute["values"])
            except Exception as error:  # Prosimos's own messages for case_attributes
                problems.append(f"catalogue {name!r}: attributes[{number}]: {type(error).__name__}: {error}")
    ids = [object_id for name, entry in catalogue.items() for object_id in _catalogue_ids(name, entry)]
    repeated = sorted({object_id for object_id in ids if ids.count(object_id) > 1})
    if repeated:
        problems.append(f"catalogue ids are repeated: {repeated}")
    like_case_ids = sorted({object_id for object_id in ids for name in names
                            if re.fullmatch(rf"{re.escape(name)}-\d+", object_id)})
    if like_case_ids:
        problems.append(f"catalogue ids look like case ids (<process name>-<number>): {like_case_ids}")
    growing = [name for name, entry in catalogue.items() if isinstance(entry, GrowingCatalogue)]
    for name in growing:
        if name in names:
            problems.append(f"growing catalogue {name!r} is named like a process; its ids would look like case ids")
        like_created = sorted({object_id for object_id in ids if re.fullmatch(rf"{re.escape(name)}-\d+", object_id)})
        if like_created:
            problems.append(f"catalogue ids look like the ids growing catalogue {name!r} creates "
                            f"({name}-<number>): {like_created}")
        creators = [reference for reference in references if reference.catalogue == name and reference.creates]
        if len(creators) != 1:
            problems.append(f"growing catalogue {name!r} needs exactly one reference that creates its objects "
                            f'("new" or new_share), got {len(creators)}')
    object_types = {spec.ocel_object_type: spec.name for spec in processes if spec.ocel_object_type is not None}
    for name in catalogue:
        if name in object_types:
            problems.append(f"catalogue {name!r} is the object type of process {object_types[name]}")
    specs = {spec.name: spec for spec in processes}
    tasks_of = {}
    for number, reference in enumerate(references):
        if reference.catalogue not in catalogue:
            problems.append(f"references[{number}] names unknown catalogue {reference.catalogue!r}")
        if reference.process not in names:
            problems.append(f"references[{number}] names unknown process {reference.process!r}")
        elif reference.link_events and reference.process in specs:  # extra engines have no BPMN to check
            if reference.process not in tasks_of:
                graph = parse_simulation_model(specs[reference.process].bpmn_path)
                tasks_of[reference.process] = {key for info in graph.element_info.values() if info.type == BPMN.TASK
                                               for key in (info.id, info.name)}
            unknown = [task for task in reference.link_events if task not in tasks_of[reference.process]]
            if unknown:
                problems.append(f"references[{number}] link_events names {unknown}, which are no tasks of "
                                f"process {reference.process}")
        if reference.pick is not None:
            problems.extend(f"references[{number}] {problem}"
                            for problem in _pick_problems(reference, catalogue.get(reference.catalogue), specs))
    picked = [(reference.process, reference.case_attribute) for reference in references if reference.pick is not None]
    for process, attribute in sorted({pair for pair in picked if picked.count(pair) > 1}):
        problems.append(f"more than one reference picks {attribute} of process {process}")
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
    if reference.process not in specs:
        return [f"picks for {reference.process}, an engine not built from a ProcessSpec, which can't take a pick"]
    with open(specs[reference.process].json_path) as file:
        settings = json.load(file)
    declared = {attribute.get("name") for attribute in settings.get("case_attributes", [])}
    if reference.case_attribute in declared:
        return [f"picks {reference.case_attribute}, which process {reference.process} also draws in its "
                f"case_attributes"]
    # a copied value would replace the picked one (with "new", the replaced object is never written), and an
    # appended attribute is a list
    messages = settings.get("messages")
    consume = messages.get("consume", []) if isinstance(messages, dict) else []
    for key, verb in (("copy", "copies"), ("append", "appends to")):
        found_at = [entry.get("event_id") for entry in consume if isinstance(entry, dict)
                    and isinstance(entry.get(key), dict) and reference.case_attribute in entry[key]]
        if found_at:
            return [f"picks {reference.case_attribute}, which process {reference.process} also {verb} from a "
                    f"message (consume entries at {found_at})"]
    return []


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
    ProcessSpec (e.g. scripted test engines), so that groups can name them too. catalogue maps a name
    (the OCEL object type) to the objects with no events of their own, and references say which case
    attributes hold their ids, and may pick them (see _checked_catalogue for what is rejected). A catalogue is
    a list of CatalogueObject, a CataloguePool or a GrowingCatalogue.
    """

    processes: List[ProcessSpec]
    start_datetime: datetime
    seed: Optional[int] = None
    consumer_groups: Optional[Dict[str, List[str]]] = None
    extra_processes: Tuple[str, ...] = ()
    catalogue: Dict[str, Union[List[CatalogueObject], CataloguePool, GrowingCatalogue]] = field(default_factory=dict)
    references: Tuple[Reference, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "extra_processes", tuple(self.extra_processes))
        object.__setattr__(self, "references", tuple(self.references))
        names = [spec.name for spec in self.processes] + list(self.extra_processes)
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Process names must be unique, repeated: {duplicates}")

        object.__setattr__(self, "consumer_groups", _checked_consumer_groups(names, self.consumer_groups))
        _checked_catalogue(self.processes, names, self.catalogue, self.references)

    @classmethod
    def from_json(cls, path) -> "SimulationConfig":
        """
        Load a configuration file such as
            {"processes": [{"name": "Sales", "bpmn_path": "sales.bpmn",
                            "json_path": "sales.json", "total_cases": 100}, ...],
             "seed": 42, "start_time": "2024-01-01T09:00:00+00:00",
             "consumer_groups": {"Sales": ["Sales"], ...}}
        BPMN and JSON paths are relative to the configuration file's folder. seed and
        consumer_groups are optional; a start time without a time zone is taken as UTC. A process
        started by messages has no total_cases. A process's optional object_type names the OCEL object
        type of its cases (by default its name); null writes no objects and no events for it. Its
        optional qualifier (by default the object type) and qualifier_by_activity ({task id: qualifier})
        qualify each event's link to its own case's object, and carry_links (false by default) makes every
        later event of a case also link the objects the case is linked to through its messages.
        The optional catalogue ({name: [{"id": ..., "attributes": {...}}]}, {name: {"count": ...,
        "attributes": [as in case_attributes]}} for a generated pool, or {name: {"attributes": [...]}} for a
        growing one) declares objects with no events of their own, and references ([{"process", "case_attribute", "catalogue", "o2o", "reverse", "link_events",
        "pick"}], the last four optional) say which case attributes hold their ids, and may pick them.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        missing = [key for key in ("processes", "start_time") if key not in data]
        if missing:
            raise ValueError(f"{path}: missing {missing}")

        folder = path.parent
        processes = []
        for entry in data["processes"]:
            missing = [key for key in ("name", "bpmn_path", "json_path") if key not in entry]
            if missing:
                raise ValueError(f"{path}: process {entry.get('name', '?')} is missing {missing}")
            processes.append(ProcessSpec(
                entry["name"], str(folder / entry["bpmn_path"]), str(folder / entry["json_path"]), entry.get("total_cases"),
                entry.get("object_type", ""), entry.get("qualifier"),
                tuple(sorted(entry.get("qualifier_by_activity", {}).items())), entry.get("carry_links", False)
            ))

        start = parse_datetime(data["start_time"], True)
        if start.tzinfo is None:
            start = pytz.utc.localize(start)

        catalogue = {name: _catalogue(path, name, entries) for name, entries in data.get("catalogue", {}).items()}
        references = [_reference(path, number, entry) for number, entry in enumerate(data.get("references", []))]

        return cls(processes, start, data.get("seed"), data.get("consumer_groups"), (), catalogue, references)


def _catalogue(path, name, entries):
    if isinstance(entries, dict) and "count" not in entries:
        return GrowingCatalogue(tuple(entries.get("attributes", [])))
    if isinstance(entries, dict):
        return CataloguePool(entries.get("count"), tuple(entries.get("attributes", [])))
    return [_catalogue_object(path, name, number, entry) for number, entry in enumerate(entries)]


def _catalogue_object(path, name, number, entry):
    if not isinstance(entry.get("id"), str) or not entry["id"]:
        raise ValueError(f"{path}: catalogue.{name}[{number}]: 'id' must be a non-empty string")
    return CatalogueObject(entry["id"], tuple(entry.get("attributes", {}).items()))


def _reference(path, number, entry):
    missing = [key for key in ("process", "case_attribute", "catalogue") if key not in entry]
    if missing:
        raise ValueError(f"{path}: references[{number}] is missing {missing}")
    link_events, pick = entry.get("link_events"), entry.get("pick")
    # anything else but "uniform" and "new" is rejected when the configuration is checked
    if isinstance(pick, dict) and list(pick) == ["weights"] and isinstance(pick["weights"], dict):
        pick = tuple(pick["weights"].items())
    elif isinstance(pick, dict) and list(pick) == ["new_share"]:
        pick = NewShare(pick["new_share"])
    return Reference(entry["process"], entry["case_attribute"], entry["catalogue"], entry.get("o2o"),
                     entry.get("reverse", False), None if link_events is None else tuple(link_events), pick)


class Verdict(Enum):
    """An engine's answer to one offered message."""

    CLAIMED = "claimed"
    DISCARDED = "discarded"
    PENDING = "pending"


@dataclass
class StalledCase:
    """A case still waiting for a message when the run ended."""

    case_id: str
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
    case_id: str
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
    case_id: str
    element_id: str
    qualifier: str = ""
    task_rows: Tuple[int, ...] = ()
    o2o: Optional[str] = None  # qualifier of the OCEL link from its case's object to the other end's, if any


@dataclass
class CaseObject:
    """One case of an engine, as an object of the OCEL output: its id and its case attributes over time,
    as (attribute, value, time): the values it had when it was created, then every later change."""

    case_id: str
    attributes: List[Tuple[str, Any, datetime]] = field(default_factory=list)


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
    case_id: str
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


class ProsimosEngine(SimulationEngine):
    """A Prosimos simulation (SimBPMEnv) seen through the SimulationEngine interface. It publishes
    the messages its model lists under 'publish', each at the time the case passes the event, and
    subscribes to the types under 'consume': a case reaching such a catch event waits there until a
    delivered message matches (docs/messaging.md).

    Prosimos writes its warnings to one list shared by every engine (warning_logger) and draws its
    random numbers from the global Python and NumPy generators. Each method therefore hands Prosimos
    this engine's own warning list and generator states for the duration of the call, so two engines'
    warnings never mix and one engine's draws don't depend on which other engines run beside it.

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
            sim_setup = SimDiffSetup(spec.bpmn_path, spec.json_path, False, spec.total_cases, start_datetime, picks)
            self._env = SimBPMEnv(sim_setup, None, log_writer, process_name=spec.name)
        tasks = {element_id for element_id, element in sim_setup.bpmn_graph.element_info.items()
                 if element.type is BPMN.TASK}
        unknown = sorted(task_id for task_id, _ in spec.qualifier_by_activity if task_id not in tasks)
        if unknown:
            raise ValueError(f"{spec.name}: qualifier_by_activity names {', '.join(unknown)}, which "
                             f"{'is not a task' if len(unknown) == 1 else 'are not tasks'} of its model")

    @contextmanager
    def _own_globals(self):
        shared_warnings, shared_python, shared_numpy = warning_logger.warnings_queue, random.getstate(), np.random.get_state()
        warning_logger.warnings_queue = self._warnings
        random.setstate(self._python_state)
        np.random.set_state(self._numpy_state)
        try:
            yield
        finally:
            self._python_state, self._numpy_state = random.getstate(), np.random.get_state()
            warning_logger.warnings_queue = shared_warnings
            random.setstate(shared_python)
            np.random.set_state(shared_numpy)

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
            stalled = [StalledCase(self._env.case_id(parked_event.p_case), place, message_types,
                                   parked_event.enabled_datetime, *self._env.collected(parked_event))
                       for parked_event, place, message_types in self._env.stalled_waits()]
            objects = [CaseObject(self._env.case_id(p_case), history)
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
        out of the OCEL output attaches nothing."""
        attach = self._spec.ocel_object_type is not None
        rows_of_case = defaultdict(list)
        for row, (p_case, element_id, enabled, completed) in enumerate(self._env.logged_rows):
            rows_of_case[p_case].append((row, element_id, enabled, completed))
        warned = set()

        def task_rows(p_case, element_id, time, forward):
            if not attach:
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
                side, case = "after" if forward else "before", self._env.case_id(p_case)
                if not tasks:
                    reason = f"{element_id} has no task {side} it in the model"
                elif forward:  # e.g. the case was still waiting further on when the run ended
                    reason = f"case {case} reached no task after {element_id} before the run ended"
                else:
                    reason = f"case {case} passed no task before {element_id}"
                warning_logger.add_warning(f"{reason}; the OCEL links of its messages there are dropped")
            return tuple(found)

        published = [PublishRecord(message.id, self._env.case_id(p_case), element_id, point.link_qualifier,
                                   task_rows(p_case, element_id, when, forward=False), point.o2o)
                     for message, p_case, element_id, point, when in self._published]
        claimed = [ClaimRecord(message_id, self._env.case_id(p_case), element_id, point.link_qualifier,
                               task_rows(p_case, element_id, when, forward=True), point.o2o)
                   for message_id, p_case, element_id, point, when in self._claimed]
        return published, claimed


class _Pick:
    """A case attribute that a reference picks (pick) from its catalogue's ids. It has what the engine uses of
    Prosimos's CaseAttribute, name and get_next_value(), so it is drawn with the case's other attributes, when
    the case is created; its generator is its own, seeded from the seed, the process and the attribute, so it
    doesn't change the engine's other draws."""

    def __init__(self, name: str, ids: List[str], weights: Optional[List[float]], seed: Optional[int], process: str):
        self.name, self._ids, self._weights = name, ids, weights
        self._generator = random.Random()
        self._generator.setstate(_engine_random_states(seed, f"pick/{process}/{name}")[0])

    def get_next_value(self):
        return self._generator.choices(self._ids, self._weights)[0]


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


def _picks(config: "SimulationConfig", catalogue: Dict[str, List[CatalogueObject]]) -> Dict[str, List[_Pick]]:
    """Process name -> the attributes its references pick."""
    picks = defaultdict(list)
    for reference in config.references:
        if reference.creates:
            share = reference.pick.share if isinstance(reference.pick, NewShare) else 1
            picks[reference.process].append(_NewPick(reference.case_attribute, reference.catalogue, share, config.seed,
                                                     reference.process))
        elif reference.pick is not None:
            ids = [obj.id for obj in catalogue[reference.catalogue]]
            weights = None if reference.pick == "uniform" else [dict(reference.pick).get(object_id, 0) for object_id in ids]
            picks[reference.process].append(_Pick(reference.case_attribute, ids, weights, config.seed, reference.process))
    return picks


def _drawn_catalogue(config: "SimulationConfig") -> Dict[str, List[CatalogueObject]]:
    """Catalogue name -> its objects when the run starts: a listed catalogue's as given, a pool's with their
    attributes drawn (see _drawn_objects), none yet for a growing catalogue."""
    return {name: list(entry) if isinstance(entry, list) else _drawn_objects(config.seed, name, entry.attributes,
                                                                            _catalogue_ids(name, entry))
            for name, entry in config.catalogue.items()}


def _grown_catalogue(config: "SimulationConfig", catalogue: Dict[str, List[CatalogueObject]],
                     picks: Dict[str, List[_Pick]]) -> Dict[str, List[CatalogueObject]]:
    """The catalogue after the run: each growing catalogue with the objects its pick created, <name>-1 to
    <name>-<created>, their attributes drawn in creation order (see _drawn_objects). Drawing them now rather
    than when each was created gives the same values, since nothing else draws from their generators."""
    grown = dict(catalogue)
    for pick in itertools.chain.from_iterable(picks.values()):
        if isinstance(pick, _NewPick):
            grown[pick.catalogue] = _drawn_objects(config.seed, pick.catalogue, config.catalogue[pick.catalogue].attributes,
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
            "stalled": [{"process": process, "case_id": case.case_id, "event_id": case.event_id,
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
    stepping engine's group are offered to their group's members in random order until one
    claims them. A copy is removed when claimed, or when every member subscribed to its type has
    discarded it; nobody is offered a copy they already discarded. Member order comes from the
    orchestrator's own generator, never the global random module the engines use.
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

    def resolve(pooled):
        pool.remove(pooled)
        copies_left[pooled.message.id] -= 1
        if copies_left[pooled.message.id] == 0 and pooled.message.id not in claimed_ids:
            message = pooled.message
            warn(f"{message.type} {message.id} from {message.source} was discarded by every recipient")

    while True:
        due = [(t, name) for name, engine in engines.items() if (t := engine.next_event_time()) is not None]
        if not due:
            break
        now, name = min(due)
        published = engines[name].step()
        report.executed.append((now, name))

        new_copies = []
        for message in published:
            message.id, message.source, message.time = f"m{next(next_id)}", name, now
            report.published.append(message)
            groups = routing.get(message.type, [])
            if not groups:
                warn(f"{message.type} {message.id} from {name} has no subscribers")
            copies_left[message.id] = len(groups)
            for group in groups:
                pooled = _PooledCopy(copy.deepcopy(message), group)
                pool.append(pooled)
                new_copies.append(pooled)
                report.copies.append((message.id, group))

        own_group = group_of[name]
        to_offer = new_copies + [c for c in pool if c.group == own_group and c not in new_copies]
        for pooled in to_offer:
            message = pooled.message
            recipients = [m for m in consumer_groups[pooled.group] if message.type in subscriptions[m]]
            candidates = [m for m in recipients if m not in pooled.discarded_by]
            rng.shuffle(candidates)
            for member in candidates:
                verdict = engines[member].deliver(message, now)
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
            publishers[record.message_id] = CaseElement(name, record.case_id, record.element_id, record.qualifier,
                                                        record.task_rows, record.o2o)
        for record in engine_report.claimed:
            claimers[record.message_id].append(CaseElement(name, record.case_id, record.element_id, record.qualifier,
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
    catalogue = _drawn_catalogue(config)
    picks = _picks(config, catalogue)
    engines: Dict[str, SimulationEngine] = {}
    for spec in sorted(config.processes, key=lambda p: p.name):
        log_writer = merged_log.writer_for(spec.name) if merged_log is not None else None
        engines[spec.name] = ProsimosEngine(spec, config.start_datetime, log_writer, config.seed,
                                            tuple(picks.get(spec.name, ())))
    engines.update(extra_engines)

    report = run_engines(engines, config.consumer_groups, config.seed)
    catalogue = _grown_catalogue(config, catalogue, picks)

    if log_out_path is not None:
        merged_log.write(log_out_path)
    if ocel_out_path is not None:
        processes = {spec.name: OcelProcess(spec.ocel_object_type, spec.qualifier, dict(spec.qualifier_by_activity),
                                            spec.carry_links) for spec in config.processes}
        static_objects, reference_object_links, reference_event_links = _reference_links(
            config, catalogue, processes, report.objects, merged_log.rows, report.logged_elements,
            report.warnings.append)
        object_links = _object_links(report.message_records, processes)
        for object_id, links in reference_object_links.items():
            object_links.setdefault(object_id, []).extend(links)
        write_ocel(ocel_out_path, processes, report.objects, merged_log.rows, report.logged_elements,
                   _message_links(report.message_records, processes), object_links, static_objects,
                   reference_event_links)

    return report


def _message_links(records: List[MessageRecord], processes: Dict[str, OcelProcess]):
    """(process, row of that process) -> [(object id, qualifier)]: the tasks a publisher's records attach to
    link the object of every case that claimed the message, and the tasks a claimer's records attach to link
    the publisher's object, each with its own end's qualifier. Ends with no task, or whose other end's process
    writes no objects, link nothing."""
    def has_object(element):
        return element.process in processes and processes[element.process].object_type is not None

    links = defaultdict(list)
    for record in records:
        if record.publisher is None:
            continue
        publisher = record.publisher
        for claimer in record.claimers:
            for row in publisher.task_rows if has_object(claimer) else ():
                links[(publisher.process, row)].append((claimer.case_id, publisher.qualifier))
            for row in claimer.task_rows if has_object(publisher) else ():
                links[(claimer.process, row)].append((publisher.case_id, claimer.qualifier))
    return dict(links)


def _object_links(records: List[MessageRecord], processes: Dict[str, OcelProcess]):
    """Object id -> [(object id, qualifier)]: the OCEL links between objects that the messages' o2o keys give.
    A publish entry's o2o links the publisher's object to the object of every case that claimed the
    message, a consume entry's links the claimer's object to the publisher's. A link is written once, however
    many messages give it; ends whose process writes no objects link nothing."""
    def has_object(element):
        return element.process in processes and processes[element.process].object_type is not None

    links = defaultdict(list)

    def link(source, target, qualifier):
        if qualifier and has_object(source) and has_object(target) and (target.case_id, qualifier) not in links[source.case_id]:
            links[source.case_id].append((target.case_id, qualifier))

    for record in records:
        for claimer in record.claimers if record.publisher is not None else ():
            link(record.publisher, claimer, record.publisher.o2o)
            link(claimer, record.publisher, claimer.o2o)
    return dict(links)


def _reference_links(config: SimulationConfig, catalogue: Dict[str, List[CatalogueObject]],
                     processes: Dict[str, OcelProcess], objects, rows, logged_elements, warn):
    """
    The catalogue's objects (catalogue: as _grown_catalogue gives them), as (catalogue name, id, [(attribute,
    value, time)]): every object of a listed catalogue, and the objects of a pool or a growing catalogue that some
    case's referenced attribute names; their attributes are timed at the start of the run, a growing catalogue's
    when the first case took its id; and the links its references give, joined with the case objects and log rows
    after the run:
    - object id -> [(object id, qualifier)]: a reference with o2o links the object of every case of its process
      to the catalogue object of each value its attribute took (reverse: from the catalogue object);
    - (process, row of that process) -> [(object id, qualifier)]: each event of those cases named in
      link_events (every event without the key) links the catalogue object of the value the attribute had when
      the event completed, with the attribute's name as qualifier.
    A value that is not in its catalogue gets no link, and a warning, once per attribute and value. Processes
    that write no objects link nothing.
    """
    ids_of = {name: {obj.id for obj in catalogue_objects} for name, catalogue_objects in catalogue.items()}
    history = {case_object.case_id: case_object.attributes for _, case_object in objects}
    object_links, event_links, warned, referred = defaultdict(list), defaultdict(list), set(), set()
    first_taken = {}  # (catalogue name, id) -> when the first case took it

    def in_catalogue(reference, value):
        if str(value) in ids_of[reference.catalogue]:
            return True
        if (reference.case_attribute, value) not in warned:
            warned.add((reference.case_attribute, value))
            warn(f"case attribute {reference.case_attribute} has the value {value!r}, which is not in catalogue "
                 f"{reference.catalogue}; no OCEL link is written for it")
        return False

    def values(object_id, attribute, until=None):
        """The values the case's attribute took, in order, or only the one it had at until."""
        taken = [value for name, value, time in history.get(object_id, [])
                 if name == attribute and value is not None and (until is None or time <= until)]
        return taken[-1:] if until is not None else list(dict.fromkeys(taken))

    def link(source, target, qualifier):
        if (target, qualifier) not in object_links[source]:
            object_links[source].append((target, qualifier))

    for reference in config.references:
        has_objects = reference.process in processes and processes[reference.process].object_type is not None
        for process, case_object in objects:
            if process != reference.process:
                continue
            for value in values(case_object.case_id, reference.case_attribute):
                if in_catalogue(reference, value):
                    referred.add((reference.catalogue, str(value)))
                    taken = min(time for name, taken_value, time in history[case_object.case_id]
                                if name == reference.case_attribute and taken_value == value)
                    key = (reference.catalogue, str(value))
                    first_taken[key] = min(taken, first_taken.get(key, taken))
                    if has_objects and reference.o2o:
                        ends = (str(value), case_object.case_id) if reference.reverse else (case_object.case_id, str(value))
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
            for value in values(f"{process}-{row['case_id']}", reference.case_attribute, _as_datetime(row["end_time"])):
                if in_catalogue(reference, value):
                    event_links[(process, number)].append((str(value), reference.case_attribute))

    def time_of(name, object_id):
        growing = isinstance(config.catalogue[name], GrowingCatalogue)
        return first_taken[(name, object_id)] if growing else config.start_datetime

    static_objects = [(name, obj.id, [(attribute, value, time_of(name, obj.id)) for attribute, value in obj.attributes])
                      for name, catalogue_objects in catalogue.items() for obj in catalogue_objects
                      if isinstance(config.catalogue[name], list) or (name, obj.id) in referred]
    return static_objects, dict(object_links), dict(event_links)
