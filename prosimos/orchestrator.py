import copy
import itertools
import random
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from prosimos.control_flow_manager import BPMN
from prosimos.messaging_parser import ConsumePoint, PublishPoint
from prosimos.orchestration.catalogue import CountFrom, _Pick, _drawn_catalogues, _grown_catalogues, _picks
from prosimos.orchestration.config import ProcessSpec, SimulationConfig, _checked_consumer_groups
from prosimos.orchestration.ocel_writer import OcelProcess, write_ocel
from prosimos.orchestration.output import (_MergedLog, _message_links, _object_links, _plain, _reference_links,
                                         _tag_along)
from prosimos.orchestration.prosimos_engine import load_process
from prosimos.orchestration.protocol import (CaseElement, CaseObject, ClaimRecord, EngineReport, Message,
                                           MessageRecord, PublishRecord, SimulationEngine, StalledCase, Verdict)
from prosimos.orchestration.random_states import _engine_random_states
from prosimos.simulation_engine import SimBPMEnv
from prosimos.warning_logger import warning_logger


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
    prosimos/orchestration/ocel_writer.py).
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
