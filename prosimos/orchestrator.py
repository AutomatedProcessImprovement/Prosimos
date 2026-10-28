import copy
import itertools
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from prosimos.orchestration.catalogue import _drawn_catalogues, _grown_catalogues, _picks
from prosimos.orchestration.config import SimulationConfig, _checked_consumer_groups
from prosimos.orchestration.ocel_writer import OcelProcess, write_ocel
from prosimos.orchestration.output import (_MergedLog, _message_links, _object_links, _plain, _reference_links,
                                         _tag_along)
from prosimos.orchestration.prosimos_engine import ProsimosEngine
from prosimos.orchestration.protocol import (CaseElement, CaseObject, EngineReport, Message, MessageRecord,
                                           SimulationEngine, StalledCase, Verdict)
from prosimos.orchestration.random_generators import _generators


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
