"""The run loop: a shared clock that steps the engines and routes their messages (docs/orchestrator.md)."""
import copy
import itertools
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from prosimos.orchestration.config import checked_consumer_groups, checked_load_balancing
from prosimos.orchestration.load_balancing import offer_order
from prosimos.orchestration.quotes import ask_members
from prosimos.orchestration.output import plain
from prosimos.orchestration.protocol import (CaseElement, CaseObject, EngineReport, Message, MessageRecord,
                                           SimulationEngine, StalledCase, Verdict)
from prosimos.orchestration.random_generators import generators


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
        """The whole report as plain lists and dicts, e.g. to save as JSON; messages by id, except in published."""
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
                                 "publishers": [plain(publisher) for publisher in record.publishers],
                                 "claimers": [plain(claimer) for claimer in record.claimers]}
                                for record in self.message_records],
        }


@dataclass(eq=False)
class _PooledCopy:
    """A group's copy of a message, waiting to be claimed."""

    message: Message
    group: str
    discarded_by: Set[str] = field(default_factory=set)


def _check_one_group(message, groups):
    """A quote type reaches exactly one group: the members of that group answer its quotes."""
    if len(groups) > 1:
        raise ValueError(f"{message.type} {message.id} from {message.source} asks a quote of type "
                         f"{message.quote.quote_type}, which the groups {groups} subscribe to: a quote type must "
                         f"reach exactly one group")


def orchestrate(
    engines: Dict[str, SimulationEngine],
    consumer_groups: Optional[Dict[str, List[str]]] = None,
    seed: Optional[int] = None,
    load_balancing: Optional[Dict[str, Dict[str, Union[Dict[str, float], List[str]]]]] = None,
) -> RunReport:
    """Runs built engines, keyed by process name, on one clock (docs/orchestrator.md, "Message routing"). The
    order members are offered a message in follows their group's policy in load_balancing (random without one),
    drawn from the orchestrator's own generator, not the engines'."""
    names = sorted(engines)
    consumer_groups = checked_consumer_groups(names, consumer_groups)
    load_balancing = checked_load_balancing(consumer_groups, load_balancing)
    group_of = {member: group for group, members in consumer_groups.items() for member in members}
    subscriptions = {name: set(engines[name].subscriptions()) for name in names}
    routing = {}
    for group in sorted(consumer_groups):
        for message_type in sorted({t for member in consumer_groups[group] for t in subscriptions[member]}):
            routing.setdefault(message_type, []).append(group)

    rng = random.Random(f"orchestrator-{seed}") if seed is not None else random.Random()
    report = RunReport()
    # only the requester's settings say which types are quote types, so each is checked at its first quote
    checked_quote_types: Set[str] = set()
    pool: List[_PooledCopy] = []
    copies_left: Dict[str, int] = {}
    claimed_ids: Set[str] = set()
    next_id = itertools.count(1)
    next_line_id = itertools.count(1)

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
        that may draw from them itself: any that doesn't put its own states there (manages_global_generators)."""
        if not getattr(engines[name], "manages_global_generators", False):
            generators.give_back()
        return engines[name]

    # nothing here draws from the global generators, so the engines may leave their states in them
    generators.keep = True
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
                if message.quote is not None:
                    for line in message.quote.lines:
                        line.id = f"l{next(next_line_id)}"
                    quote_type = message.quote.quote_type
                    groups = routing.get(quote_type, [])
                    if quote_type not in checked_quote_types:
                        _check_one_group(message, groups)
                        checked_quote_types.add(quote_type)
                    # a quote isn't pooled: the members of its one group are asked at once, never the requester
                    members = [member for group in groups for member in consumer_groups[group]
                               if quote_type in subscriptions[member] and member != name]
                    ask_members(message, members, engine, now)
                    continue
                # a message never goes back to its publisher, so a group with no other subscriber gets no copy
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
                for member in offer_order(load_balancing.get(pooled.group), candidates, rng):
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
        generators.keep = False
        generators.give_back()

    report.unclaimed = [(pooled.group, pooled.message) for pooled in pool]
    # the engines' records are only joined here, once the run is over: nothing during the run uses them
    publishers, claimers = defaultdict(list), defaultdict(list)
    for name in names:
        engine_report = engines[name].finish()
        if not isinstance(engine_report, EngineReport):
            raise TypeError(f"{name}.finish() must return an EngineReport, got {engine_report!r}")
        report.stalled.extend((name, case) for case in engine_report.stalled)
        report.engine_warnings.extend((name, warning) for warning in engine_report.warnings)
        report.objects.extend((name, case_object) for case_object in engine_report.objects)
        report.logged_elements[name] = engine_report.logged_elements
        for record in engine_report.published:
            publishers[record.message_id].append(CaseElement(name, record.object_id, record.element_id,
                                                             record.qualifier, record.task_rows, record.o2o,
                                                             record.carry))
        for record in engine_report.claimed:
            claimers[record.message_id].append(CaseElement(name, record.object_id, record.element_id, record.qualifier,
                                                           record.task_rows, record.o2o, record.carry))
    report.message_records = [MessageRecord(message.id, message.type, publishers[message.id], claimers[message.id])
                              for message in report.published]
    return report
