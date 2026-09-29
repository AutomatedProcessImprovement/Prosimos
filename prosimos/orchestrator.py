import csv
import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytz

from prosimos.simulation_engine import SimBPMEnv
from prosimos.simulation_properties_parser import parse_datetime
from prosimos.simulation_setup import SimDiffSetup


@dataclass(frozen=True)
class ProcessSpec:
    name: str
    bpmn_path: str
    json_path: str
    total_cases: int


@dataclass
class Message:
    """One object passed between processes. The publishing process sets type and attributes;
    the orchestrator sets id, source and time when it stamps the message."""

    type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    source: Optional[str] = None
    time: Optional[datetime] = None


@dataclass(frozen=True)
class SimulationConfig:
    """
    Everything needed for one multi-process run. Each consumer group maps a group name to the
    processes in it; without consumer_groups, every process is its own group. Every process must
    be in exactly one group, and groups may only name known processes.
    """

    processes: List[ProcessSpec]
    start_datetime: datetime
    seed: Optional[int] = None
    consumer_groups: Optional[Dict[str, List[str]]] = None

    def __post_init__(self):
        names = [spec.name for spec in self.processes]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Process names must be unique, repeated: {duplicates}")

        if self.consumer_groups is None:
            object.__setattr__(self, "consumer_groups", {name: [name] for name in names})

        groups_of = {}
        for group, members in self.consumer_groups.items():
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

    @classmethod
    def from_json(cls, path) -> "SimulationConfig":
        """
        Load a configuration file such as
            {"processes": [{"name": "Sales", "bpmn_path": "sales.bpmn",
                            "json_path": "sales.json", "total_cases": 100}, ...],
             "seed": 42, "start_time": "2024-01-01T09:00:00+00:00",
             "consumer_groups": {"Sales": ["Sales"], ...}}
        BPMN and JSON paths are relative to the configuration file's folder. seed and
        consumer_groups are optional; a start time without a time zone is taken as UTC.
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
            missing = [key for key in ("name", "bpmn_path", "json_path", "total_cases") if key not in entry]
            if missing:
                raise ValueError(f"{path}: process {entry.get('name', '?')} is missing {missing}")
            processes.append(ProcessSpec(
                entry["name"], str(folder / entry["bpmn_path"]), str(folder / entry["json_path"]), entry["total_cases"]
            ))

        start = parse_datetime(data["start_time"], True)
        if start.tzinfo is None:
            start = pytz.utc.localize(start)

        return cls(processes, start, data.get("seed"), data.get("consumer_groups"))


class SimulationEngine(ABC):
    """
    The complete set of methods the orchestrator may call on an engine, as defined by the
    Orchestrator Protocol; nothing else may cross that boundary, and no process may read another
    process's data by any other route. Engines never call the orchestrator. Messages and their
    ids are typed loosely until the message format is implemented. See docs/orchestrator.md.
    """

    @abstractmethod
    def subscriptions(self) -> List[str]:
        """The message types this process consumes, from its model configuration. Called once, at setup."""

    @abstractmethod
    def next_event_time(self) -> Optional[datetime]:
        """When is your next event due? None when there is nothing to do right now (cases waiting
        for a message don't count)."""

    @abstractmethod
    def step(self) -> List[Any]:
        """Perform exactly one event and return the messages it published; the only way to publish."""

    @abstractmethod
    def deliver(self, msgs: List[Any], now: datetime) -> Tuple[List[Any], List[Any]]:
        """Offer the engine its pending messages at time now. Returns (claimed ids, discarded ids):
        claimed messages are used by a case, taking effect at now; discarded ones will never be
        used. Any other message stays pending and is offered again later."""


class ProsimosEngine(SimulationEngine):
    """A Prosimos simulation (SimBPMEnv) seen through the SimulationEngine interface. It doesn't
    exchange messages yet: it subscribes to nothing, publishes nothing and claims nothing."""

    def __init__(self, spec: ProcessSpec, start_datetime: datetime, log_writer=None):
        sim_setup = SimDiffSetup(spec.bpmn_path, spec.json_path, False, spec.total_cases, start_datetime)
        self._env = SimBPMEnv(sim_setup, None, log_writer)

    def subscriptions(self) -> List[str]:
        return []

    def next_event_time(self) -> Optional[datetime]:
        return self._env.next_event_time()

    def step(self) -> List[Any]:
        self._env.step()
        # the engine buffers its log rows; hand them over now so nobody outside the engine
        # has to reach into it to flush them at the end
        self._env.log_writer.force_write()
        return []

    def deliver(self, msgs: List[Any], now: datetime) -> Tuple[List[Any], List[Any]]:
        return [], []


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


def run_orchestrator(config: SimulationConfig, log_out_path: Optional[str] = None) -> List[Tuple[datetime, str]]:
    """
    Simulate the configured processes side by side on one shared clock, repeatedly stepping the
    engine whose next_event_time() is earliest. Ties are broken by process name, so runs given
    the same seed are repeatable; without a seed each run draws different random values.
    When log_out_path is given, every process's events are written to that one CSV, sorted
    by start time, with the process name as the first column.
    Returns the executed (event time, process name) pairs in execution order.
    """
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)

    merged_log = _MergedLog() if log_out_path is not None else None

    # engines share the global random generators, and an engine draws from them the first
    # time it is asked for its next event, so build and query them in name order rather
    # than input order to keep the result independent of how the list was written
    engines: Dict[str, SimulationEngine] = {}
    for spec in sorted(config.processes, key=lambda p: p.name):
        log_writer = merged_log.writer_for(spec.name) if merged_log is not None else None
        engines[spec.name] = ProsimosEngine(spec, config.start_datetime, log_writer)

    executed = []
    while True:
        due = [(t, name) for name, engine in engines.items() if (t := engine.next_event_time()) is not None]
        if not due:
            break
        event_time, name = min(due)
        engines[name].step()
        executed.append((event_time, name))

    if merged_log is not None:
        merged_log.write(log_out_path)

    return executed
