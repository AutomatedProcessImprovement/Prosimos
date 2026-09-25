import csv
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from prosimos.simulation_engine import SimBPMEnv
from prosimos.simulation_setup import SimDiffSetup


@dataclass(frozen=True)
class ProcessSpec:
    name: str
    bpmn_path: str
    json_path: str
    total_cases: int


class SimulationEngine(ABC):
    """
    The complete set of methods the orchestrator may call on an engine; nothing else may cross
    that boundary, and no process may read another process's data by any other route. Only
    next_event_time() and step() exist so far. The types of the three planned methods are
    provisional until what passes between processes is agreed. See docs/orchestrator.md.
    """

    @abstractmethod
    def next_event_time(self) -> Optional[datetime]:
        """When is your next event due? None when the engine has nothing left to do."""

    @abstractmethod
    def step(self) -> None:
        """Perform exactly one event."""

    @abstractmethod
    def pending_publish(self) -> List[Any]:
        """Did that event produce anything to send out? (planned)"""

    @abstractmethod
    def blocked_on(self) -> Optional[Any]:
        """Is your next event waiting, and for what? None when it isn't. (planned)"""

    @abstractmethod
    def inject(self, objects: List[Any]) -> None:
        """Here are the things you were waiting for. (planned)"""


class ProsimosEngine(SimulationEngine):
    """A Prosimos simulation (SimBPMEnv) seen through the SimulationEngine interface."""

    def __init__(self, spec: ProcessSpec, start_datetime: datetime, log_writer=None):
        sim_setup = SimDiffSetup(spec.bpmn_path, spec.json_path, False, spec.total_cases, start_datetime)
        self._env = SimBPMEnv(sim_setup, None, log_writer)

    def next_event_time(self) -> Optional[datetime]:
        return self._env.next_event_time()

    def step(self) -> None:
        self._env.step()
        # the engine buffers its log rows; hand them over now so nobody outside the engine
        # has to reach into it to flush them at the end
        self._env.log_writer.force_write()

    def pending_publish(self) -> List[Any]:
        raise NotImplementedError("pending_publish() is planned but not implemented yet")

    def blocked_on(self) -> Optional[Any]:
        raise NotImplementedError("blocked_on() is planned but not implemented yet")

    def inject(self, objects: List[Any]) -> None:
        raise NotImplementedError("inject() is planned but not implemented yet")


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


def run_orchestrator(
    processes: List[ProcessSpec],
    start_datetime: datetime,
    seed: Optional[int] = None,
    log_out_path: Optional[str] = None,
) -> List[Tuple[datetime, str]]:
    """
    Simulate several processes side by side on one shared clock, repeatedly stepping the
    engine whose next_event_time() is earliest. Ties are broken by process name, so runs given
    the same seed are repeatable; without a seed each run draws different random values.
    An engine's next event isn't always its earliest one (timers and case priorities jump
    ahead), so engines aren't strictly kept in step; see docs/orchestrator.md.
    When log_out_path is given, every process's events are written to that one CSV, sorted
    by start time, with the process name as the first column.
    Returns the executed (event time, process name) pairs in execution order.
    """
    names = [spec.name for spec in processes]
    if len(set(names)) != len(names):
        raise ValueError(f"Process names must be unique, got {names}")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    merged_log = _MergedLog() if log_out_path is not None else None

    # engines share the global random generators, and an engine draws from them the first
    # time it is asked for its next event, so build and query them in name order rather
    # than input order to keep the result independent of how the list was written
    engines: Dict[str, SimulationEngine] = {}
    for spec in sorted(processes, key=lambda p: p.name):
        log_writer = merged_log.writer_for(spec.name) if merged_log is not None else None
        engines[spec.name] = ProsimosEngine(spec, start_datetime, log_writer)

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
