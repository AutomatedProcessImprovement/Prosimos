import csv
import random
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from prosimos.simulation_engine import SimBPMEnv
from prosimos.simulation_setup import SimDiffSetup


@dataclass(frozen=True)
class ProcessSpec:
    name: str
    bpmn_path: str
    json_path: str
    total_cases: int


def run_orchestrator(
    processes: List[ProcessSpec],
    start_datetime: datetime,
    seed: Optional[int] = None,
    log_out_dir: Optional[str] = None,
) -> List[Tuple[datetime, str]]:
    """
    Simulate several processes side by side on one shared clock, always executing the
    globally earliest event next. Ties are broken by process name, so runs given the same
    seed are repeatable; without a seed each run draws different random values.
    Writes one event log per process to log_out_dir/<name>.csv when a directory is given.
    Returns the executed (event time, process name) pairs in execution order.
    """
    names = [spec.name for spec in processes]
    if len(set(names)) != len(names):
        raise ValueError(f"Process names must be unique, got {names}")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    with ExitStack() as open_files:
        # engines share the global random generators, and an engine draws from them the first
        # time it is asked for its next event, so build and query them in name order rather
        # than input order to keep the result independent of how the list was written
        engines = {}
        for spec in sorted(processes, key=lambda p: p.name):
            log_writer = None
            if log_out_dir is not None:
                log_file = open_files.enter_context(
                    open(Path(log_out_dir) / f"{spec.name}.csv", mode="w", newline="", encoding="utf-8")
                )
                log_writer = csv.writer(log_file)
            sim_setup = SimDiffSetup(spec.bpmn_path, spec.json_path, False, spec.total_cases, start_datetime)
            engines[spec.name] = SimBPMEnv(sim_setup, None, log_writer)

        executed = []
        while True:
            due = [(t, name) for name, engine in engines.items() if (t := engine.next_event_time()) is not None]
            if not due:
                break
            event_time, name = min(due)
            engines[name].step()
            executed.append((event_time, name))

        for engine in engines.values():
            engine.log_writer.force_write()

    return executed
