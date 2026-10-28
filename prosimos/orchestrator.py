import random
from typing import Dict, Optional

from prosimos.orchestration.catalogue import _drawn_catalogues, _grown_catalogues, _picks
from prosimos.orchestration.config import SimulationConfig
from prosimos.orchestration.ocel_writer import OcelProcess, write_ocel
from prosimos.orchestration.orchestrator import RunReport, run_engines
from prosimos.orchestration.output import _MergedLog, _message_links, _object_links, _reference_links, _tag_along
from prosimos.orchestration.prosimos_engine import ProsimosEngine
from prosimos.orchestration.protocol import SimulationEngine


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
