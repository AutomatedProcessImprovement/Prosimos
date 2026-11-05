"""A whole configuration's run: catalogues, picks, engines, the run loop and the output (docs/orchestrator.md)."""
import random
from typing import Dict, Optional

from prosimos.orchestration.catalogue.picks import drawn_catalogues, grown_catalogues, picks_of
from prosimos.orchestration.catalogue.timetable import Timetable
from prosimos.orchestration.config import SimulationConfig
from prosimos.orchestration.ocel_writer import OcelProcess, write_ocel
from prosimos.orchestration.orchestrator import RunReport, orchestrate
from prosimos.orchestration.output import MergedLog, message_links, message_object_links, reference_links, tag_along
from prosimos.orchestration.prosimos_engine import ProsimosEngine
from prosimos.orchestration.protocol import SimulationEngine


def run_multiprocess_simulation(config: SimulationConfig, log_out_path: Optional[str] = None,
                                extra_engines: Optional[Dict[str, SimulationEngine]] = None,
                                ocel_out_path: Optional[str] = None) -> RunReport:
    """Simulates the configured processes on one clock (see orchestrate), and writes the merged CSV log and the OCEL
    file to the paths given (docs/orchestrator.md, "Output"). extra_engines: ready-made engines, keyed by process
    name, exactly config.extra_processes; they write no log."""
    extra_engines = dict(extra_engines or {})
    if sorted(extra_engines) != sorted(config.extra_processes):
        raise ValueError(f"extra_engines must be exactly the configuration's extra_processes "
                         f"{sorted(config.extra_processes)}, got {sorted(extra_engines)}")

    merged_log = MergedLog() if log_out_path is not None or ocel_out_path is not None else None

    # without a seed, every generator is seeded from the global ones, so the order is fixed: all catalogues, then
    # all picks, then the engines in name order, whatever the order of the configuration
    drawn = {organization.name: drawn_catalogues(organization, config.seed) for organization in config.organizations}
    timetables = {organization.name: Timetable(organization.catalogue_changes, config.start_datetime)
                  for organization in config.organizations}
    picks = {spec.name: picks_of(spec, config.seed, drawn[organization.name], timetables[organization.name])
             for organization in config.organizations for spec in organization.processes}
    engines: Dict[str, SimulationEngine] = {}
    for spec in sorted(config.processes, key=lambda p: p.name):
        log_writer = merged_log.writer_for(spec.name) if merged_log is not None else None
        engines[spec.name] = ProsimosEngine(spec, config.start_datetime, log_writer, config.seed,
                                            tuple(picks[spec.name]))
    engines.update(extra_engines)

    report = orchestrate(engines, config.consumer_groups, config.seed, config.load_balancing)
    # the OCEL output needs every organization's catalogue objects, the growing ones as the run left them
    catalogue = {name: objects for organization in config.organizations
                 for name, objects in grown_catalogues(organization, config.seed, drawn[organization.name],
                                                        [pick for spec in organization.processes
                                                         for pick in picks[spec.name]]).items()}

    if log_out_path is not None:
        merged_log.write(log_out_path)
    if ocel_out_path is not None:
        processes = {spec.name: OcelProcess(spec.object_type, spec.qualifier, dict(spec.qualifier_by_activity),
                                            spec.ocel_attributes) for spec in config.processes}
        static_objects, reference_object_links, reference_event_links = reference_links(
            config, catalogue, processes, report.objects, merged_log.rows, report.logged_elements,
            report.warnings.append, {name: timetables[organization.name] for organization in config.organizations
                                     for name in organization.catalogues})
        object_links = message_object_links(report.message_records, processes)
        for source, links in reference_object_links.items():
            object_links.setdefault(source, []).extend(links)
        write_ocel(ocel_out_path, processes, report.objects, merged_log.rows, report.logged_elements,
                   message_links(report.message_records, processes), object_links, static_objects,
                   reference_event_links, tag_along(config, catalogue, report.objects))

    return report
