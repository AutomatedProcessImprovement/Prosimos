"""The run's output: the merged log and the links of the OCEL file (docs/orchestrator.md, "Output")."""
import csv
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from prosimos.orchestration.catalogue.model import CatalogueObject, GrowingCatalogue
from prosimos.orchestration.catalogue.timetable import Timetable
from prosimos.orchestration.config import SimulationConfig
from prosimos.orchestration.ocel_writer import MessageEvent, OcelProcess
from prosimos.orchestration.protocol import CaseElement, MessageRecord


class MergedLog:
    """Collects every process's rows and writes them to one CSV, sorted by start time, process name first. The
    header is the union of the processes' columns; a process leaves the others blank."""

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

        # rows arrive in execution order, not start-time order; the stable sort keeps the execution order of ties
        rows = sorted(self._rows, key=lambda row: (_as_datetime(row[1]["start_time"]), row[0]))

        with open(path, mode="w", newline="", encoding="utf-8") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(["process", *columns])
            for process_name, values in rows:
                writer.writerow([process_name, *(values.get(c, "") for c in columns)])


def _as_datetime(value):
    # timestamps come as datetimes, or as strings when they have no fractional seconds (see verify_miliseconds)
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


def plain(element: CaseElement) -> Dict[str, Any]:
    return {**asdict(element), "at": _iso(element.at), "enabled": _iso(element.enabled)}


def _iso(moment):
    return None if moment is None else moment.isoformat()


def message_events(records: List[MessageRecord], processes: Dict[str, OcelProcess]) -> List[MessageEvent]:
    """The OCEL events of the ends with a qualifier, in a process with an object type: one per case, element and
    moment, linked to the objects at the other end of every message it sent or took then, e.g. one for all the
    messages of a fan-out, or for those a collect took at once."""
    def has_object(element):
        return element.process in processes and processes[element.process].object_type is not None

    events = {}  # (process, object id, element id, moment) -> its event, in the order of their first message
    for record in records:
        for end, others in [*((publisher, record.claimers) for publisher in record.publishers),
                            *((claimer, record.publishers) for claimer in record.claimers)]:
            if end.at is not None and has_object(end):
                event = events.setdefault((end.process, end.object_id, end.element_id, end.at), MessageEvent(
                    end.process, end.object_id, end.element_id, end.activity, end.at, end.enabled))
                event.links.extend(((other.process, other.object_id), end.qualifier, end.carry)
                                   for other in others if has_object(other))
    return list(events.values())


def message_object_links(records: List[MessageRecord], processes: Dict[str, OcelProcess]):
    """(process, object id) -> [((process, object id), qualifier)]: the links between objects the messages' o2o keys
    give, each once: a publish entry's from the publisher, a consume entry's from the claimer."""
    def has_object(element):
        return element.process in processes and processes[element.process].object_type is not None

    links = defaultdict(list)

    def link(source, target, qualifier):
        if not (qualifier and has_object(source) and has_object(target)):
            return
        end = (target.process, target.object_id)
        if (end, qualifier) not in links[(source.process, source.object_id)]:
            links[(source.process, source.object_id)].append((end, qualifier))

    for record in records:
        for publisher in record.publishers:
            for claimer in record.claimers:
                link(publisher, claimer, publisher.o2o)
                link(claimer, publisher, claimer.o2o)
    return dict(links)


def reference_links(config: SimulationConfig, catalogue: Dict[str, List[CatalogueObject]],
                    processes: Dict[str, OcelProcess], objects, rows, logged_elements, warn,
                    timetables: Optional[Dict[str, Timetable]] = None):
    """The catalogue objects to write, as (catalogue name, id, [(attribute, value, time)]), and the links the
    references give: object -> [(object, qualifier)] and (process, row) -> [(object id, qualifier)] (docs/catalogue.md,
    "The links"). timetables: catalogue name -> its organization's timetable; None: values that never change."""
    ids_of = {name: {obj.id for obj in catalogue_objects} for name, catalogue_objects in catalogue.items()}
    history = {(process, case_object.object_id): case_object.attributes for process, case_object in objects}
    object_links, event_links, warned, referred = defaultdict(list), defaultdict(list), set(), set()
    first_taken = {}  # (catalogue name, id) -> when the first case took it

    def in_catalogue(reference, value):
        if str(value) in ids_of[reference.catalogue]:
            return True
        if (reference.object_attribute, value) not in warned:
            warned.add((reference.object_attribute, value))
            warn(f"object attribute {reference.object_attribute} has the value {value!r}, which is not in catalogue "
                 f"{reference.catalogue}; no OCEL link is written for it")
        return False

    def values(case, attribute, until=None):
        """The ids the attribute of case named, in order and each once: all of them, or those of its value at until."""
        taken = [value for name, value, time in history.get(case, [])
                 if name == attribute and value is not None and (until is None or time <= until)]
        return list(dict.fromkeys(entry for value in (taken[-1:] if until is not None else taken)
                                  for entry in _entries(value)))

    def link(source, target, qualifier):
        if (target, qualifier) not in object_links[source]:
            object_links[source].append((target, qualifier))

    for reference in config.references:
        has_objects = reference.process in processes and processes[reference.process].object_type is not None
        for process, case_object in objects:
            if process != reference.process:
                continue
            case = (process, case_object.object_id)
            for value in values(case, reference.object_attribute):
                if in_catalogue(reference, value):
                    referred.add((reference.catalogue, str(value)))
                    taken = min(time for name, taken_value, time in history[case]
                                if name == reference.object_attribute and value in _entries(taken_value))
                    key = (reference.catalogue, str(value))
                    first_taken[key] = min(taken, first_taken.get(key, taken))
                    if has_objects and reference.o2o:
                        ends = (str(value), case) if reference.reverse else (case, str(value))
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
            case = (process, f"{processes[process].object_type}-{row['case_id']}")
            for value in values(case, reference.object_attribute, _as_datetime(row["end_time"])):
                if in_catalogue(reference, value):
                    event_links[(process, number)].append((str(value), reference.object_attribute))

    def time_of(name, object_id):
        growing = isinstance(config.catalogue[name], GrowingCatalogue)
        return first_taken[(name, object_id)] if growing else config.start_datetime

    end = max((_as_datetime(row["end_time"]) for _, row in rows), default=config.start_datetime)

    def timed(name, object_id, attributes):
        """An object's attributes, (attribute, value, time): its values when written, then at each change up to the
        end of the run, its last event."""
        time = time_of(name, object_id)
        if timetables is None:
            return [(attribute, value, time) for attribute, value in attributes]
        timetable = timetables[name]
        written = [(attribute, timetable.value(name, attribute, object_id, value, time), time)
                   for attribute, value in attributes]
        changed = [(moment, number, attribute, timetable.value(name, attribute, object_id, value, moment))
                   for number, (attribute, value) in enumerate(attributes)
                   for moment in timetable.changed_at(name, attribute, object_id, time, end)]
        changed.sort(key=lambda found: found[:2])  # in time, and at one moment in the attributes' order
        return written + [(attribute, value, moment) for moment, _, attribute, value in changed]

    static_objects = [(name, obj.id, timed(name, obj.id, obj.attributes))
                      for name, catalogue_objects in catalogue.items() for obj in catalogue_objects
                      if isinstance(config.catalogue[name], list) or (name, obj.id) in referred]
    return static_objects, dict(object_links), dict(event_links)


def tag_along(config: SimulationConfig, catalogue: Dict[str, List[CatalogueObject]], objects):
    """(process, object id) -> [(qualifier, [(time, [catalogue object id, ...])])]: for each reference with
    tag_along, the catalogue objects the case's object named, from when (see write_ocel)."""
    ids_of = {name: {obj.id for obj in catalogue_objects} for name, catalogue_objects in catalogue.items()}
    tagged = defaultdict(list)
    for reference in config.references:
        if not reference.tag_along:
            continue
        for process, case_object in objects:
            if process == reference.process:
                timeline = []
                for name, value, time in case_object.attributes:
                    if name == reference.object_attribute and value is not None:
                        named = list(dict.fromkeys(str(entry) for entry in _entries(value)
                                                   if str(entry) in ids_of[reference.catalogue]))
                        if named:
                            timeline.append((time, named))
                tagged[(process, case_object.object_id)].append((reference.object_attribute, timeline))
    return dict(tagged)


def _entries(value):
    """The catalogue ids a referenced attribute's value names: a list's entries, or the value itself."""
    return [entry for entry in value if entry is not None] if isinstance(value, (list, tuple)) else [value]
