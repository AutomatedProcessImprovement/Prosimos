"""
The run's output (docs/orchestrator.md, "Output"): the merged log of every process, and the links of the OCEL file
between events and objects.
"""
import csv
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List

from prosimos.orchestration.catalogue import CatalogueObject, GrowingCatalogue
from prosimos.orchestration.config import SimulationConfig
from prosimos.orchestration.ocel_writer import OcelProcess
from prosimos.orchestration.protocol import CaseElement, MessageRecord


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


def _plain(element: CaseElement) -> Dict[str, Any]:
    return {**asdict(element), "task_rows": list(element.task_rows)}


def _message_links(records: List[MessageRecord], processes: Dict[str, OcelProcess]):
    """(process, row of that process) -> [((process, object id), qualifier)]: the tasks a publisher's records
    attach to link the object of every case that claimed the message, and the tasks a claimer's records attach to
    link the publisher's object, each with its own end's qualifier. Ends without a qualifier or a task, or whose other
    end's process writes no objects, link nothing."""
    def has_object(element):
        return element.process in processes and processes[element.process].object_type is not None

    links = defaultdict(list)
    for record in records:
        if record.publisher is None:
            continue
        publisher = record.publisher
        for claimer in record.claimers:
            for row in publisher.task_rows if has_object(claimer) else ():
                links[(publisher.process, row)].append(((claimer.process, claimer.object_id), publisher.qualifier))
            for row in claimer.task_rows if has_object(publisher) else ():
                links[(claimer.process, row)].append(((publisher.process, publisher.object_id), claimer.qualifier))
    return dict(links)


def _object_links(records: List[MessageRecord], processes: Dict[str, OcelProcess]):
    """(process, object id) -> [((process, object id), qualifier)]: the OCEL links between objects that the
    messages' o2o keys give.
    A publish entry's o2o links the publisher's object to the object of every case that claimed the
    message, a consume entry's links the claimer's object to the publisher's. A link is written once, however
    many messages give it; ends whose process writes no objects link nothing."""
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
        for claimer in record.claimers if record.publisher is not None else ():
            link(record.publisher, claimer, record.publisher.o2o)
            link(claimer, record.publisher, claimer.o2o)
    return dict(links)


def _reference_links(config: SimulationConfig, catalogue: Dict[str, List[CatalogueObject]],
                     processes: Dict[str, OcelProcess], objects, rows, logged_elements, warn):
    """
    The catalogue's objects (catalogue: every organization's, as _grown_catalogues gives them), as (catalogue name, id, [(attribute,
    value, time)]): every object of a listed catalogue, and the objects of a pool or a growing catalogue that some
    case's referenced attribute names; their attributes are timed at the start of the run, a growing catalogue's
    when the first case took its id; and the links its references give, joined with the case objects and log rows
    after the run:
    - object -> [(object, qualifier)]: a reference with o2o links the object of every case of its process, as
      (process, object id), to the catalogue object of each value its attribute took, as its id (reverse: from
      the catalogue object);
    - (process, row of that process) -> [(object id, qualifier)]: each event of those cases named in
      link_events (every event without the key) links the catalogue object of the value the attribute had when
      the event completed, with the attribute's name as qualifier.
    A value that is a list (appended, or picked) stands for each of its entries, and an entry it repeats is linked
    once. A value that is not in its catalogue gets no link, and a warning, once per attribute and value.
    Processes that write no objects link nothing.
    """
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
        """The ids the attribute of case, (process, object id), named, in order and each once: all it ever named, or
        only those of the value it had at until."""
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

    static_objects = [(name, obj.id, [(attribute, value, time_of(name, obj.id)) for attribute, value in obj.attributes])
                      for name, catalogue_objects in catalogue.items() for obj in catalogue_objects
                      if isinstance(config.catalogue[name], list) or (name, obj.id) in referred]
    return static_objects, dict(object_links), dict(event_links)


def _tag_along(config: SimulationConfig, catalogue: Dict[str, List[CatalogueObject]], objects):
    """(process, object id) -> [(qualifier, [(time, [catalogue object id, ...])])]: for each reference with
    tag_along, the catalogue objects its attribute named on the case's object, from when, with the attribute's name
    as qualifier: one for a single value, each entry once for a list. The OCEL output links every event of another
    case that links the object to those named when the event completed (see write_ocel). Ids not in the catalogue
    are left out, and a value with none at all; _reference_links warns about them."""
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
    """The catalogue ids a referenced attribute's value names: a list's entries (an appended or picked list), or the
    value itself."""
    return [entry for entry in value if entry is not None] if isinstance(value, (list, tuple)) else [value]
