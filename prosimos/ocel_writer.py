"""
Writes a multi-process run as an OCEL 2.0 JSON file (docs/orchestrator.md): one object per case, typed by its
process's object type, and one event per task, linked to its case's object and to the objects on the other end
of its messages, and objects to each other where a message entry asks for it (o2o); plus the catalogue's
objects, which have no events of their own, linked where a reference says a case attribute holds their ids, and,
with tag_along, wherever an event of another case links that case's object.

Engines number their cases each on their own, so two processes of one object type both have an item-0. The writer
names the objects of the output: <object type>-0, <object type>-1, ... per object type across all its processes
(object_ids). A case's object comes in named (process, its id in the engine), a catalogue object by its id.
"""
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple

import pytz

OCEL_TYPES = {bool: "boolean", int: "integer", float: "float", datetime: "time", str: "string"}


@dataclass
class OcelProcess:
    """How one process appears in the OCEL output."""

    object_type: Optional[str]  # of its cases; None leaves the process out
    qualifier: Optional[str] = None  # of each event's link to its own case's object; None means object_type
    qualifier_by_activity: Dict[str, str] = field(default_factory=dict)  # task id -> qualifier, overriding it
    carry_links: bool = False  # later events of a case also link the objects its messages linked
    attributes: Tuple[str, ...] = ()  # the attributes written to its objects; the others are left out


def object_ids(processes, objects):
    """(process, object id in its engine) -> the object's id in the output: <object type>-<number>, numbered per
    object type across all its processes in order of creation. Objects created at the same moment go by the order
    of their processes in processes, and those of one process by its own case number. processes and objects are
    as write_ocel takes them; a process left out has no ids."""
    order = {name: place for place, name in enumerate(processes)}
    of_type = defaultdict(list)
    for place, (process, case_object) in enumerate(objects):
        object_type = _settings(processes, process).object_type
        if object_type is not None:
            of_type[object_type].append((case_object.created, order[process], place, process, case_object.object_id))
    return {(process, object_id): f"{object_type}-{number}" for object_type, found in of_type.items()
            for number, (*_, process, object_id) in enumerate(sorted(found))}


def write_ocel(path, processes, objects, rows, logged_elements=None, message_links=None, object_links=None,
               static_objects=None, reference_links=None, tag_along=None):
    """
    processes: process -> OcelProcess, or its object type (None to leave the process out), in the order of the
    configuration;
    objects: (process, case object with its object_id in its engine, creation time and attributes as (attribute,
    value, time)) for every case; only the attributes its process's OcelProcess lists are written (see _written);
    rows: (process, {column: value}) for every logged task, as in the merged CSV log, each process's in the
    order it logged them;
    logged_elements: process -> the element id of each of its rows, in order (for qualifier_by_activity);
    message_links: (process, row of that process) -> [((process, object id), qualifier)], the links the event of
    that row gets through its messages;
    object_links: object -> [(object, qualifier)], its links to other objects, each a case's object as (process,
    object id) or a catalogue object as its id;
    static_objects: (object type, id, attributes as (attribute, value, time)) for every object with no events
    of its own, all its attributes written;
    reference_links: (process, row of that process) -> [(object id, qualifier)], the links the event of that
    row gets to such objects (never carried by carry_links);
    tag_along: (process, object id) -> [(qualifier, [(time, object id)])], the objects with no events of their own
    that a case's object names, each from its time on: every event of another case that links the case's object,
    through its messages or carry_links, also links the one named when the event completed, with that qualifier.
    """
    processes = {name: _settings(processes, name) for name in processes}
    ids = object_ids(processes, objects)

    def named(end):
        return ids[end] if isinstance(end, tuple) else end

    logged_elements, message_links = logged_elements or {}, message_links or {}
    reference_links = reference_links or {}
    tagged = {ids[end]: timelines for end, timelines in (tag_along or {}).items() if end in ids}
    named_links = defaultdict(list)
    for source, targets in (object_links or {}).items():
        named_links[named(source)].extend((named(target), qualifier) for target, qualifier in targets)
    case_objects = [(processes[process].object_type, ids[(process, case_object.object_id)],
                     _written(case_object, processes[process].attributes))
                    for process, case_object in objects
                    if process in processes and processes[process].object_type is not None]
    ocel_objects, attribute_values = [], {}
    for object_type, object_id, object_attributes in case_objects + list(static_objects or []):
        attributes = [{"name": name, "value": _json_value(value), "time": _time(time)}
                      for name, value, time in object_attributes]
        for attribute in attributes:
            attribute_values.setdefault(object_type, {}).setdefault(attribute["name"], []).append(attribute["value"])
        attribute_values.setdefault(object_type, {})
        ocel_objects.append({"id": object_id, "type": object_type, "attributes": attributes,
                             "relationships": [{"objectId": target, "qualifier": qualifier}
                                               for target, qualifier in named_links.get(object_id, [])]})

    # every row becomes an event, linked to its own case's object and to the objects its messages linked
    logged, rows_so_far = [], {}
    for process, row in rows:
        number = rows_so_far[process] = rows_so_far.get(process, -1) + 1  # the row's place in its process's log
        settings = processes.get(process)
        if settings is None or settings.object_type is None:
            continue
        element_id = logged_elements[process][number] if process in logged_elements else None
        own_qualifier = settings.qualifier_by_activity.get(element_id, settings.qualifier or settings.object_type)
        logged.append({"time": _as_datetime(row["end_time"]), "process": process, "number": number, "row": row,
                       "own": (ids[(process, f"{settings.object_type}-{row['case_id']}")], own_qualifier),
                       "links": [(ids[end], qualifier) for end, qualifier in message_links.get((process, number), [])],
                       "carried": [],
                       "references": list(reference_links.get((process, number), []))})

    # with carry_links, every later event of a case also links what its earlier events linked through messages
    linked_so_far = {}
    for event in sorted(logged, key=lambda event: (event["time"], event["number"])):
        if processes[event["process"]].carry_links:
            case_links = linked_so_far.setdefault((event["process"], event["row"]["case_id"]), [])
            event["carried"] = list(case_links)
            case_links.extend(event["links"])

    # the rows come in the order events were executed; events are listed in time order, a task's
    # event at the time it was completed
    events = []
    for number, event in enumerate(sorted(logged, key=lambda event: (event["time"], event["process"])), start=1):
        # with tag_along, an event linking another case's object also links what that object named then
        tags = []
        for object_id, _ in [*event["links"], *event["carried"]]:
            if object_id == event["own"][0]:
                continue
            for qualifier, timeline in tagged.get(object_id, []):
                named = [value for time, value in timeline if time <= event["time"]]
                if named:
                    tags.append((named[-1], qualifier))
        relationships = []
        for object_id, qualifier in [event["own"], *event["links"], *event["carried"], *event["references"], *tags]:
            relationship = {"objectId": object_id, "qualifier": qualifier}
            if relationship not in relationships:
                relationships.append(relationship)
        events.append({
            "id": f"e{number}",
            "type": event["row"]["activity"],
            "time": _time(event["time"]),
            "attributes": [{"name": "resource", "value": event["row"]["resource"]}],
            "relationships": relationships,
        })

    ocel = {
        "objectTypes": [{"name": object_type, "attributes": [{"name": name, "type": _ocel_type(values)}
                                                             for name, values in sorted(names.items())]}
                        for object_type, names in sorted(attribute_values.items())],
        "eventTypes": [{"name": activity, "attributes": [{"name": "resource", "type": "string"}]}
                       for activity in sorted({event["type"] for event in events})],
        "objects": ocel_objects,
        "events": events,
    }
    with open(path, "w", encoding="utf-8") as ocel_file:
        json.dump(ocel, ocel_file, indent=1, ensure_ascii=False)


def _written(case_object, listed):
    """The values of a case's object that the output writes: those of the listed attributes, without the starting 0
    of a sum or count no message had written yet, and of several values of one attribute at the same moment only
    the last (aggregate writes its running total after every message, and a case may take many at once)."""
    only_started = set(case_object.only_started)
    values = []
    for name, value, time in case_object.attributes:
        if name in only_started:  # its first value is the one it was created with
            only_started.discard(name)
        elif name in listed:
            values.append((name, value, time))
    last = {(name, time): place for place, (name, _, time) in enumerate(values)}
    return [value for place, value in enumerate(values) if last[(value[0], value[2])] == place]


def _settings(processes, process):
    settings = processes.get(process)
    return settings if isinstance(settings, OcelProcess) else OcelProcess(settings)


def _json_value(value):
    if hasattr(value, "item"):  # a NumPy number, e.g. a drawn continuous case attribute
        value = value.item()
    if isinstance(value, list):  # an appended attribute: OCEL attribute values are single values, so JSON text
        return json.dumps([_json_value(element) for element in value])
    return _time(value) if isinstance(value, datetime) else value


def _ocel_type(values):
    """The OCEL attribute type of a column of values: integers mixed with floats are floats; any other mix,
    or a type OCEL doesn't know, is a string. Times are already strings here, so they are strings too."""
    types = {OCEL_TYPES.get(type(value), "string") for value in values if value is not None}
    if types == {"integer", "float"}:
        return "float"
    return types.pop() if len(types) == 1 else "string"


def _time(moment):
    """ISO 8601 in UTC, as OCEL files write it, e.g. 2023-04-03T10:08:18.000000Z."""
    return moment.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _as_datetime(value):
    # the engine hands over timestamps as datetimes, or as preformatted strings when they have no
    # fractional seconds
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)
