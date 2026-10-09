"""
Writes a multi-process run as an OCEL 2.0 JSON file (docs/orchestrator.md): one object per case, typed by its
process's object type, and one event per task, linked to its case's object and to the objects on the other end
of its messages.
"""
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

import pytz

OCEL_TYPES = {bool: "boolean", int: "integer", float: "float", datetime: "time", str: "string"}


@dataclass
class OcelProcess:
    """How one process appears in the OCEL output."""

    object_type: Optional[str]  # of its cases; None leaves the process out
    qualifier: Optional[str] = None  # of each event's link to its own case's object; None means object_type
    qualifier_by_activity: Dict[str, str] = field(default_factory=dict)  # task id -> qualifier, overriding it
    carry_links: bool = False  # later events of a case also link the objects its messages linked


def write_ocel(path, processes, objects, rows, logged_elements=None, message_links=None):
    """
    processes: process -> OcelProcess, or its object type (None to leave the process out);
    objects: (process, case object with case_id and attributes as (attribute, value, time)) for every case;
    rows: (process, {column: value}) for every logged task, as in the merged CSV log, each process's in the
    order it logged them;
    logged_elements: process -> the element id of each of its rows, in order (for qualifier_by_activity);
    message_links: (process, row of that process) -> [(object id, qualifier)], the links the event of that
    row gets through its messages.
    """
    processes = {name: process if isinstance(process, OcelProcess) else OcelProcess(process)
                 for name, process in processes.items()}
    logged_elements, message_links = logged_elements or {}, message_links or {}
    ocel_objects, attribute_values = [], {}
    for process, case_object in objects:
        object_type = processes[process].object_type if process in processes else None
        if object_type is None:
            continue
        attributes = [{"name": name, "value": _json_value(value), "time": _time(time)}
                      for name, value, time in case_object.attributes]
        for attribute in attributes:
            attribute_values.setdefault(object_type, {}).setdefault(attribute["name"], []).append(attribute["value"])
        attribute_values.setdefault(object_type, {})
        ocel_objects.append({"id": case_object.case_id, "type": object_type, "attributes": attributes,
                             "relationships": []})

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
                       "own": (f"{process}-{row['case_id']}", own_qualifier),
                       "links": list(message_links.get((process, number), [])), "carried": []})

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
        relationships = []
        for object_id, qualifier in [event["own"], *event["links"], *event["carried"]]:
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


def _json_value(value):
    if hasattr(value, "item"):  # a NumPy number, e.g. a drawn continuous case attribute
        value = value.item()
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
