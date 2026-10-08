"""
Writes a multi-process run as an OCEL 2.0 JSON file (docs/orchestrator.md): one object per case, typed by its
process's object type, and one event per task, linked to its case's object.
"""
import json
from datetime import datetime

import pytz

OCEL_TYPES = {bool: "boolean", int: "integer", float: "float", datetime: "time", str: "string"}


def write_ocel(path, object_types, objects, rows):
    """
    object_types: process -> the object type of its cases, or None to leave the process out;
    objects: (process, case object with case_id and attributes as (attribute, value, time)) for every case;
    rows: (process, {column: value}) for every logged task, as in the merged CSV log.
    """
    ocel_objects, attribute_values = [], {}
    for process, case_object in objects:
        object_type = object_types.get(process)
        if object_type is None:
            continue
        attributes = [{"name": name, "value": _json_value(value), "time": _time(time)}
                      for name, value, time in case_object.attributes]
        for attribute in attributes:
            attribute_values.setdefault(object_type, {}).setdefault(attribute["name"], []).append(attribute["value"])
        attribute_values.setdefault(object_type, {})
        ocel_objects.append({"id": case_object.case_id, "type": object_type, "attributes": attributes,
                             "relationships": []})

    # the rows come in the order events were executed; events are listed in time order, a task's
    # event at the time it was completed
    logged = sorted(((process, row) for process, row in rows if object_types.get(process) is not None),
                    key=lambda logged_row: (_as_datetime(logged_row[1]["end_time"]), logged_row[0]))
    events = []
    for number, (process, row) in enumerate(logged, start=1):
        events.append({
            "id": f"e{number}",
            "type": row["activity"],
            "time": _time(_as_datetime(row["end_time"])),
            "attributes": [{"name": "resource", "value": row["resource"]}],
            "relationships": [{"objectId": f"{process}-{row['case_id']}", "qualifier": object_types[process]}],
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
