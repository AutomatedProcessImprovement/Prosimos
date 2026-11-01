"""
Writes a multi-process run as an OCEL 2.0 JSON file (docs/orchestrator.md, "Output"). A case's object is written
only when an event refers to it, as pm4py would leave it out on reading; catalogue objects always are. Engines
number their cases on their own, so the writer names the objects <object type>-<number> per object type across its
processes (object_ids).
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
    attributes: Tuple[str, ...] = ()  # the attributes written to its objects; the others are left out


def object_ids(processes, objects):
    """(process, object id in its engine) -> its id in the output, numbered per object type in order of creation;
    at the same moment by the order of processes, then by case number."""
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
    """Writes the OCEL file. processes: process -> OcelProcess, or its object type, in the configuration's order;
    objects, rows and logged_elements as the run report and the merged log hold them; the links and the catalogue
    objects (static_objects) as output.py builds them. With tag_along, an event linking a case's object also links
    the catalogue objects that object named when the event completed."""
    processes = {name: _settings(processes, name) for name in processes}
    logged_elements, message_links = logged_elements or {}, message_links or {}
    referred = _referred(processes, rows, message_links)
    objects = [(process, case_object) for process, case_object in objects
               if (process, case_object.object_id) in referred]
    ids = object_ids(processes, objects)

    def named(end):
        return ids[end] if isinstance(end, tuple) else end

    reference_links = reference_links or {}
    tagged = {ids[end]: timelines for end, timelines in (tag_along or {}).items() if end in ids}
    named_links = defaultdict(list)
    for source, targets in (object_links or {}).items():
        if source in ids or not isinstance(source, tuple):  # none from or to an object left out
            named_links[named(source)].extend((named(target), qualifier) for target, qualifier in targets
                                              if target in ids or not isinstance(target, tuple))
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
        links = message_links.get((process, number), [])
        logged.append({"time": _as_datetime(row["end_time"]), "process": process, "number": number, "row": row,
                       "own": (ids[(process, f"{settings.object_type}-{row['case_id']}")], own_qualifier),
                       "links": [(ids[end], qualifier) for end, qualifier, _ in links],
                       "carries": [(ids[end], qualifier, carry) for end, qualifier, carry in links if carry],
                       "activities": {row["activity"], element_id},
                       "carried": [],
                       "references": list(reference_links.get((process, number), []))})

    # a link whose message entry carries it passes on to the later events of the case: every one, or those it names
    carried_so_far = {}
    for event in sorted(logged, key=lambda event: (event["time"], event["number"])):
        case_links = carried_so_far.setdefault((event["process"], event["row"]["case_id"]), [])
        event["carried"] = [(object_id, qualifier) for object_id, qualifier, carry in case_links
                            if carry is True or event["activities"] & set(carry)]
        case_links.extend(event["carries"])

    # events are listed in time order, a task's at its completion; the rows came in execution order
    events = []
    for number, event in enumerate(sorted(logged, key=lambda event: (event["time"], event["process"])), start=1):
        # with tag_along, an event linking another case's object also links what that object named then
        tags = []
        for object_id, _ in [*event["links"], *event["carried"]]:
            if object_id == event["own"][0]:
                continue
            for qualifier, timeline in tagged.get(object_id, []):
                named = [values for time, values in timeline if time <= event["time"]]
                if named:
                    tags.extend((value, qualifier) for value in named[-1])
        relationships = []
        for object_id, qualifier in [event["own"], *event["links"], *event["carried"], *event["references"], *tags]:
            relationship = {"objectId": object_id, "qualifier": qualifier}
            if relationship not in relationships:
                relationships.append(relationship)
        events.append({
            "id": f"e{number}",
            "type": event["row"]["activity"],
            "time": _time(event["time"]),
            # our extension of OCEL, whose events are atomic: when the task started and when it was enabled
            "attributes": [{"name": "resource", "value": event["row"]["resource"]},
                           {"name": "start_time", "value": _time(_as_datetime(event["row"]["start_time"]))},
                           {"name": "enable_time", "value": _time(_as_datetime(event["row"]["enable_time"]))}],
            "relationships": relationships,
        })

    ocel = {
        "objectTypes": [{"name": object_type, "attributes": [{"name": name, "type": _ocel_type(values)}
                                                             for name, values in sorted(names.items())]}
                        for object_type, names in sorted(attribute_values.items())],
        "eventTypes": [{"name": activity, "attributes": [{"name": "resource", "type": "string"},
                                                         {"name": "start_time", "type": "time"},
                                                         {"name": "enable_time", "type": "time"}]}
                       for activity in sorted({event["type"] for event in events})],
        "objects": ocel_objects,
        "events": events,
    }
    with open(path, "w", encoding="utf-8") as ocel_file:
        json.dump(ocel, ocel_file, indent=1, ensure_ascii=False)


def _referred(processes, rows, message_links):
    """(process, object id) of every case's object some event refers to: its own, or one its messages link."""
    referred, rows_so_far = set(), {}
    for process, row in rows:
        number = rows_so_far[process] = rows_so_far.get(process, -1) + 1
        settings = processes.get(process)
        if settings is not None and settings.object_type is not None:
            referred.add((process, f"{settings.object_type}-{row['case_id']}"))
            referred.update(end for end, *_ in message_links.get((process, number), []))
    return referred


def _written(case_object, listed):
    """The values of a case's object the output writes: the listed attributes, without None or a start no message
    wrote, and of one attribute's values at one moment only the last (aggregate writes a total per message)."""
    only_started = set(case_object.only_started)
    values = []
    for name, value, time in case_object.attributes:
        if name in only_started:  # its first value is the one it was created with
            only_started.discard(name)
        elif name in listed and value is not None:
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
    """The OCEL type of a column of values: integers mixed with floats are floats; any other mix is a string."""
    types = {OCEL_TYPES.get(type(value), "string") for value in values if value is not None}
    if types == {"integer", "float"}:
        return "float"
    return types.pop() if len(types) == 1 else "string"


def _time(moment):
    """ISO 8601 in UTC, as OCEL files write it, e.g. 2023-04-03T10:08:18.000000Z."""
    return moment.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _as_datetime(value):
    # timestamps come as datetimes, or as strings when they have no fractional seconds
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)
