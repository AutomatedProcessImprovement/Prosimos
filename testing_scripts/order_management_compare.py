"""
Compares a simulated Order Management run (testing_scripts/assets/order_management, docs/order-management.md)
with the real OCEL 2.0 Order Management log, side by side.

    poetry run python testing_scripts/order_management_compare.py <order-management.json> <simulated_ocel.json>

The real log isn't part of the repository; give its path. The simulated file is what start-orchestration writes
with --ocel_out_path. Both are read the same way, from their objects, events and links; the simulated file names
its object types in the singular (order), the real log in the plural (orders), so they are read under the real
log's names.
"""
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, time, timedelta

TWENTY_DAYS = timedelta(days=20)
HOUR = timedelta(hours=1)
# the staff's calendar in the model, weekday (Monday 0) -> its working periods: working hours are counted in it,
# in both files
_SHORT = [(time(7, 12), time(10, 48)), (time(12, 13, 48), time(16, 22, 12))]
_FULL = [(time(7), time(11)), (time(12), time(16, 36))]
CALENDAR = {0: _SHORT, 1: _FULL, 2: _FULL, 3: _FULL, 4: _SHORT}
# the staff pools of the real log (its employees' roles) and the model's resources in each
STAFF_POOLS = {"Sales": "Sales", "Warehousing": "Warehousing", "Packer": "Warehousing", "Shipment": "Shipment"}
# the activities of each object type that enable one another
ACTIVITIES = {"orders": {"place order", "confirm order", "payment reminder", "pay order"},
              "items": {"place order", "item out of stock", "reorder item", "pick item", "create package"},
              "packages": {"create package", "send package", "failed delivery", "package delivered"}}
GAPS = [("place order", "confirm order"), ("confirm order", "pay order"), ("payment reminder", "pay order"),
        ("place order", "item out of stock"), ("item out of stock", "reorder item"),
        ("place order", "pick item"), ("reorder item", "pick item"), ("create package", "send package"),
        ("send package", "failed delivery"), ("failed delivery", "failed delivery"),
        ("send package", "package delivered"), ("failed delivery", "package delivered")]
# the Shipment's gaps: about half end the day they start and half the next, so their median falls in the night
# between the two and jumps between them from run to run; they are compared by the share that ends the same day
SAME_DAY_GAPS = GAPS[GAPS.index(("create package", "send package")):]


def _shares(counts, buckets):
    """counts: value -> how many; buckets: (label, test) pairs. Returns label -> share of the total."""
    total = sum(counts.values())
    return {label: sum(n for value, n in counts.items() if test(value)) / total for label, test in buckets}


def _summary(values, buckets=()):
    """The metrics of one per-object count: its mean, median, maximum and the share in each bucket."""
    metrics = {"mean": statistics.mean(values), "median": statistics.median(values), "max": max(values)}
    metrics.update(_shares(Counter(values), buckets))
    return metrics


def _values_per_object(ocel):
    """Per object type and attribute, how many values its objects have: min, median and max, an object without the
    attribute counting 0. Section "<object type> <attribute> values per object"."""
    counts = defaultdict(lambda: defaultdict(Counter))  # type -> attribute -> object -> values
    for obj in ocel["objects"]:
        for attribute in obj.get("attributes", []):
            counts[obj["type"]][attribute["name"]][obj["id"]] += 1
    objects = Counter(obj["type"] for obj in ocel["objects"])
    sections = {}
    for kind, attributes in sorted(counts.items()):
        for name, per_object in sorted(attributes.items()):
            values = list(per_object.values()) + [0] * (objects[kind] - len(per_object))
            sections[f"{kind} {name} values per object"] = {"min": min(values), "median": statistics.median(values),
                                                            "max": max(values)}
    return sections


def _time(event):
    return datetime.fromisoformat(event["time"].replace("Z", "+00:00"))


def working_seconds(start, end):
    """The seconds from start to end that fall in CALENDAR; 0 if end isn't after start."""
    total, day = 0.0, start.date()
    while day <= end.date():
        for period in CALENDAR.get(day.weekday(), []):
            opens, closes = (datetime.combine(day, moment, tzinfo=start.tzinfo) for moment in period)
            total += max((min(end, closes) - max(start, opens)).total_seconds(), 0)
        day += timedelta(days=1)
    return total


def _read(ocel_path):
    """The OCEL file, its object types under the real log's names."""
    with open(ocel_path) as file:
        ocel = json.load(file)
    for obj in ocel["objects"]:
        obj["type"] = REAL_LOG_TYPES.get(obj["type"], obj["type"])
    return ocel


def _timelines(ocel):
    """object -> [(time, activity, event id)]: its events of its type's ACTIVITIES, in time order."""
    object_type = {obj["id"]: obj["type"] for obj in ocel["objects"]}
    timelines = defaultdict(list)
    for event in ocel["events"]:
        for relation in event["relationships"]:
            if event["type"] in ACTIVITIES.get(object_type.get(relation["objectId"]), ()):
                timelines[relation["objectId"]].append((_time(event), event["type"], event["id"]))
    for timeline in timelines.values():
        timeline.sort()
    return timelines


def gaps(ocel_path, working=False):
    """(enabling activity, activity) -> the gaps in seconds, of every object whose events of its type's ACTIVITIES
    have the enabling activity right before the activity; with working, only their hours in CALENDAR."""
    found = defaultdict(list)
    for timeline in _timelines(_read(ocel_path)).values():
        for (before, enabling, _), (moment, activity, _) in zip(timeline, timeline[1:]):
            found[(enabling, activity)].append(working_seconds(before, moment) if working
                                               else (moment - before).total_seconds())
    return found


def staff_tasks(ocel):
    """person -> (staff pool, [(completion time, event id)] of the tasks they did, in time order). The real log
    links each task to its people, objects of type employees with a role; the model names its resource in the
    event's resource attribute, e.g. Warehousing_3, its pool STAFF_POOLS of the part before the number."""
    role = {obj["id"]: attribute["value"] for obj in ocel["objects"] if obj["type"] == "employees"
            for attribute in obj["attributes"] if attribute["name"] == "role"}
    pool_of, tasks = {}, defaultdict(list)
    for event in ocel["events"]:
        people = [(relation["objectId"], role[relation["objectId"]]) for relation in event["relationships"]
                  if relation["objectId"] in role]
        people += [(attribute["value"], STAFF_POOLS[str(attribute["value"]).rsplit("_", 1)[0]])
                   for attribute in event.get("attributes", []) if attribute["name"] == "resource"
                   and str(attribute["value"]).rsplit("_", 1)[0] in STAFF_POOLS]
        for person, pool in people:
            pool_of[person] = pool
            tasks[person].append((_time(event), event["id"]))
    return {person: (pool_of[person], sorted(done)) for person, done in tasks.items()}


def _staff(ocel, timelines):
    """Per staff pool, over the weekdays from the first to the last placed order: its people, the tasks they
    complete per working day, and their utilisation, the hours worked over the hours in CALENDAR. Both files have
    completion times only, so a task is taken to start as soon as its person is free (completed the task before)
    and the task is enabled (its object's event before it completed), whichever is later, as the model starts
    tasks; only its hours in CALENDAR count. A person's first task, with neither, starts at the start of its day."""
    enabled = {}
    for timeline in timelines.values():
        for (before, *_), (_, _, event_id) in zip(timeline, timeline[1:]):
            enabled[event_id] = max(enabled.get(event_id, before), before)
    placed = [_time(event) for event in ocel["events"] if event["type"] == "place order"]
    first_day, last_day = min(placed).date(), max(placed).date()
    days = sum((first_day + timedelta(days=n)).weekday() < 5 for n in range((last_day - first_day).days + 1))
    zone = placed[0].tzinfo
    available = working_seconds(datetime.combine(first_day, time(), tzinfo=zone),
                                datetime.combine(last_day + timedelta(days=1), time(), tzinfo=zone))
    people = staff_tasks(ocel)
    sections = {}
    for pool in sorted(set(STAFF_POOLS.values())):
        members = [person for person, (of, _) in people.items() if of == pool]
        done, worked = 0, 0.0
        for person in members:
            previous = None
            for moment, event_id in people[person][1]:
                if first_day <= moment.date() <= last_day:
                    known = [when for when in (previous, enabled.get(event_id)) if when is not None]
                    starts = max(known) if known else datetime.combine(moment.date(), time(), tzinfo=moment.tzinfo)
                    done, worked = done + 1, worked + working_seconds(starts, moment)
                previous = moment
        sections[f"{pool} staff"] = {"people": len(members), "tasks per working day": done / days,
                                     "utilisation": worked / (len(members) * available)}
    return sections


def _same_day(timelines, enabling, activity):
    """The share of the objects' gaps from the enabling activity right before the activity to it that end on the
    day they start (UTC)."""
    days = [(before.date(), moment.date()) for timeline in timelines.values()
            for (before, previous, _), (moment, current, _) in zip(timeline, timeline[1:])
            if (previous, current) == (enabling, activity)]
    return sum(start == end for start, end in days) / len(days)


def _displacement(timelines, order, enabling, activity):
    """How far from first-in-first-out the activity is done: its objects whose event right before it is the
    enabling one, ranked by when that completed (ties in the order of the file's objects) and by when the activity
    completed; the median of how many places each object's two ranks differ."""
    done = [(before, order[obj], moment) for obj, timeline in timelines.items()
            for (before, previous, _), (moment, current, _) in zip(timeline, timeline[1:])
            if (previous, current) == (enabling, activity)]
    by_enabled = sorted(range(len(done)), key=lambda n: done[n][:2])
    by_done = {n: rank for rank, n in enumerate(sorted(range(len(done)), key=lambda n: (done[n][2], done[n][1])))}
    return {"objects": len(done),
            "median (places)": statistics.median(abs(rank - by_done[n]) for rank, n in enumerate(by_enabled))}


def _hours(durations):
    """Mean, median and quartiles of durations, in hours."""
    hours = [duration / HOUR for duration in durations]
    first, median, third = statistics.quantiles(hours, n=4)
    return {"mean (h)": statistics.mean(hours), "first quartile (h)": first, "median (h)": median,
            "third quartile (h)": third}


ITEM_BUCKETS = [(f"{n}", lambda v, n=n: v == n) for n in range(1, 8)] + [("8+", lambda v: v >= 8)]
REMINDER_BUCKETS = [(f"{n}", lambda v, n=n: v == n) for n in range(0, 4)] + [("4+", lambda v: v >= 4)]
FAILED_BUCKETS = [(f"{n}", lambda v, n=n: v == n) for n in range(0, 3)] + [("3+", lambda v: v >= 3)]
EVENTS_WITH_ITEMS = ("place order", "create package", "send package")
REAL_LOG_TYPES = {"order": "orders", "item": "items", "package": "packages", "customer": "customers",
                  "product": "products"}


def metrics(ocel_path):
    """The comparison's metrics of one OCEL file, section -> metric -> value."""
    ocel = _read(ocel_path)
    object_type = {obj["id"]: obj["type"] for obj in ocel["objects"]}
    related = lambda element, kind, qualifier=None: [
        relation["objectId"] for relation in element["relationships"]
        if object_type.get(relation["objectId"]) == kind and qualifier in (None, relation["qualifier"])]
    objects = lambda kind: [obj for obj in ocel["objects"] if obj["type"] == kind]
    order_items = {obj["id"]: related(obj, "items", "comprises") for obj in objects("orders")}
    package_items = {obj["id"]: related(obj, "items", "contains") for obj in objects("packages")}
    order_of = {item: order for order, items in order_items.items() for item in items}
    customer_of = {order: customer["id"] for customer in objects("customers")
                   for order in related(customer, "orders", "places")}

    times = defaultdict(lambda: defaultdict(list))  # object -> activity -> times
    items_per_event = defaultdict(list)  # activity -> items linked to each of its events
    for event in ocel["events"]:
        for kind in ("orders", "items", "packages"):
            for obj in related(event, kind):
                times[obj][event["type"]].append(_time(event))
        if event["type"] in EVENTS_WITH_ITEMS:
            items_per_event[event["type"]].append(len(related(event, "items")))

    first = lambda obj, activity: min(times[obj][activity])
    delivered = {package: first(package, "package delivered") for package in package_items}
    first_delivery = {}
    for package, items in package_items.items():
        for item in items:
            order = order_of[item]
            first_delivery[order] = min(first_delivery.get(order, delivered[package]), delivered[package])
    packaged = {item: first(package, "create package") for package, items in package_items.items() for item in items}
    customer_packages = Counter(customer_of[order_of[items[0]]] for items in package_items.values())
    links = Counter(f"{obj['type']} {relation['qualifier']} {object_type.get(relation['objectId'])}"
                    for obj in ocel["objects"] for relation in obj["relationships"])
    items = list(order_of)
    timelines = _timelines(ocel)
    position = {obj["id"]: place for place, obj in enumerate(ocel["objects"])}
    clock, working = gaps(ocel_path), gaps(ocel_path, working=True)

    return {
        "counts": {"orders": len(order_items), "items": len(items), "packages": len(package_items),
                   "customers": len(objects("customers")), "products": len(objects("products")),
                   "picked items left unpacked": sum(bool(times[item]["pick item"]) and item not in packaged
                                                     for item in items)},
        "items per order": _summary([len(order_items[order]) for order in order_items], ITEM_BUCKETS),
        "reminders per order": _summary([len(times[order]["payment reminder"]) for order in order_items],
                                        REMINDER_BUCKETS),
        "items per package": _summary([len(contents) for contents in package_items.values()]),
        "customers per package": _summary([len({customer_of[order_of[item]] for item in contents})
                                           for contents in package_items.values()],
                                          [("more than 1", lambda v: v > 1)]),
        "packages per customer": _summary([customer_packages[customer["id"]] for customer in objects("customers")]),
        "failed deliveries per package": _summary([len(times[package]["failed delivery"])
                                                   for package in package_items], FAILED_BUCKETS),
        **{f"items per {activity}": _summary(items_per_event[activity]) for activity in EVENTS_WITH_ITEMS},
        "object-to-object links": dict(sorted(links.items())),
        **_values_per_object(ocel),
        **_staff(ocel, timelines),
        "FIFO displacement of pick item (in stock)": _displacement(timelines, position, "place order", "pick item"),
        "FIFO displacement of send package": _displacement(timelines, position, "create package", "send package"),
        **{f"gap {enabling} -> {activity}": {"gaps": len(clock[(enabling, activity)]),
                                             "same day": _same_day(timelines, enabling, activity)}
           if (enabling, activity) in SAME_DAY_GAPS else
           {"gaps": len(clock[(enabling, activity)]),
            "median (h)": statistics.median(clock[(enabling, activity)]) / 3600,
            "median in working hours (h)": statistics.median(working[(enabling, activity)]) / 3600}
           for enabling, activity in GAPS if clock[(enabling, activity)]},
        "wait from pick item to create package": _hours([packaged[item] - first(item, "pick item")
                                                         for item in items if item in packaged]),
        "shares": {
            "orders paid after 20 days": statistics.mean(
                first(order, "pay order") - first(order, "confirm order") > TWENTY_DAYS for order in order_items),
            "items out of stock": statistics.mean(bool(times[item]["item out of stock"]) for item in items),
            "items picked before their order is confirmed": statistics.mean(
                first(item, "pick item") < first(order_of[item], "confirm order") for item in items),
            "orders paid after their first delivery": statistics.mean(
                first(order, "pay order") > first_delivery[order] for order in order_items
                if order in first_delivery),
        },
    }


def _format(value, kind):
    if value is None:
        return "-"
    if kind == "share":
        return f"{value:.1%}"
    if kind == "hours":
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".") if isinstance(value, float) else f"{value:,}"


def _kind(section, key):
    if section in ("counts", "object-to-object links") or key in ("people", "objects", "gaps"):
        return "count"
    if key.endswith("(h)"):
        return "hours"
    return "number" if key in ("mean", "median", "min", "max", "tasks per working day", "median (places)") else "share"


def comparison_rows(real, simulated):
    """(metric, real, simulated) rows; counts, means, medians, minima, maxima and hours as numbers, the rest as
    shares. A section only one of them has, e.g. an attribute only the real log writes, is "-" on the other side."""
    rows = []
    for section in list(real) + [section for section in simulated if section not in real]:
        real_section, simulated_section = real.get(section, {}), simulated.get(section, {})
        keys = list(real_section) + [key for key in simulated_section if key not in real_section]
        for key in keys:
            label = key if section in ("counts", "shares", "object-to-object links") else f"{section}: {key}"
            kind = _kind(section, key)
            rows.append((label, _format(real_section.get(key), kind), _format(simulated_section.get(key), kind)))
    return rows


def main(real_path, simulated_path):
    rows = comparison_rows(metrics(real_path), metrics(simulated_path))
    width = max(len(label) for label, _, _ in rows)
    print(f"{'':{width}}  {'real log':>10}  {'simulated':>10}")
    for label, real, simulated in rows:
        print(f"{label:{width}}  {real:>10}  {simulated:>10}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
