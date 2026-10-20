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
from datetime import datetime, timedelta

TWENTY_DAYS = timedelta(days=20)
HOUR = timedelta(hours=1)


def _shares(counts, buckets):
    """counts: value -> how many; buckets: (label, test) pairs. Returns label -> share of the total."""
    total = sum(counts.values())
    return {label: sum(n for value, n in counts.items() if test(value)) / total for label, test in buckets}


def _summary(values, buckets=()):
    """The metrics of one per-object count: its mean, median, maximum and the share in each bucket."""
    metrics = {"mean": statistics.mean(values), "median": statistics.median(values), "max": max(values)}
    metrics.update(_shares(Counter(values), buckets))
    return metrics


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
    with open(ocel_path) as file:
        ocel = json.load(file)
    for obj in ocel["objects"]:
        obj["type"] = REAL_LOG_TYPES.get(obj["type"], obj["type"])
    object_type = {obj["id"]: obj["type"] for obj in ocel["objects"]}
    related = lambda element, kind, qualifier=None: [
        relation["objectId"] for relation in element["relationships"]
        if object_type.get(relation["objectId"]) == kind and qualifier in (None, relation["qualifier"])]
    objects = lambda kind: [obj for obj in ocel["objects"] if obj["type"] == kind]
    order_items = {obj["id"]: related(obj, "items", "comprises") for obj in objects("orders")}
    # a packing round that never took an item is a package object without items or events
    all_packages = {obj["id"]: related(obj, "items", "contains") for obj in objects("packages")}
    package_items = {package: items for package, items in all_packages.items() if items}
    order_of = {item: order for order, items in order_items.items() for item in items}
    customer_of = {order: customer["id"] for customer in objects("customers")
                   for order in related(customer, "orders", "places")}

    times = defaultdict(lambda: defaultdict(list))  # object -> activity -> times
    items_per_event = defaultdict(list)  # activity -> items linked to each of its events
    for event in ocel["events"]:
        time = datetime.fromisoformat(event["time"].replace("Z", "+00:00"))
        for kind in ("orders", "items", "packages"):
            for obj in related(event, kind):
                times[obj][event["type"]].append(time)
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

    return {
        "counts": {"orders": len(order_items), "items": len(items), "packages": len(package_items),
                   "packing rounds that took no item": len(all_packages) - len(package_items),
                   "customers": len(objects("customers")), "products": len(objects("products"))},
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
    if section in ("counts", "object-to-object links"):
        return "count"
    if key.endswith("(h)"):
        return "hours"
    return "number" if key in ("mean", "median", "max") else "share"


def comparison_rows(real, simulated):
    """(metric, real, simulated) rows; counts, means, medians, maxima and hours as numbers, the rest as shares."""
    rows = []
    for section in real:
        keys = list(real[section]) + [key for key in simulated[section] if key not in real[section]]
        for key in keys:
            label = key if section in ("counts", "shares", "object-to-object links") else f"{section}: {key}"
            kind = _kind(section, key)
            rows.append((label, _format(real[section].get(key), kind), _format(simulated[section].get(key), kind)))
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
