"""
Compares a simulated Order Management run (testing_scripts/assets/order_management, docs/order-management.md)
with the real OCEL 2.0 Order Management log, side by side.

    poetry run python testing_scripts/order_management_compare.py <order-management.json> <merged_log.csv>

The OCEL file isn't part of the repository; give its path. The merged log is what start-orchestration writes.
"""
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import pandas as pd

TWENTY_DAYS = timedelta(days=20)


def _shares(counts, buckets):
    """counts: value -> how many; buckets: (label, test) pairs. Returns label -> share of the total."""
    total = sum(counts.values())
    return {label: sum(n for value, n in counts.items() if test(value)) / total for label, test in buckets}


def _summary(values, buckets):
    """The metrics of one per-object count: its mean, median, maximum and the share in each bucket."""
    counts = Counter(values)
    metrics = {"mean": statistics.mean(values), "median": statistics.median(values), "max": max(values)}
    metrics.update(_shares(counts, buckets))
    return metrics


ITEM_BUCKETS = [(f"{n}", lambda v, n=n: v == n) for n in range(1, 8)] + [("8+", lambda v: v >= 8)]
REMINDER_BUCKETS = [(f"{n}", lambda v, n=n: v == n) for n in range(0, 4)] + [("4+", lambda v: v >= 4)]
FAILED_BUCKETS = [(f"{n}", lambda v, n=n: v == n) for n in range(0, 3)] + [("3+", lambda v: v >= 3)]


def _metrics(items_per_order, reminders_per_order, paid_after_20_days, out_of_stock, items_per_package,
             customers_per_package, failed_per_package, picked_before_confirmed, paid_after_first_delivery, counts):
    return {
        "counts": counts,
        "items per order": _summary(items_per_order, ITEM_BUCKETS),
        "reminders per order": _summary(reminders_per_order, REMINDER_BUCKETS),
        "items per package": _summary(items_per_package, []),
        "customers per package": _summary(customers_per_package, [("more than 1", lambda v: v > 1)]),
        "failed deliveries per package": _summary(failed_per_package, FAILED_BUCKETS),
        "shares": {
            "orders paid after 20 days": statistics.mean(paid_after_20_days),
            "items out of stock": statistics.mean(out_of_stock),
            "items picked before their order is confirmed": statistics.mean(picked_before_confirmed),
            "orders paid after their first delivery": statistics.mean(paid_after_first_delivery),
        },
    }


def real_metrics(ocel_path):
    with open(ocel_path) as file:
        ocel = json.load(file)
    object_type = {obj["id"]: obj["type"] for obj in ocel["objects"]}
    related = lambda element, kind, qualifier=None: [
        relation["objectId"] for relation in element["relationships"]
        if object_type.get(relation["objectId"]) == kind and qualifier in (None, relation["qualifier"])]
    order_items = {obj["id"]: related(obj, "items", "comprises") for obj in ocel["objects"] if obj["type"] == "orders"}
    package_items = {obj["id"]: related(obj, "items", "contains") for obj in ocel["objects"] if obj["type"] == "packages"}
    order_of = {item: order for order, items in order_items.items() for item in items}

    times = defaultdict(lambda: defaultdict(list))  # object -> activity -> times
    customer_of = {}
    for event in ocel["events"]:
        time = datetime.fromisoformat(event["time"].replace("Z", "+00:00"))
        for kind in ("orders", "items", "packages"):
            for obj in related(event, kind):
                times[obj][event["type"]].append(time)
        if event["type"] == "place order":
            customer_of[related(event, "orders")[0]] = related(event, "customers")[0]

    first = lambda obj, activity: min(times[obj][activity])
    delivered = {package: first(package, "package delivered") for package in package_items}
    first_delivery = defaultdict(lambda: None)
    for package, items in package_items.items():
        for item in items:
            order = order_of[item]
            if first_delivery[order] is None or delivered[package] < first_delivery[order]:
                first_delivery[order] = delivered[package]
    items = list(order_of)
    return _metrics(
        items_per_order=[len(order_items[order]) for order in order_items],
        reminders_per_order=[len(times[order]["payment reminder"]) for order in order_items],
        paid_after_20_days=[first(o, "pay order") - first(o, "confirm order") > TWENTY_DAYS for o in order_items],
        out_of_stock=[bool(times[item]["item out of stock"]) for item in items],
        items_per_package=[len(contents) for contents in package_items.values()],
        customers_per_package=[len({customer_of[order_of[item]] for item in contents})
                               for contents in package_items.values()],
        failed_per_package=[len(times[package]["failed delivery"]) for package in package_items],
        picked_before_confirmed=[first(item, "pick item") < first(order_of[item], "confirm order") for item in items],
        paid_after_first_delivery=[first(order, "pay order") > first_delivery[order] for order in order_items
                                   if first_delivery[order] is not None],
        counts={"orders": len(order_items), "items": len(items), "packages": len(package_items)},
    )


def package_contents(log):
    """(package case -> the items (order id, item index) it collected, in the order it got them; customer -> the
    items still collecting in that customer's open package at the end).

    The merged log doesn't link a package to its items, so they are reconstructed: a customer has at most one
    open package at a time, packages get consecutive case ids as they start, and each takes its first item and
    then more_items more of that customer, in the order the items were picked (an item's ItemPicked is published
    when its pick ends). A package still collecting has no rows in the log yet; its items are the customer's
    last ones."""
    picks = log[(log["process"] == "Warehouse") & (log["activity"] == "pick item")]
    packages = log[log["process"] == "Packaging"].groupby("case_id").first()
    items_by_customer = defaultdict(list)
    for _, pick in picks.sort_values(["end_time", "case_id"], kind="stable").iterrows():
        items_by_customer[pick["customer"]].append((pick["order_id"], int(pick["item_index"])))
    contents, still_collecting = {}, {}
    for customer, items in items_by_customer.items():
        position = 0
        for package, row in packages[packages["customer"] == customer].sort_index().iterrows():
            size = 1 + int(row["more_items"])
            contents[package] = items[position:position + size]
            position += size
        if position < len(items):
            still_collecting[customer] = items[position:]
    return contents, still_collecting


def simulated_metrics(log_path):
    log = pd.read_csv(log_path, parse_dates=["enable_time", "start_time", "end_time"])
    by = lambda process, activity: log[(log["process"] == process) & (log["activity"] == activity)]
    first_end = lambda process, activity: by(process, activity).groupby("case_id")["end_time"].min()

    orders = sorted(log.loc[log["process"] == "Sales", "case_id"].unique())
    order_name = {order: f"Sales-{order}" for order in orders}
    confirmed = {order_name[o]: t for o, t in first_end("Sales", "confirm order").items()}
    paid = {order_name[o]: t for o, t in first_end("Sales", "pay order").items()}
    reminders = by("Sales", "payment reminder").groupby("case_id").size()

    items = log[log["process"] == "Warehouse"].groupby("case_id").first()
    out_of_stock = set(by("Warehouse", "item out of stock")["case_id"])
    picked = first_end("Warehouse", "pick item")
    item_order = items["order_id"]

    contents, still_collecting = package_contents(log)
    created = set(contents)
    delivered = first_end("Packaging", "package delivered")
    failed = by("Packaging", "failed delivery").groupby("case_id").size()
    first_delivery = {}
    for package, package_items in contents.items():
        if package in delivered:
            for order, _ in package_items:
                first_delivery[order] = min(first_delivery.get(order, delivered[package]), delivered[package])
    customer_of = log[log["process"] == "Sales"].groupby("case_id")["customer"].first()
    customer_of = {order_name[o]: c for o, c in customer_of.items()}

    return _metrics(
        items_per_order=list(item_order.value_counts().reindex(order_name.values(), fill_value=0)),
        reminders_per_order=[int(reminders.get(order, 0)) for order in orders],
        paid_after_20_days=[paid[o] - confirmed[o] > TWENTY_DAYS for o in order_name.values() if o in paid],
        out_of_stock=[item in out_of_stock for item in items.index],
        items_per_package=[len(contents[package]) for package in sorted(created)],
        customers_per_package=[len({customer_of[order] for order, _ in contents[package]}) for package in sorted(created)],
        failed_per_package=[int(failed.get(package, 0)) for package in sorted(created)],
        picked_before_confirmed=[picked[item] < confirmed[item_order[item]] for item in items.index if item in picked],
        paid_after_first_delivery=[paid[o] > first_delivery[o] for o in order_name.values()
                                   if o in paid and o in first_delivery],
        counts={"orders": len(orders), "items": len(items), "packages": len(created),
                "packages still collecting": len(still_collecting)},
    )


def _format(value, is_share):
    if value is None:
        return "-"
    if is_share:
        return f"{value:.1%}"
    return f"{value:.2f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)


def comparison_rows(real, simulated):
    """(metric, real, simulated) rows; counts, means, medians and maxima as numbers, the rest as shares."""
    rows = []
    for section in real:
        keys = list(real[section]) + [key for key in simulated[section] if key not in real[section]]
        for key in keys:
            label = key if section in ("counts", "shares") else f"{section}: {key}"
            is_share = section != "counts" and key not in ("mean", "median", "max")
            rows.append((label, _format(real[section].get(key), is_share),
                         _format(simulated[section].get(key), is_share)))
    return rows


def main(ocel_path, log_path):
    rows = comparison_rows(real_metrics(ocel_path), simulated_metrics(log_path))
    width = max(len(label) for label, _, _ in rows)
    print(f"{'':{width}}  {'real log':>10}  {'simulated':>10}")
    for label, real, simulated in rows:
        print(f"{label:{width}}  {real:>10}  {simulated:>10}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
