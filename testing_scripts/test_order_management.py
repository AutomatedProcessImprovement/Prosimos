"""
The Order Management example (docs/order-management.md): orders, items and packages, three processes run
together on 100 orders with a fixed seed, by seven people on one calendar.
"""
import dataclasses
import json
import statistics
from collections import defaultdict
from datetime import datetime, time, timedelta
from pathlib import Path

import pandas as pd
import pm4py
import pytest
from click.testing import CliRunner

from cli.diff_res_bpsim import cli
from prosimos.orchestrator import SimulationConfig, run_orchestrator
CONFIG = "testing_scripts/assets/order_management/simulation.json"  # 2,000 orders, 1,450 packing rounds; seed 1
ORDERS = 100
# enough rounds to pack every item: restocked items are picked up to a week after the last order
ROUNDS = 100
# a reminder is due 20 days on the clock after the confirmation or the previous reminder
REMINDER_GAP = timedelta(days=20)
# the staff, each pool on the one staff calendar, Monday to Friday 9:00-17:00; the instant tasks are recorded by
# System, around the clock
STAFF = {"Sales clerk": 1, "Warehouse worker": 2, "Packer": 1, "Courier": 3}
OPENS, CLOSES = time(9), time(17)
# each task's work time, in minutes
WORK = {"place order": (0, 0), "confirm order": (5, 15), "payment reminder": (2, 5), "pay order": (2, 5),
        "item out of stock": (1, 3), "reorder item": (5, 10), "pick item": (3, 10), "create package": (10, 20),
        "send package": (60, 240), "failed delivery": (0, 0), "package delivered": (0, 0)}


def _config(orders=ORDERS, rounds=ROUNDS):
    config = SimulationConfig.from_json(CONFIG)
    cases = {"Sales": orders, "Packaging": rounds}
    return dataclasses.replace(config, processes=[
        dataclasses.replace(spec, arrivals=cases.get(spec.name, spec.arrivals)) for spec in config.processes])


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """One 100-order run: the report, the merged log, and the path of the OCEL file."""
    folder = tmp_path_factory.mktemp("order_management")
    report = run_orchestrator(_config(), folder / "log.csv", ocel_out_path=folder / "log.json")
    log = pd.read_csv(folder / "log.csv", parse_dates=["enable_time", "start_time", "end_time"])
    return report, log, folder / "log.json"


def _packages(report):
    """Package case id -> the ItemPicked messages it claimed, in the order it claimed them, with when."""
    picked = {message.id: message for message in report.published if message.type == "ItemPicked"}
    claimer = {record.message_id: record.claimers[0].object_id for record in report.message_records if record.claimers}
    packages = defaultdict(list)
    for message_id, process, time in report.claims:
        if process == "Packaging":
            packages[claimer[message_id]].append((picked[message_id], time))
    return packages


def package_items(report):
    """Package case id -> the case ids of the items it took, in the order it claimed them."""
    return {package: [message.attributes["object_id"] for message, _ in claimed]
            for package, claimed in _packages(report).items()}


def _rows(log, process, activity=None):
    rows = log[log["process"] == process]
    return rows if activity is None else rows[rows["activity"] == activity]


def _work_seconds(start, end):
    """The seconds from start to end that fall in the staff's hours."""
    total, day = 0.0, start.date()
    while day <= end.date():
        if day.weekday() < 5:
            opens, closes = (datetime.combine(day, moment, tzinfo=start.tzinfo) for moment in (OPENS, CLOSES))
            total += max((min(end, closes) - max(start, opens)).total_seconds(), 0)
        day += timedelta(days=1)
    return total


def _in_staff_hours(moment):
    return moment.weekday() < 5 and OPENS <= moment.time() <= CLOSES


def _orders(log):
    """Sales case -> (order id, items, customer)."""
    first = _rows(log, "Sales").groupby("case_id").first()
    return {case: (f"order-{case}", int(row["items"]), row["customer"]) for case, row in first.iterrows()}


def test_every_order_publishes_one_item_ordered_per_item(run):
    report, log, _ = run

    published = defaultdict(list)
    for message in report.published:
        if message.type == "ItemOrdered":
            published[message.attributes["object_id"]].append(
                (message.attributes["index"], message.attributes["customer"]))
    orders = _orders(log)
    assert len(orders) == ORDERS
    assert all(published[order] == [(index, customer) for index in range(1, items + 1)]
               for order, items, customer in orders.values())


def test_every_item_case_has_its_order_id_index_and_customer(run):
    _, log, _ = run

    items = _rows(log, "Warehouse").groupby("case_id").first()
    customers = {order: customer for order, _, customer in _orders(log).values()}
    expected = sorted((order, index) for order, count, _ in _orders(log).values() for index in range(1, count + 1))
    assert sorted(zip(items["order_id"], items["item_index"].astype(int))) == expected
    assert all(customer == customers[order] for order, customer in zip(items["order_id"], items["customer"]))


def test_every_package_holds_all_of_one_customer_s_waiting_items_oldest_first(run):
    report, _, _ = run

    position = {message.id: index for index, message in enumerate(report.published)}  # in the order published
    picked = [message for message in report.published if message.type == "ItemPicked"]
    packages = _packages(report)
    taken = set()
    # the packages in the order they claimed their first item
    for package in sorted(packages, key=lambda package: report.claims.index(
            next(claim for claim in report.claims if claim[0] == packages[package][0][0].id))):
        claimed = [message for message, _ in packages[package]]
        [time] = {time for _, time in packages[package]}  # a round takes its items at once
        waiting = [message for message in picked if message.time <= time and message.id not in taken]
        oldest = waiting[0]
        same_customer = [message for message in waiting if message.attributes["customer"] == oldest.attributes["customer"]]
        assert [message.id for message in claimed] == sorted((message.id for message in same_customer), key=position.get)
        taken.update(message.id for message in claimed)
    assert taken == {message.id for message in picked}  # every item is in a package, and in one only


def test_an_item_weighs_what_its_product_weighs_and_a_package_the_sum_of_its_items(run):
    report, _, ocel_path = run

    with open(ocel_path) as file:
        objects = {obj["id"]: obj for obj in json.load(file)["objects"]}
    attributes = lambda obj: {attribute["name"]: attribute["value"] for attribute in obj["attributes"]}
    items = [obj for obj in objects.values() if obj["type"] == "item"]
    for item in items:
        [product] = [relation["objectId"] for relation in item["relationships"] if relation["qualifier"] == "is a"]
        assert {key: attributes(item)[key] for key in ("weight", "price")} == attributes(objects[product])
    weight = {item["id"]: attributes(item)["weight"] for item in items}
    picked = [message.attributes for message in report.published if message.type == "ItemPicked"]
    assert all(message["weight"] == weight[message["object_id"]] for message in picked)  # what ItemPicked carries
    for package, contents in package_items(report).items():
        assert attributes(objects[package])["weight"] == pytest.approx(sum(weight[item] for item in contents))


def test_product_weights_are_written_only_in_the_catalogue():
    folder = Path(CONFIG).parent
    [warehouse] = [process for process in json.loads(Path(CONFIG).read_text())["processes"] if process["name"] == "Warehouse"]
    products = [product["id"] for product in warehouse["catalogues"]["product"]]
    for settings in folder.glob("*.json"):
        if settings.name != "simulation.json":
            assert not any(product in settings.read_text() for product in products), settings.name


def test_every_event_that_links_an_item_links_its_product_and_only_its_order_s_events_link_a_customer(run):
    # tag_along on the product reference; the customer reference has none, so pay order links no customer
    _, _, ocel_path = run

    with open(ocel_path) as file:
        ocel = json.load(file)
    object_type = {obj["id"]: obj["type"] for obj in ocel["objects"]}
    product_of = {obj["id"]: link["objectId"] for obj in ocel["objects"] if obj["type"] == "item"
                  for link in obj["relationships"] if link["qualifier"] == "is a"}
    with_customer = set()
    for event in ocel["events"]:
        linked = [(link["objectId"], link["qualifier"]) for link in event["relationships"]]
        items = {object_id for object_id, _ in linked if object_type[object_id] == "item"}
        assert {(object_id, qualifier) for object_id, qualifier in linked if object_type[object_id] == "product"} == {
            (product_of[item], "product") for item in items}
        if any(object_type[object_id] == "customer" for object_id, _ in linked):
            with_customer.add(event["type"])
    assert with_customer == {"place order", "confirm order"}


def test_pm4py_reads_the_simulated_file(run):
    _, _, ocel_path = run

    with open(ocel_path) as file:
        written = json.load(file)
    ocel = pm4py.read_ocel2_json(str(ocel_path))

    assert len(ocel.events) == len(written["events"])
    # pm4py leaves out objects that no event refers to; the file has none, so both count the same objects
    assert sorted(ocel.objects["ocel:oid"]) == sorted(obj["id"] for obj in written["objects"])


def test_every_order_ends_paid(run):
    report, log, _ = run

    assert set(_rows(log, "Sales", "pay order")["case_id"]) == set(_orders(log))
    assert not any(process == "Sales" for process, _ in report.stalled)


def test_after_confirmation_and_each_reminder_an_order_waits_for_the_customer_or_is_reminded_20_days_later(run):
    _, log, _ = run

    confirmed = _rows(log, "Sales", "confirm order").groupby("case_id")["end_time"].first()
    paying = _rows(log, "Sales", "pay order").groupby("case_id")["enable_time"].first()
    reminders = _rows(log, "Sales", "payment reminder").groupby("case_id")
    due = reminders["enable_time"].apply(sorted)
    done = reminders["end_time"].apply(sorted)
    assert 0 < len(due) < len(confirmed)  # some orders were reminded, not all
    waited = []
    for order in _orders(log):
        # each choice is made when the confirmation or the previous reminder is done
        chosen = [confirmed[order]] + done.get(order, [])
        assert [earlier + REMINDER_GAP for earlier in chosen[:-1]] == due.get(order, [])
        # paying: the customer's timer, then pay order
        assert paying[order] > chosen[-1]
        waited.append((paying[order] - chosen[-1]).total_seconds())
    assert 2 * 86400 < statistics.mean(waited) < 4 * 86400  # the customer pays after 3 days on average


def test_at_the_end_the_only_stalled_cases_are_packing_rounds_that_never_took_an_item(run):
    report, _, _ = run

    assert report.stalled  # with this seed, the last few rounds find no items left
    assert all(process == "Packaging" and case.event_id == "Catch_Items" for process, case in report.stalled)
    assert not {case.object_id for _, case in report.stalled} & set(_packages(report))
    assert len(report.stalled) + len(_packages(report)) == ROUNDS
    assert report.unclaimed == [] and report.discards == []


def test_seven_people_work_monday_to_friday_9_to_17_and_system_records_the_instant_tasks():
    folder = Path(CONFIG).parent
    staff, system_tasks = {}, set()
    for name in ("sales.json", "warehouse.json", "packaging.json"):
        settings = json.loads((folder / name).read_text())
        calendars = {calendar["id"]: calendar["time_periods"] for calendar in settings["resource_calendars"]}
        durations = {task["task_id"]: task["resources"][0] for task in settings["task_resource_distribution"]}
        for profile in settings["resource_profiles"]:
            for resource in profile["resource_list"]:
                if resource["id"] == "System":
                    system_tasks.update(resource["assigned_tasks"])
                    assert all(durations[task]["distribution_params"] == [{"value": 0}] for task in resource["assigned_tasks"])
                else:
                    staff[resource["id"]] = resource["amount"]
                    assert calendars[resource["calendar"]] == [
                        {"from": "MONDAY", "to": "FRIDAY", "beginTime": "09:00:00", "endTime": "17:00:00"}]

    assert staff == STAFF
    assert system_tasks == {"Place_Order", "Failed_Delivery", "Package_Delivered"}


def test_no_staff_task_starts_or_ends_outside_the_staff_s_hours(run):
    _, log, _ = run

    staff = log[log["resource"].str.rsplit("_", n=1).str[0].isin(STAFF)]
    assert set(staff["resource"].str.rsplit("_", n=1).str[0]) == set(STAFF)
    assert all(_in_staff_hours(moment.to_pydatetime()) for moment in [*staff["start_time"], *staff["end_time"]])
    # the instant tasks happen when they are due, at any hour
    instant = log[log["resource"] == "System"]
    assert (instant["start_time"] == instant["end_time"]).all() and set(instant["activity"]) == {
        "place order", "failed delivery", "package delivered"}


def test_every_task_s_work_time_is_within_its_range(run):
    _, log, _ = run

    for activity, rows in log.groupby("activity"):
        low, high = WORK[activity]
        work = [_work_seconds(start.to_pydatetime(), end.to_pydatetime()) / 60
                for start, end in zip(rows["start_time"], rows["end_time"])]
        assert low - 1e-6 <= min(work) and max(work) <= high + 1e-6, activity
    assert set(log["activity"]) == set(WORK)


def test_the_same_seed_gives_the_same_run(tmp_path):
    first = run_orchestrator(_config(), tmp_path / "first.csv")
    second = run_orchestrator(_config(), tmp_path / "second.csv")

    assert first.to_dict() == second.to_dict()
    assert (tmp_path / "first.csv").read_text() == (tmp_path / "second.csv").read_text()


def test_the_same_seed_gives_the_same_run_through_start_orchestration(tmp_path):
    # the committed configuration has 2,000 orders; this copy has 100, with the models' absolute paths
    with open(CONFIG) as file:
        settings = json.load(file)
    folder = Path(CONFIG).parent.resolve()
    for process in settings["processes"]:
        process["bpmn_path"] = str(folder / process["bpmn_path"])
        process["json_path"] = str(folder / process["json_path"])
        if process["name"] in ("Sales", "Packaging"):
            process["arrivals"] = ORDERS if process["name"] == "Sales" else ROUNDS
    config_path = tmp_path / "simulation.json"
    config_path.write_text(json.dumps(settings))

    for name in ("first", "second"):
        result = CliRunner().invoke(cli, ["start-orchestration", "--config", str(config_path),
                                          "--log_out_path", str(tmp_path / f"{name}.csv"),
                                          "--report_out_path", str(tmp_path / f"{name}.json")])
        assert result.exit_code == 0, result.output
    run_orchestrator(_config(), tmp_path / "python.csv")

    assert (tmp_path / "first.csv").read_text() == (tmp_path / "second.csv").read_text()
    assert (tmp_path / "first.json").read_text() == (tmp_path / "second.json").read_text()
    assert (tmp_path / "first.csv").read_text() == (tmp_path / "python.csv").read_text()
