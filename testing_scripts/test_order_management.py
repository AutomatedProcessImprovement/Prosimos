"""
The Order Management running example (docs/order-management.md): four processes after the OCEL 2.0 Order
Management log, run together on 100 orders with a fixed seed.
"""
import dataclasses
import json
import statistics
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pm4py
import pytest
from click.testing import CliRunner

from cli.diff_res_bpsim import cli
from prosimos.orchestrator import SimulationConfig, run_orchestrator
from testing_scripts.order_management_compare import metrics

CONFIG = "testing_scripts/assets/order_management/simulation.json"  # 2,000 orders, 1,200 packing rounds; seed 1
ORDERS = 100
ROUNDS = 60  # as many per order as the committed configuration, so that the rounds last as long as the orders
# a race timer's branch continues one microsecond after its time, so each reminder comes 20 days and one
# microsecond after the confirmation or the previous reminder (the reminder itself takes no time)
REMINDER_GAP = timedelta(days=20, microseconds=1)


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


def test_the_ocel_file_links_items_to_products_and_packages_by_the_real_log_s_qualifiers(run):
    _, log, ocel_path = run

    with open(ocel_path) as file:
        ocel = json.load(file)
    product_of = _rows(log, "Warehouse").groupby("case_id")["product"].first()
    items = [obj for obj in ocel["objects"] if obj["type"] == "item"]
    assert all(obj["relationships"] == [{"objectId": product_of[int(obj["id"].split("-")[1])], "qualifier": "is a"}]
               for obj in items)
    qualifiers = defaultdict(set)  # activity -> the qualifiers of its events' links to packages
    for event in ocel["events"]:
        for relation in event["relationships"]:
            if relation["objectId"].startswith("package-"):
                qualifiers[event["type"]].add(relation["qualifier"])
    # pick item doesn't link the package: the Warehouse's ItemPicked entry has no qualifier, as in the real log
    assert dict(qualifiers) == {"create package": {"creates"},
                                "send package": {"shipped package"}, "failed delivery": {"shipped package"},
                                "package delivered": {"shipped package"}}


# the object types each event type of the real log links to, apart from its employees (resources here) and products
REAL_LOG_LINKS = {
    "place order": {"order", "item", "customer"}, "confirm order": {"order", "item", "customer"},
    "payment reminder": {"order", "item"}, "pay order": {"order", "item"},
    "item out of stock": {"item"}, "reorder item": {"item"}, "pick item": {"item"},
    "create package": {"package", "item"}, "send package": {"package", "item"},
    "failed delivery": {"package", "item"}, "package delivered": {"package", "item"},
}


def test_each_event_type_links_the_object_types_the_real_log_links(run):
    _, _, ocel_path = run

    with open(ocel_path) as file:
        ocel = json.load(file)
    object_type = {obj["id"]: obj["type"] for obj in ocel["objects"]}
    linked = defaultdict(set)
    for event in ocel["events"]:
        linked[event["type"]].update(object_type[relation["objectId"]] for relation in event["relationships"])

    assert {activity: types - {"product"} for activity, types in linked.items()} == REAL_LOG_LINKS


def test_pm4py_reads_the_simulated_file(run):
    _, _, ocel_path = run

    with open(ocel_path) as file:
        written = json.load(file)
    ocel = pm4py.read_ocel2_json(str(ocel_path))

    assert len(ocel.events) == len(written["events"])
    # pm4py leaves out objects that no event refers to: here, the packing rounds that never took an item
    referred_to = {relation["objectId"] for event in written["events"] for relation in event["relationships"]}
    assert set(ocel.objects["ocel:oid"]) == referred_to
    assert len(referred_to) < len(written["objects"])


def test_the_comparison_script_counts_from_the_simulated_file_what_the_engine_did(run):
    report, _, ocel_path = run

    packages = _packages(report)
    measured = metrics(ocel_path)

    picked = [message for message in report.published if message.type == "ItemPicked"]
    assert measured["counts"] == {"orders": ORDERS, "items": len(picked), "packages": len(packages),
                                  "packing rounds that took no item": ROUNDS - len(packages),
                                  "customers": 15, "products": 20}
    assert measured["items per package"]["mean"] == statistics.mean(len(claimed) for claimed in packages.values())
    assert measured["object-to-object links"] == {"customers places orders": ORDERS, "items is a products": len(picked),
                                                  "orders comprises items": len(picked),
                                                  "packages contains items": len(picked)}


def test_every_order_ends_paid(run):
    report, log, _ = run

    assert set(_rows(log, "Sales", "pay order")["case_id"]) == set(_orders(log))
    assert not any(process == "Sales" for process, _ in report.stalled)


def test_reminders_come_exactly_20_days_apart_and_only_while_unpaid(run):
    _, log, _ = run

    confirmed = _rows(log, "Sales", "confirm order").groupby("case_id")["end_time"].first()
    paid = _rows(log, "Sales", "pay order").groupby("case_id")["enable_time"].first()  # when the payment came
    reminders = _rows(log, "Sales", "payment reminder").groupby("case_id")["enable_time"].apply(sorted)
    assert len(reminders) > 0  # some orders were reminded
    for order in _orders(log):
        # the race starts when the order is confirmed, and again after each reminder
        times = [confirmed[order]] + reminders.get(order, [])
        assert all(later - earlier == REMINDER_GAP for earlier, later in zip(times, times[1:]))
        # every reminder came before the payment, and the payment came before the next one was due
        assert times[-1] < paid[order] < times[-1] + REMINDER_GAP


def test_at_the_end_the_only_stalled_cases_are_packing_rounds_that_never_took_an_item(run):
    report, _, _ = run

    assert report.stalled  # with this seed, the last few rounds find no items left
    assert all(process == "Packaging" and case.event_id == "Catch_Items" for process, case in report.stalled)
    assert not {case.object_id for _, case in report.stalled} & set(_packages(report))
    assert len(report.stalled) + len(_packages(report)) == ROUNDS
    assert report.unclaimed == [] and report.discards == []


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
