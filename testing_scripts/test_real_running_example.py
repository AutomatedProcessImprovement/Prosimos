"""
The running example (docs/running-example.md): Sales, Billing and the two warehouses, all real Prosimos models
loaded from one configuration file and run with run_multiprocess_simulation.
"""
import csv
import json
from collections import Counter
from datetime import datetime

import pytest

from prosimos.orchestration.config import SimulationConfig
from prosimos.orchestration.simulation import run_multiprocess_simulation

CONFIG = "testing_scripts/assets/running_example/simulation.json"  # seed 1, 20 orders
SERVES = {"TartuWarehouse": {"Tartu", "Tapa"}, "TallinnWarehouse": {"Tallinn", "Tapa"}}


def _run(log_path):
    report = run_multiprocess_simulation(SimulationConfig.from_json(CONFIG), log_path)
    with open(log_path, encoding="utf-8") as file:
        return report, list(csv.DictReader(file))


@pytest.fixture
def run(tmp_path):
    report, rows = _run(tmp_path / "merged_log.csv")
    city = {f"order-{row['case_id']}": row["city"] for row in rows if row["process"] == "Sales"}
    return report, rows, city


def _order_ids(rows, process):
    return [row["order_id"] for row in rows if row["process"] == process]


def test_the_merged_log_has_rows_from_all_four_processes(run):
    _, rows, city = run

    assert {row["process"] for row in rows} == {"Sales", "Billing", "TartuWarehouse", "TallinnWarehouse"}
    assert len(city) == 20 and set(city.values()) == {"Tartu", "Tallinn", "Tapa", "Pärnu"}


def test_billing_has_one_case_per_order_parnu_included(run):
    _, rows, city = run

    assert sorted(_order_ids(rows, "Billing")) == sorted(city)


def test_each_order_from_a_served_city_has_one_warehouse_case_and_is_closed_after_its_shipment(run):
    report, rows, city = run
    packed_by = {order_id: warehouse for warehouse in SERVES for order_id in _order_ids(rows, warehouse)}
    shipments = {m.attributes["order_id"]: m for m in report.published if m.type == "Shipment"}
    closed = {f"order-{row['case_id']}": datetime.fromisoformat(row["start_time"])
              for row in rows if row["process"] == "Sales" and row["activity"] == "Close order"}

    served = sorted(order for order, order_city in city.items() if order_city != "Pärnu")
    assert {city[order] for order in served} == {"Tartu", "Tallinn", "Tapa"}
    assert sorted(closed) == served and sorted(shipments) == served  # only these are shipped and closed
    warehouse_cases = _order_ids(rows, "TartuWarehouse") + _order_ids(rows, "TallinnWarehouse")
    assert sorted(warehouse_cases) == served  # exactly one warehouse case per served order
    for order in served:
        assert city[order] in SERVES[packed_by[order]]
        assert shipments[order].source == packed_by[order]
        assert closed[order] >= shipments[order].time
    assert {packed_by[order] for order in served if city[order] == "Tapa"} == set(SERVES)  # Tapa goes to either


def test_parnu_orders_are_the_only_stalled_cases(run):
    report, _, city = run

    parnu = sorted(order for order, order_city in city.items() if order_city == "Pärnu")
    assert len(parnu) == 3
    # each still waiting for its shipment
    assert [(process, case.object_id, case.event_id, case.message_types) for process, case in report.stalled] == [
        ("Sales", order, "Catch_Shipment", ["Shipment"]) for order in parnu]
    assert report.warnings == [] and report.engine_warnings == [] and report.unclaimed == []


def test_the_same_seed_gives_the_same_run(tmp_path):
    first = _run(tmp_path / "first.csv")
    second = _run(tmp_path / "second.csv")

    assert first == second


def test_the_ocel_output_names_objects_by_type_and_numbers_shipments_across_both_warehouses(tmp_path):
    run_multiprocess_simulation(SimulationConfig.from_json(CONFIG), None, ocel_out_path=tmp_path / "log.json")
    with open(tmp_path / "log.json", encoding="utf-8") as file:
        objects = json.load(file)["objects"]

    ids = Counter(obj["type"] for obj in objects)
    assert ids == {"order": 20, "invoice": 20, "shipment": 17}  # 12 shipments from Tartu, 5 from Tallinn
    assert sorted(obj["id"] for obj in objects) == sorted(
        f"{object_type}-{number}" for object_type, count in ids.items() for number in range(count))


def test_the_ocel_output_links_each_invoice_and_shipment_to_its_order(tmp_path):
    run_multiprocess_simulation(SimulationConfig.from_json(CONFIG), None, ocel_out_path=tmp_path / "log.json")
    with open(tmp_path / "log.json", encoding="utf-8") as file:
        ocel = json.load(file)

    # every invoice bills one order and every shipment ships one; each order is billed once and shipped at most once
    order_of = {obj["id"]: [(link["objectId"], link["qualifier"]) for link in obj["relationships"]]
                for obj in ocel["objects"] if obj["type"] in ("invoice", "shipment")}
    assert all(len(links) == 1 for links in order_of.values())
    billed = [order for (order, qualifier), in order_of.values() if qualifier == "bills"]
    shipped = [order for (order, qualifier), in order_of.values() if qualifier == "ships"]
    assert sorted(billed) == sorted(f"order-{number}" for number in range(20))
    assert len(set(shipped)) == len(shipped) == 17
    # each event links the order of its own invoice or shipment, and closing an order links the shipment that ships it
    for event in ocel["events"]:
        links = {link["qualifier"]: link["objectId"] for link in event["relationships"]}
        if event["type"] == "Create invoice":
            assert order_of[links["invoice"]] == [(links["billed order"], "bills")]
        elif event["type"] == "Pack order":
            assert order_of[links["shipment"]] == [(links["packed order"], "ships")]
        elif event["type"] == "Close order":
            assert order_of[links["received shipment"]] == [(links["order"], "ships")]
