"""
The running example with real Prosimos engines only (docs/running-example.md): Sales, Billing and the
two warehouses, all loaded from one configuration file and run with run_orchestrator.
"""
import csv
from datetime import datetime

import pytest

from prosimos.orchestrator import SimulationConfig, run_orchestrator

CONFIG = "testing_scripts/assets/running_example/simulation.json"  # seed 1, 20 orders
SERVES = {"TartuWarehouse": {"Tartu", "Tapa"}, "TallinnWarehouse": {"Tallinn", "Tapa"}}


def _run(log_path):
    report = run_orchestrator(SimulationConfig.from_json(CONFIG), log_path)
    with open(log_path, encoding="utf-8") as file:
        return report, list(csv.DictReader(file))


@pytest.fixture
def run(tmp_path):
    report, rows = _run(tmp_path / "merged_log.csv")
    city = {f"Sales-{row['case_id']}": row["city"] for row in rows if row["process"] == "Sales"}
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
    closed = {f"Sales-{row['case_id']}": datetime.fromisoformat(row["start_time"])
              for row in rows if row["process"] == "Sales" and row["activity"] == "Close order"}

    served = sorted(order for order, order_city in city.items() if order_city != "Pärnu")
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
    assert [(process, case.case_id) for process, case in report.stalled] == [("Sales", order) for order in parnu]
    assert report.warnings == [] and report.engine_warnings == [] and report.unclaimed == []


def test_the_same_seed_gives_the_same_run(tmp_path):
    first = _run(tmp_path / "first.csv")
    second = _run(tmp_path / "second.csv")

    assert first == second
