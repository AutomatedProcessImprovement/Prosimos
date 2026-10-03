"""
The first real-engine version of the running example: a Prosimos Sales model publishes OrderPlaced
and waits for its Shipment, next to the fake Billing and warehouses (testing_scripts/real_sales_scenario.py).
"""
import csv
from datetime import datetime

import pytest

from testing_scripts.real_sales_scenario import run_with_real_sales

SEED = 1
CASES = 20  # with this seed: 8 Tartu, 2 Tallinn, 7 Tapa and 3 Pärnu orders
WAREHOUSE_OF = {"Tartu": {"TartuWarehouse"}, "Tallinn": {"TallinnWarehouse"},
                "Tapa": {"TartuWarehouse", "TallinnWarehouse"}}


def _read(log_path):
    with open(log_path, encoding="utf-8") as file:
        return list(csv.DictReader(file))


@pytest.fixture
def run(tmp_path):
    report = run_with_real_sales(SEED, CASES, tmp_path / "merged_log.csv")
    rows = _read(tmp_path / "merged_log.csv")
    city = {f"Sales-{row['case_id']}": row["city"] for row in rows}
    return report, rows, city


def test_every_order_from_a_city_with_a_warehouse_is_closed_after_its_shipment(run):
    report, rows, city = run
    shipped = {m.attributes["order_id"]: m for m in report.published if m.type == "Shipment"}
    closed = {f"Sales-{row['case_id']}": datetime.fromisoformat(row["start_time"])
              for row in rows if row["activity"] == "Close order"}

    served = sorted(order for order, order_city in city.items() if order_city != "Pärnu")
    assert {city[order] for order in served} == {"Tartu", "Tallinn", "Tapa"}
    assert sorted(closed) == served and sorted(shipped) == served
    for order in served:
        assert closed[order] >= shipped[order].time
        assert shipped[order].source in WAREHOUSE_OF[city[order]]
    assert all(row["process"] == "Sales" for row in rows)  # only Sales is a real engine with a log


def test_parnu_orders_appear_as_stalled_cases(run):
    report, rows, city = run
    parnu = sorted(order for order, order_city in city.items() if order_city == "Pärnu")

    assert len(parnu) == 3
    assert [(process, case.case_id, case.event_id, case.message_types) for process, case in report.stalled] == [
        ("Sales", order, "Catch_Shipment", ["Shipment"]) for order in parnu]
    assert report.unclaimed == [] and report.warnings == [] and report.engine_warnings == []


def test_the_same_seed_gives_the_same_run(tmp_path):
    first = run_with_real_sales(SEED, CASES, tmp_path / "first.csv")
    second = run_with_real_sales(SEED, CASES, tmp_path / "second.csv")

    assert first == second
    assert _read(tmp_path / "first.csv") == _read(tmp_path / "second.csv")
