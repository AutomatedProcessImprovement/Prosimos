"""
OCEL 2.0 output (docs/orchestrator.md): one object per case, typed by its process's object_type, and one event per
task, linked to its case's object.
"""
import json
from collections import Counter
from datetime import datetime

import pandas as pd
import pytest
import pytz

from prosimos.orchestrator import ProcessSpec, ProsimosEngine, SimulationConfig, run_engines, run_orchestrator
from prosimos.ocel_writer import write_ocel
from testing_scripts.scripted_engine import ScriptedEngine
from testing_scripts.test_order_management import _config

START = pytz.utc.localize(datetime(2024, 1, 1, 9))


@pytest.fixture(scope="module")
def order_management(tmp_path_factory):
    """A 100-order run of the Order Management example, written as CSV and as OCEL."""
    folder = tmp_path_factory.mktemp("ocel")
    report = run_orchestrator(_config(100), folder / "log.csv", ocel_out_path=folder / "log.json")
    log = pd.read_csv(folder / "log.csv", parse_dates=["end_time"])
    with open(folder / "log.json") as file:
        ocel = json.load(file)
    return report, log, ocel, folder / "log.json"


def test_orders_items_and_packages_are_objects_one_per_case(order_management):
    report, log, ocel, _ = order_management

    objects = Counter(obj["type"] for obj in ocel["objects"])
    cases_in_log = log.groupby("process")["case_id"].nunique()
    still_collecting = sum(process == "Packaging" for process, _ in report.stalled)  # no rows in the log yet
    assert objects == Counter({"orders": cases_in_log["Sales"], "items": cases_in_log["Warehouse"],
                               "packages": cases_in_log["Packaging"] + still_collecting})
    assert not any(obj["id"].startswith("Customer-") for obj in ocel["objects"])  # object_type: null


def test_there_is_one_event_per_task_linked_to_its_case(order_management):
    _, log, ocel, _ = order_management

    events = Counter((event["type"], event["time"], event["attributes"][0]["value"],
                      event["relationships"][0]["objectId"]) for event in ocel["events"])
    rows = Counter((row.activity, row.end_time.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ"), row.resource,
                    f"{row.process}-{row.case_id}") for row in log.itertuples())
    assert events == rows
    assert [event["id"] for event in ocel["events"]] == [f"e{n}" for n in range(1, len(log) + 1)]
    assert [event["time"] for event in ocel["events"]] == sorted(event["time"] for event in ocel["events"])


def test_object_and_event_types_are_listed_with_their_attributes(order_management):
    _, _, ocel, _ = order_management

    assert {t["name"]: t["attributes"] for t in ocel["objectTypes"]} == {
        "items": [{"name": "customer", "type": "string"}, {"name": "item_index", "type": "integer"},
                  {"name": "order_id", "type": "string"}],
        "orders": [{"name": "customer", "type": "string"}, {"name": "items", "type": "integer"}],
        "packages": [{"name": "customer", "type": "string"}, {"name": "more_items", "type": "integer"}],
    }
    assert {t["name"] for t in ocel["eventTypes"]} == {
        "place order", "confirm order", "payment reminder", "pay order", "item out of stock", "reorder item",
        "pick item", "create package", "send package", "failed delivery", "package delivered"}
    assert all(t["attributes"] == [{"name": "resource", "type": "string"}] for t in ocel["eventTypes"])


def test_pm4py_reads_the_file(order_management):
    pm4py = pytest.importorskip("pm4py")  # not a dependency of Prosimos
    _, log, _, path = order_management

    ocel = pm4py.read_ocel2_json(str(path))

    assert len(ocel.events) == len(log)
    assert set(ocel.objects["ocel:type"]) == {"orders", "items", "packages"}


def test_the_object_type_defaults_to_the_process_name_and_null_leaves_the_process_out(tmp_path):
    config_path = tmp_path / "simulation.json"
    config_path.write_text(json.dumps({"start_time": "2024-01-01T09:00:00+00:00", "processes": [
        {"name": "Sales", "bpmn_path": "s.bpmn", "json_path": "s.json", "total_cases": 1},
        {"name": "Customer", "bpmn_path": "c.bpmn", "json_path": "c.json", "object_type": None},
        {"name": "Warehouse", "bpmn_path": "w.bpmn", "json_path": "w.json", "object_type": "items"}]}))

    config = SimulationConfig.from_json(config_path)

    assert [spec.ocel_object_type for spec in config.processes] == ["Sales", None, "items"]


def test_a_value_copied_at_a_catch_event_is_a_later_attribute_value(tmp_path):
    # orders wait at the dock; a truck at 09:30 takes the first two and copies its id into them
    orders = ProsimosEngine(ProcessSpec("Orders", "testing_scripts/assets/messaging/orders_and_trucks.bpmn",
                                        "testing_scripts/assets/messaging/orders_and_trucks.json", 2), START, None,
                            seed=1)
    carrier = ScriptedEngine("Carrier")
    carrier.publish_at(START.replace(minute=30), "Truck", dock="Tartu", case_id="Carrier-0")
    report = run_engines({"Orders": orders, "Carrier": carrier}, None, 1)

    write_ocel(tmp_path / "log.json", {"Orders": "orders"}, report.objects, [])

    with open(tmp_path / "log.json") as file:
        objects = {obj["id"]: obj["attributes"] for obj in json.load(file)["objects"]}
    assert objects["Orders-0"] == [{"name": "truck_id", "value": "Carrier-0", "time": "2024-01-01T09:30:00.000000Z"}]
