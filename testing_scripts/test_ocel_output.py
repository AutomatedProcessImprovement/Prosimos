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
    never_took_an_item = sum(process == "Packaging" for process, _ in report.stalled)  # no rows in the log
    assert objects == Counter({"order": cases_in_log["Sales"], "item": cases_in_log["Warehouse"],
                               "package": cases_in_log["Packaging"] + never_took_an_item,
                               "customer": 15, "product": 20})  # the catalogue's
    assert not any(obj["id"].startswith("Customer-") for obj in ocel["objects"])  # object_type: null


def test_there_is_one_event_per_task_linked_to_its_case(order_management):
    _, log, ocel, _ = order_management

    events = Counter((event["type"], event["time"], event["attributes"][0]["value"],
                      event["relationships"][0]["objectId"]) for event in ocel["events"])
    object_type = {"Sales": "order", "Warehouse": "item", "Packaging": "package"}
    rows = Counter((row.activity, row.end_time.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ"), row.resource,
                    f"{object_type[row.process]}-{row.case_id}") for row in log.itertuples())
    assert events == rows
    assert [event["id"] for event in ocel["events"]] == [f"e{n}" for n in range(1, len(log) + 1)]
    assert [event["time"] for event in ocel["events"]] == sorted(event["time"] for event in ocel["events"])


def test_object_and_event_types_are_listed_with_their_attributes(order_management):
    _, _, ocel, _ = order_management

    assert {t["name"]: t["attributes"] for t in ocel["objectTypes"]} == {
        "item": [{"name": "customer", "type": "string"}, {"name": "item_index", "type": "integer"},
                  {"name": "order_id", "type": "string"}, {"name": "price", "type": "float"},
                  {"name": "product", "type": "string"}, {"name": "weight", "type": "float"}],
        "order": [{"name": "customer", "type": "string"}, {"name": "items", "type": "integer"}],
        "package": [{"name": "customer", "type": "string"}, {"name": "item_ids", "type": "string"},
                     {"name": "weight", "type": "float"}],
        "customer": [{"name": "segment", "type": "string"}],
        "product": [{"name": "price", "type": "float"}, {"name": "weight", "type": "float"}],
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
    assert set(ocel.objects["ocel:type"]) == {"order", "item", "package", "customer", "product"}


def test_a_process_has_an_object_type_only_where_it_is_given(tmp_path):
    config_path = tmp_path / "simulation.json"
    processes = [{"name": "Sales", "bpmn_path": "s.bpmn", "json_path": "s.json", "arrivals": 1},
                 {"name": "Warehouse", "bpmn_path": "w.bpmn", "json_path": "w.json", "object_type": "item"}]
    config_path.write_text(json.dumps({"start_time": "2024-01-01T09:00:00+00:00", "processes": processes}))

    assert [spec.object_type for spec in SimulationConfig.from_json(config_path).processes] == [None, "item"]

    processes[0]["object_type"] = None
    config_path.write_text(json.dumps({"start_time": "2024-01-01T09:00:00+00:00", "processes": processes}))
    with pytest.raises(ValueError, match=r"processes\[0\]: 'object_type' must be a non-empty string, got None"):
        SimulationConfig.from_json(config_path)


@pytest.mark.parametrize("object_type, object_id", [("order", "order-0"), (None, "Sales-0")])
def test_an_object_is_named_after_its_type_or_without_one_after_its_process(object_type, object_id):
    # Sales announces each order with OrderPlaced{object_id, city}
    sales = ProsimosEngine(ProcessSpec("Sales", "testing_scripts/assets/messaging/sales.bpmn",
                                       "testing_scripts/assets/messaging/sales.json", 1, object_type), START, seed=1)
    published = []
    while sales.next_event_time() is not None:
        published.extend(sales.step())

    assert [message.attributes["object_id"] for message in published] == [object_id]


def test_two_processes_with_the_same_object_type_are_rejected():
    # both would name their objects order-0, order-1, ...
    with pytest.raises(ValueError, match=r"Object types must be unique, repeated: \['order'\]"):
        SimulationConfig([ProcessSpec("Sales", "s.bpmn", "s.json", 1, "order"),
                          ProcessSpec("Returns", "r.bpmn", "r.json", 1, "order")], START)


def test_a_value_copied_at_a_catch_event_is_a_later_attribute_value(tmp_path):
    # orders wait at the dock; a truck at 09:30 takes the first two and copies its id into them
    orders = ProsimosEngine(ProcessSpec("Orders", "testing_scripts/assets/messaging/orders_and_trucks.bpmn",
                                        "testing_scripts/assets/messaging/orders_and_trucks.json", 2), START, None,
                            seed=1)
    carrier = ScriptedEngine("Carrier")
    carrier.publish_at(START.replace(minute=30), "Truck", dock="Tartu", object_id="Carrier-0")
    report = run_engines({"Orders": orders, "Carrier": carrier}, None, 1)

    write_ocel(tmp_path / "log.json", {"Orders": "orders"}, report.objects, [])

    with open(tmp_path / "log.json") as file:
        objects = {obj["id"]: obj["attributes"] for obj in json.load(file)["objects"]}
    assert objects["Orders-0"] == [{"name": "truck_id", "value": "Carrier-0", "time": "2024-01-01T09:30:00.000000Z"}]
