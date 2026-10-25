"""
OCEL 2.0 output (docs/orchestrator.md): one object per case, typed by its process's object_type, and one event per
task, linked to its case's object.
"""
import dataclasses
import json
from collections import Counter
from datetime import datetime

import pandas as pd
import pytest
import pytz

from prosimos.orchestrator import (CaseObject, ProcessSpec, ProsimosEngine, SimulationConfig, run_engines,
                                   run_orchestrator)
from prosimos.ocel_writer import OcelProcess, object_ids, write_ocel
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
    _, log, ocel, _ = order_management

    objects = Counter(obj["type"] for obj in ocel["objects"])
    cases_in_log = log.groupby("process")["case_id"].nunique()
    # a packing round that never took an item has no events, so it is left out
    assert objects == Counter({"order": cases_in_log["Sales"], "item": cases_in_log["Warehouse"],
                               "package": cases_in_log["Packaging"], "customer": 15, "product": 20})
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

    # only the listed attributes: none of the attributes that route messages (order_id, item_index, item_ids, items)
    assert {t["name"]: t["attributes"] for t in ocel["objectTypes"]} == {
        "item": [{"name": "price", "type": "float"}, {"name": "weight", "type": "float"}],
        "order": [],
        "package": [{"name": "weight", "type": "float"}],
        "customer": [],
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


def test_processes_of_one_object_type_get_ids_numbered_across_them_by_creation_time(tmp_path):
    # Sales and Returns both make orders and each numbers its own from order-0; the output numbers them together
    config = SimulationConfig([ProcessSpec(name, "testing_scripts/assets/messaging/sales.bpmn",
                                           "testing_scripts/assets/messaging/sales.json", 4, "order")
                               for name in ("Sales", "Returns")], START, 1)
    report = run_orchestrator(config, tmp_path / "log.csv", ocel_out_path=tmp_path / "log.json")

    assert sorted(case.object_id for _, case in report.objects) == [f"order-{case}" for case in range(4) for _ in "SR"]
    in_creation_order = sorted(report.objects, key=lambda pair: (pair[1].created, pair[0] != "Sales"))
    ids = {(process, case.object_id): f"order-{number}" for number, (process, case) in enumerate(in_creation_order)}
    assert [process for process, _ in in_creation_order] != ["Sales"] * 4 + ["Returns"] * 4  # the processes interleave
    with open(tmp_path / "log.json") as file:
        ocel = json.load(file)
    assert sorted(obj["id"] for obj in ocel["objects"]) == sorted(f"order-{number}" for number in range(8))
    # each object's events are its case's rows of the log
    events = Counter((event["relationships"][0]["objectId"], event["type"], event["time"]) for event in ocel["events"])
    log = pd.read_csv(tmp_path / "log.csv", parse_dates=["end_time"])
    rows = Counter((ids[(process, f"order-{case}")], activity, end.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
                   for process, case, activity, end
                   in log[["process", "case_id", "activity", "end_time"]].itertuples(index=False))
    assert events == rows


def test_objects_created_at_the_same_moment_go_by_process_order_then_case_number():
    later = START.replace(minute=1)
    objects = [("A", CaseObject("x-0", later)), ("A", CaseObject("x-1", later)),
               ("B", CaseObject("x-0", later)), ("B", CaseObject("x-1", START))]

    assert object_ids({"B": OcelProcess("x"), "A": OcelProcess("x"), "C": OcelProcess(None)}, objects) == {
        ("B", "x-1"): "x-0", ("B", "x-0"): "x-1", ("A", "x-0"): "x-2", ("A", "x-1"): "x-3"}


def _row(process, case, end):
    """A logged task of the case, so that the writer writes its object."""
    return process, {"case_id": case, "activity": "a", "resource": "R", "end_time": end}


def test_a_value_copied_at_a_catch_event_is_a_later_attribute_value(tmp_path):
    # orders wait at the dock; a truck at 09:30 takes the first two and copies its id into them
    orders = ProsimosEngine(ProcessSpec("Orders", "testing_scripts/assets/messaging/orders_and_trucks.bpmn",
                                        "testing_scripts/assets/messaging/orders_and_trucks.json", 2, "order"),
                            START, None, seed=1)
    carrier = ScriptedEngine("Carrier")
    carrier.publish_at(START.replace(minute=30), "Truck", dock="Tartu", object_id="Carrier-0")
    report = run_engines({"Orders": orders, "Carrier": carrier}, None, 1)

    write_ocel(tmp_path / "log.json", {"Orders": OcelProcess("order", attributes=("truck_id",))}, report.objects,
               [_row("Orders", 0, START.replace(minute=30))])

    with open(tmp_path / "log.json") as file:
        objects = {obj["id"]: obj["attributes"] for obj in json.load(file)["objects"]}
    assert objects["order-0"] == [{"name": "truck_id", "value": "Carrier-0", "time": "2024-01-01T09:30:00.000000Z"}]



# objects without events

def test_an_object_no_event_refers_to_is_left_out_with_its_links_but_catalogue_objects_are_written(tmp_path):
    # p-0 has an event, which links q-1 through a message; p-1 and q-0 have no event and no message links them
    objects = [("P", CaseObject(f"p-{case}", START)) for case in range(2)] + [
        ("Q", CaseObject(f"q-{case}", START.replace(minute=case))) for case in range(2)]
    write_ocel(tmp_path / "log.json", {"P": OcelProcess("p"), "Q": OcelProcess("q")}, objects,
               [_row("P", 0, START.replace(minute=5))], None, {("P", 0): [(("Q", "q-1"), "item")]},
               {("P", "p-0"): [(("Q", "q-0"), "waits for"), (("Q", "q-1"), "ships")],
                ("P", "p-1"): [("c1", "placed by")], "c1": [(("P", "p-0"), "places"), (("P", "p-1"), "places")]},
               [("customer", "c1", []), ("customer", "c2", [])])

    with open(tmp_path / "log.json") as file:
        ocel = json.load(file)
    # the objects written are numbered without gaps: Q's case 1 is q-0
    assert {obj["id"]: [(link["objectId"], link["qualifier"]) for link in obj["relationships"]]
            for obj in ocel["objects"]} == {"p-0": [("q-0", "ships")], "q-0": [], "c1": [("p-0", "places")], "c2": []}
    assert [[link["objectId"] for link in event["relationships"]] for event in ocel["events"]] == [["p-0", "q-0"]]


# one value per attribute and moment

def test_of_several_values_at_one_moment_only_the_last_is_written_and_a_starting_0_is_left_out(tmp_path):
    later = START.replace(minute=30)
    # weight starts at 0 and takes 3 messages at 09:00 and one at 09:30; total is a real 0, written by a message
    package = CaseObject("package-0", START, [
        ("weight", 0, START), ("total", 0, START), ("weight", 0.5, START), ("weight", 1.2, START),
        ("weight", 2.0, START), ("weight", 2.7, later)], only_started=("weight",))

    write_ocel(tmp_path / "log.json", {"Packaging": OcelProcess("package", attributes=("weight", "total"))},
               [("Packaging", package)], [_row("Packaging", 0, later)])

    with open(tmp_path / "log.json") as file:
        [written] = json.load(file)["objects"]
    assert [(value["name"], value["value"], value["time"][11:16]) for value in written["attributes"]] == [
        ("total", 0, "09:00"), ("weight", 2.0, "09:00"), ("weight", 2.7, "09:30")]


def test_every_order_management_package_with_items_has_one_weight_the_sum_of_its_items(order_management):
    _, _, ocel, _ = order_management
    weights = {obj["id"]: [value["value"] for value in obj["attributes"] if value["name"] == "weight"]
               for obj in ocel["objects"]}
    packages = [obj for obj in ocel["objects"] if obj["type"] == "package" and obj["relationships"]]

    assert packages
    for package in packages:
        items = [link["objectId"] for link in package["relationships"] if link["qualifier"] == "contains"]
        assert weights[package["id"]] == [pytest.approx(sum(weights[item][0] for item in items))]


# ocel_attributes

@pytest.mark.parametrize("listed, written", [(("city",), {"city"}), ((), set())])
def test_only_the_listed_attributes_become_object_attributes(tmp_path, listed, written):
    # Sales draws a city for every order; its objects carry it only if ocel_attributes lists it
    config = SimulationConfig([ProcessSpec("Sales", "testing_scripts/assets/messaging/sales.bpmn",
                                           "testing_scripts/assets/messaging/sales.json", 3, "order",
                                           ocel_attributes=listed)], START, 1)
    run_orchestrator(config, None, ocel_out_path=tmp_path / "log.json")

    with open(tmp_path / "log.json") as file:
        orders = [obj for obj in json.load(file)["objects"] if obj["type"] == "order"]
    assert len(orders) == 3 and all({attribute["name"] for attribute in obj["attributes"]} == written for obj in orders)


@pytest.mark.parametrize("process, name", [
    ("Sales", "items"),  # drawn in object_attributes
    ("Sales", "customer"),  # picked
    ("Warehouse", "price"),  # brought with with
    ("Warehouse", "order_id"),  # copied
    ("Packaging", "item_ids"),  # appended
    ("Packaging", "weight"),  # aggregated
])
def test_ocel_attributes_may_name_every_attribute_the_objects_have(process, name):
    config = _config(10)
    config = SimulationConfig([dataclasses.replace(spec, ocel_attributes=(name,)) if spec.name == process else spec
                               for spec in config.processes], config.start_datetime, config.seed)

    assert next(spec for spec in config.processes if spec.name == process).ocel_attributes == (name,)


def test_ocel_attributes_naming_an_attribute_the_objects_never_have_are_rejected():
    config = _config(10)
    processes = [dataclasses.replace(spec, ocel_attributes=("weight", "colour")) if spec.name == "Warehouse" else spec
                 for spec in config.processes]

    with pytest.raises(ValueError, match=r"Invalid ocel_attributes: processes\[1\].ocel_attributes names \['colour'\], "
                                         r"which the objects of process Warehouse never have"):
        SimulationConfig(processes, config.start_datetime, config.seed)


def test_ocel_attributes_that_arent_a_list_of_names_are_rejected(tmp_path):
    config_path = tmp_path / "simulation.json"
    config_path.write_text(json.dumps({"start_time": "2024-01-01T09:00:00+00:00", "processes": [
        {"name": "Sales", "bpmn_path": "s.bpmn", "json_path": "s.json", "arrivals": 1, "ocel_attributes": "city"}]}))

    with pytest.raises(ValueError, match=r"processes\[0\]: 'ocel_attributes' must be a list of attribute names, got 'city'"):
        SimulationConfig.from_json(config_path)
