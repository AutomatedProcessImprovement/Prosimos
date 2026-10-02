"""
Message start events (docs/messaging.md): a consume entry on the start event makes messages start
the process's cases. Loading and validation only.
"""
import json
from datetime import datetime

import pytest
import pytz

from prosimos.exceptions import InvalidSimScenarioException
from prosimos.messaging_parser import ConditionTerm, ConsumePoint
from prosimos.orchestrator import ProcessSpec, SimulationConfig
from prosimos.simulation_setup import SimDiffSetup

ASSETS = "testing_scripts/assets/messaging"
WAREHOUSE = f"{ASSETS}/tartu_warehouse.bpmn"  # message start -> Pack order -> throw Shipment -> end
WAREHOUSE_JSON = f"{ASSETS}/tartu_warehouse.json"  # no arrival settings; starts a case per Tartu or Tapa order
START = pytz.utc.localize(datetime(2024, 1, 1, 9))


def _load(bpmn, json_path, total_cases=None):
    return SimDiffSetup(bpmn, json_path, False, total_cases, START)


def _warehouse_settings(tmp_path, change):
    with open(WAREHOUSE_JSON) as file:
        settings = json.load(file)
    change(settings)
    path = tmp_path / "warehouse.json"
    path.write_text(json.dumps(settings))
    return str(path)


def _start_entry(settings):
    return settings["messages"]["consume"][0]


def test_a_warehouse_started_by_orders_loads():
    setup = _load(WAREHOUSE, WAREHOUSE_JSON)

    assert setup.messaging.started_by_messages
    assert setup.messaging.consume == (ConsumePoint(
        "Start_Order", "OrderPlaced",
        ((ConditionTerm("city", "=", value="Tartu"),), (ConditionTerm("city", "=", value="Tapa"),)),
        copy=(("order_id", "case_id"),), starts_case=True),)
    assert setup.messaging.publish[0].attributes == ("order_id",)  # declared by the copy
    assert setup.total_num_cases is None


def test_a_process_started_by_messages_takes_no_total_cases():
    with pytest.raises(InvalidSimScenarioException, match="tartu_warehouse is started by messages, so it takes no "
                                                          r"total_cases \(got 0\)"):
        _load(WAREHOUSE, WAREHOUSE_JSON, total_cases=0)


def test_a_process_not_started_by_messages_needs_total_cases_and_an_arrival_distribution(tmp_path):
    with pytest.raises(InvalidSimScenarioException, match="sales needs total_cases: it isn't started by messages"):
        _load(f"{ASSETS}/sales.bpmn", f"{ASSETS}/sales.json")

    with open(f"{ASSETS}/sales.json") as file:
        settings = json.load(file)
    del settings["arrival_time_distribution"]
    no_arrivals = tmp_path / "sales.json"
    no_arrivals.write_text(json.dumps(settings))
    with pytest.raises(InvalidSimScenarioException, match="needs an arrival_time_distribution: it isn't started"):
        _load(f"{ASSETS}/sales.bpmn", str(no_arrivals), total_cases=5)


def test_a_start_condition_may_not_use_case_attributes(tmp_path):
    on_case = _warehouse_settings(tmp_path, lambda settings: _start_entry(settings)["condition"].append(
        [{"attribute": "order_id", "comparison": "=", "case_attribute": "order_id"}]))

    with pytest.raises(InvalidSimScenarioException, match="the condition of a start event may only use fixed values "
                                                          "and source, not case_attribute"):
        _load(WAREHOUSE, on_case)


def test_a_start_condition_may_use_source(tmp_path):
    from_sales = _warehouse_settings(tmp_path, lambda settings: _start_entry(settings).update(
        condition=[[{"attribute": "source", "comparison": "=", "value": "Sales"}]]))

    assert _load(WAREHOUSE, from_sales).messaging.consume[0].condition[0][0].attribute == "source"


@pytest.mark.parametrize("copy, reason", [
    ({"case_id": "order_id"}, "'copy' can't set case_id, which is reserved"),
    (["order_id"], "'copy' must map case attribute names to message attribute names"),
    ({"order_id": ""}, "'copy' must map case attribute names to message attribute names"),
])
def test_copy_maps_case_attributes_to_message_attributes(tmp_path, copy, reason):
    settings = _warehouse_settings(tmp_path, lambda settings: _start_entry(settings).update(copy=copy))

    with pytest.raises(InvalidSimScenarioException, match=reason):
        _load(WAREHOUSE, settings)


def test_without_copy_publishing_order_id_is_rejected(tmp_path):
    no_copy = _warehouse_settings(tmp_path, lambda settings: _start_entry(settings).pop("copy"))

    with pytest.raises(InvalidSimScenarioException, match="'order_id' not declared as a case, global or event attribute"):
        _load(WAREHOUSE, no_copy)


def test_copy_is_allowed_on_a_catch_event_too(tmp_path):
    with open(f"{ASSETS}/sales.json") as file:
        settings = json.load(file)
    settings["messages"]["consume"][0]["copy"] = {"warehouse": "source"}
    path = tmp_path / "sales.json"
    path.write_text(json.dumps(settings))

    point = _load(f"{ASSETS}/sales.bpmn", str(path), total_cases=5).messaging.consume[0]

    assert (point.copy, point.starts_case) == ((("warehouse", "source"),), False)


def test_a_configuration_can_leave_out_total_cases(tmp_path):
    config_file = tmp_path / "simulation.json"
    config_file.write_text(json.dumps({"processes": [
        {"name": "TartuWarehouse", "bpmn_path": "warehouse.bpmn", "json_path": "warehouse.json"}],
        "start_time": "2024-01-01T09:00:00+00:00"}))

    assert SimulationConfig.from_json(config_file).processes[0].total_cases is None
    assert ProcessSpec("TartuWarehouse", "warehouse.bpmn", "warehouse.json").total_cases is None
