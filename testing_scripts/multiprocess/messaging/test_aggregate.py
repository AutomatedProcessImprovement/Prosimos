"""
aggregate (docs/messaging.md): a consume entry's aggregate writes a function (sum, count, min or max) over the
messages claimed into the case, updated at every claim.
"""
import json

import pandas as pd
import pytest

from prosimos.exceptions import InvalidSimScenarioException
from prosimos.messaging.parser import declared_attributes
from prosimos.orchestration.config import ProcessSpec, SimulationConfig
from prosimos.orchestration.prosimos_engine import ProsimosEngine
from prosimos.orchestration.protocol import Message
from prosimos.orchestration.simulation import run_multiprocess_simulation
from testing_scripts.multiprocess.catalogue.test_catalogue import _config_with
from testing_scripts.multiprocess.messaging.test_append import _run as _run_orders
from testing_scripts.multiprocess.messaging.test_count_over_list import PACKAGES, PACKAGES_JSON, START, at
from testing_scripts.multiprocess.orchestrator_settings import one_organization
from testing_scripts.multiprocess.scripted_engine import ScriptedEngine


def _settings(tmp_path, aggregate, change=lambda settings: None):
    """Packages (started by an ItemPicked, collecting 2 more) with aggregate on both of its consume entries."""
    with open(PACKAGES_JSON) as file:
        settings = json.load(file)
    for entry in settings["messages"]["consume"]:
        entry["aggregate"] = aggregate
    change(settings)
    path = tmp_path / "packages.json"
    path.write_text(json.dumps(settings))
    return str(path)


def _run(tmp_path, sent, aggregate):
    """A package sent an ItemPicked with each of sent's attributes by a Picking engine at 09:00, 09:10, ...; the
    report and the CSV log."""
    picking = ScriptedEngine("Picking")
    for minutes, attributes in zip(range(0, 60, 10), sent):
        picking.publish_at(at(f"09:{minutes:02}"), "ItemPicked", **attributes)
    config = SimulationConfig(one_organization([ProcessSpec("Packages", PACKAGES, _settings(tmp_path, aggregate), None,
                                                            "package", ocel_attributes=tuple(aggregate))]),
                              START, 1, extra_processes=["Picking"])
    report = run_multiprocess_simulation(config, tmp_path / "log.csv", {"Picking": picking},
                                         ocel_out_path=tmp_path / "log.json")
    return report, pd.read_csv(tmp_path / "log.csv")


def _history(report, attribute):
    return [(value, time.strftime("%H:%M")) for _, case_object in report.objects if case_object.object_id == "package-0"
            for name, value, time in case_object.attributes if name == attribute]


def _ocel_history(tmp_path, object_id, attribute):
    with open(tmp_path / "log.json") as file:
        objects = {obj["id"]: obj for obj in json.load(file)["objects"]}
    return [(value["value"], value["time"][11:16]) for value in objects[object_id]["attributes"]
            if value["name"] == attribute]


WEIGHTS = [{"weight": 0.8}, {"weight": 2.1}, {"weight": 1.2}]


def test_a_package_of_items_weighing_1_2_0_8_and_2_1_weighs_4_1(tmp_path):
    report, _ = _run(tmp_path, WEIGHTS, {"weight": {"sum": "weight"}})

    # at every claim, the first at the start event. The sum is rounded once, at the end: added one by one,
    # 0.8 + 2.1 + 1.2 is 4.1000000000000005 (0.8 + 2.1 is 2.9000000000000004 either way)
    assert _history(report, "weight") == [(0.8, "09:00"), (pytest.approx(2.9), "09:10"), (4.1, "09:20")]


def test_the_ocel_output_keeps_the_value_a_starting_message_wrote(tmp_path):
    # the package is created by its first ItemPicked, which writes 0.8 into it: a value, not a starting 0
    _run(tmp_path, WEIGHTS, {"weight": {"sum": "weight"}, "items": {"count": "weight"}})

    assert _ocel_history(tmp_path, "package-0", "weight") == [
        (0.8, "09:00"), (pytest.approx(2.9), "09:10"), (4.1, "09:20")]
    assert _ocel_history(tmp_path, "package-0", "items") == [(1, "09:00"), (2, "09:10"), (3, "09:20")]


@pytest.mark.parametrize("function, values", [
    ("count", [1, 2, 3]),
    ("min", [0.8, 0.8, 0.8]),
    ("max", [0.8, 2.1, 2.1]),
])
def test_count_min_and_max(tmp_path, function, values):
    report, _ = _run(tmp_path, WEIGHTS, {"result": {function: "weight"}})

    assert _history(report, "result") == list(zip(values, ["09:00", "09:10", "09:20"]))


def test_a_sum_of_whole_numbers_is_a_whole_number(tmp_path):
    report, _ = _run(tmp_path, [{"weight": 1}, {"weight": 2}, {"weight": 3}], {"weight": {"sum": "weight"}})

    assert [type(value) for value, _ in _history(report, "weight")] == [int, int, int]
    assert _history(report, "weight")[-1] == (6, "09:20")


def test_the_value_is_a_column_of_the_csv_log_and_an_ocel_object_attribute(tmp_path):
    _, log = _run(tmp_path, WEIGHTS, {"weight": {"sum": "weight"}})
    with open(tmp_path / "log.json") as file:
        ocel = json.load(file)

    assert log.set_index("activity")["weight"]["Pack"] == 4.1
    package = next(obj for obj in ocel["objects"] if obj["id"] == "package-0")
    assert [attribute["value"] for attribute in package["attributes"] if attribute["name"] == "weight"] == [
        0.8, pytest.approx(2.9), 4.1]
    assert {"name": "weight", "type": "float"} in next(
        object_type for object_type in ocel["objectTypes"] if object_type["name"] == "package")["attributes"]


def test_a_sum_or_count_starts_at_0_and_a_min_or_max_has_no_value_before_the_first_message(tmp_path):
    def collecting_only(settings):
        del settings["messages"]["consume"][0]["aggregate"]

    engine = ProsimosEngine(ProcessSpec("Packages", PACKAGES, _settings(
        tmp_path, {"total": {"sum": "weight"}, "items": {"count": "weight"}, "lightest": {"min": "weight"}},
        collecting_only)), START, seed=1)
    engine.deliver(Message("ItemPicked", {"weight": 0.8}, source="Picking"), START)

    values = engine._env.sim_setup.bpmn_graph.all_attributes[0]
    assert (values["total"], values["items"], "lightest" in values) == (0, 0, False)


def test_a_planned_case_starts_with_a_sum_of_0_too(tmp_path):
    # an order created by its arrival schedule at 09:00, collecting 3 ItemReady at Catch_Items from 09:10
    def aggregating(settings):
        settings["messages"]["consume"][0]["aggregate"] = {"weight": {"sum": "weight"}}

    report, _, _ = _run_orders(tmp_path, WEIGHTS, aggregating, ocel_attributes=("weight",))

    assert [(value, time.strftime("%H:%M")) for _, case_object in report.objects if case_object.object_id == "order-0"
            for name, value, time in case_object.attributes if name == "weight"] == [
        (0, "09:00"), (0.8, "09:20"), (pytest.approx(2.9), "09:30"), (4.1, "09:40")]



def test_the_ocel_output_leaves_out_a_planned_cases_starting_0(tmp_path):
    # the run report and the CSV log keep it (above); the OCEL output starts at the first message
    def aggregating(settings):
        settings["messages"]["consume"][0]["aggregate"] = {"weight": {"sum": "weight"}}

    _, log, _ = _run_orders(tmp_path, WEIGHTS, aggregating, ocel_attributes=("weight",))

    assert _ocel_history(tmp_path, "order-0", "weight") == [
        (0.8, "09:20"), (pytest.approx(2.9), "09:30"), (4.1, "09:40")]
    assert 0 in log["weight"].tolist()


@pytest.mark.parametrize("function", ["sum", "count"])
def test_a_message_without_the_attribute_is_skipped_with_one_warning_per_event_and_attribute(tmp_path, function):
    report, _ = _run(tmp_path, [{"weight": 0.8}, {}, {}], {"result": {function: "weight"}})

    assert _history(report, "result")[-1] == ({"sum": 0.8, "count": 1}[function], "09:00")
    assert [warning for _, warning in report.engine_warnings if "aggregate" in warning] == [
        "ItemPicked message accepted at Catch_More_Items has no weight to aggregate into result; result is left "
        "unchanged"]


def test_a_value_that_is_not_a_number_is_skipped_with_one_warning_except_for_count(tmp_path):
    sent = [{"weight": 0.8}, {"weight": True}, {"weight": "light"}]

    report, _ = _run(tmp_path, sent, {"total": {"sum": "weight"}, "items": {"count": "weight"}})

    assert (_history(report, "total")[-1][0], _history(report, "items")[-1][0]) == (0.8, 3)
    assert [warning for _, warning in report.engine_warnings if "number" in warning] == [
        "ItemPicked message accepted at Catch_More_Items has weight True, which isn't a number to sum into total; "
        "total is left unchanged"]


# load errors

def _rejected(tmp_path, aggregate, message, change=lambda settings: None):
    with pytest.raises(InvalidSimScenarioException, match=message):
        ProsimosEngine(ProcessSpec("Packages", PACKAGES, _settings(tmp_path, aggregate, change)), START, seed=1)


@pytest.mark.parametrize("aggregate, message", [
    (["weight"], r"'aggregate' must map object attribute names to \{<function>: <message attribute>\}"),
    ({"weight": "weight"}, r"'aggregate' must map object attribute names to"),
    ({"weight": {"avg": "weight"}}, r"\.aggregate\.weight: unknown key 'avg'; allowed: count, initial, max, min, sum"),
    ({"weight": {}}, r"aggregate 'weight' must be one function and a message attribute"),
    ({"weight": {"sum": "weight", "max": "weight"}}, r"aggregate 'weight' must be one function and a message attribute"),
    ({"weight": {"sum": 3}}, r"aggregate 'weight' must be one function and a message attribute"),
    ({"object_id": {"sum": "weight"}}, r"'aggregate' can't set object_id"),
])
def test_a_malformed_aggregate_is_rejected(tmp_path, aggregate, message):
    _rejected(tmp_path, aggregate, r"messages.consume\[0\](: )?" + message)


@pytest.mark.parametrize("key", ["copy", "append"])
def test_the_same_attribute_in_copy_or_append_and_aggregate_of_one_entry_is_rejected(tmp_path, key):
    def both(settings):
        settings["messages"]["consume"][0].pop("append")
        settings["messages"]["consume"][0][key] = {"weight": "weight"}

    _rejected(tmp_path, {"weight": {"sum": "weight"}},
              rf"messages.consume\[0\]: 'weight' is in both {key} and aggregate", both)


def test_aggregating_into_an_attribute_declared_in_object_attributes_is_rejected(tmp_path):
    def drawn(settings):
        settings["object_attributes"] = [{"name": "weight", "type": "continuous",
                                        "values": {"distribution_name": "fix", "distribution_params": [{"value": 1}]}}]

    _rejected(tmp_path, {"weight": {"sum": "weight"}},
              r"messages.consume\[0\]: aggregate writes 'weight', which object_attributes declares", drawn)


def test_aggregating_into_an_attribute_another_entry_appends_to_is_rejected(tmp_path):
    def other(settings):
        del settings["messages"]["consume"][0]["aggregate"], settings["messages"]["consume"][1]["append"]

    _rejected(tmp_path, {"item_ids": {"count": "object_id"}},
              r"messages.consume\[1\]: aggregate writes 'item_ids', which messages.consume\[0\] appends to", other)


def test_two_entries_aggregating_into_one_attribute_with_different_functions_are_rejected(tmp_path):
    def different(settings):
        settings["messages"]["consume"][1]["aggregate"] = {"weight": {"max": "weight"}}

    _rejected(tmp_path, {"weight": {"sum": "weight"}},
              r"messages.consume\[1\]: aggregate writes 'weight' with max, but messages.consume\[0\] with sum",
              different)


def test_an_aggregated_attribute_counts_as_declared_so_messages_can_publish_it():
    assert "weight" in declared_attributes({"messages": {"consume": [{"aggregate": {"weight": {"sum": "weight"}}}]}})


def test_picking_an_attribute_the_process_aggregates_into_is_rejected(tmp_path):
    with open("testing_scripts/assets/order_management/packaging.json") as file:
        settings = json.load(file)
    settings["messages"]["consume"][0]["aggregate"] = {"segment": {"max": "customer"}}
    packaging = tmp_path / "packaging.json"
    packaging.write_text(json.dumps(settings))
    path = _config_with(tmp_path, catalogues={"customer": {"count": 10}}, references={
        "Packaging": [{"object_attribute": "segment", "catalogue": "customer", "pick": "uniform"}]})
    data = json.loads(path.read_text())
    next(process for process in data["organizations"][0]["processes"] if process["name"] == "Packaging")["json_path"] = str(packaging)
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match=r"processes\[2\].references\[0\] picks segment, which process Packaging also aggregates "
                                         r"into from a message \(consume entries at \['Catch_Items'\]\)"):
        SimulationConfig.from_json(path)
