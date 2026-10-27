"""
append (docs/messaging.md): a consume entry's append adds the value of every message it claims to a list in the
case, which starts empty, where copy would overwrite it.
"""
import json

import pandas as pd
import pytest

from prosimos.exceptions import InvalidSimScenarioException
from prosimos.messaging_parser import declared_attributes
from prosimos.orchestration.protocol import Message, Verdict
from prosimos.orchestrator import ProcessSpec, ProsimosEngine, SimulationConfig, run_orchestrator
from testing_scripts.scripted_engine import ScriptedEngine
from testing_scripts.test_catalogue import _config_with
from testing_scripts.test_collect import ORDERS, START, _settings, at
from testing_scripts.test_count_over_list import PACKAGES, PACKAGES_JSON
from testing_scripts.test_order_management import _config, package_items


def _appending(settings):
    settings["messages"]["consume"][0]["append"] = {"item_ids": "item_id"}


def _run(tmp_path, sent, change=_appending, ocel_attributes=("item_ids",)):
    """One order (order-0, 3 items) waiting at Catch_Items from 09:10, and a Picking engine sending it an
    ItemReady with each of sent's attributes, at 09:20, 09:30, ...; the report, the CSV log and the OCEL file."""
    picking = ScriptedEngine("Picking")
    for minutes, attributes in zip(range(20, 60, 10), sent):
        picking.publish_at(at(f"09:{minutes}"), "ItemReady", order_id="order-0", **attributes)
    config = SimulationConfig([ProcessSpec("Orders", ORDERS, _settings(tmp_path, change), 1, "order",
                                           ocel_attributes=ocel_attributes)], START, 1,
                              extra_processes=["Picking"])
    report = run_orchestrator(config, tmp_path / "log.csv", {"Picking": picking}, ocel_out_path=tmp_path / "log.json")
    with open(tmp_path / "log.json") as file:
        return report, pd.read_csv(tmp_path / "log.csv"), json.load(file)


def _history(report, case_id, attribute):
    return [(value, time.strftime("%H:%M")) for _, case_object in report.objects if case_object.object_id == case_id
            for name, value, time in case_object.attributes if name == attribute]


def test_a_case_that_collects_3_items_ends_with_exactly_their_ids_in_claim_order(tmp_path):
    report, _, _ = _run(tmp_path, [{"item_id": "i17"}, {"item_id": "i23"}, {"item_id": "i12"}])

    # empty when the order is created, then one value per claim; copy, beside it, keeps only the last
    assert _history(report, "order-0", "item_ids") == [
        ([], "09:00"), (["i17"], "09:20"), (["i17", "i23"], "09:30"), (["i17", "i23", "i12"], "09:40")]
    assert _history(report, "order-0", "last_item")[-1] == ("i12", "09:40")


def test_the_list_is_a_column_of_the_csv_log(tmp_path):
    _, log, _ = _run(tmp_path, [{"item_id": "i17"}, {"item_id": "i23"}, {"item_id": "i12"}])

    rows = log.set_index("activity")["item_ids"]

    # as JSON text; Pick ran before any item came, Pack after the third
    assert rows["Pick"] == "[]" and rows["Pack"] == '["i17", "i23", "i12"]'


def test_the_list_is_an_ocel_object_attribute(tmp_path):
    _, _, ocel = _run(tmp_path, [{"item_id": "i17"}, {"item_id": "i23"}, {"item_id": "i12"}])

    order = next(obj for obj in ocel["objects"] if obj["id"] == "order-0")

    # OCEL attribute values are single values, so the list is JSON text, at every change
    assert [(attribute["value"], attribute["time"][11:16]) for attribute in order["attributes"]
            if attribute["name"] == "item_ids"] == [
        ("[]", "09:00"), ('["i17"]', "09:20"), ('["i17", "i23"]', "09:30"), ('["i17", "i23", "i12"]', "09:40")]
    assert {"name": "item_ids", "type": "string"} in next(
        object_type for object_type in ocel["objectTypes"] if object_type["name"] == "order")["attributes"]


def test_a_message_without_the_attribute_adds_nothing_and_warns_once_per_event_and_attribute(tmp_path):
    report, _, _ = _run(tmp_path, [{"item_id": "i17"}, {}, {}])

    assert _history(report, "order-0", "item_ids")[-1] == (["i17"], "09:20")
    assert [(process, warning) for process, warning in report.engine_warnings if "append" in warning] == [
        ("Orders", "ItemReady message accepted at Catch_Items has no item_id to append to item_ids; nothing is added "
                   "to item_ids")]


def test_a_start_event_appends_the_message_that_starts_the_case():
    # a package, started by its first item, collecting 2 more
    engine = ProsimosEngine(ProcessSpec("Packages", PACKAGES, PACKAGES_JSON), START, seed=1)

    for minutes, item in ((0, "Warehouse-5"), (20, "Warehouse-6"), (30, "Warehouse-7")):
        now = at(f"09:{minutes:02}")
        while engine.next_event_time() is not None and engine.next_event_time() <= now:
            engine.step()
        assert engine.deliver(Message("ItemPicked", {"object_id": item}, source="Warehouse"), now) is Verdict.CLAIMED

    package = next(case_object for case_object in engine.finish().objects if case_object.object_id == "Packages-0")
    assert [value for name, value, _ in package.attributes if name == "item_ids"] == [
        ["Warehouse-5"], ["Warehouse-5", "Warehouse-6"], ["Warehouse-5", "Warehouse-6", "Warehouse-7"]]


def test_every_order_management_package_lists_the_items_it_took_in_claim_order():
    report = run_orchestrator(_config(100), None)

    packages = {case_object.object_id: [value for attribute, value, _ in case_object.attributes if attribute == "item_ids"][-1]
                for name, case_object in report.objects if name == "Packaging"}

    took = package_items(report)
    assert len(took) > 30
    assert {package: items for package, items in packages.items() if items} == took


# load errors

def _rejected(tmp_path, change, message):
    with pytest.raises(InvalidSimScenarioException, match=message):
        ProsimosEngine(ProcessSpec("Orders", ORDERS, _settings(tmp_path, change), 1), START, seed=1)


def test_the_same_attribute_in_copy_and_append_of_one_entry_is_rejected(tmp_path):
    def both(settings):
        settings["messages"]["consume"][0]["append"] = {"last_item": "item_id"}

    _rejected(tmp_path, both, r"messages.consume\[0\]: 'last_item' is in both copy and append")


def test_appending_to_an_attribute_declared_in_object_attributes_is_rejected(tmp_path):
    def drawn(settings):
        settings["messages"]["consume"][0]["append"] = {"items": "item_id"}

    _rejected(tmp_path, drawn, r"messages.consume\[0\]: append adds to 'items', which object_attributes declares")


def test_appending_to_an_attribute_another_entry_copies_into_is_rejected(tmp_path):
    with open(PACKAGES_JSON) as file:
        settings = json.load(file)
    settings["messages"]["consume"][0]["copy"] = {"item_ids": "object_id"}
    del settings["messages"]["consume"][0]["append"]
    (tmp_path / "packages.json").write_text(json.dumps(settings))

    with pytest.raises(InvalidSimScenarioException, match=r"messages.consume\[1\]: append adds to 'item_ids', which "
                                                          r"messages.consume\[0\] copies into"):
        ProsimosEngine(ProcessSpec("Packages", PACKAGES, str(tmp_path / "packages.json")), START)


@pytest.mark.parametrize("append, message", [
    (["item_ids"], "'append' must map object attribute names to message attribute names"),
    ({"item_ids": 3}, "'append' must map object attribute names to message attribute names"),
    ({"object_id": "item_id"}, "'append' can't set object_id"),
])
def test_a_malformed_append_is_rejected(tmp_path, append, message):
    def malformed(settings):
        settings["messages"]["consume"][0]["append"] = append

    _rejected(tmp_path, malformed, message)


def test_an_appended_attribute_counts_as_declared_so_messages_can_publish_it():
    assert "item_ids" in declared_attributes({"messages": {"consume": [{"append": {"item_ids": "object_id"}}]}})


def test_picking_an_attribute_the_process_appends_to_is_rejected(tmp_path):
    path = _config_with(tmp_path, catalogues={"Packaging": {"customer": {"count": 10}}}, references={
        "Packaging": [{"object_attribute": "item_ids", "catalogue": "customer", "pick": "uniform"}]})

    with pytest.raises(ValueError, match=r"processes\[2\].references\[0\] picks item_ids, which process Packaging also appends to "
                                         r"from a message \(consume entries at \['Catch_Items'\]\)"):
        SimulationConfig.from_json(path)
