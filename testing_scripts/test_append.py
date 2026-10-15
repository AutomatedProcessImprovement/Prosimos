"""
append (docs/messaging.md): a consume entry's append adds the value of every message it claims to a list in the
case, which starts empty, where copy would overwrite it.
"""
import json

import pandas as pd
import pytest

from prosimos.exceptions import InvalidSimScenarioException
from prosimos.messaging_parser import declared_attributes
from prosimos.orchestrator import Message, ProcessSpec, ProsimosEngine, SimulationConfig, Verdict, run_orchestrator
from testing_scripts.scripted_engine import ScriptedEngine
from testing_scripts.test_catalogue import _config_with
from testing_scripts.test_collect import ORDERS, START, _settings, at
from testing_scripts.test_order_management import _config, _PackageContents

OM = "testing_scripts/assets/order_management"


def _appending(settings):
    settings["messages"]["consume"][0]["append"] = {"item_ids": "item_id"}


def _run(tmp_path, sent, change=_appending):
    """One order (Orders-0, 3 items) waiting at Catch_Items from 09:10, and a Picking engine sending it an
    ItemReady with each of sent's attributes, at 09:20, 09:30, ...; the report, the CSV log and the OCEL file."""
    picking = ScriptedEngine("Picking")
    for minutes, attributes in zip(range(20, 60, 10), sent):
        picking.publish_at(at(f"09:{minutes}"), "ItemReady", order_id="Orders-0", **attributes)
    config = SimulationConfig([ProcessSpec("Orders", ORDERS, _settings(tmp_path, change), 1, "orders")], START, 1,
                              extra_processes=["Picking"])
    report = run_orchestrator(config, tmp_path / "log.csv", {"Picking": picking}, ocel_out_path=tmp_path / "log.json")
    with open(tmp_path / "log.json") as file:
        return report, pd.read_csv(tmp_path / "log.csv"), json.load(file)


def _history(report, case_id, attribute):
    return [(value, time.strftime("%H:%M")) for _, case_object in report.objects if case_object.case_id == case_id
            for name, value, time in case_object.attributes if name == attribute]


def test_a_case_that_collects_3_items_ends_with_exactly_their_ids_in_claim_order(tmp_path):
    report, _, _ = _run(tmp_path, [{"item_id": "i17"}, {"item_id": "i23"}, {"item_id": "i12"}])

    # empty when the order is created, then one value per claim; copy, beside it, keeps only the last
    assert _history(report, "Orders-0", "item_ids") == [
        ([], "09:00"), (["i17"], "09:20"), (["i17", "i23"], "09:30"), (["i17", "i23", "i12"], "09:40")]
    assert _history(report, "Orders-0", "last_item")[-1] == ("i12", "09:40")


def test_the_list_is_a_column_of_the_csv_log(tmp_path):
    _, log, _ = _run(tmp_path, [{"item_id": "i17"}, {"item_id": "i23"}, {"item_id": "i12"}])

    rows = log.set_index("activity")["item_ids"]

    # as JSON text; Pick ran before any item came, Pack after the third
    assert rows["Pick"] == "[]" and rows["Pack"] == '["i17", "i23", "i12"]'


def test_the_list_is_an_ocel_object_attribute(tmp_path):
    _, _, ocel = _run(tmp_path, [{"item_id": "i17"}, {"item_id": "i23"}, {"item_id": "i12"}])

    order = next(obj for obj in ocel["objects"] if obj["id"] == "Orders-0")

    # OCEL attribute values are single values, so the list is JSON text, at every change
    assert [(attribute["value"], attribute["time"][11:16]) for attribute in order["attributes"]
            if attribute["name"] == "item_ids"] == [
        ("[]", "09:00"), ('["i17"]', "09:20"), ('["i17", "i23"]', "09:30"), ('["i17", "i23", "i12"]', "09:40")]
    assert {"name": "item_ids", "type": "string"} in next(
        object_type for object_type in ocel["objectTypes"] if object_type["name"] == "orders")["attributes"]


def test_a_message_without_the_attribute_adds_nothing_and_warns_once_per_event_and_attribute(tmp_path):
    report, _, _ = _run(tmp_path, [{"item_id": "i17"}, {}, {}])

    assert _history(report, "Orders-0", "item_ids")[-1] == (["i17"], "09:20")
    assert [(process, warning) for process, warning in report.engine_warnings if "append" in warning] == [
        ("Orders", "ItemReady message accepted at Catch_Items has no item_id to append to item_ids; nothing is added "
                   "to item_ids")]


def test_a_start_event_appends_the_message_that_starts_the_case(tmp_path):
    # an Order Management package, started by its first item, collecting 2 more of the same customer
    with open(f"{OM}/packaging.json") as file:
        settings = json.load(file)
    more_items = next(attribute for attribute in settings["case_attributes"] if attribute["name"] == "more_items")
    more_items["values"] = [{"key": 2, "value": 1}]
    (tmp_path / "packaging.json").write_text(json.dumps(settings))
    engine = ProsimosEngine(ProcessSpec("Packaging", f"{OM}/packaging.bpmn", str(tmp_path / "packaging.json")), START,
                            seed=1)

    for minutes, item in ((0, "Warehouse-5"), (20, "Warehouse-6"), (30, "Warehouse-7")):
        now = at(f"09:{minutes:02}")
        while engine.next_event_time() is not None and engine.next_event_time() <= now:
            engine.step()
        message = Message("ItemPicked", {"order_id": "Sales-1", "item_index": minutes, "customer": "c1", "case_id": item},
                          source="Warehouse")
        assert engine.deliver(message, now) is Verdict.CLAIMED

    package = next(case_object for case_object in engine.finish().objects if case_object.case_id == "Packaging-0")
    assert [value for name, value, _ in package.attributes if name == "item_ids"] == [
        ["Warehouse-5"], ["Warehouse-5", "Warehouse-6"], ["Warehouse-5", "Warehouse-6", "Warehouse-7"]]


def test_every_order_management_package_lists_the_items_it_took_in_claim_order(tmp_path):
    monkeypatch = pytest.MonkeyPatch()
    contents = _PackageContents(monkeypatch)
    try:
        report = run_orchestrator(_config(100), None)
    finally:
        monkeypatch.undo()
    item_of = {(values["order_id"], values["item_index"]): case_object.case_id for name, case_object in report.objects
               if name == "Warehouse" for values in [{attribute: value for attribute, value, _ in case_object.attributes}]}

    packages = {case_object.case_id: [value for attribute, value, _ in case_object.attributes if attribute == "item_ids"][-1]
                for name, case_object in report.objects if name == "Packaging"}

    assert len(packages) > 30
    for package, took in contents.items.items():
        assert packages[f"Packaging-{package}"] == [item_of[(order_id, index)] for order_id, index, _ in took]


# load errors

def _rejected(tmp_path, change, message):
    with pytest.raises(InvalidSimScenarioException, match=message):
        ProsimosEngine(ProcessSpec("Orders", ORDERS, _settings(tmp_path, change), 1), START, seed=1)


def test_the_same_attribute_in_copy_and_append_of_one_entry_is_rejected(tmp_path):
    def both(settings):
        settings["messages"]["consume"][0]["append"] = {"last_item": "item_id"}

    _rejected(tmp_path, both, r"messages.consume\[0\]: 'last_item' is in both copy and append")


def test_appending_to_an_attribute_declared_in_case_attributes_is_rejected(tmp_path):
    def drawn(settings):
        settings["messages"]["consume"][0]["append"] = {"items": "item_id"}

    _rejected(tmp_path, drawn, r"messages.consume\[0\]: append adds to 'items', which case_attributes declares")


def test_appending_to_an_attribute_another_entry_copies_into_is_rejected(tmp_path):
    with open(f"{OM}/packaging.json") as file:
        settings = json.load(file)
    settings["messages"]["consume"][0]["copy"]["item_ids"] = "case_id"
    del settings["messages"]["consume"][0]["append"]
    (tmp_path / "packaging.json").write_text(json.dumps(settings))

    with pytest.raises(InvalidSimScenarioException, match=r"messages.consume\[1\]: append adds to 'item_ids', which "
                                                          r"messages.consume\[0\] copies into"):
        ProsimosEngine(ProcessSpec("Packaging", f"{OM}/packaging.bpmn", str(tmp_path / "packaging.json")), START)


@pytest.mark.parametrize("append, message", [
    (["item_ids"], "'append' must map case attribute names to message attribute names"),
    ({"item_ids": 3}, "'append' must map case attribute names to message attribute names"),
    ({"case_id": "item_id"}, "'append' can't set case_id"),
])
def test_a_malformed_append_is_rejected(tmp_path, append, message):
    def malformed(settings):
        settings["messages"]["consume"][0]["append"] = append

    _rejected(tmp_path, malformed, message)


def test_an_appended_attribute_counts_as_declared_so_messages_can_publish_it():
    assert "item_ids" in declared_attributes({"messages": {"consume": [{"append": {"item_ids": "case_id"}}]}})


def test_picking_an_attribute_the_process_appends_to_is_rejected(tmp_path):
    path = _config_with(tmp_path, catalogue={"customers": {"count": 10}}, references=[
        {"process": "Packaging", "case_attribute": "item_ids", "catalogue": "customers", "pick": "uniform"}])

    with pytest.raises(ValueError, match=r"references\[0\] picks item_ids, which process Packaging also appends to "
                                         r"from a message \(consume entries at \['Start_Package', 'Catch_More_Items'\]\)"):
        SimulationConfig.from_json(path)
