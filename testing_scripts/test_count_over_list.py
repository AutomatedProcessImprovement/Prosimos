"""
count over a list (docs/messaging.md): a publish entry's {"list": ..., "as": ...} count publishes one message per
entry of a list in the publishing case, each carrying its own entry.
"""
import json
from datetime import datetime

import pytest
import pytz

from prosimos.exceptions import InvalidSimScenarioException
from prosimos.orchestrator import ProcessSpec, ProsimosEngine, SimulationConfig, run_orchestrator
from testing_scripts.scripted_engine import ScriptedEngine

ASSETS = "testing_scripts/assets/messaging"
# an item arrives every 10 minutes from 09:00; after Pick (5 min) it publishes ItemPicked with its case_id and
# waits for the Packed whose item_id is its case_id, then is shipped
ITEMS = f"{ASSETS}/packed_items.bpmn"
ITEMS_JSON = f"{ASSETS}/packed_items.json"
# an ItemPicked starts a package, which collects 2 more, appending each item's case_id to item_ids; after Pack
# (5 min) it publishes one Packed per entry of item_ids, as item_id
PACKAGES = f"{ASSETS}/packages.bpmn"
PACKAGES_JSON = f"{ASSETS}/packages.json"
START = pytz.utc.localize(datetime(2024, 1, 1, 9))


def at(hh_mm):
    hours, minutes = map(int, hh_mm.split(":"))
    return START.replace(hour=hours, minute=minutes)


def _settings(tmp_path, change):
    with open(PACKAGES_JSON) as file:
        settings = json.load(file)
    change(settings)
    path = tmp_path / "packages.json"
    path.write_text(json.dumps(settings))
    return str(path)


def _packed(report):
    return [(message.time.strftime("%H:%M"), message.attributes) for message in report.published
            if message.type == "Packed"]


def test_a_package_of_3_items_publishes_packed_to_each_and_each_item_gets_exactly_its_own():
    config = SimulationConfig([ProcessSpec("Items", ITEMS, ITEMS_JSON, 6, "items"),
                               ProcessSpec("Packages", PACKAGES, PACKAGES_JSON, None, "packages")], START, 1)

    report = run_orchestrator(config, None)

    assert _packed(report) == [
        ("09:30", {"case_id": "Packages-0", "index": 1, "item_id": "Items-0"}),
        ("09:30", {"case_id": "Packages-0", "index": 2, "item_id": "Items-1"}),
        ("09:30", {"case_id": "Packages-0", "index": 3, "item_id": "Items-2"}),
        ("10:00", {"case_id": "Packages-1", "index": 1, "item_id": "Items-3"}),
        ("10:00", {"case_id": "Packages-1", "index": 2, "item_id": "Items-4"}),
        ("10:00", {"case_id": "Packages-1", "index": 3, "item_id": "Items-5"})]
    item_of = {message.id: message.attributes["item_id"] for message in report.published if message.type == "Packed"}
    assert sorted((item_of[record.message_id], [claimer.case_id for claimer in record.claimers])
                  for record in report.message_records if record.type == "Packed") == [
        (f"Items-{item}", [f"Items-{item}"]) for item in range(6)]
    assert report.stalled == [] and report.engine_warnings == []


def _run(tmp_path, sent, change=lambda settings: None):
    """Packages, sent an ItemPicked with each of sent's attributes by a Picking engine at 09:00, 09:10, ...;
    the report."""
    picking = ScriptedEngine("Picking")
    for minutes, attributes in zip(range(0, 60, 10), sent):
        picking.publish_at(at(f"09:{minutes:02}"), "ItemPicked", **attributes)
    config = SimulationConfig([ProcessSpec("Packages", PACKAGES, _settings(tmp_path, change), None)], START, 1,
                              extra_processes=["Picking"])
    return run_orchestrator(config, None, {"Picking": picking})


def test_the_messages_follow_the_list_in_claim_order(tmp_path):
    report = _run(tmp_path, [{"case_id": "w17"}, {"case_id": "w23"}, {"case_id": "w12"}])

    assert [(attributes["index"], attributes["item_id"]) for _, attributes in _packed(report)] == [
        (1, "w17"), (2, "w23"), (3, "w12")]


def _copying_the_list(settings):
    """One ItemPicked carrying a whole list makes the package, which copies it into item_ids."""
    start, more = settings["messages"]["consume"]
    del start["append"], more["append"]
    start["copy"] = {"item_ids": "items"}
    more["collect"] = 0


def test_a_list_copied_from_a_message_is_walked_too(tmp_path):
    report = _run(tmp_path, [{"items": ["w5", "w6"]}], _copying_the_list)

    assert _packed(report) == [("09:05", {"case_id": "Packages-0", "index": 1, "item_id": "w5"}),
                               ("09:05", {"case_id": "Packages-0", "index": 2, "item_id": "w6"})]


@pytest.mark.parametrize("sent, kind", [
    ({"items": []}, "an empty list item_ids (got [])"),
    ({}, "no list item_ids (got None)"),
    ({"items": "w5"}, "no list item_ids (got 'w5')"),
])
def test_a_missing_or_empty_list_publishes_nothing_with_one_warning_per_event(tmp_path, sent, kind):
    report = _run(tmp_path, [sent, sent], _copying_the_list)

    assert _packed(report) == []
    assert [warning for _, warning in report.engine_warnings if "Throw_Packed" in warning] == [
        f"case 0 passes Throw_Packed with {kind}; it publishes no Packed message"]


# load errors

def _rejected(tmp_path, count, message, attributes=("case_id", "index")):
    def change(settings):
        settings["messages"]["publish"][0]["count"] = count
        settings["messages"]["publish"][0]["attributes"] = list(attributes)

    with pytest.raises(InvalidSimScenarioException, match=r"messages.publish\[0\]: " + message):
        ProsimosEngine(ProcessSpec("Packages", PACKAGES, _settings(tmp_path, change)), START, seed=1)


def test_a_count_over_a_list_without_as_is_rejected(tmp_path):
    _rejected(tmp_path, {"list": "item_ids"}, r"'count' over a list needs 'as'")


@pytest.mark.parametrize("name", ["case_id", "index"])
def test_as_equal_to_a_reserved_name_is_rejected(tmp_path, name):
    _rejected(tmp_path, {"list": "item_ids", "as": name}, rf"'count' as can't be {name}, which is reserved")


@pytest.mark.parametrize("count, message", [
    ({"list": "weights", "as": "item_id"}, r"'count' list 'weights' is not a declared case, global or event attribute"),
    ({"list": "item_ids", "as": ""}, r"'count' as must be a non-empty string"),
    ({"list": "item_ids", "as": 3}, r"'count' as must be a non-empty string"),
    ({"list": "item_ids", "as": "item_id", "case_attribute": "items"}, r"'count' over a list has only 'list' and 'as'"),
    ({"as": "item_id"}, r"'count' must be either"),
])
def test_an_invalid_count_over_a_list_is_rejected(tmp_path, count, message):
    _rejected(tmp_path, count, message)


def test_as_also_listed_in_attributes_is_rejected(tmp_path):
    def listed(settings):
        settings["messages"]["consume"][0]["copy"] = {"item_id": "case_id"}  # so that item_id is declared
        settings["messages"]["publish"][0]["attributes"].append("item_id")

    with pytest.raises(InvalidSimScenarioException, match=r"messages.publish\[0\]: 'item_id' is both in "
                                                          r"'attributes' and count's 'as'"):
        ProsimosEngine(ProcessSpec("Packages", PACKAGES, _settings(tmp_path, listed)), START, seed=1)
