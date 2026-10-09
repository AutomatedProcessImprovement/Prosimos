"""
Object-to-object links (docs/orchestrator.md, "OCEL output"): an o2o key on a message entry links the objects of
the cases on the two ends of each message: on a publish entry the publisher's object to each claimer's, on a
consume entry the claimer's object to the publisher's.
"""
import json
from datetime import datetime

import pytest
import pytz

from prosimos.exceptions import InvalidSimScenarioException
from prosimos.ocel_writer import OcelProcess
from prosimos.orchestrator import CaseElement, MessageRecord, ProcessSpec, ProsimosEngine, _object_links, run_orchestrator
from testing_scripts.test_message_records import _verdicts_run
from testing_scripts.test_ocel_links import _items, _items_of_orders, _items_of_packages
from testing_scripts.test_order_management import _config, _PackageContents

ASSETS = "testing_scripts/assets/messaging"
START = pytz.utc.localize(datetime(2024, 1, 1, 9))


@pytest.fixture(scope="module")
def order_management(tmp_path_factory):
    """A 100-order run of the Order Management example written as OCEL, and what each package took."""
    monkeypatch = pytest.MonkeyPatch()
    contents = _PackageContents(monkeypatch)
    path = tmp_path_factory.mktemp("object_links") / "log.json"
    try:
        run_orchestrator(_config(100), None, ocel_out_path=path)
    finally:
        monkeypatch.undo()
    with open(path) as file:
        ocel = json.load(file)
    return ocel, contents.items, path


def _related(ocel, object_type):
    """Object id -> [(object id, qualifier)] of every object of the type."""
    return {obj["id"]: [(link["objectId"], link["qualifier"]) for link in obj["relationships"]]
            for obj in ocel["objects"] if obj["type"] == object_type}


def test_every_order_comprises_exactly_its_items(order_management):
    # publish side: Sales's ItemOrdered entry has "o2o": "comprises", from the order to each item it started
    ocel, _, _ = order_management
    items_of_order = _items_of_orders(ocel)

    orders = _related(ocel, "orders")
    assert len(orders) == 100
    for order, links in orders.items():
        assert sorted(links) == sorted((item, "comprises") for item in items_of_order[order])
    assert sum(len(links) for links in orders.values()) > 300


def test_every_package_contains_exactly_its_items(order_management):
    # consume side: Packaging's ItemPicked entries, at the start event (its first item) and at the catch event
    # (the rest), have "o2o": "contains", from the package to each item whose message it claimed
    ocel, contents, _ = order_management
    items_of_package = _items_of_packages(ocel, contents)

    packages = _related(ocel, "packages")
    assert set(packages) == set(items_of_package)
    for package, links in packages.items():
        assert sorted(links) == sorted((item, "contains") for item in items_of_package[package])
    assert any(len(links) > 1 for links in packages.values())


def test_a_link_goes_only_in_its_entrys_direction_and_no_key_gives_none(order_management):
    # items are on the other end of both links, and Warehouse's own entries have no o2o key
    ocel, _, _ = order_management

    assert _items(ocel)
    assert all(links == [] for links in _related(ocel, "items").values())


def test_a_run_without_o2o_keys_links_no_objects():
    # the verdict models pass ItemOrdered, ItemReady and Shipped between orders and items, with no o2o key
    report, _ = _verdicts_run()

    assert report.message_records
    assert _object_links(report.message_records, {"Sales": OcelProcess("orders"), "Picking": OcelProcess("items")}) == {}


def _record(publisher_o2o=None, claimer_o2o=None, claimer_process="P"):
    return MessageRecord("m1", "Go", CaseElement("S", "S-0", "Throw_Go", o2o=publisher_o2o),
                         [CaseElement(claimer_process, f"{claimer_process}-0", "Catch_Go", o2o=claimer_o2o),
                          CaseElement(claimer_process, f"{claimer_process}-1", "Catch_Go", o2o=claimer_o2o)])


PROCESSES = {"S": OcelProcess("s"), "P": OcelProcess("p"), "Q": OcelProcess(None)}


def test_a_publish_entrys_o2o_links_the_publisher_to_each_claimer():
    assert _object_links([_record(publisher_o2o="sends")], PROCESSES) == {"S-0": [("P-0", "sends"), ("P-1", "sends")]}


def test_a_consume_entrys_o2o_links_each_claimer_to_the_publisher():
    assert _object_links([_record(claimer_o2o="from")], PROCESSES) == {"P-0": [("S-0", "from")], "P-1": [("S-0", "from")]}


def test_a_link_is_written_once_however_many_messages_give_it():
    assert _object_links([_record(publisher_o2o="sends"), _record(publisher_o2o="sends")], PROCESSES) == {
        "S-0": [("P-0", "sends"), ("P-1", "sends")]}


def test_no_link_to_or_from_a_process_without_objects():
    assert _object_links([_record("sends", "from", claimer_process="Q")], PROCESSES) == {}


def test_pm4py_reads_the_object_links(order_management):
    pm4py = pytest.importorskip("pm4py")
    ocel, _, path = order_management

    read = pm4py.read_ocel2_json(str(path))

    assert len(read.o2o) == sum(len(obj["relationships"]) for obj in ocel["objects"])
    assert set(read.o2o["ocel:qualifier"]) == {"comprises", "contains"}


def test_an_o2o_key_must_be_a_non_empty_string(tmp_path):
    with open(f"{ASSETS}/orders_and_trucks.json") as file:
        settings = json.load(file)
    settings["messages"]["consume"][0]["o2o"] = ""
    (tmp_path / "orders.json").write_text(json.dumps(settings))
    spec = ProcessSpec("Orders", f"{ASSETS}/orders_and_trucks.bpmn", str(tmp_path / "orders.json"), 1)

    with pytest.raises(InvalidSimScenarioException, match=r"messages.consume\[0\]: 'o2o' must be a non-empty string"):
        ProsimosEngine(spec, START, None, seed=1)
