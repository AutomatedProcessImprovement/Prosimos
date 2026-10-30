"""
The catalogue of objects with no events of their own, and the references that link to it (docs/catalogue.md):
a reference says a case attribute's value is the id of a catalogue object, or a list of such ids, each linked as a
single value is, and the OCEL output links the case's object and the events named in link_events to it, and, with
tag_along, every event of another case that links the case's object.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import pytz

from prosimos.orchestration.catalogue import CatalogueObject, Reference
from prosimos.orchestration.config import ProcessSpec, SimulationConfig
from prosimos.orchestration.ocel_writer import OcelProcess, write_ocel
from prosimos.orchestration.output import _reference_links, _tag_along
from prosimos.orchestration.protocol import CaseObject
from prosimos.orchestration.simulation import run_multiprocess_simulation
from testing_scripts.multiprocess.examples.test_order_management import CONFIG, _config
from testing_scripts.multiprocess.messaging.test_collect import ORDERS, _settings, at
from testing_scripts.multiprocess.orchestrator_settings import one_organization
from testing_scripts.multiprocess.scripted_engine import ScriptedEngine

START = pytz.utc.localize(datetime(2024, 1, 1, 9))


@pytest.fixture(scope="module")
def order_management(tmp_path_factory):
    """A 100-order run of the Order Management example written as OCEL, and its report."""
    path = tmp_path_factory.mktemp("catalogue") / "log.json"
    report = run_multiprocess_simulation(_config(100), None, ocel_out_path=path)
    with open(path) as file:
        return json.load(file), report, path


def _of_type(ocel, object_type):
    return {obj["id"]: obj for obj in ocel["objects"] if obj["type"] == object_type}


def _customers_of_orders(report):
    """Order id -> its customer, from the run report: the OCEL file doesn't write the attribute."""
    return {case_object.object_id: value for process, case_object in report.objects if process == "Sales"
            for attribute, value, _ in case_object.attributes if attribute == "customer"}


def test_the_run_writes_every_customer_of_the_catalogue(order_management):
    ocel, _, _ = order_management

    customers = _of_type(ocel, "customer")

    assert sorted(customers) == sorted(f"c{n}" for n in range(1, 16))
    # without attributes
    assert all(customer["attributes"] == [] for customer in customers.values())
    assert {"name": "customer", "attributes": []} in ocel["objectTypes"]


def test_every_customer_places_exactly_its_orders(order_management):
    # "reverse": true turns the link around: from the customer to the order
    ocel, report, _ = order_management
    orders = _of_type(ocel, "order")
    customer_of = _customers_of_orders(report)

    for customer_id, customer in _of_type(ocel, "customer").items():
        assert sorted(link["objectId"] for link in customer["relationships"]) == sorted(
            order_id for order_id in orders if customer_of[order_id] == customer_id)
        assert {link["qualifier"] for link in customer["relationships"]} <= {"places"}
    assert sum(len(customer["relationships"]) for customer in _of_type(ocel, "customer").values()) == 100
    assert not any(link["objectId"].startswith("c") for order in orders.values() for link in order["relationships"])


def test_place_and_confirm_order_link_their_customer_and_pay_order_does_not(order_management):
    ocel, report, _ = order_management
    customer_of = _customers_of_orders(report)

    def customers_of(event):
        return [link for link in event["relationships"] if link["objectId"] in _of_type(ocel, "customer")]

    linked = {"place order": 0, "confirm order": 0}
    for event in ocel["events"]:
        if event["type"] in linked:
            order = next(link["objectId"] for link in event["relationships"] if link["qualifier"] == "order")
            assert customers_of(event) == [{"objectId": customer_of[order], "qualifier": "customer"}]
            linked[event["type"]] += 1
        elif event["type"] == "pay order":
            assert customers_of(event) == []
    assert linked == {"place order": 100, "confirm order": 100}


def test_pm4py_reads_the_customers(order_management):
    pm4py = pytest.importorskip("pm4py")
    _, _, path = order_management

    read = pm4py.read_ocel2_json(str(path))

    assert (read.objects["ocel:type"] == "customer").sum() == 15
    assert (read.o2o["ocel:qualifier"] == "places").sum() == 100
    assert (read.relations["ocel:qualifier"] == "customer").sum() == 200


# the join on its own, with one process P, the Sales model, whose cases are objects of type p

SALES = str(Path(CONFIG).parent / "sales.bpmn")


def _links(references, objects, rows, processes=None, catalogues=None):
    config = SimulationConfig(one_organization([ProcessSpec(
        "P", SALES, "p.json", 1, catalogues=catalogues or {"customer": [CatalogueObject("c1"), CatalogueObject("c2")]},
        references=tuple(references))]), START)
    warnings = []
    static_objects, object_links, event_links = _reference_links(
        config, config.catalogue, processes or {"P": OcelProcess("p")}, objects, rows, {"P": ["Place_Order", "Confirm_Order"]},
        warnings.append)
    return static_objects, object_links, event_links, warnings


def _row(case, activity, minutes):
    return "P", {"case_id": case, "activity": activity, "end_time": START + timedelta(minutes=minutes)}


ONE_CASE = [("P", CaseObject("p-0", START, [("customer", "c1", START)]))]
TWO_ROWS = [_row(0, "place order", 10), _row(0, "confirm order", 20)]


def test_without_link_events_every_event_of_the_case_links_its_object():
    _, _, event_links, _ = _links([Reference("customer", "customer")], ONE_CASE, TWO_ROWS)

    assert event_links == {("P", 0): [("c1", "customer")], ("P", 1): [("c1", "customer")]}


def test_link_events_may_name_a_task_by_its_id():
    _, _, event_links, _ = _links([Reference("customer", "customer", link_events=("Confirm_Order",))], ONE_CASE,
                                  TWO_ROWS)

    assert event_links == {("P", 1): [("c1", "customer")]}


def test_without_reverse_the_link_goes_from_the_case_object_to_the_catalogue_object():
    _, object_links, _, _ = _links([Reference("customer", "customer", o2o="placed by", link_events=())],
                                   ONE_CASE, TWO_ROWS)

    assert object_links == {("P", "p-0"): [("c1", "placed by")]}


def test_an_event_links_the_value_the_attribute_had_when_it_completed():
    # the customer changes from c1 to c2 between the two events; the o2o link goes to both
    objects = [("P", CaseObject("p-0", START, [("customer", "c1", START),
                                               ("customer", "c2", START + timedelta(minutes=15))]))]

    _, object_links, event_links, _ = _links([Reference("customer", "customer", o2o="placed by")], objects, TWO_ROWS)

    assert event_links == {("P", 0): [("c1", "customer")], ("P", 1): [("c2", "customer")]}
    assert object_links == {("P", "p-0"): [("c1", "placed by"), ("c2", "placed by")]}


def test_a_value_not_in_the_catalogue_gets_no_link_and_one_warning_per_attribute_and_value():
    objects = [("P", CaseObject(f"p-{case}", START, [("customer", "c9", START)])) for case in range(2)]
    rows = [_row(0, "place order", 10), _row(1, "place order", 10)]

    _, object_links, event_links, warnings = _links([Reference("customer", "customer", o2o="placed by")],
                                                    objects, rows)

    assert object_links == {} and event_links == {}
    assert warnings == ["object attribute customer has the value 'c9', which is not in catalogue customer; "
                        "no OCEL link is written for it"]


def test_an_empty_link_events_list_and_no_o2o_link_nothing_but_the_catalogue_is_still_written():
    static_objects, object_links, event_links, warnings = _links(
        [Reference("customer", "customer", link_events=())], ONE_CASE, TWO_ROWS)

    assert object_links == {} and event_links == {} and warnings == []
    assert [object_id for _, object_id, _ in static_objects] == ["c1", "c2"]


def test_a_process_without_objects_links_nothing_but_the_catalogue_is_still_written():
    static_objects, object_links, event_links, _ = _links(
        [Reference("customer", "customer", o2o="placed by")], ONE_CASE, TWO_ROWS, processes={"P": OcelProcess(None)},
        catalogues={"customer": [CatalogueObject("c1", (("segment", "retail"),))]})

    assert object_links == {} and event_links == {}
    assert static_objects == [("customer", "c1", [("segment", "retail", START)])]


# a list of ids: each entry links as a single value does

THREE = {"customer": [CatalogueObject("c1"), CatalogueObject("c2"), CatalogueObject("c3")]}


def test_a_list_links_each_of_its_ids_on_the_events_and_between_the_objects():
    objects = [("P", CaseObject("p-0", START, [("customer", ["c1", "c2", "c3"], START)]))]

    _, object_links, event_links, warnings = _links([Reference("customer", "customer", o2o="placed by")], objects,
                                                    TWO_ROWS, catalogues=THREE)

    assert event_links == {("P", n): [("c1", "customer"), ("c2", "customer"), ("c3", "customer")] for n in (0, 1)}
    assert object_links == {("P", "p-0"): [("c1", "placed by"), ("c2", "placed by"), ("c3", "placed by")]}
    assert warnings == []


def test_an_id_a_list_repeats_is_linked_once_per_event_and_once_between_the_objects():
    objects = [("P", CaseObject("p-0", START, [("customer", ["c2", "c1", "c2"], START)]))]

    _, object_links, event_links, _ = _links([Reference("customer", "customer", o2o="placed by", reverse=True)],
                                             objects, TWO_ROWS, catalogues=THREE)

    assert event_links == {("P", n): [("c2", "customer"), ("c1", "customer")] for n in (0, 1)}
    assert object_links == {"c2": [(("P", "p-0"), "placed by")], "c1": [(("P", "p-0"), "placed by")]}


def test_an_event_links_the_ids_of_the_list_the_attribute_held_when_it_completed():
    # the list grows from [c1] to [c1, c2] between the two events, as append grows it; an empty list links nothing
    objects = [("P", CaseObject("p-0", START, [("customer", [], START), ("customer", ["c1"], START + timedelta(minutes=5)),
                                               ("customer", ["c1", "c2"], START + timedelta(minutes=15))]))]

    _, object_links, event_links, _ = _links([Reference("customer", "customer", o2o="placed by")], objects,
                                             [_row(0, "place order", 2), *TWO_ROWS], catalogues=THREE)

    assert event_links == {("P", 1): [("c1", "customer")], ("P", 2): [("c1", "customer"), ("c2", "customer")]}
    assert object_links == {("P", "p-0"): [("c1", "placed by"), ("c2", "placed by")]}


def test_an_id_of_a_list_not_in_the_catalogue_gets_no_link_and_one_warning_and_the_others_are_linked():
    objects = [("P", CaseObject("p-0", START, [("customer", ["c1", "c9"], START)]))]

    _, object_links, event_links, warnings = _links([Reference("customer", "customer", o2o="placed by")], objects,
                                                    TWO_ROWS)

    assert event_links == {("P", n): [("c1", "customer")] for n in (0, 1)}
    assert object_links == {("P", "p-0"): [("c1", "placed by")]}
    assert warnings == ["object attribute customer has the value 'c9', which is not in catalogue customer; "
                        "no OCEL link is written for it"]


def test_a_list_appended_from_messages_links_each_item_it_collected(tmp_path):
    # an order collects three ItemReady messages and appends their item ids to item_ids, which a reference ties to
    # a catalogue of items: Pick runs before any item came, Pack after the third
    def appending(settings):
        settings["messages"]["consume"][0]["append"] = {"item_ids": "item_id"}

    picking = ScriptedEngine("Picking")
    for minutes, item in zip((20, 30, 40), ("i17", "i23", "i12")):
        picking.publish_at(at(f"09:{minutes}"), "ItemReady", order_id="order-0", item_id=item)
    config = SimulationConfig(one_organization([ProcessSpec(
        "Orders", ORDERS, _settings(tmp_path, appending), 1, "order",
        catalogues={"item": [CatalogueObject(item) for item in ("i12", "i17", "i23", "i99")]},
        references=(Reference("item_ids", "item", o2o="contains"),))]), START, 1, extra_processes=["Picking"])
    run_multiprocess_simulation(config, None, {"Picking": picking}, ocel_out_path=tmp_path / "log.json")
    with open(tmp_path / "log.json") as file:
        ocel = json.load(file)

    order = next(obj for obj in ocel["objects"] if obj["id"] == "order-0")
    assert [(link["objectId"], link["qualifier"]) for link in order["relationships"]] == [
        ("i17", "contains"), ("i23", "contains"), ("i12", "contains")]
    linked = {event["type"]: [link["objectId"] for link in event["relationships"] if link["qualifier"] == "item_ids"]
              for event in ocel["events"]}
    assert linked == {"Pick": [], "Pack": ["i17", "i23", "i12"]}


def _instant_row(process, activity, at):
    """A logged task of the process's case 0, enabled, started and completed at at."""
    return process, {"case_id": 0, "activity": activity, "resource": "R", "enable_time": at, "start_time": at,
                     "end_time": at}


# tag_along: P's object p-0 names a product (tag_along, none of its own events) and a colour (every own event);
# Q's case q-0 links p-0 through a message at its first event and, with carry_links, at its second

def test_with_tag_along_every_event_of_another_case_that_links_the_object_links_its_catalogue_object(tmp_path):
    config = SimulationConfig(one_organization([
        ProcessSpec("P", SALES, "p.json", 1,
                    catalogues={"product": [CatalogueObject("phone"), CatalogueObject("tablet")],
                                                 "colour": [CatalogueObject("red")]},
                    references=(Reference("product", "product", link_events=(), tag_along=True),
                                                 Reference("colour", "colour"))),
        ProcessSpec("Q", SALES, "q.json", 1)]), START)
    processes = {"P": OcelProcess("p"), "Q": OcelProcess("q", carry_links=True)}
    # the product changes from phone to tablet between Q's two events
    objects = [("P", CaseObject("p-0", START, [("product", "phone", START), ("colour", "red", START),
                                               ("product", "tablet", START + timedelta(minutes=15))])),
               ("Q", CaseObject("q-0", START))]
    rows = [_instant_row(process, activity, START + timedelta(minutes=end))
            for process, activity, end in (("P", "place order", 5), ("Q", "a", 10), ("Q", "b", 20))]
    _, _, event_links = _reference_links(config, config.catalogue, processes, objects, rows, {}, [].append)

    write_ocel(tmp_path / "log.json", processes, objects, rows, None, {("Q", 0): [(("P", "p-0"), "item")]}, None,
               None, event_links, _tag_along(config, config.catalogue, objects))
    with open(tmp_path / "log.json") as file:
        events = [[(link["objectId"], link["qualifier"]) for link in event["relationships"]]
                  for event in json.load(file)["events"]]

    assert events == [
        [("p-0", "p"), ("red", "colour")],  # its own event: link_events decides, not tag_along
        [("q-0", "q"), ("p-0", "item"), ("phone", "product")],  # the product p-0 named when the event completed
        [("q-0", "q"), ("p-0", "item"), ("tablet", "product")]]  # no colour: its reference has no tag_along


def test_with_tag_along_an_event_links_each_id_of_the_list_the_object_held_once(tmp_path):
    config = SimulationConfig(one_organization([
        ProcessSpec("P", SALES, "p.json", 1,
                    catalogues={"product": [CatalogueObject("phone"), CatalogueObject("tablet")]},
                    references=(Reference("products", "product", link_events=(), tag_along=True),)),
        ProcessSpec("Q", SALES, "q.json", 1)]), START)
    processes = {"P": OcelProcess("p"), "Q": OcelProcess("q")}
    objects = [("P", CaseObject("p-0", START, [("products", ["phone", "tablet", "phone"], START)])),
               ("Q", CaseObject("q-0", START))]
    rows = [_instant_row(process, activity, START + timedelta(minutes=end))
            for process, activity, end in (("P", "place order", 5), ("Q", "a", 10))]

    write_ocel(tmp_path / "log.json", processes, objects, rows, None, {("Q", 0): [(("P", "p-0"), "item")]}, None,
               None, None, _tag_along(config, config.catalogue, objects))
    with open(tmp_path / "log.json") as file:
        events = [[(link["objectId"], link["qualifier"]) for link in event["relationships"]]
                  for event in json.load(file)["events"]]

    assert events[1] == [("q-0", "q"), ("p-0", "item"), ("phone", "products"), ("tablet", "products")]


# load errors

def _config_with(tmp_path, catalogues=None, references=None, **changes):
    """The Order Management configuration with the given top-level changes. catalogues ({process: {name: objects}})
    and references ({process: [reference, ...]}), when given, replace those of every process: a process they don't
    name has none. Given references also leave out every process's ocel_attributes, which name attributes the
    committed references bring."""
    with open(CONFIG) as file:
        data = json.load(file)
    data.update(changes)
    for process in data["organizations"][0]["processes"]:  # the paths are relative to the configuration's folder
        process["bpmn_path"], process["json_path"] = (str(Path(CONFIG).resolve().parent / process[key]) for key in ("bpmn_path", "json_path"))
        for key, given in (("catalogues", catalogues), ("references", references)):
            if given is not None:
                process.pop(key, None)
                if process["name"] in given:
                    process[key] = given[process["name"]]
        if references is not None:
            process.pop("ocel_attributes", None)  # they name attributes the references bring
    path = tmp_path / "simulation.json"
    path.write_text(json.dumps(data))
    return path


def test_a_reference_to_an_unknown_catalogue_is_rejected(tmp_path):
    path = _config_with(tmp_path, references={"Sales": [{"object_attribute": "customer", "catalogue": "clients"}]})

    with pytest.raises(ValueError, match=r"processes\[0\].references\[0\] names unknown catalogue 'clients'"):
        SimulationConfig.from_json(path)


def test_a_reference_to_another_processs_catalogue_is_rejected(tmp_path):
    # the items know their order's customer only from ItemOrdered, which copies it under any name they choose
    path = _config_with(tmp_path, references={"Warehouse": [{"object_attribute": "customer", "catalogue": "customer"}]})

    with pytest.raises(ValueError, match=r"processes\[1\].references\[0\] names catalogue 'customer' of process Sales; "
                                         r"a reference names a catalogue of its own process"):
        SimulationConfig.from_json(path)


def test_a_catalogue_declared_by_two_processes_is_rejected(tmp_path):
    # its name is an object type of the OCEL output
    path = _config_with(tmp_path, catalogues={"Sales": {"customer": [{"id": "c1"}]},
                                              "Warehouse": {"customer": [{"id": "c2"}]}}, references={})

    with pytest.raises(ValueError, match=r"catalogue 'customer' is declared by more than one process: Sales, Warehouse"):
        SimulationConfig.from_json(path)


def test_duplicate_ids_are_rejected_across_all_catalogues(tmp_path):
    path = _config_with(tmp_path, catalogues={"Sales": {"customer": [{"id": "c1"}, {"id": "x"}]},
                                              "Warehouse": {"product": [{"id": "x"}]}}, references={})

    with pytest.raises(ValueError, match=r"catalogue ids are repeated: \['x'\]"):
        SimulationConfig.from_json(path)


def test_a_catalogue_object_without_an_id_is_rejected(tmp_path):
    path = _config_with(tmp_path, catalogues={"Sales": {"customer": [{"attributes": {"segment": "retail"}}]}}, references={})

    with pytest.raises(ValueError, match=r"processes\[0\].catalogues.customer\[0\]: 'id' must be a non-empty string"):
        SimulationConfig.from_json(path)


def test_a_reference_missing_a_key_is_rejected(tmp_path):
    path = _config_with(tmp_path, references={"Sales": [{"catalogue": "customer"}]})

    with pytest.raises(ValueError, match=r"processes\[0\].references\[0\] is missing \['object_attribute'\]"):
        SimulationConfig.from_json(path)


def test_a_link_events_name_that_is_no_task_of_the_process_is_rejected(tmp_path):
    # a typo would otherwise silently link nothing
    path = _config_with(tmp_path, references={"Sales": [{"object_attribute": "customer", "catalogue": "customer",
                                                         "link_events": ["place order", "confrim order", "Throw_ItemOrdered"]}]})

    with pytest.raises(ValueError, match=r"processes\[0\].references\[0\] link_events names \['confrim order', 'Throw_ItemOrdered'\], "
                                         r"which are no tasks of process Sales"):
        SimulationConfig.from_json(path)


def test_catalogue_ids_that_look_like_object_ids_are_rejected(tmp_path):
    # objects of the processes are named <object type>-<number>, e.g. order-1
    path = _config_with(tmp_path, catalogues={"Sales": {"customer": [{"id": "order-1"}, {"id": "item-12"}, {"id": "order-x"},
                                                                    {"id": "Sales-1"}]}},
                        references={})

    with pytest.raises(ValueError, match=r"catalogue ids look like object ids \(<object type>-<number>\): "
                                         r"\['item-12', 'order-1'\]"):
        SimulationConfig.from_json(path)


def test_a_catalogue_named_like_a_processs_object_type_is_rejected(tmp_path):
    path = _config_with(tmp_path, catalogues={"Sales": {"order": [{"id": "o1"}]}}, references={})

    with pytest.raises(ValueError, match=r"catalogue 'order' is the object type of process Sales"):
        SimulationConfig.from_json(path)
