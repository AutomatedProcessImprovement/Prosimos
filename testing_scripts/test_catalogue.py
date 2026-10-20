"""
The catalogue of objects with no events of their own, and the references that link to it (docs/catalogue.md):
a reference says a case attribute's value is the id of a catalogue object, and the OCEL output links the case's
object and the events named in link_events to it.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import pytz

from prosimos.ocel_writer import OcelProcess
from prosimos.orchestrator import (CaseObject, CatalogueObject, ProcessSpec, Reference, SimulationConfig,
                                   _reference_links, run_orchestrator)
from testing_scripts.test_order_management import CONFIG, _config

START = pytz.utc.localize(datetime(2024, 1, 1, 9))


@pytest.fixture(scope="module")
def order_management(tmp_path_factory):
    """A 100-order run of the Order Management example written as OCEL, and its report."""
    path = tmp_path_factory.mktemp("catalogue") / "log.json"
    report = run_orchestrator(_config(100), None, ocel_out_path=path)
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
    assert customers["c3"]["attributes"] == [{"name": "segment", "value": "business", "time": "2023-04-03T06:00:00.000000Z"}]
    assert {"name": "customer", "attributes": [{"name": "segment", "type": "string"}]} in ocel["objectTypes"]


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
    config = SimulationConfig([ProcessSpec(
        "P", SALES, "p.json", 1, catalogues=catalogues or {"customer": [CatalogueObject("c1"), CatalogueObject("c2")]},
        references=tuple(references))], START)
    warnings = []
    static_objects, object_links, event_links = _reference_links(
        config, config.catalogue, processes or {"P": OcelProcess("p")}, objects, rows, {"P": ["Place_Order", "Confirm_Order"]},
        warnings.append)
    return static_objects, object_links, event_links, warnings


def _row(case, activity, minutes):
    return "P", {"case_id": case, "activity": activity, "end_time": START + timedelta(minutes=minutes)}


ONE_CASE = [("P", CaseObject("p-0", [("customer", "c1", START)]))]
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

    assert object_links == {"p-0": [("c1", "placed by")]}


def test_an_event_links_the_value_the_attribute_had_when_it_completed():
    # the customer changes from c1 to c2 between the two events; the o2o link goes to both
    objects = [("P", CaseObject("p-0", [("customer", "c1", START), ("customer", "c2", START + timedelta(minutes=15))]))]

    _, object_links, event_links, _ = _links([Reference("customer", "customer", o2o="placed by")], objects, TWO_ROWS)

    assert event_links == {("P", 0): [("c1", "customer")], ("P", 1): [("c2", "customer")]}
    assert object_links == {"p-0": [("c1", "placed by"), ("c2", "placed by")]}


def test_a_value_not_in_the_catalogue_gets_no_link_and_one_warning_per_attribute_and_value():
    objects = [("P", CaseObject(f"p-{case}", [("customer", "c9", START)])) for case in range(2)]
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


# load errors

def _config_with(tmp_path, catalogues=None, references=None, **changes):
    """The Order Management configuration with the given top-level changes. catalogues ({process: {name: objects}})
    and references ({process: [reference, ...]}), when given, replace those of every process: a process they don't
    name has none. Given references also leave out every process's ocel_attributes, which name attributes the
    committed references bring."""
    with open(CONFIG) as file:
        data = json.load(file)
    data.update(changes)
    for process in data["processes"]:  # the paths are relative to the configuration's folder
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

    with pytest.raises(ValueError, match=r"processes\[2\].references\[0\] names catalogue 'customer' of process Sales; "
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
