"""
with (docs/catalogue.md, "With"): a picking reference gives the case starting values from the objects it picked,
as {attribute: source}: a plain name copies the picked object's attribute, under any name; {function: attribute}
computes over the picked objects, as aggregate does over messages. A reference that doesn't pick gives them from
the object its attribute names when it gets its value: copied from the message that starts the case, or drawn.
"""
import dataclasses
import json

import pytest

from prosimos.orchestration.catalogue import CatalogueObject, Computed, Reference, _Brought, _Pick
from prosimos.orchestration.config import Organization, ProcessSpec, SimulationConfig
from prosimos.orchestration.simulation import run_multiprocess_simulation
from testing_scripts.multiprocess.catalogue.test_catalogue import _config_with
from testing_scripts.multiprocess.catalogue.test_pick import (CUSTOMERS, PICKING, POOL, PRODUCTS_LIST, RATED, _loans,
                                                              _run, _starting_values)
from testing_scripts.multiprocess.catalogue.test_pick_count import PRODUCTS, SALES, _at_creation, _sales_picking
from testing_scripts.multiprocess.examples.test_order_management import CONFIG
from testing_scripts.multiprocess.messaging.test_collect import START, at
from testing_scripts.multiprocess.orchestrator_settings import one_organization
from testing_scripts.multiprocess.scripted_engine import ScriptedEngine

PRICE = {obj["id"]: obj["attributes"]["price"] for obj in PRODUCTS_LIST}


def _brought(pick, catalogue, with_attributes):
    """The values with gives the next case: the pick draws, then what it brings, in order, as the engine draws them."""
    values = {obj.id: dict(obj.attributes) for obj in catalogue}
    brought = [_Brought(name, pick, {object_id: attributes[source.source if isinstance(source, Computed) else source]
                                     for object_id, attributes in values.items()},
                        source.function if isinstance(source, Computed) else None)
               for name, source in with_attributes]
    picked = pick.get_next_value()
    return picked, {bringing.name: bringing.get_next_value() for bringing in brought}


# copy

def test_a_plain_name_copies_the_attribute_and_may_rename_it(tmp_path):
    config = _loans(200, RATED, "uniform")
    [loans] = config.processes
    reference = dataclasses.replace(loans.references[0], with_attributes=(("segment", "segment"), ("credit", "limit")))
    config = dataclasses.replace(config,
                                 organizations=one_organization([dataclasses.replace(loans, references=(reference,))],
                                                                config.catalogue))

    report, _, _ = _run(config, tmp_path)

    rated = {obj.id: dict(obj.attributes) for obj in RATED}
    values = _starting_values(report, "Loans").values()
    assert len(values) == 200
    assert all(case["segment"] == rated[case["customer"]]["segment"]
               and case["credit"] == rated[case["customer"]]["limit"] and "limit" not in case for case in values)


# functions

THREE = [CatalogueObject(object_id, (("price", price), ("name", object_id))) for object_id, price in
         (("a", 10), ("b", 2.5), ("c", 4))]
ALL_FOUR = (("total", Computed("sum", "price")), ("lines", Computed("count", "price")),
            ("cheapest", Computed("min", "price")), ("dearest", Computed("max", "price")))


def test_the_functions_over_a_list_pick_count_a_repeated_object_each_time():
    pick = _Pick("products", ["a", "b", "c"], [1, 1, 1], seed=1, process="P", count=4)

    picked, values = next((picked, values) for picked, values in (_brought(pick, THREE, ALL_FOUR) for _ in range(20))
                          if len(set(picked)) < 4)  # one with a repeat

    prices = [dict(dict((obj.id, obj.attributes) for obj in THREE)[object_id])["price"] for object_id in picked]
    assert values == {"total": sum(prices), "lines": 4, "cheapest": min(prices), "dearest": max(prices)}


def test_a_sum_of_whole_numbers_stays_whole_and_others_add_up_in_any_order():
    # as aggregate: a sum of whole numbers is a whole number; any other is rounded once, at the end
    whole = _Pick("products", ["a", "c"], None, seed=1, process="P", count=3)
    assert isinstance(_brought(whole, THREE, ALL_FOUR[:1])[1]["total"], int)
    prices = {"x": 1.2, "y": 0.8, "z": 2.1}

    class Picked:  # a pick that picked these, in this order
        def __init__(self, last):
            self.last = last

    assert {_Brought("total", Picked(order), prices, "sum").get_next_value()
            for order in (["x", "y", "z"], ["z", "y", "x"], ["y", "x", "z"])} == {4.1}


def test_over_no_objects_a_sum_and_a_count_are_0_and_a_min_and_a_max_have_no_value():
    pick = _Pick("products", ["a"], None, seed=1, process="P", count=0)

    assert _brought(pick, THREE, ALL_FOUR) == ([], {"total": 0, "lines": 0, "cheapest": None, "dearest": None})


def test_a_function_on_a_single_pick_works_over_the_one_object():
    pick = _Pick("product", ["a", "b", "c"], None, seed=1, process="P")

    picked, values = _brought(pick, THREE, ALL_FOUR)

    price = dict(dict((obj.id, obj.attributes) for obj in THREE)[picked])["price"]
    assert values == {"total": price, "lines": 1, "cheapest": price, "dearest": price}


# in a run: each order picks one product per item and starts with their total price

@pytest.fixture(scope="module")
def priced_run(tmp_path_factory):
    path = tmp_path_factory.mktemp("with")
    return _run(_sales_picking(path, {**PRODUCTS, "count": {"object_attribute": "items"},
                                      "with": {"price": {"sum": "price"}, "dearest": {"max": "price"}}}), path)


def test_every_order_starts_with_the_sum_of_its_products_prices(priced_run):
    report, log, _ = priced_run

    products, prices = _at_creation(report, "Sales", "products"), _at_creation(report, "Sales", "price")
    dearest = _at_creation(report, "Sales", "dearest")

    assert len(products) == 100
    assert any(len(set(picked)) < len(picked) for picked in products.values())  # some order repeats a product
    assert all(prices[order] == pytest.approx(sum(PRICE[product] for product in products[order]))
               and dearest[order] == max(PRICE[product] for product in products[order]) for order in products)
    # a normal case value from then on: a column of the log
    sales = log[log["process"] == "Sales"].groupby("case_id").first()
    assert all(value == pytest.approx(prices[f"order-{case}"]) for case, value in sales["price"].items())


def test_a_computed_with_doesnt_change_the_picks_or_the_rest_of_the_run(tmp_path, priced_run):
    _, priced, _ = priced_run
    _, plain, _ = _run(_sales_picking(tmp_path, {**PRODUCTS, "count": {"object_attribute": "items"}}), tmp_path,
                       "plain")

    assert priced.drop(columns=["price", "dearest"]).equals(plain)


def test_order_management_spells_its_with_as_an_object():
    [warehouse] = [spec for spec in SimulationConfig.from_json("testing_scripts/assets/order_management/simulation.json")
                   .processes if spec.name == "Warehouse"]

    assert [reference.with_attributes for reference in warehouse.references] == [(("weight", "weight"),
                                                                                   ("price", "price"))]


# load errors

def test_a_plain_copy_on_a_list_pick_is_rejected(tmp_path):
    with pytest.raises(ValueError, match=rf"{SALES} copies \['price'\] on a pick with count; a list pick needs a "
                                         r'function over the picked objects, e.g. \{"sum": "price"\}'):
        _sales_picking(tmp_path, {**PRODUCTS, "count": 3, "with": {"price": "price", "total": {"sum": "price"}}})


@pytest.mark.parametrize("function", ["sum", "min", "max"])
def test_a_sum_min_or_max_of_an_attribute_that_isnt_a_number_is_rejected(tmp_path, function):
    products = json.loads(json.dumps(PRODUCTS_LIST))
    products[2]["attributes"]["price"] = "39.99"  # text, not a number

    path = _config_with(tmp_path, catalogues={"customer": CUSTOMERS, "product": products},
                        references={"Sales": [{**PICKING, "pick": "uniform"},
                                              {**PRODUCTS, "count": 3, "with": {"price": {function: "price"}}}]})

    with pytest.raises(ValueError, match=rf"{SALES} with sums, or takes the min or max of, \['price'\], which not "
                                         r"every object of catalogue 'product' has as a number"):
        SimulationConfig.from_json(path)


def test_a_count_of_an_attribute_that_isnt_a_number_is_allowed(tmp_path):
    # count only needs the attribute to be there
    customers = [{"id": customer["id"], "attributes": {"region": "north"}} for customer in CUSTOMERS]
    path = _config_with(tmp_path, catalogues={"customer": customers},
                        references={"Sales": [{**PICKING, "pick": "uniform", "with": {"regions": {"count": "region"}}}]})

    SimulationConfig.from_json(path)


POOL_OF_TEXT = {"count": 10, "attributes": [{"name": "segment", "type": "discrete",
                                             "values": [{"key": "retail", "value": 0.8},
                                                        {"key": "business", "value": 0.2}]}]}


def test_a_sum_of_a_pool_attribute_that_draws_text_is_rejected(tmp_path):
    path = _config_with(tmp_path, catalogues={"customer": POOL_OF_TEXT},
                        references={"Sales": [{**PICKING, "pick": "uniform", "with": {"segments": {"max": "segment"}}}]})

    with pytest.raises(ValueError, match=r"processes\[0\].references\[0\] with sums, or takes the min or max of, "
                                         r"\['segment'\], which not every object of catalogue 'customer' has as a number"):
        SimulationConfig.from_json(path)


def test_a_sum_of_a_pool_attribute_that_draws_numbers_is_allowed(tmp_path):
    path = _config_with(tmp_path, catalogues={"customer": {"count": 10, "attributes": list(POOL.attributes)}},
                        references={"Sales": [{**PICKING, "pick": "uniform",
                                               "with": {"limit": {"sum": "credit_limit"}}}]})

    SimulationConfig.from_json(path)


# with for a received value: a reference without pick

PRODUCT_OBJECTS = [CatalogueObject(obj["id"], tuple(obj["attributes"].items())) for obj in PRODUCTS_LIST]
WEIGHT = {obj["id"]: obj["attributes"]["weight"] for obj in PRODUCTS_LIST}


def _received(report, process):
    """Case id -> its values when it was created, and when that was."""
    return {case.object_id: ({name: value for name, value, time in case.attributes if time == case.created},
                             case.created)
            for name, case in report.objects if name == process}


def test_an_item_started_by_a_message_naming_its_product_gets_its_weight_and_price_at_creation(tmp_path):
    # Picking sends one ItemOrdered per product; the Warehouse copies product from it, and its reference, without a
    # pick, brings that product's weight and price into the item when it is created
    folder = CONFIG.replace("simulation.json", "")
    with open(folder + "warehouse.json") as file:
        warehouse = json.load(file)
    warehouse["messages"]["consume"][0]["copy"]["product"] = "product"
    (tmp_path / "warehouse.json").write_text(json.dumps(warehouse))
    reference = Reference("product", "product", o2o="is a", with_attributes=(("weight", "weight"), ("price", "price")))
    shop = Organization("Shop", [ProcessSpec("Warehouse", folder + "warehouse.bpmn", str(tmp_path / "warehouse.json"),
                                             None, "item", ocel_attributes=("weight", "price"),
                                             references=(reference,))], {"product": PRODUCT_OBJECTS})
    picking = ScriptedEngine("Picking")
    sent = ["Kindle", "iPad", "Kindle", "Echo Dot"]
    for number, product in enumerate(sent):
        picking.publish_at(at(f"09:{10 * number:02}"), "ItemOrdered", object_id=f"order-{number}", index=1,
                           customer="c1", product=product)

    report = run_multiprocess_simulation(SimulationConfig([shop], START, 1, extra_processes=["Picking"]),
                                         tmp_path / "log.csv", {"Picking": picking},
                                         ocel_out_path=tmp_path / "log.json")

    items = _received(report, "Warehouse")
    assert sorted(values["product"] for values, _ in items.values()) == sorted(sent)
    assert all(values["weight"] == WEIGHT[values["product"]] and values["price"] == PRICE[values["product"]]
               for values, _ in items.values())
    with open(tmp_path / "log.json") as file:
        objects = {obj["id"]: obj for obj in json.load(file)["objects"] if obj["type"] == "item"}
    # written on the item, at its creation
    assert all({(value["name"], value["value"]) for value in objects[item]["attributes"]} ==
               {("weight", WEIGHT[values["product"]]), ("price", PRICE[values["product"]])} for item, (values, _) in
               items.items())


def test_a_drawn_value_brings_its_object_attributes_at_creation(tmp_path):
    # Sales draws each order's city in its object_attributes; a city catalogue gives the order its city's region
    regions = {"Tartu": "south", "Tallinn": "north", "Tapa": "north", "Pärnu": "west"}
    cities = [CatalogueObject(city, (("region", region),)) for city, region in regions.items()]
    reference = Reference("city", "city", with_attributes=(("region", "region"),))
    shop = Organization("Shop", [ProcessSpec("Sales", "testing_scripts/assets/messaging/sales.bpmn",
                                             "testing_scripts/assets/messaging/sales.json", 20, "order",
                                             references=(reference,))], {"city": cities})

    report = run_multiprocess_simulation(SimulationConfig([shop], START, 1), tmp_path / "log.csv")

    orders = [values for values, _ in _received(report, "Sales").values()]
    assert len(orders) == 20 and len({order["city"] for order in orders}) > 1
    assert all(order["region"] == regions[order["city"]] for order in orders)
