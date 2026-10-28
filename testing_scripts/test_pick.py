"""
Picking (docs/catalogue.md, "Pick" and "Pool"): with pick, a reference chooses the
catalogue object of each case when the case is created, "uniform" or by weights; a catalogue can also be a pool,
its objects generated with attributes drawn at the start, of which only the referred ones are written.
"""
import dataclasses
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
import pytz

from prosimos.orchestration.catalogue import CatalogueObject, CataloguePool, Reference
from prosimos.orchestrator import ProcessSpec, SimulationConfig, _Pick, _drawn_catalogues, run_orchestrator
from testing_scripts.test_catalogue import _config_with
from testing_scripts.test_order_management import CONFIG, _config

ASSETS = Path("testing_scripts/assets")
LOANS = (str(ASSETS / "LoanApp_sequential_9-5.bpmn"), str(ASSETS / "LoanApp_arrival_fix_10.json"))  # no object attributes
START = pytz.utc.localize(datetime(2022, 6, 21, 9))
POOL = CataloguePool(1500, (
    {"name": "segment", "type": "discrete",
     "values": [{"key": "retail", "value": 0.8}, {"key": "business", "value": 0.2}]},
    {"name": "credit_limit", "type": "continuous",
     "values": {"distribution_name": "norm",
                "distribution_params": [{"value": 5000}, {"value": 1000}, {"value": 1000}, {"value": 9000}]}}))
THREE = [CatalogueObject("c1"), CatalogueObject("c2"), CatalogueObject("c3")]


def _loans(cases, catalogue, pick, seed=1, with_attributes=()):
    """A config of one process, Loans, whose cases pick a customer."""
    return SimulationConfig([ProcessSpec("Loans", *LOANS, cases, "loan", ocel_attributes=("customer", *with_attributes),
                                         catalogues={"customer": catalogue},
                                         references=(Reference("customer", "customer", o2o="applies for", reverse=True,
                                                               link_events=(), pick=pick,
                                                               with_attributes=tuple((name, name) for name in
                                                                                     with_attributes)),))],
                            START, seed)


def _run(config, tmp_path, name="log"):
    report = run_orchestrator(config, tmp_path / f"{name}.csv", ocel_out_path=tmp_path / f"{name}.json")
    with open(tmp_path / f"{name}.json") as file:
        return report, pd.read_csv(tmp_path / f"{name}.csv"), json.load(file)


def _customers_of_cases(report, process):
    """Case id -> the customer it picked."""
    return {case_object.object_id: value for name, case_object in report.objects if name == process
            for attribute, value, _ in case_object.attributes if attribute == "customer"}


# Order Management

def test_order_management_picks_its_customer_and_sales_json_no_longer_draws_it(tmp_path):
    with open(ASSETS / "order_management" / "sales.json") as file:
        assert "customer" not in {attribute["name"] for attribute in json.load(file)["object_attributes"]}
    [sales] = [spec for spec in SimulationConfig.from_json(CONFIG).processes if spec.name == "Sales"]
    assert [reference.pick for reference in sales.references] == ["uniform"]

    report, log, _ = _run(_config(100), tmp_path)

    customers = _customers_of_cases(report, "Sales")
    assert len(customers) == 100 and set(customers.values()) <= {f"c{n}" for n in range(1, 16)}
    assert len(set(customers.values())) > 10  # uniform over 15
    # a normal case attribute from then on: a column of the log, and copied to items by ItemOrdered
    sales = log[log["process"] == "Sales"]
    assert sales["customer"].notna().all()
    assert set(_customers_of_cases(report, "Warehouse").values()) <= set(customers.values())


# weights

def test_weights_give_their_shares():
    pick = _Pick("customer", ["c1", "c2", "c3"], [3, 1, 1], seed=1, process="P")

    shares = Counter(pick.get_next_value() for _ in range(30000))

    assert {customer: round(count / 30000, 2) for customer, count in shares.items()} == {"c1": 0.6, "c2": 0.2, "c3": 0.2}


def test_a_run_with_weights_picks_in_their_shares_and_never_an_id_left_out(tmp_path):
    catalogue = THREE + [CatalogueObject("c4")]
    report, _, _ = _run(_loans(1000, catalogue, (("c1", 3), ("c2", 1), ("c3", 1))), tmp_path)

    shares = Counter(_customers_of_cases(report, "Loans").values())

    assert set(shares) == {"c1", "c2", "c3"}  # c4 has no weight
    assert abs(shares["c1"] / 1000 - 0.6) < 0.05 and abs(shares["c2"] / 1000 - 0.2) < 0.05


# pools

@pytest.fixture(scope="module")
def pool_run(tmp_path_factory):
    """2,000 loans, each picking its customer from a pool of 1,500."""
    return _run(_loans(2000, POOL, "uniform"), tmp_path_factory.mktemp("pool"))


def test_a_pool_of_1500_with_2000_cases_writes_only_the_picked_customers(pool_run):
    report, _, ocel = pool_run

    picked = set(_customers_of_cases(report, "Loans").values())
    written = {obj["id"]: obj for obj in ocel["objects"] if obj["type"] == "customer"}

    assert set(written) == picked
    assert 900 < len(picked) < 1300  # uniform: about 1500 * (1 - e^(-2000/1500)) = 1105 different customers
    assert picked <= {f"customer-{number}" for number in range(1, 1501)}
    for customer in written.values():
        attributes = {attribute["name"]: attribute["value"] for attribute in customer["attributes"]}
        assert attributes["segment"] in ("retail", "business") and 1000 <= attributes["credit_limit"] <= 9000
        assert customer["relationships"]  # each applies for at least one loan
    assert sum(len(customer["relationships"]) for customer in written.values()) == 2000


def test_a_pools_attributes_are_drawn_in_their_shares(pool_run):
    _, _, ocel = pool_run

    segments = Counter(attribute["value"] for obj in ocel["objects"] if obj["type"] == "customer"
                       for attribute in obj["attributes"] if attribute["name"] == "segment")

    assert abs(segments["retail"] / sum(segments.values()) - 0.8) < 0.04


def test_the_same_seed_gives_the_same_pool_and_the_same_picks(tmp_path):
    first, _, first_ocel = _run(_loans(300, POOL, "uniform"), tmp_path, "first")
    again, _, again_ocel = _run(_loans(300, POOL, "uniform"), tmp_path, "again")
    other, _, _ = _run(_loans(300, POOL, "uniform", seed=2), tmp_path, "other")

    def customers(ocel):
        return [obj for obj in ocel["objects"] if obj["type"] == "customer"]

    assert _customers_of_cases(first, "Loans") == _customers_of_cases(again, "Loans")
    assert customers(first_ocel) == customers(again_ocel)
    assert _customers_of_cases(first, "Loans") != _customers_of_cases(other, "Loans")


def test_the_pool_is_drawn_from_the_seed():
    def drawn(seed=1):
        return _drawn_catalogues(_loans(1, POOL, "uniform").processes[0], seed)["customer"]

    pool = drawn()

    assert pool == drawn()
    assert pool != drawn(seed=2)
    assert [obj.id for obj in pool] == [f"customer-{number}" for number in range(1, 1501)]


def test_picking_doesnt_change_the_processs_other_draws(tmp_path):
    # the loans also draw a size from the engine's generator when a case is created, as the customer would be
    # drawn by object_attributes; the picks have a generator of their own, so the sizes and the rest of the log
    # are the same with and without the pick
    with open(LOANS[1]) as file:
        settings = json.load(file)
    settings["object_attributes"] = [{"name": "size", "type": "discrete",
                                    "values": [{"key": "small", "value": 0.5}, {"key": "large", "value": 0.5}]}]
    (tmp_path / "loans.json").write_text(json.dumps(settings))
    spec = ProcessSpec("Loans", LOANS[0], str(tmp_path / "loans.json"), 200, "loan", catalogues={"customer": THREE},
                       references=(Reference("customer", "customer", pick="uniform"),))

    _, with_pick, _ = _run(SimulationConfig([spec], START, 1), tmp_path, "with")
    _, without, _ = _run(SimulationConfig([dataclasses.replace(spec, references=())], START, 1), tmp_path, "without")

    assert set(with_pick["customer"]) == {"c1", "c2", "c3"} and "customer" not in without
    assert set(without["size"]) == {"small", "large"}
    assert with_pick.drop(columns="customer").equals(without)


# with

RATED = [CatalogueObject("c1", (("segment", "retail"), ("limit", 2000))),
         CatalogueObject("c2", (("segment", "business"), ("limit", 9000.5))),
         CatalogueObject("c3", (("segment", "retail"), ("limit", 500)))]


def _starting_values(report, process):
    """Case id -> its attribute values when it was created."""
    return {case_object.object_id: {name: value for name, value, time in case_object.attributes
                                  if time == case_object.attributes[0][2]}
            for process_name, case_object in report.objects if process_name == process}


def test_a_case_starts_with_the_attributes_with_names_of_the_object_it_picks(tmp_path):
    report, log, ocel = _run(_loans(200, RATED, (("c1", 1), ("c2", 1), ("c3", 1)), with_attributes=("segment", "limit")),
                             tmp_path)

    rated = {obj.id: dict(obj.attributes) for obj in RATED}
    values = _starting_values(report, "Loans")
    assert {case["customer"] for case in values.values()} == {"c1", "c2", "c3"}
    assert all(case == {"customer": case["customer"], **rated[case["customer"]]} for case in values.values())
    # normal case values from then on: columns of the log and attributes of the case's OCEL object
    assert all(rated[customer] == {"segment": segment, "limit": limit}
               for customer, segment, limit in zip(log["customer"], log["segment"], log["limit"]))
    loans = [obj for obj in ocel["objects"] if obj["type"] == "loan"]
    assert {attribute["name"] for obj in loans for attribute in obj["attributes"]} == {"customer", "segment", "limit"}


def test_a_pool_object_brings_the_attributes_drawn_for_it(tmp_path):
    report, _, ocel = _run(_loans(300, POOL, "uniform", with_attributes=("credit_limit",)), tmp_path)

    drawn = {obj["id"]: {attribute["name"]: attribute["value"] for attribute in obj["attributes"]}
             for obj in ocel["objects"] if obj["type"] == "customer"}
    values = _starting_values(report, "Loans").values()
    assert len(values) == 300
    assert all(case["credit_limit"] == drawn[case["customer"]]["credit_limit"] for case in values)


def test_bringing_attributes_doesnt_change_what_is_picked_or_the_rest_of_the_log(tmp_path):
    picks = (("c1", 1), ("c2", 1), ("c3", 1))
    _, bringing, _ = _run(_loans(200, RATED, picks, with_attributes=("segment", "limit")), tmp_path, "with")
    _, plain, _ = _run(_loans(200, RATED, picks), tmp_path, "without")

    assert bringing.drop(columns=["segment", "limit"]).equals(plain)


# load errors

PICKING = {"object_attribute": "customer", "catalogue": "customer"}
OM = json.loads(Path(CONFIG).read_text())
CUSTOMERS, PRODUCTS_LIST = (next(entry["catalogues"][name] for entry in OM["processes"] if name in entry.get("catalogues", {}))
                            for name in ("customer", "product"))


def test_weights_naming_unknown_ids_are_rejected(tmp_path):
    path = _config_with(tmp_path, references={"Sales": [{**PICKING, "pick": {"weights": {"c1": 3, "c99": 1}}}]})

    with pytest.raises(ValueError, match=r"processes\[0\].references\[0\] weights name \['c99'\], which are no ids of "
                                         r"catalogue 'customer'"):
        SimulationConfig.from_json(path)


def test_weights_on_a_pool_are_rejected(tmp_path):
    path = _config_with(tmp_path, catalogues={"Sales": {"customer": {"count": 10}}},
                        references={"Sales": [{**PICKING, "pick": {"weights": {"customer-1": 1}}}]})

    with pytest.raises(ValueError, match=r"processes\[0\].references\[0\] has weights on pool 'customer'"):
        SimulationConfig.from_json(path)


@pytest.mark.parametrize("count", [0, -5, "10", 2.5, None])
def test_a_pool_count_below_1_or_not_a_whole_number_is_rejected(tmp_path, count):
    path = _config_with(tmp_path, catalogues={"Sales": {"customer": {"count": count}}},
                        references={"Sales": [{**PICKING, "pick": "uniform"}]})

    with pytest.raises(ValueError, match=r"catalogue 'customer': count must be a whole number of at least 1"):
        SimulationConfig.from_json(path)


def test_pool_attributes_prosimos_cant_parse_are_rejected(tmp_path):
    path = _config_with(tmp_path, catalogues={"Sales": {"customer": {"count": 10, "attributes": [
        {"name": "segment", "type": "discrete", "values": [{"key": "retail", "value": 0.5}]}]}}}, references={})

    with pytest.raises(ValueError, match=r"catalogue 'customer': attributes\[0\]: .*probabilities' sum should be equal to 1"):
        SimulationConfig.from_json(path)


def test_picking_an_attribute_also_in_object_attributes_is_rejected(tmp_path):
    with open(CONFIG.replace("simulation.json", "sales.json")) as file:
        sales = json.load(file)
    sales["object_attributes"].append({"name": "customer", "type": "discrete", "values": [{"key": "c1", "value": 1}]})
    (tmp_path / "sales.json").write_text(json.dumps(sales))
    path = _config_with(tmp_path, references={"Sales": [{**PICKING, "pick": "uniform"}]})
    data = json.loads(path.read_text())
    next(process for process in data["processes"] if process["name"] == "Sales")["json_path"] = str(tmp_path / "sales.json")
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match=r"processes\[0\].references\[0\] picks customer, which process Sales also draws "
                                         r"in its object_attributes"):
        SimulationConfig.from_json(path)


def test_picking_an_attribute_the_process_copies_from_a_message_is_rejected(tmp_path):
    # the Warehouse copies customer from ItemOrdered; a pick from a customer catalogue of its own would be replaced
    # by the copied value
    path = _config_with(tmp_path, catalogues={"Warehouse": {"customer": CUSTOMERS}},
                        references={"Warehouse": [{**PICKING, "pick": "uniform"}]})

    with pytest.raises(ValueError, match=r"processes\[1\].references\[0\] picks customer, which process Warehouse also "
                                         r"copies from a message \(consume entries at \['Start_Item'\]\)"):
        SimulationConfig.from_json(path)


def test_an_unknown_pick_is_rejected(tmp_path):
    path = _config_with(tmp_path, references={"Sales": [{**PICKING, "pick": "random"}]})

    with pytest.raises(ValueError, match=r'processes\[0\].references\[0\] pick must be "uniform", \{"weights"'):
        SimulationConfig.from_json(path)


def test_weights_that_are_negative_or_all_zero_are_rejected(tmp_path):
    path = _config_with(tmp_path, references={"Sales": [{**PICKING, "pick": {"weights": {"c1": 0, "c2": 0}}}]})

    with pytest.raises(ValueError, match=r"processes\[0\].references\[0\] weights must be numbers of at least 0, not all 0"):
        SimulationConfig.from_json(path)


def test_two_references_picking_the_same_attribute_are_rejected(tmp_path):
    path = _config_with(tmp_path, references={"Sales": [{**PICKING, "pick": "uniform"}, {**PICKING, "pick": "uniform"}]})

    with pytest.raises(ValueError, match=r"more than one reference picks customer of process Sales"):
        SimulationConfig.from_json(path)


PRODUCTS = {"object_attribute": "product", "catalogue": "product", "pick": "uniform"}
WAREHOUSE = r"processes\[1\].references"


def _products_with(tmp_path, names, *other_references, attributes=None, process="Warehouse"):
    """The Order Management configuration whose only catalogue is the products, with weight and price and
    attributes if given, and whose only references are a product pick with this with (a list of names: each copied
    under its own name) and the other references, all in process."""
    if isinstance(names, list):
        names = {name: name for name in names}
    products = json.loads(json.dumps(PRODUCTS_LIST))
    if attributes is not None:
        for product in products:
            product["attributes"].update(attributes)
    return _config_with(tmp_path, catalogues={process: {"product": products}},
                        references={process: [{**PRODUCTS, "with": names}, *other_references]})


def test_with_without_a_pick_is_rejected(tmp_path):
    path = _config_with(tmp_path, references={"Warehouse": [{"object_attribute": "product", "catalogue": "product",
                                                             "with": {"weight": "weight"}}]})

    with pytest.raises(ValueError, match=rf"{WAREHOUSE}\[0\] has with but no pick; a catalogue object brings its "
                                         r"attributes only when the reference picks it"):
        SimulationConfig.from_json(path)


def test_with_on_a_growing_catalogue_is_rejected(tmp_path):
    path = _config_with(tmp_path, catalogues={"Warehouse": {"product": {"attributes": [
        {"name": "weight", "type": "discrete", "values": [{"key": 1, "value": 1}]}]}}},
                        references={"Warehouse": [{**PRODUCTS, "pick": "new", "with": {"weight": "weight"}}]})

    with pytest.raises(ValueError, match=rf"{WAREHOUSE}\[0\] has with on growing catalogue 'product', whose objects "
                                         r"don't exist yet when a case is created"):
        SimulationConfig.from_json(path)


def test_with_naming_an_attribute_not_every_object_has_is_rejected(tmp_path):
    products = json.loads(json.dumps(PRODUCTS_LIST))
    products[3]["attributes"]["volume"] = 2.5  # only one product has a volume
    path = _config_with(tmp_path, catalogues={"Warehouse": {"product": products}},
                        references={"Warehouse": [{**PRODUCTS, "with": {"weight": "weight", "volume": "volume",
                                                                        "colour": "colour"}}]})

    with pytest.raises(ValueError, match=rf"{WAREHOUSE}\[0\] with names \['volume', 'colour'\], which not every object "
                                         r"of catalogue 'product' has"):
        SimulationConfig.from_json(path)


def test_with_naming_an_attribute_a_pool_doesnt_draw_is_rejected(tmp_path):
    path = _config_with(tmp_path, catalogues={"Warehouse": {"product": {"count": 5, "attributes": [
        {"name": "weight", "type": "discrete", "values": [{"key": 1, "value": 1}]}]}}},
                        references={"Warehouse": [{**PRODUCTS, "with": {"weight": "weight", "price": "price"}}]})

    with pytest.raises(ValueError, match=rf"{WAREHOUSE}\[0\] with names \['price'\], which not every object of "
                                         r"catalogue 'product' has"):
        SimulationConfig.from_json(path)


@pytest.mark.parametrize("names", [
    ["weight", "price"],  # a list, which with no longer is
    "weight", {"weight": 3}, {"": "weight"}, {"weight": ""},
    {"price": {"avg": "price"}}, {"price": {"sum": "price", "max": "price"}}, {"price": {"sum": 3}}])
def test_with_that_doesnt_map_names_to_an_attribute_or_a_function_of_one_is_rejected(tmp_path, names):
    path = _config_with(tmp_path, catalogues={"Warehouse": {"product": PRODUCTS_LIST}},
                        references={"Warehouse": [{**PRODUCTS, "with": names}]})

    with pytest.raises(ValueError, match=rf"{WAREHOUSE}\[0\] with must map attribute names to an attribute of the "
                                         r'picked object or to one function of one, \{"sum" \| "count" \| "min" \| '
                                         r'"max": <attribute>\}, got'):
        SimulationConfig.from_json(path)


def test_bringing_an_attribute_the_process_draws_in_its_object_attributes_is_rejected(tmp_path):
    with open(CONFIG.replace("simulation.json", "warehouse.json")) as file:
        warehouse = json.load(file)
    warehouse["object_attributes"] = [{"name": "price", "type": "discrete", "values": [{"key": 1, "value": 1}]}]
    (tmp_path / "warehouse.json").write_text(json.dumps(warehouse))
    path = _products_with(tmp_path, ["weight", "price"])
    data = json.loads(path.read_text())
    next(process for process in data["processes"] if process["name"] == "Warehouse")["json_path"] = \
        str(tmp_path / "warehouse.json")
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match=rf"{WAREHOUSE}\[0\] brings price, which process Warehouse also draws in its "
                                         r"object_attributes"):
        SimulationConfig.from_json(path)


def test_bringing_an_attribute_the_process_picks_is_rejected(tmp_path):
    path = _products_with(tmp_path, ["weight", "product"], attributes={"product": "same"})

    with pytest.raises(ValueError, match=rf"{WAREHOUSE}\[0\] brings product, which process Warehouse also picks"):
        SimulationConfig.from_json(path)


def test_two_withs_bringing_the_same_attribute_to_a_process_are_rejected(tmp_path):
    gift = {"object_attribute": "gift", "catalogue": "product", "pick": "uniform", "with": {"weight": "weight"}}
    path = _products_with(tmp_path, ["weight"], gift)

    with pytest.raises(ValueError, match=rf"{WAREHOUSE}\[0\] brings weight, which another with also brings to process "
                                         rf"Warehouse; {WAREHOUSE}\[1\] brings weight, which another with"):
        SimulationConfig.from_json(path)


@pytest.mark.parametrize("process, index, name, verb, event", [
    ("Warehouse", 1, "customer", "copies", "Start_Item"),
    ("Packaging", 2, "item_ids", "appends to", "Catch_Items"),
    ("Packaging", 2, "weight", "aggregates into", "Catch_Items"),
])
def test_bringing_an_attribute_the_process_sets_from_a_message_is_rejected(tmp_path, process, index, name, verb, event):
    path = _products_with(tmp_path, [name], attributes={name: "x"}, process=process)

    with pytest.raises(ValueError, match=rf"processes\[{index}\].references\[0\] brings {name}, which process {process} "
                                         rf"also {verb} from a message \(consume entries at \['{event}'\]\)"):
        SimulationConfig.from_json(path)


def test_bringing_object_id_is_rejected(tmp_path):
    path = _products_with(tmp_path, ["object_id"], attributes={"object_id": "x"})

    with pytest.raises(ValueError, match=rf"{WAREHOUSE}\[0\] brings object_id, which is reserved for the object's own id"):
        SimulationConfig.from_json(path)
