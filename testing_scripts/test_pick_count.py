"""
A pick with count (docs/catalogue.md, "Count"): the reference picks that many catalogue objects, each one
independently by the same rule, into a list, which the OCEL output links entry by entry.
"""
import dataclasses
import json
from collections import Counter

import pandas as pd
import pytest

from prosimos.orchestration.catalogue import CountFrom, _Pick
from prosimos.orchestration.config import SimulationConfig
from prosimos.orchestrator import run_orchestrator
from testing_scripts.test_catalogue import _config_with
from testing_scripts.test_pick import CUSTOMERS, PICKING, PRODUCTS_LIST

PRODUCTS = {"object_attribute": "products", "catalogue": "product", "pick": "uniform"}
SALES = r"processes\[0\].references\[1\]"


def _sales_picking(tmp_path, products, orders=100):
    """Order Management's Sales alone, with 100 orders: it picks its customer and, with the reference products, its
    products from the 20. Without the Warehouse, which picks its own products, ItemOrdered finds no
    recipient."""
    path = _config_with(tmp_path, catalogues={"Sales": {"customer": CUSTOMERS, "product": PRODUCTS_LIST}},
                        references={"Sales": [{**PICKING, "pick": "uniform"}, *([products] if products else [])]})
    config = SimulationConfig.from_json(path)
    return dataclasses.replace(config, processes=[
        dataclasses.replace(spec, arrivals=orders if spec.name == "Sales" else spec.arrivals)
        for spec in config.processes if spec.name == "Sales"], consumer_groups=None)


def _run(tmp_path, products, name="log"):
    report = run_orchestrator(_sales_picking(tmp_path, products), tmp_path / f"{name}.csv",
                              ocel_out_path=tmp_path / f"{name}.json")
    with open(tmp_path / f"{name}.json") as file:
        return report, pd.read_csv(tmp_path / f"{name}.csv"), json.load(file)


def _at_creation(report, process, attribute):
    """Object id -> the value its attribute was created with."""
    return {case_object.object_id: next(value for name, value, _ in case_object.attributes if name == attribute)
            for name, case_object in report.objects if name == process}


def test_a_fixed_count_picks_that_many_ids_into_a_list():
    pick = _Pick("products", ["a", "b", "c"], None, seed=1, process="P", count=2)

    picked = [pick.get_next_value() for _ in range(100)]

    assert all(len(value) == 2 and set(value) <= {"a", "b", "c"} for value in picked)


def test_each_pick_is_independent_so_an_id_can_come_more_than_once():
    pick = _Pick("products", ["a", "b"], None, seed=1, process="P", count=5)

    assert any(len(set(pick.get_next_value())) < 5 for _ in range(10))  # five picks among two ids repeat one


def test_a_count_of_0_picks_an_empty_list():
    assert _Pick("products", ["a", "b"], None, seed=1, process="P", count=0).get_next_value() == []


def test_the_picks_follow_the_weights_over_many_cases():
    pick = _Pick("products", ["a", "b", "c"], [3, 1, 1], seed=1, process="P", count=3)

    shares = Counter(product for _ in range(10000) for product in pick.get_next_value())

    assert {product: round(count / 30000, 2) for product, count in shares.items()} == {"a": 0.6, "b": 0.2, "c": 0.2}


def test_without_count_a_pick_gives_a_single_value_as_before():
    with_count = _Pick("product", ["a", "b", "c"], [3, 1, 1], seed=1, process="P", count=1)
    without = _Pick("product", ["a", "b", "c"], [3, 1, 1], seed=1, process="P")

    # the same draws; with count each is a list of one
    assert [[without.get_next_value()] for _ in range(20)] == [with_count.get_next_value() for _ in range(20)]


class _Drawn:
    """A case attribute the engine draws, standing in for Prosimos's CaseAttribute."""

    def __init__(self, name, values):
        self.name, self._values = name, iter(values)

    def get_next_value(self):
        return next(self._values)


def test_a_count_read_from_an_attribute_takes_the_value_drawn_for_the_same_case():
    items = _Drawn("items", [3, 1, 4])
    pick = _Pick("products", ["a", "b"], None, seed=1, process="P", count=CountFrom("items"))
    pick.read_count_from(items)

    # the engine draws items, then the pick, for each case in turn
    drawn = [(items.get_next_value(), pick.get_next_value()) for _ in range(3)]

    assert [(count, len(products)) for count, products in drawn] == [(3, 3), (1, 1), (4, 4)]


def test_a_count_attribute_without_a_whole_number_picks_one_with_one_warning(monkeypatch):
    warnings = []
    monkeypatch.setattr("prosimos.orchestrator.warning_logger.add_warning", warnings.append)
    items = _Drawn("items", [2.5, -1])
    pick = _Pick("products", ["a", "b"], None, seed=1, process="P", count=CountFrom("items"))
    pick.read_count_from(items)

    picked = [(items.get_next_value(), pick.get_next_value())[1] for _ in range(2)]

    assert [len(products) for products in picked] == [1, 1]
    assert warnings == ["items is 2.5, no whole number of at least 0; the pick of products picks one"]


# in a run: Sales picks one product per item, the count read from items, which sales.json draws

@pytest.fixture(scope="module")
def picking_run(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("pick_count"), {**PRODUCTS, "count": {"object_attribute": "items"},
                                                        "o2o": "contains"})


def test_every_order_picks_as_many_products_as_it_has_items(picking_run):
    report, _, _ = picking_run

    items, products = _at_creation(report, "Sales", "items"), _at_creation(report, "Sales", "products")

    assert len(products) == 100
    ids = {product["id"] for product in PRODUCTS_LIST}
    assert all(len(products[order]) == items[order] and set(products[order]) <= ids for order in products)


def test_an_order_links_each_product_it_picked(picking_run):
    report, _, ocel = picking_run
    products = _at_creation(report, "Sales", "products")

    for order in (obj for obj in ocel["objects"] if obj["type"] == "order"):
        assert [link["objectId"] for link in order["relationships"] if link["qualifier"] == "contains"] == list(
            dict.fromkeys(products[order["id"]]))  # a product it picked twice once
    place_order = next(event for event in ocel["events"] if event["type"] == "place order")
    order = place_order["relationships"][0]["objectId"]
    assert [link["objectId"] for link in place_order["relationships"] if link["qualifier"] == "products"] == list(
        dict.fromkeys(products[order]))


def test_the_list_is_a_column_of_the_log(picking_run):
    report, log, _ = picking_run

    sales = log[log["process"] == "Sales"].groupby("case_id").first()

    assert all(json.loads(value) == _at_creation(report, "Sales", "products")[f"order-{case}"]
               for case, value in sales["products"].items())


def test_a_count_pick_doesnt_change_the_rest_of_the_run(tmp_path, picking_run):
    _, picking, _ = picking_run
    _, plain, _ = _run(tmp_path, None, "plain")

    pd.testing.assert_frame_equal(picking.drop(columns="products"), plain)


# load errors

@pytest.mark.parametrize("products, problem", [
    ({"object_attribute": "products", "catalogue": "product", "count": 3},
     "has count but no pick; count says how many objects the pick chooses"),
    ({**PRODUCTS, "count": -1}, r'count must be a whole number of at least 0 or \{"object_attribute": <object '
                                r"attribute>\}, got -1"),
    ({**PRODUCTS, "count": 2.5}, "count must be a whole number of at least 0"),
    ({**PRODUCTS, "count": "3"}, "count must be a whole number of at least 0"),
    ({**PRODUCTS, "count": True}, "count must be a whole number of at least 0"),
    ({**PRODUCTS, "count": {"object_attribute": "lines"}},
     "count reads 'lines', which process Sales doesn't draw in its object_attributes; the count is read when the "
     "case is created"),
])
def test_an_invalid_count_is_rejected(tmp_path, products, problem):
    with pytest.raises(ValueError, match=rf"{SALES} {problem}"):
        _sales_picking(tmp_path, products)


@pytest.mark.parametrize("pick", ["new", {"new_share": 0.5}])
def test_count_on_a_growing_catalogue_is_rejected(tmp_path, pick):
    path = _config_with(tmp_path, catalogues={"Sales": {"customer": CUSTOMERS, "product": {"attributes": []}}},
                        references={"Sales": [{**PICKING, "pick": "uniform"}, {**PRODUCTS, "pick": pick, "count": 3}]})

    with pytest.raises(ValueError, match=rf'{SALES} has count with "new" or new_share; count works with "uniform" or '
                                         r"weights"):
        SimulationConfig.from_json(path)


def test_a_count_object_with_another_key_is_rejected(tmp_path):
    with pytest.raises(ValueError, match=r"processes\[0\].references\[1\].count: unknown key 'attribute'; allowed: "
                                         r"object_attribute$"):
        _sales_picking(tmp_path, {**PRODUCTS, "count": {"attribute": "items"}})
