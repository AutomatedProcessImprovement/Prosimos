"""
Growing catalogues (docs/catalogue.md, "Growing"): a catalogue with only attributes starts empty,
and a reference that picks "new" creates an object for every case, or with new_share for that share of them,
the rest reusing one created so far.
"""
import dataclasses
from collections import Counter

import pytest

from prosimos.orchestrator import GrowingCatalogue, NewShare, Reference, SimulationConfig, _NewPick
from testing_scripts.test_catalogue import _config_with
from testing_scripts.test_order_management import _config
from testing_scripts.test_pick import POOL, PICKING, _customers_of_cases, _loans, _run

CUSTOMERS = GrowingCatalogue(POOL.attributes)  # segment and credit_limit


def _order_management(orders, pick):
    """Order Management whose customers are a growing catalogue, with the items referring to the order's
    customer that ItemOrdered carries (the items still pick their products)."""
    config = _config(orders)
    customers, products = config.references
    return dataclasses.replace(config, catalogue={**config.catalogue, "customer": CUSTOMERS}, references=[
        dataclasses.replace(customers, pick=pick), products,
        Reference("Warehouse", "customer", "customer", o2o="ordered by", link_events=())])


def _created(ocel):
    return {obj["id"]: obj for obj in ocel["objects"] if obj["type"] == "customer"}


# new

@pytest.fixture(scope="module")
def new_run(tmp_path_factory):
    return _run(_order_management(100, "new"), tmp_path_factory.mktemp("new"))


def test_with_new_100_orders_give_100_customers_each_placing_one_order(new_run):
    report, _, ocel = new_run

    customers = _created(ocel)

    assert sorted(customers) == sorted(f"customer-{number}" for number in range(1, 101))
    for customer in customers.values():
        assert [link["qualifier"] for link in customer["relationships"]] == ["places"]
    assert sorted(_customers_of_cases(report, "Sales").values()) == sorted(customers)


def test_ids_are_numbered_in_creation_order(new_run):
    report, _, _ = new_run

    # Sales' planned cases are created in case order
    assert _customers_of_cases(report, "Sales") == {f"order-{case}": f"customer-{case + 1}" for case in range(100)}


def test_a_new_object_gets_its_attributes_timed_when_it_is_created(new_run):
    report, _, ocel = new_run
    created = {case_object.object_id: time for name, case_object in report.objects if name == "Sales"
               for attribute, _, time in case_object.attributes if attribute == "customer"}

    customer = _created(ocel)["customer-7"]

    assert {attribute["name"] for attribute in customer["attributes"]} == {"segment", "credit_limit"}
    assert {attribute["time"] for attribute in customer["attributes"]} == {created["order-6"].strftime("%Y-%m-%dT%H:%M:%S.%fZ")}


def test_a_reference_without_pick_refers_to_the_objects_another_process_created(new_run):
    # the items carry their order's new customer, from ItemOrdered
    report, _, ocel = new_run
    customer_of_item = _customers_of_cases(report, "Warehouse")

    items = {obj["id"]: obj for obj in ocel["objects"] if obj["type"] == "item"}

    assert len(items) > 100
    for item_id, item in items.items():
        assert [link["objectId"] for link in item["relationships"] if link["qualifier"] == "ordered by"] == \
            [customer_of_item[item_id]]
    assert set(customer_of_item.values()) <= set(_customers_of_cases(report, "Sales").values())
    assert not any("not in catalogue" in warning for warning in report.warnings)


# new_share

def test_new_share_creates_in_its_share_and_reuses_uniformly():
    pick = _NewPick("customer", "customer", 0.3, seed=1, process="P")

    values = [pick.get_next_value() for _ in range(30000)]

    assert values[0] == "customer-1"  # the first case always creates one
    assert round(len(set(values)) / 30000, 2) == 0.3 and pick.created == len(set(values))


@pytest.fixture(scope="module")
def share_run(tmp_path_factory):
    """2,000 loans, 30% of them creating a customer."""
    return _run(_loans(2000, CUSTOMERS, NewShare(0.3)), tmp_path_factory.mktemp("share"))


def test_with_new_share_about_30_percent_of_2000_orders_create_a_customer(share_run):
    report, _, ocel = share_run

    picked = _customers_of_cases(report, "Loans")
    customers = _created(ocel)

    assert abs(len(customers) / 2000 - 0.3) < 0.03
    assert set(customers) == set(picked.values()) == {f"customer-{number}" for number in range(1, len(customers) + 1)}
    assert sum(len(customer["relationships"]) for customer in customers.values()) == 2000


def test_the_first_case_creates_and_every_later_id_is_new_or_created_before(share_run):
    report, _, _ = share_run

    picked = _customers_of_cases(report, "Loans")
    highest = 0
    for case in range(2000):
        number = int(picked[f"loan-{case}"].split("-")[1])
        assert number <= highest + 1  # either one created so far, or the next new one
        highest = max(highest, number)
    assert picked["loan-0"] == "customer-1"


def test_reuse_picks_uniformly_among_those_created_so_far(share_run):
    # the first customers are reused most, since they are among the candidates longest
    report, _, _ = share_run

    orders = Counter(_customers_of_cases(report, "Loans").values())

    first, last = sum(orders[f"customer-{number}"] for number in range(1, 51)), \
        sum(orders[f"customer-{number}"] for number in range(len(orders) - 49, len(orders) + 1))
    assert first > 3 * last


def test_the_same_seed_gives_the_same_objects(tmp_path):
    first, _, first_ocel = _run(_loans(300, CUSTOMERS, NewShare(0.3)), tmp_path, "first")
    again, _, again_ocel = _run(_loans(300, CUSTOMERS, NewShare(0.3)), tmp_path, "again")
    other, _, other_ocel = _run(_loans(300, CUSTOMERS, NewShare(0.3), seed=2), tmp_path, "other")

    assert _customers_of_cases(first, "Loans") == _customers_of_cases(again, "Loans")
    assert _created(first_ocel) == _created(again_ocel)
    assert _customers_of_cases(first, "Loans") != _customers_of_cases(other, "Loans")
    assert _created(first_ocel) != _created(other_ocel)


# load errors

GROWING = {"customer": {"attributes": []}}


def test_a_growing_catalogue_reads_from_attributes_without_a_count(tmp_path):
    path = _config_with(tmp_path, catalogue=GROWING, references=[{**PICKING, "pick": {"new_share": 0.3}}])

    config = SimulationConfig.from_json(path)

    assert config.catalogue == {"customer": GrowingCatalogue()}
    assert config.references[0].pick == NewShare(0.3)


def _rejected(tmp_path, message, catalogue=None, references=None):
    path = _config_with(tmp_path, catalogue=catalogue or GROWING, references=references or [{**PICKING, "pick": "new"}])
    with pytest.raises(ValueError, match=message):
        SimulationConfig.from_json(path)


def test_uniform_or_weights_on_a_growing_catalogue_are_rejected(tmp_path):
    _rejected(tmp_path, r"references\[0\] picks from growing catalogue 'customer', which only takes \"new\"",
              references=[{**PICKING, "pick": "uniform"}])


@pytest.mark.parametrize("catalogue", [{"customer": [{"id": "c1"}]}, {"customer": {"count": 10}}])
def test_new_on_a_listed_catalogue_or_a_pool_is_rejected(tmp_path, catalogue):
    _rejected(tmp_path, r"references\[0\] picks \"new\" or new_share from catalogue 'customer', which is no growing",
              catalogue=catalogue)


@pytest.mark.parametrize("share", [0, -0.1, 1.5, "0.3", True, None])
def test_a_new_share_not_above_0_and_at_most_1_is_rejected(tmp_path, share):
    _rejected(tmp_path, r"references\[0\] new_share must be a number above 0 and at most 1",
              references=[{**PICKING, "pick": {"new_share": share}}])


def test_a_growing_catalogue_that_no_reference_creates_from_is_rejected(tmp_path):
    _rejected(tmp_path, r"growing catalogue 'customer' needs exactly one reference that creates its objects .*got 0",
              references=[{"process": "Warehouse", "object_attribute": "customer", "catalogue": "customer"}])


def test_two_references_creating_from_one_growing_catalogue_are_rejected(tmp_path):
    _rejected(tmp_path, r"growing catalogue 'customer' needs exactly one reference that creates its objects .*got 2",
              references=[{**PICKING, "pick": "new"},
                          {"process": "Warehouse", "object_attribute": "buyer", "catalogue": "customer", "pick": "new"}])


def test_a_growing_catalogue_named_like_an_object_type_is_rejected(tmp_path):
    # its ids, order-1, order-2, ..., would be those of Sales's orders
    _rejected(tmp_path, r"catalogue 'order' is the object type of process Sales",
              catalogue={"order": {"attributes": []}}, references=[{**PICKING, "catalogue": "order", "pick": "new"}])


def test_ids_like_those_a_growing_catalogue_creates_are_rejected(tmp_path):
    _rejected(tmp_path, r"catalogue ids look like the ids growing catalogue 'customer' creates \(customer-<number>\): "
                        r"\['customer-2'\]",
              catalogue={**GROWING, "vips": [{"id": "customer-2"}, {"id": "customer-x"}]})


def test_growing_catalogue_attributes_prosimos_cant_parse_are_rejected(tmp_path):
    _rejected(tmp_path, r"catalogue 'customer': attributes\[0\]: .*probabilities' sum should be equal to 1",
              catalogue={"customer": {"attributes": [
                  {"name": "segment", "type": "discrete", "values": [{"key": "retail", "value": 0.5}]}]}})
