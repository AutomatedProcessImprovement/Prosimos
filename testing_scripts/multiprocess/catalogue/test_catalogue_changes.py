"""
Catalogue changes (docs/catalogue.md, "Changes"): an organization's catalogue_changes change its catalogues'
attributes on the clock, every period from the start of the run or once at a moment. They are worked out in advance
into a timetable, and whoever reads a value reads the one valid at that moment: a pick's 'with' when the case is
created, a received 'with' at the claim, a growing catalogue's new object when it is created. The OCEL file writes a
changing attribute's value at every change, up to the end of the run.
"""
import dataclasses
import json
from datetime import datetime, timedelta

import pytest

from prosimos.orchestration.catalogue.model import CatalogueChange, CatalogueObject, GrowingCatalogue, Reference
from prosimos.orchestration.catalogue.timetable import Timetable
from prosimos.orchestration.config import Organization, ProcessSpec, SimulationConfig
from prosimos.orchestration.simulation import run_multiprocess_simulation
from testing_scripts.multiprocess.catalogue.test_catalogue import config_with, with_partner
from testing_scripts.multiprocess.catalogue.test_pick import PICKING, START as LOANS_START, loans_config, run
from testing_scripts.multiprocess.catalogue.test_pick_quantity import PRODUCTS, at_creation, sales_picking
from testing_scripts.multiprocess.catalogue.test_with import PRICE, received, warehouse_receiving
from testing_scripts.multiprocess.messaging.test_collect import ORDERS, START, at, settings_with
from testing_scripts.multiprocess.scripted_engine import ScriptedEngine

DAY = timedelta(days=1)


def _price(*changes):
    """The price of a product whose price is 100 before any change, as a function of the moment it is read at."""
    timetable = Timetable(changes, START)
    return lambda when: timetable.value("product", "price", "Kindle", 100, when)


# the timetable

def test_a_repeated_change_is_counted_from_the_start_of_the_run():
    price = _price(CatalogueChange("product", "price", "multiply", 1.5, every=timedelta(days=60)))

    assert [price(START + days * DAY) for days in (0, 59, 60, 119, 120, 365)] == [
        100, 100, 150, 150, 225, 100 * 1.5 ** 6]


def test_a_read_exactly_at_a_change_sees_the_new_value():
    price = _price(CatalogueChange("product", "price", "add", 10, at=START + 3 * DAY))

    assert [price(START + 3 * DAY - timedelta(microseconds=1)), price(START + 3 * DAY)] == [100, 110]


def test_reads_in_any_order_see_the_same_values():
    change = CatalogueChange("product", "price", "multiply", 2, every=timedelta(hours=1))
    moments = [START + timedelta(minutes=minutes) for minutes in (300, 30, 600, 60, 59, 61, 0)]
    forwards, backwards = _price(change), _price(change)

    in_order = {moment: forwards(moment) for moment in sorted(moments)}

    assert {moment: backwards(moment) for moment in moments} == in_order
    assert in_order[START + timedelta(minutes=600)] == 100 * 2 ** 10


@pytest.mark.parametrize("first, then, expected", [
    (CatalogueChange("product", "price", "add", 10, at=START + DAY),
     CatalogueChange("product", "price", "multiply", 2, at=START + DAY), 220),
    (CatalogueChange("product", "price", "multiply", 2, at=START + DAY),
     CatalogueChange("product", "price", "add", 10, at=START + DAY), 210),
])
def test_changes_at_the_same_moment_apply_in_the_listed_order(first, then, expected):
    assert _price(first, then)(START + DAY) == expected


def test_a_repeated_and_a_one_off_change_at_the_same_moment_apply_in_the_listed_order():
    price = _price(CatalogueChange("product", "price", "set", 50, at=START + 2 * DAY),
                   CatalogueChange("product", "price", "add", 1, every=DAY))

    assert [price(START + days * DAY) for days in (1, 2, 3)] == [101, 51, 52]


def test_round_rounds_after_every_change_and_without_it_values_arent_rounded():
    every = timedelta(days=60)
    rounded = Timetable((CatalogueChange("product", "price", "multiply", 1.03, every=every, round=2),), START)
    exact = Timetable((CatalogueChange("product", "price", "multiply", 1.03, every=every),), START)

    assert [rounded.value("product", "price", "iPhone X", 699, START + days * DAY) for days in (60, 120)] == [
        719.97, 741.57]  # 719.97 * 1.03 = 741.5691
    assert exact.value("product", "price", "iPhone X", 699, START + 120 * DAY) == 699 * 1.03 * 1.03


def test_a_change_with_objects_leaves_the_other_objects_and_attributes_as_they_are():
    timetable = Timetable((CatalogueChange("product", "price", "multiply", 0.8, at=START, objects=("Kindle",)),),
                          START)

    assert [timetable.value("product", "price", "Kindle", 100, START + DAY),
            timetable.value("product", "price", "Echo", 100, START + DAY),
            timetable.value("product", "weight", "Kindle", 0.5, START + DAY)] == [80, 100, 0.5]


# the readers in a run

def _with_changes(config, *changes):
    """config, of one organization, with changes as its catalogue_changes."""
    [organization] = config.organizations
    return dataclasses.replace(config, organizations=[dataclasses.replace(organization, catalogue_changes=changes)])


@pytest.fixture(scope="module")
def price_rise(tmp_path_factory):
    """Order Management's Sales alone, 100 orders, each picking one product per item and starting with their total
    price; every product's price rises by 10% every three days."""
    path = tmp_path_factory.mktemp("price_rise")
    config = sales_picking(path, {**PRODUCTS, "quantity": {"object_attribute": "items"},
                                  "with": {"price": {"sum": "price"}}})
    rise = CatalogueChange("product", "price", "multiply", 1.1, every=3 * DAY)
    return run(_with_changes(config, rise), path) + (config.start_datetime,)


def _created(report, process):
    return {case.object_id: case.created for name, case in report.objects if name == process}


def test_a_pick_reads_the_values_valid_when_the_case_is_created(price_rise):
    report, _, _, start = price_rise
    products, prices, created = (at_creation(report, "Sales", "products"), at_creation(report, "Sales", "price"),
                                 _created(report, "Sales"))

    rises = {order: (created[order] - start) // (3 * DAY) for order in products}

    assert len(set(rises.values())) > 3  # orders placed between several rises
    assert all(prices[order] == pytest.approx(sum(PRICE[product] for product in products[order]) * 1.1 ** rises[order])
               for order in products)


def test_a_value_already_given_to_a_case_doesnt_change(price_rise):
    report, log, _, _ = price_rise

    # an order keeps the price it was placed at
    assert all([name for name, _, _ in case.attributes].count("price") == 1
               for process, case in report.objects if process == "Sales")
    assert log[log["process"] == "Sales"].groupby("case_id")["price"].nunique().max() == 1


def test_the_rest_of_the_run_is_as_without_the_changes(tmp_path, price_rise):
    _, changed, _, _ = price_rise
    _, unchanged, _ = run(sales_picking(tmp_path, {**PRODUCTS, "quantity": {"object_attribute": "items"},
                                                   "with": {"price": {"sum": "price"}}}), tmp_path, "unchanged")

    assert changed.drop(columns="price").equals(unchanged.drop(columns="price"))


def test_a_received_value_brings_the_values_valid_when_the_case_is_created(tmp_path):
    # items created at 09:00, 09:10, 09:20 and 09:30; from 09:10 on, a Kindle costs 100
    sent = ["Kindle", "Kindle", "iPad", "Kindle"]
    report = warehouse_receiving(tmp_path, sent, [CatalogueChange("product", "price", "set", 100, at=at("09:10"),
                                                                  objects=("Kindle",))])

    items = sorted(received(report, "Warehouse").values(), key=lambda item: item[1])
    assert [(values["product"], values["price"], created.strftime("%H:%M")) for values, created in items] == [
        ("Kindle", PRICE["Kindle"], "09:00"), ("Kindle", 100, "09:10"), ("iPad", PRICE["iPad"], "09:20"),
        ("Kindle", 100, "09:30")]


def test_a_value_copied_at_a_catch_brings_the_values_valid_at_the_claim(tmp_path):
    # order-0 claims an ItemReady at 09:20, 09:30 and 09:40, copying last_item; from 09:30 on, every item's price is
    # doubled, so the claim at 09:30 already sees it
    prices = {"i17": 2.5, "i23": 4, "i12": 10}
    reference = Reference("last_item", "item", with_attributes=(("last_price", "price"),))
    shop = Organization("Shop", [ProcessSpec("Orders", ORDERS, settings_with(tmp_path, lambda settings: None), 1,
                                             "order", references=(reference,))],
                        {"item": [CatalogueObject(item, (("price", price),)) for item, price in prices.items()]},
                        (CatalogueChange("item", "price", "multiply", 2, at=at("09:30")),))
    picking = ScriptedEngine("Picking")
    for minutes, item in zip((20, 30, 40), prices):
        picking.publish_at(at(f"09:{minutes}"), "ItemReady", order_id="order-0", item_id=item)

    report = run_multiprocess_simulation(SimulationConfig([shop], START, 1, extra_processes=["Picking"]), None,
                                         {"Picking": picking})

    [order] = [case for name, case in report.objects if name == "Orders"]
    assert [(value, time.strftime("%H:%M")) for name, value, time in order.attributes
            if name == "last_price" and value is not None] == [(2.5, "09:20"), (8, "09:30"), (20, "09:40")]


def _parsed(time):
    return datetime.fromisoformat(time.replace("Z", "+00:00"))


def _timed(ocel, object_type, attribute):
    """Object id -> the attribute's values as the OCEL file writes them, (value, time), in order."""
    return {obj["id"]: [(value["value"], _parsed(value["time"])) for value in obj["attributes"]
                        if value["name"] == attribute] for obj in ocel["objects"] if obj["type"] == object_type}


def _end(ocel):
    """The end of the run: its last event."""
    return max(_parsed(event["time"]) for event in ocel["events"])


@pytest.fixture(scope="module")
def growing_run(tmp_path_factory):
    """30 loans, each creating a customer, who draws a credit limit of 1000; 100 more every hour from the start of
    the run."""
    customers = GrowingCatalogue(({"name": "credit_limit", "type": "discrete", "values": [{"key": 1000, "value": 1}]},))
    config = _with_changes(loans_config(30, customers, "new"),
                           CatalogueChange("customer", "credit_limit", "add", 100, every=timedelta(hours=1)))
    _, _, ocel = run(config, tmp_path_factory.mktemp("growing"))
    return ocel


def _hours(moment):
    return (moment - LOANS_START) // timedelta(hours=1)


def test_a_growing_catalogues_new_object_starts_from_the_value_valid_when_it_is_created(growing_run):
    limits = _timed(growing_run, "customer", "credit_limit")

    first = [limit[0] for limit in limits.values()]
    assert len(first) == 30 and len({_hours(created) for _, created in first}) > 3
    assert all(value == 1000 + 100 * _hours(created) for value, created in first)


# the OCEL output

def test_the_ocel_file_has_a_timed_value_at_every_change_up_to_the_end_of_the_run(price_rise):
    _, _, ocel, start = price_rise
    rises = (_end(ocel) - start) // (3 * DAY)

    prices = _timed(ocel, "product", "price")

    assert rises > 3 and len(prices) == len(PRICE)
    for product, timed in prices.items():
        assert [time for _, time in timed] == [start + rise * 3 * DAY for rise in range(rises + 1)]
        assert [value for value, _ in timed] == pytest.approx([PRICE[product] * 1.1 ** rise
                                                               for rise in range(rises + 1)])


def test_a_new_object_gets_the_changes_after_its_creation(growing_run):
    end = _end(growing_run)

    for timed in _timed(growing_run, "customer", "credit_limit").values():
        _, created = timed[0]
        assert timed[1:] == [(1000 + 100 * hour, LOANS_START + hour * timedelta(hours=1))
                             for hour in range(_hours(created) + 1, _hours(end) + 1)]


def test_changes_at_one_moment_give_one_timed_value_and_changes_after_the_run_none(tmp_path):
    config = sales_picking(tmp_path, {**PRODUCTS, "quantity": {"object_attribute": "items"}})
    start = config.start_datetime
    _, _, ocel = run(_with_changes(config, *(CatalogueChange("product", "price", operation, value, at=at,
                                                             objects=("Kindle",)) for operation, value, at in (
        ("add", 10, start + DAY), ("multiply", 2, start + DAY), ("set", 1, start + 1000 * DAY)))), tmp_path)

    prices = _timed(ocel, "product", "price")

    assert prices["Kindle"] == [(PRICE["Kindle"], start), ((PRICE["Kindle"] + 10) * 2, start + DAY)]
    assert prices["Echo"] == [(PRICE["Echo"], start)]


# load errors

CHANGE = {"catalogue": "product", "attribute": "price", "every": {"days": 60}, "multiply": 1.03}
WHERE = r"organizations\[0\].catalogue_changes\[0\]"


def _rejected(tmp_path, change, message, partner=None, catalogues=None, references=None):
    path = config_with(tmp_path, catalogues=catalogues, references=references)
    if partner is not None:
        path = with_partner(path, "Warehouse", partner)
    data = json.loads(path.read_text())
    data["organizations"][0]["catalogue_changes"] = [change]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        SimulationConfig.from_json(path)


@pytest.mark.parametrize("change, message", [
    ({key: value for key, value in CHANGE.items() if key != "attribute"}, r"is missing \['attribute'\]"),
    ({**CHANGE, "catalogue": ""}, r"'catalogue' must be a non-empty string, got ''"),
    ({key: value for key, value in CHANGE.items() if key != "every"},
     r"needs exactly one of every \(repeated\) and at \(once\), got neither"),
    ({**CHANGE, "at": "2024-12-01T00:00:00+00:00"},
     r"needs exactly one of every \(repeated\) and at \(once\), got every and at"),
    ({key: value for key, value in CHANGE.items() if key != "multiply"},
     r"needs exactly one of multiply, add and set, got none"),
    ({**CHANGE, "add": 5}, r"needs exactly one of multiply, add and set, got multiply, add"),
    ({**CHANGE, "multiply": "1.03"}, r"multiply must be a number, got '1.03'"),
    ({**CHANGE, "every": {"days": 0}},
     r'\.every must be \{"weeks" \| "days" \| "hours": n\}, n a number above 0, got \{\'days\': 0\}'),
    ({**CHANGE, "every": {"days": 30, "hours": 12}}, r'\.every must be \{"weeks" \| "days" \| "hours": n\}'),
    ({**CHANGE, "every": 60}, r'\.every must be \{"weeks" \| "days" \| "hours": n\}'),
    ({key: value for key, value in {**CHANGE, "at": "the first of December"}.items() if key != "every"},
     r"at must be a date and time, e.g. \"2024-12-01T00:00:00\+00:00\", got 'the first of December'"),
    ({**CHANGE, "round": 1.5}, r"round must be a whole number of at least 0 \(decimal places\), got 1.5"),
    ({**{key: value for key, value in CHANGE.items() if key != "multiply"}, "set": "cheap", "round": 2},
     r"round needs a number to round, and set gives 'cheap'"),
    ({**CHANGE, "objects": []}, r"objects must be a non-empty list of object ids, got \[\]"),
])
def test_a_malformed_change_is_rejected(tmp_path, change, message):
    _rejected(tmp_path, change, rf"{WHERE}(: | )?{message}")


def test_a_change_of_another_organizations_catalogue_is_rejected(tmp_path):
    # the Warehouse, now a partner's, takes Shop's products with it into Partner; Shop changes them
    _rejected(tmp_path, CHANGE, rf"{WHERE} changes catalogue 'product' of organization Partner; an organization "
                                r"changes its own catalogues",
              partner={"product": [{"id": "Kindle", "attributes": {"price": 129}}]},
              catalogues={"customer": [{"id": "c1"}]}, references={})


def test_a_change_of_an_unknown_catalogue_is_rejected(tmp_path):
    _rejected(tmp_path, {**CHANGE, "catalogue": "products"}, rf"{WHERE} changes unknown catalogue 'products'")


def test_a_change_of_an_attribute_the_objects_dont_have_is_rejected(tmp_path):
    _rejected(tmp_path, {**CHANGE, "attribute": "colour"},
              rf"{WHERE} changes 'colour', which objects \['iPhone X', .*\] of catalogue 'product' don't have")


def test_a_change_of_an_attribute_a_pool_doesnt_draw_is_rejected(tmp_path):
    _rejected(tmp_path, {**CHANGE, "catalogue": "customer", "attribute": "colour"},
              rf"{WHERE} changes 'colour', which catalogue 'customer' doesn't draw in its attributes",
              catalogues={"customer": {"count": 10, "attributes": [
                  {"name": "segment", "type": "discrete", "values": [{"key": "retail", "value": 1}]}]}},
              references={})


@pytest.mark.parametrize("operation", ["multiply", "add"])
def test_multiply_or_add_on_an_attribute_that_isnt_a_number_is_rejected(tmp_path, operation):
    change = {"catalogue": "product", "attribute": "name", "at": "2024-06-01T00:00:00+00:00", operation: 2}
    _rejected(tmp_path, change, rf"{WHERE} {operation} needs a number, and 'name' of objects \['b'\] of catalogue "
                                r"'product' isn't one",
              catalogues={"product": [{"id": "a", "attributes": {"name": 1}}, {"id": "b", "attributes": {"name": "B"}}]},
              references={})


def test_multiply_on_a_pool_attribute_that_draws_text_is_rejected(tmp_path):
    _rejected(tmp_path, {**CHANGE, "catalogue": "customer", "attribute": "segment"},
              rf"{WHERE} multiply needs a number, and catalogue 'customer' doesn't always draw 'segment' as one",
              catalogues={"customer": {"count": 10, "attributes": [
                  {"name": "segment", "type": "discrete", "values": [{"key": "retail", "value": 1}]}]}},
              references={})


def test_set_on_an_attribute_that_isnt_a_number_is_allowed(tmp_path):
    path = config_with(tmp_path, catalogues={"customer": [{"id": "c1", "attributes": {"segment": "retail"}}]},
                       references={})
    data = json.loads(path.read_text())
    data["organizations"][0]["catalogue_changes"] = [{"catalogue": "customer", "attribute": "segment",
                                                      "at": "2024-06-01T00:00:00+00:00", "set": "business"}]
    path.write_text(json.dumps(data))

    [change] = SimulationConfig.from_json(path).organizations[0].catalogue_changes

    assert (change.operation, change.value, change.at.isoformat()) == ("set", "business", "2024-06-01T00:00:00+00:00")


def test_a_change_naming_unknown_objects_is_rejected(tmp_path):
    _rejected(tmp_path, {**CHANGE, "objects": ["Kindle", "Walkman"]},
              rf"{WHERE} names objects \['Walkman'\], which are no ids of catalogue 'product'")


def test_a_change_naming_objects_of_a_growing_catalogue_is_rejected(tmp_path):
    _rejected(tmp_path, {"catalogue": "customer", "attribute": "credit_limit", "every": {"weeks": 4}, "add": 100,
                         "objects": ["customer-1"]},
              rf"{WHERE} names objects \['customer-1'\] of growing catalogue 'customer', which has none when the run "
              r"starts; without objects, the change applies to every object it creates",
              catalogues={"customer": {"attributes": [
                  {"name": "credit_limit", "type": "discrete", "values": [{"key": 1000, "value": 1}]}]}},
              references={"Sales": [{**PICKING, "pick": "new"}]})
