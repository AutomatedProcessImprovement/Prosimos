"""
initial (docs/messaging.md, "aggregate", and docs/catalogue.md, "With"): a number already in the pile before any
value, which a sum, count, min or max works over together with the claimed messages (aggregate) or the picked
objects (with).
"""
import pytest

from prosimos.exceptions import InvalidSimScenarioException
from prosimos.orchestration.catalogue import _Brought
from prosimos.orchestration.config import ProcessSpec, SimulationConfig
from prosimos.orchestration.prosimos_engine import ProsimosEngine
from prosimos.orchestration.protocol import Message
from testing_scripts.test_aggregate import WEIGHTS, _history, _ocel_history, _rejected, _run, _settings
from testing_scripts.test_append import _run as _run_orders
from testing_scripts.test_catalogue import _config_with
from testing_scripts.test_count_over_list import PACKAGES, START
from testing_scripts.test_pick import CUSTOMERS, PICKING, PRODUCTS_LIST
from testing_scripts.test_pick_count import PRODUCTS, SALES, _at_creation, _sales_picking
from testing_scripts.test_with import PRICE
from testing_scripts.test_with import _run as _run_picking

# the examples of docs/catalogue.md: initial 5, over the values 476 and 99.99, and over none
OVER_TWO = {"sum": 580.99, "count": 7, "min": 5, "max": 476}


class _Picked:
    """A pick that picked these ids for the latest case."""

    def __init__(self, last):
        self.last = last


# with

@pytest.mark.parametrize("function, expected", OVER_TWO.items())
def test_with_works_over_initial_and_the_picked_objects(function, expected):
    brought = _Brought("price", _Picked(["iPad", "Echo"]), {"iPad": 476, "Echo": 99.99}, function, 5)

    assert brought.get_next_value() == pytest.approx(expected)


@pytest.mark.parametrize("function", ["sum", "count", "min", "max"])
def test_with_over_no_objects_gives_initial(function):
    assert _Brought("price", _Picked([]), {"iPad": 476}, function, 5).get_next_value() == 5


def test_with_works_over_initial_and_a_single_pick():
    brought = _Brought("price", _Picked("iPad"), {"iPad": 476}, "sum", 5)

    assert brought.get_next_value() == 481


def test_without_initial_with_stays_as_before():
    assert [_Brought("price", _Picked([]), {}, function).get_next_value()
            for function in ("sum", "count", "min", "max")] == [0, 0, None, None]


def test_every_order_costs_its_products_prices_plus_a_fee(tmp_path):
    # an order's price: the sum of its items' prices plus a fee of 5.00
    report, _, _ = _run_picking(_sales_picking(tmp_path, {**PRODUCTS, "count": {"object_attribute": "items"},
                                                          "with": {"price": {"sum": "price", "initial": 5.0}}}),
                                tmp_path)

    products, prices = _at_creation(report, "Sales", "products"), _at_creation(report, "Sales", "price")

    assert len(prices) == 100
    assert all(prices[order] == pytest.approx(5.0 + sum(PRICE[product] for product in products[order]))
               for order in prices)


@pytest.mark.parametrize("initial", ["5", True, None])
def test_with_an_initial_that_isnt_a_number_is_rejected(tmp_path, initial):
    with pytest.raises(ValueError, match=rf"{SALES} with 'price' initial must be a number, got {initial!r}"):
        _sales_picking(tmp_path, {**PRODUCTS, "count": 3, "with": {"price": {"sum": "price", "initial": initial}}})


def test_initial_on_a_plain_copy_is_rejected(tmp_path):
    path = _config_with(tmp_path, catalogues={"Sales": {"customer": CUSTOMERS, "product": PRODUCTS_LIST}},
                        references={"Sales": [{**PICKING, "pick": "uniform"},
                                              {**PRODUCTS, "with": {"price": {"initial": 5}}}]})

    with pytest.raises(ValueError, match=rf"{SALES} with 'price' has initial but no function; initial is the first "
                                         r"value a sum, count, min or max works over, and a plain copy has none"):
        SimulationConfig.from_json(path)


# aggregate: a package collects items weighing 0.8, 2.1 and 1.2, the first starting it

@pytest.mark.parametrize("function, values", [
    ("sum", [5.8, 7.9, 9.1]),
    ("count", [6, 7, 8]),
    ("min", [0.8, 0.8, 0.8]),  # initial takes part, as a ceiling
    ("max", [5, 5, 5]),  # and as a floor
])
def test_aggregate_works_over_initial_and_the_claimed_messages(tmp_path, function, values):
    report, _ = _run(tmp_path, WEIGHTS, {"result": {function: "weight", "initial": 5}})

    assert _history(report, "result") == [(pytest.approx(value), at) for value, at in
                                          zip(values, ["09:00", "09:10", "09:20"])]


@pytest.mark.parametrize("function", ["sum", "count", "min", "max"])
def test_aggregate_starts_with_initial_before_the_first_message(tmp_path, function):
    def collecting_only(settings):
        del settings["messages"]["consume"][0]["aggregate"]

    engine = ProsimosEngine(ProcessSpec("Packages", PACKAGES, _settings(
        tmp_path, {"result": {function: "weight", "initial": 5}}, collecting_only)), START, seed=1)
    engine.deliver(Message("ItemPicked", {"weight": 0.8}, source="Picking"), START)

    assert engine._env.sim_setup.bpmn_graph.all_attributes[0]["result"] == 5


def test_the_ocel_output_leaves_out_an_aggregates_initial_until_the_first_message(tmp_path):
    # an order created by its arrival schedule at 09:00, collecting 3 ItemReady at Catch_Items from 09:20; the run
    # report and the CSV log keep the initial value, the OCEL output starts at the first message
    def aggregating(settings):
        settings["messages"]["consume"][0]["aggregate"] = {"weight": {"sum": "weight", "initial": 0.5},
                                                           "heaviest": {"max": "weight", "initial": 1}}

    report, log, _ = _run_orders(tmp_path, WEIGHTS, aggregating, ocel_attributes=("weight", "heaviest"))

    history = [(name, value, time.strftime("%H:%M")) for _, case_object in report.objects
               if case_object.object_id == "order-0" for name, value, time in case_object.attributes
               if name in ("weight", "heaviest")]
    assert [(value, at) for name, value, at in history if name == "weight"] == [
        (0.5, "09:00"), (1.3, "09:20"), (pytest.approx(3.4), "09:30"), (4.6, "09:40")]
    assert [(value, at) for name, value, at in history if name == "heaviest"] == [
        (1, "09:00"), (1, "09:20"), (2.1, "09:30"), (2.1, "09:40")]
    assert _ocel_history(tmp_path, "order-0", "weight") == [(1.3, "09:20"), (pytest.approx(3.4), "09:30"),
                                                            (4.6, "09:40")]
    assert _ocel_history(tmp_path, "order-0", "heaviest") == [(1, "09:20"), (2.1, "09:30"), (2.1, "09:40")]
    assert 0.5 in log["weight"].tolist()


@pytest.mark.parametrize("initial", ["5", True, None])
def test_an_aggregate_initial_that_isnt_a_number_is_rejected(tmp_path, initial):
    _rejected(tmp_path, {"weight": {"sum": "weight", "initial": initial}},
              rf"messages.consume\[0\]: aggregate 'weight' initial must be a number, got {initial!r}")


def test_an_aggregate_with_initial_but_no_function_is_rejected(tmp_path):
    _rejected(tmp_path, {"weight": {"initial": 5}},
              r"messages.consume\[0\]: aggregate 'weight' must be one function and a message attribute")


def test_two_entries_aggregating_into_one_attribute_with_different_initials_are_rejected(tmp_path):
    def different(settings):
        settings["messages"]["consume"][1]["aggregate"] = {"weight": {"sum": "weight", "initial": 1}}

    _rejected(tmp_path, {"weight": {"sum": "weight"}},
              r"messages.consume\[1\]: aggregate writes 'weight' with initial 1, but messages.consume\[0\] with "
              r"initial None: an attribute starts from one value", different)
