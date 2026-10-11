"""
Picking (docs/orchestrator.md, "Picking the object" and "Generated pools"): with pick, a reference chooses the
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

from prosimos.orchestrator import (CatalogueObject, CataloguePool, ProcessSpec, Reference, SimulationConfig, _Pick,
                                   _drawn_catalogue, run_orchestrator)
from testing_scripts.test_catalogue import _config_with
from testing_scripts.test_order_management import CONFIG, _config

ASSETS = Path("testing_scripts/assets")
LOANS = (str(ASSETS / "LoanApp_sequential_9-5.bpmn"), str(ASSETS / "LoanApp_arrival_fix_10.json"))  # no case_attributes
START = pytz.utc.localize(datetime(2022, 6, 21, 9))
POOL = CataloguePool(1500, (
    {"name": "segment", "type": "discrete",
     "values": [{"key": "retail", "value": 0.8}, {"key": "business", "value": 0.2}]},
    {"name": "credit_limit", "type": "continuous",
     "values": {"distribution_name": "norm",
                "distribution_params": [{"value": 5000}, {"value": 1000}, {"value": 1000}, {"value": 9000}]}}))
THREE = [CatalogueObject("c1"), CatalogueObject("c2"), CatalogueObject("c3")]


def _loans(cases, catalogue, pick, seed=1):
    """A config of one process, Loans, whose cases pick a customer."""
    return SimulationConfig([ProcessSpec("Loans", *LOANS, cases, "loans")], START, seed, catalogue={"customers": catalogue},
                            references=[Reference("Loans", "customer", "customers", o2o="applies for", reverse=True,
                                                  link_events=(), pick=pick)])


def _run(config, tmp_path, name="log"):
    report = run_orchestrator(config, tmp_path / f"{name}.csv", ocel_out_path=tmp_path / f"{name}.json")
    with open(tmp_path / f"{name}.json") as file:
        return report, pd.read_csv(tmp_path / f"{name}.csv"), json.load(file)


def _customers_of_cases(report, process):
    """Case id -> the customer it picked."""
    return {case_object.case_id: value for name, case_object in report.objects if name == process
            for attribute, value, _ in case_object.attributes if attribute == "customer"}


# Order Management

def test_order_management_picks_its_customer_and_sales_json_no_longer_draws_it(tmp_path):
    with open(ASSETS / "order_management" / "sales.json") as file:
        assert "customer" not in {attribute["name"] for attribute in json.load(file)["case_attributes"]}
    config = SimulationConfig.from_json(CONFIG)
    assert [reference.pick for reference in config.references if reference.process == "Sales"] == ["uniform"]

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
    written = {obj["id"]: obj for obj in ocel["objects"] if obj["type"] == "customers"}

    assert set(written) == picked
    assert 900 < len(picked) < 1300  # uniform: about 1500 * (1 - e^(-2000/1500)) = 1105 different customers
    assert picked <= {f"customers-{number}" for number in range(1, 1501)}
    for customer in written.values():
        attributes = {attribute["name"]: attribute["value"] for attribute in customer["attributes"]}
        assert attributes["segment"] in ("retail", "business") and 1000 <= attributes["credit_limit"] <= 9000
        assert customer["relationships"]  # each applies for at least one loan
    assert sum(len(customer["relationships"]) for customer in written.values()) == 2000


def test_a_pools_attributes_are_drawn_in_their_shares(pool_run):
    _, _, ocel = pool_run

    segments = Counter(attribute["value"] for obj in ocel["objects"] if obj["type"] == "customers"
                       for attribute in obj["attributes"] if attribute["name"] == "segment")

    assert abs(segments["retail"] / sum(segments.values()) - 0.8) < 0.04


def test_the_same_seed_gives_the_same_pool_and_the_same_picks(tmp_path):
    first, _, first_ocel = _run(_loans(300, POOL, "uniform"), tmp_path, "first")
    again, _, again_ocel = _run(_loans(300, POOL, "uniform"), tmp_path, "again")
    other, _, _ = _run(_loans(300, POOL, "uniform", seed=2), tmp_path, "other")

    def customers(ocel):
        return [obj for obj in ocel["objects"] if obj["type"] == "customers"]

    assert _customers_of_cases(first, "Loans") == _customers_of_cases(again, "Loans")
    assert customers(first_ocel) == customers(again_ocel)
    assert _customers_of_cases(first, "Loans") != _customers_of_cases(other, "Loans")


def test_the_pool_is_drawn_from_the_seed():
    pool = _drawn_catalogue(_loans(1, POOL, "uniform"))["customers"]

    assert pool == _drawn_catalogue(_loans(1, POOL, "uniform"))["customers"]
    assert pool != _drawn_catalogue(_loans(1, POOL, "uniform", seed=2))["customers"]
    assert [obj.id for obj in pool] == [f"customers-{number}" for number in range(1, 1501)]


def test_picking_doesnt_change_the_processs_other_draws(tmp_path):
    # the loans also draw a size from the engine's generator when a case is created, as the customer would be
    # drawn by case_attributes; the picks have a generator of their own, so the sizes and the rest of the log
    # are the same with and without the pick
    with open(LOANS[1]) as file:
        settings = json.load(file)
    settings["case_attributes"] = [{"name": "size", "type": "discrete",
                                    "values": [{"key": "small", "value": 0.5}, {"key": "large", "value": 0.5}]}]
    (tmp_path / "loans.json").write_text(json.dumps(settings))
    spec = ProcessSpec("Loans", LOANS[0], str(tmp_path / "loans.json"), 200, "loans")
    picking = SimulationConfig([spec], START, 1, catalogue={"customers": THREE},
                               references=[Reference("Loans", "customer", "customers", pick="uniform")])

    _, with_pick, _ = _run(picking, tmp_path, "with")
    _, without, _ = _run(dataclasses.replace(picking, references=()), tmp_path, "without")

    assert set(with_pick["customer"]) == {"c1", "c2", "c3"} and "customer" not in without
    assert set(without["size"]) == {"small", "large"}
    assert with_pick.drop(columns="customer").equals(without)


# load errors

PICKING = {"process": "Sales", "case_attribute": "customer", "catalogue": "customers"}


def test_weights_naming_unknown_ids_are_rejected(tmp_path):
    path = _config_with(tmp_path, references=[{**PICKING, "pick": {"weights": {"c1": 3, "c99": 1}}}])

    with pytest.raises(ValueError, match=r"references\[0\] weights name \['c99'\], which are no ids of catalogue 'customers'"):
        SimulationConfig.from_json(path)


def test_weights_on_a_pool_are_rejected(tmp_path):
    path = _config_with(tmp_path, catalogue={"customers": {"count": 10}},
                        references=[{**PICKING, "pick": {"weights": {"customers-1": 1}}}])

    with pytest.raises(ValueError, match=r"references\[0\] has weights on pool 'customers'"):
        SimulationConfig.from_json(path)


@pytest.mark.parametrize("count", [0, -5, "10", 2.5, None])
def test_a_pool_count_below_1_or_not_a_whole_number_is_rejected(tmp_path, count):
    path = _config_with(tmp_path, catalogue={"customers": {"count": count}}, references=[{**PICKING, "pick": "uniform"}])

    with pytest.raises(ValueError, match=r"catalogue 'customers': count must be a whole number of at least 1"):
        SimulationConfig.from_json(path)


def test_pool_attributes_prosimos_cant_parse_are_rejected(tmp_path):
    path = _config_with(tmp_path, catalogue={"customers": {"count": 10, "attributes": [
        {"name": "segment", "type": "discrete", "values": [{"key": "retail", "value": 0.5}]}]}}, references=[])

    with pytest.raises(ValueError, match=r"catalogue 'customers': attributes\[0\]: .*probabilities' sum should be equal to 1"):
        SimulationConfig.from_json(path)


def test_picking_an_attribute_also_in_case_attributes_is_rejected(tmp_path):
    with open(CONFIG.replace("simulation.json", "sales.json")) as file:
        sales = json.load(file)
    sales["case_attributes"].append({"name": "customer", "type": "discrete", "values": [{"key": "c1", "value": 1}]})
    (tmp_path / "sales.json").write_text(json.dumps(sales))
    path = _config_with(tmp_path, references=[{**PICKING, "pick": "uniform"}])
    data = json.loads(path.read_text())
    next(process for process in data["processes"] if process["name"] == "Sales")["json_path"] = str(tmp_path / "sales.json")
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match=r"references\[0\] picks customer, which process Sales also draws in its case_attributes"):
        SimulationConfig.from_json(path)


def test_a_pick_other_than_uniform_or_weights_is_rejected(tmp_path):
    path = _config_with(tmp_path, references=[{**PICKING, "pick": "random"}])

    with pytest.raises(ValueError, match=r'references\[0\] pick must be "uniform" or \{"weights"'):
        SimulationConfig.from_json(path)


def test_weights_that_are_negative_or_all_zero_are_rejected(tmp_path):
    path = _config_with(tmp_path, references=[{**PICKING, "pick": {"weights": {"c1": 0, "c2": 0}}}])

    with pytest.raises(ValueError, match=r"references\[0\] weights must be numbers of at least 0, not all 0"):
        SimulationConfig.from_json(path)


def test_two_references_picking_the_same_attribute_are_rejected(tmp_path):
    path = _config_with(tmp_path, references=[{**PICKING, "pick": "uniform"}, {**PICKING, "pick": "uniform"}])

    with pytest.raises(ValueError, match=r"more than one reference picks customer of process Sales"):
        SimulationConfig.from_json(path)


def test_a_pick_on_an_extra_engine_is_rejected():
    with pytest.raises(ValueError, match=r"references\[0\] picks for Scripted, an engine not built from a ProcessSpec"):
        SimulationConfig([], START, extra_processes=("Scripted",), catalogue={"customers": THREE},
                         references=[Reference("Scripted", "customer", "customers", pick="uniform")])
