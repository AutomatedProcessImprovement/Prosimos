"""
load_balancing (docs/orchestrator.md, "Configuration"): each consumer group may choose the order its members are
offered a message, weighted or preferred; a group without a policy stays random. Loading and load errors only.
"""
import json
from datetime import datetime
from pathlib import Path

import pytest
import pytz

from prosimos.orchestration.config import SimulationConfig

RUNNING_EXAMPLE = "testing_scripts/assets/running_example/simulation.json"
START = pytz.utc.localize(datetime(2024, 1, 1, 9))
WAREHOUSES = ["Tartu", "Riga", "Tallinn"]


def _from_file(tmp_path, load_balancing):
    """The running example's configuration with load_balancing added, read from a file."""
    with open(RUNNING_EXAMPLE) as file:
        data = json.load(file)
    folder = Path(RUNNING_EXAMPLE).resolve().parent
    for organization in data["organizations"]:
        for process in organization["processes"]:
            process["bpmn_path"], process["json_path"] = (str(folder / process[key]) for key in ("bpmn_path",
                                                                                               "json_path"))
    data["load_balancing"] = load_balancing
    path = tmp_path / "simulation.json"
    path.write_text(json.dumps(data))
    return SimulationConfig.from_json(path)


def _scripted(load_balancing, groups=None):
    """Scripted Sales and three warehouses in a group, with load_balancing."""
    return SimulationConfig([], START, consumer_groups=groups or {"Warehouses": WAREHOUSES, "Sales": ["Sales"]},
                            extra_processes=["Sales", *WAREHOUSES], load_balancing=load_balancing)


@pytest.mark.parametrize("policy", [{"weighted": {"TartuWarehouse": 8, "TallinnWarehouse": 2}},
                                    {"preferred": ["TallinnWarehouse", "TartuWarehouse"]}])
def test_both_policies_load_from_a_configuration_file(tmp_path, policy):
    assert _from_file(tmp_path, {"Warehouses": policy}).load_balancing == {"Warehouses": policy}


def test_without_load_balancing_every_group_stays_random():
    assert SimulationConfig.from_json(RUNNING_EXAMPLE).load_balancing == {}


def test_a_group_of_extra_engines_can_have_a_policy():
    weighted = {"Warehouses": {"weighted": {"Tartu": 5, "Riga": 0.5, "Tallinn": 0}}}

    assert _scripted(weighted).load_balancing == weighted


@pytest.mark.parametrize("load_balancing, problem", [
    ({"Warehouse": {"preferred": WAREHOUSES}},
     r"group 'Warehouse' is no consumer group \(groups: \['Sales', 'Warehouses'\]\)"),
    ({"Warehouses": {"fastest": WAREHOUSES}}, r"group 'Warehouses': unknown policy 'fastest'; allowed: preferred, "
                                              r"weighted"),
    ({"Warehouses": {}}, r"group 'Warehouses' needs exactly one policy, \{\"weighted\": \{<member>: <weight>\}\} or "
                         r"\{\"preferred\": \[<members>\]\}, got \{\}"),
    ({"Warehouses": {"preferred": WAREHOUSES, "weighted": {"Tartu": 1, "Riga": 1, "Tallinn": 1}}},
     r"group 'Warehouses' needs exactly one policy"),
    ({"Warehouses": ["Tartu"]}, r"group 'Warehouses' needs exactly one policy, .*, got \['Tartu'\]"),
    ({"Sales": {"preferred": ["Sales"]}}, r"group 'Sales' has one member, Sales, so there is nothing to choose"),
    ("random", r"it must map consumer groups to a policy, got 'random'"),
])
def test_an_invalid_group_or_policy_is_rejected(load_balancing, problem):
    with pytest.raises(ValueError, match=r"^Invalid load_balancing: .*" + problem):
        _scripted(load_balancing)


@pytest.mark.parametrize("weights, problem", [
    ({"Tartu": 8, "Riga": 2}, r"gives no weight to \['Tallinn'\]"),
    ({"Tartu": 8, "Riga": 2, "Tallinn": 1, "Narva": 1}, r"names \['Narva'\], which are not members of the group "
                                                        r"\['Tartu', 'Riga', 'Tallinn'\]"),
    ({"Tartu": 8, "Riga": -1, "Tallinn": "2"}, r"weights must be numbers of at least 0, got \{'Riga': -1, "
                                               r"'Tallinn': '2'\}"),
    ({"Tartu": True, "Riga": 1, "Tallinn": 1}, r"weights must be numbers of at least 0, got \{'Tartu': True\}"),
    ({"Tartu": 0, "Riga": 0, "Tallinn": 0}, r"weights are all 0: no member would be asked first"),
    (["Tartu", "Riga", "Tallinn"], r"must map each member to its weight, got \['Tartu', 'Riga', 'Tallinn'\]"),
])
def test_invalid_weights_are_rejected(weights, problem):
    with pytest.raises(ValueError, match=r"^Invalid load_balancing: group 'Warehouses': weighted " + problem):
        _scripted({"Warehouses": {"weighted": weights}})


@pytest.mark.parametrize("ranking", [["Tartu", "Riga"], ["Tartu", "Riga", "Tallinn", "Riga"],
                                     ["Tartu", "Riga", "Narva"], "Tartu"])
def test_preferred_must_rank_each_member_once(ranking):
    with pytest.raises(ValueError, match=r"^Invalid load_balancing: group 'Warehouses': preferred must list each "
                                         r"member of the group once, \['Tartu', 'Riga', 'Tallinn'\], got "):
        _scripted({"Warehouses": {"preferred": ranking}})


def test_without_consumer_groups_each_process_is_a_group_of_one():
    with pytest.raises(ValueError, match=r"group 'Tartu' has one member, Tartu, so there is nothing to choose"):
        SimulationConfig([], START, extra_processes=WAREHOUSES, load_balancing={"Tartu": {"preferred": ["Tartu"]}})


def test_every_problem_is_reported_at_once():
    with pytest.raises(ValueError, match=r"group 'Shop' is no consumer group .*; group 'Sales' has one member"):
        _scripted({"Shop": {"preferred": []}, "Sales": {"preferred": ["Sales"]}})
