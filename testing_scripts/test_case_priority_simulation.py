import json

import pandas as pd

from test_discovery import assets_path
from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation
from testing_scripts.test_batching import _setup_arrival_distribution
from testing_scripts.test_case_priority_is_true import BUSINESS, REGULAR

NOT_KNOWN = "Not Known"


def test__no_batching_only_priority__correct_log(assets_path):
    """
    Input:      Generate 4 cases simultaneously.
                Assigning three client_types based on distribution
    Execute:    Whole simulation.
    Verifying:  Order of the executed tasks aligns with the client_types:
                BUSINESS - first, REGULAR - second,
                NOT_KNOWN - does not have any priority accordingly to the rules
                but being executed the last based on the default priority level.
    """

    model_path = assets_path / "timer_with_task.bpmn"
    json_path = assets_path / "timer_with_task.json"
    sim_stats = assets_path / "timer_with_task_stats.csv"
    sim_logs = assets_path / "timer_with_task_logs.csv"

    start_string = "2022-06-21 13:22:30.035185+03:00"

    # provide setup for the arrival distribution
    arrival_distr = {
        "distribution_name": "fix",
        "distribution_params": [{"value": 0}, {"value": 0}, {"value": 1}],
    }

    # provide setup for the case attributes generation
    case_attributes = [
        {
            "name": "client_type",
            "type": "discrete",
            "values": [
                {"key": REGULAR, "value": 0.5},
                {"key": BUSINESS, "value": 0.25},
                {"key": NOT_KNOWN, "value": 0.25},
            ],
        },
        {
            "name": "loan_amount",
            "type": "continuous",
            "values": {
                "distribution_name": "norm",
                "distribution_params": [
                    {"value": 150},
                    {"value": 100},
                    {"value": 0},
                    {"value": 400},
                ],
            },
        },
    ]

    # provide setup for the prioritisation rules generation
    prioritisation_rules = [
        {
            "priority_level": 1,
            "rules": [
                [{"attribute": "client_type", "condition": "=", "value": BUSINESS}],
            ],
        },
        {
            "priority_level": 2,
            "rules": [
                [{"attribute": "client_type", "condition": "=", "value": REGULAR}]
            ],
        },
    ]

    while True:
        _setup_and_write_arrival_distr_case_attr_priority_rules(
            json_path, arrival_distr, case_attributes, prioritisation_rules
        )

        # ====== ACT ======
        _ = run_diff_res_simulation(
            start_string, 5, model_path, json_path, sim_stats, sim_logs, True
        )

        # ====== ASSERT ======
        df = pd.read_csv(sim_logs)

        unique_client_types = df["client_type"].unique()
        if unique_client_types.size == 3:
            break

    # replace the value by priority assigned to it
    # verify that the newly mapped values are ordered ascendingly
    df = df.replace([BUSINESS, REGULAR, NOT_KNOWN], [0, 1, 2])
    assert df[
        "client_type"
    ].is_monotonic_increasing, (
        f"Cases are executed not accordingly to the priority level"
    )


def _setup_and_write_arrival_distr_case_attr_priority_rules(
    json_path: str, new_arrival_dist, case_attributes, new_priority_rules
):
    with open(json_path, "r") as f:
        json_dict = json.load(f)

    _setup_arrival_distribution(json_dict, new_arrival_dist)
    json_dict["case_attributes"] = case_attributes
    json_dict["prioritization_rules"] = new_priority_rules

    with open(json_path, "w+") as json_file:
        json.dump(json_dict, json_file)
