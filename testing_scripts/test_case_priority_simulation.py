import json

import pandas as pd
import pytest
from prosimos.simulation_properties_parser import (
    BATCH_PROCESSING_SECTION,
    CASE_ATTRIBUTES_SECTION,
    PRIORITISATION_RULES_SECTION,
)

from test_discovery import assets_path
from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation
from testing_scripts.test_batching import _setup_arrival_distribution
from testing_scripts.test_case_priority_is_true import BUSINESS, REGULAR

NOT_KNOWN = "Not Known"

# setup for the distribution of the arrival rate
def ARRIVAL_DISTR(value_seconds):
    return {
        "distribution_name": "fix",
        "distribution_params": [{"value": value_seconds}, {"value": 0}, {"value": 1}],
    }


# setup for the case attributes generation
THREE_CLIENT_TYPES_ATTRS = [
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

# setup for the prioritisation rules generation
ORDERED_NOT_INTERSECTED_PRIORITISATION_RULES = [
    {
        "priority_level": 1,
        "rules": [
            [{"attribute": "client_type", "comparison": "=", "value": BUSINESS}],
        ],
    },
    {
        "priority_level": 2,
        "rules": [[{"attribute": "client_type", "comparison": "=", "value": REGULAR}]],
    },
]

NOT_ORDERED_INTERSECTED_PRIORITISATION_RULES = [
    {
        "priority_level": 2,
        "rules": [
            [{"attribute": "client_type", "comparison": "=", "value": REGULAR}],
            [{"attribute": "client_type", "comparison": "=", "value": BUSINESS}],
        ],
    },
    {
        "priority_level": 1,
        "rules": [
            [{"attribute": "client_type", "comparison": "=", "value": BUSINESS}],
        ],
    },
]

no_batching_test_cases = [
    # there are two rules that are not intersected
    # they are sorted by the priority level
    (ORDERED_NOT_INTERSECTED_PRIORITISATION_RULES),
    # there are two rules and they intersect
    # (cases with client_type = BUSINESS match both of the priority level)
    # additionally, the rule with lower priority level is the first one in the list
    (NOT_ORDERED_INTERSECTED_PRIORITISATION_RULES),
]


@pytest.mark.parametrize(
    "prioritisation_rules",
    no_batching_test_cases,
)
def test__no_batching_only_priority__correct_log(assets_path, prioritisation_rules):
    """
    Input:      Generate 4 cases simultaneously.
                Assigning three client_types based on distribution
    Execute:    Whole simulation.
    Verifying:  Order of the executed tasks aligns with the client_types:
                BUSINESS - first, REGULAR - second,
                NOT_KNOWN - does not have any priority accordingly to the rules
                but being executed the last based on the default priority level.
    """

    json_path = assets_path / "timer_with_task.json"

    _setup_and_write_arrival_distr_case_attr_priority_rules(
        json_path, ARRIVAL_DISTR(0), THREE_CLIENT_TYPES_ATTRS, prioritisation_rules, []
    )

    # ====== ACT ======
    df = _run_simulation_until_all_client_types_present(assets_path, 20)

    # ====== ASSERT ======
    # 1) replace the value by priority assigned to it
    # 2) verify that the newly mapped values are ordered ascendingly
    df = df.replace([BUSINESS, REGULAR, NOT_KNOWN], [0, 1, 2])
    _verify_column_values_increase(
        df, "client_type", "Activities", "by priorily level of cases"
    )


def test__batching_and_prioritiation__correct_order_inside_batch(assets_path):
    """
    Input:          Batch executes when there are 4 items
                    Three types of client_types associated with each case
    Verifying:      Order of the activity execution inside the batch execution
                    follows the priority rules defined in the simulation scenario
    """
    json_path = assets_path / "timer_with_task.json"

    batch_processing = [
        {
            "task_id": "Activity_002wpuc",
            "type": "Parallel",
            "batch_frequency": 1.0,
            "size_distrib": [{"key": "1", "value": 0}, {"key": "2", "value": 1}],
            "duration_distrib": [{"key": "2", "value": 0.8}],
            "firing_rules": [[{"attribute": "size", "comparison": "=", "value": 4}]],
        }
    ]

    _setup_and_write_arrival_distr_case_attr_priority_rules(
        json_path,
        ARRIVAL_DISTR(1200),
        THREE_CLIENT_TYPES_ATTRS,
        ORDERED_NOT_INTERSECTED_PRIORITISATION_RULES,
        batch_processing,
    )

    # ====== ACT ======
    df = _run_simulation_until_all_client_types_present(assets_path, 10)

    # ====== ASSERT ======
    # 1) replace the value by priority assigned to it
    # 2) group value by the same way they were batched
    # 3) verify that the order INSIDE the batch follow the defined priority in the simulation scenario
    df = df.replace([BUSINESS, REGULAR, NOT_KNOWN], [0, 1, 2])
    logs_d_task = df[df["activity"] == "Task 1"]
    grouped_by_start_and_resource = logs_d_task.groupby(by=["start_time"])
    for _, item in grouped_by_start_and_resource:
        _verify_column_values_increase(
            item,
            "client_type",
            "Activities inside the batch",
            "by priority level of cases",
        )


def test__batching_and_prioritiation__correct_order_outside_batch(assets_path):
    # """
    # Input:            Batch executes when there are 4 items.
    #                   Three types of client_types associated with each case.
    #                   All 20 process instances start at the same moment.
    # Expected:         Order of the activity execution outside the batch execution.
    #                   This means that, first, the priority should be calculated for the whole batch.
    #                   After this, we insert the batch to the task queue with the calculated priority.
    # Verifying:        Batches are executed according to the "general" priority level of the batch.
    #                   Events are not impacted by the priority and
    #                   are executed according to the start_time.
    # """
    json_path = assets_path / "timer_with_task.json"

    batch_processing = [
        {
            "task_id": "Activity_002wpuc",
            "type": "Parallel",
            "batch_frequency": 1.0,
            "size_distrib": [{"key": "1", "value": 0}, {"key": "2", "value": 1}],
            "duration_distrib": [{"key": "2", "value": 0.8}],
            "firing_rules": [[{"attribute": "size", "comparison": "=", "value": 4}]],
        }
    ]

    _setup_and_write_arrival_distr_case_attr_priority_rules(
        json_path,
        ARRIVAL_DISTR(0),
        THREE_CLIENT_TYPES_ATTRS,
        ORDERED_NOT_INTERSECTED_PRIORITISATION_RULES,
        batch_processing,
    )

    # ====== ACT ======
    df = _run_simulation_until_all_client_types_present(assets_path, 20)

    df = df.replace([BUSINESS, REGULAR, NOT_KNOWN], [0, 1, 2])
    grouped_by_task_name_and_start = df.groupby(by=["activity", "start_time"])

    # the list will contain
    # all batched task reduced to one row with the highest priority
    priority_calc_list = []

    for _, group in grouped_by_task_name_and_start:
        if group.iloc[0]["activity"] == "Task 1":
            # activity "Task 1" is batched
            # so we reduce all tasks in the batch to one row with the highest priority

            # minimum value of client_type refers to the highest priority of the case
            highest_priority = group.iloc[0]["client_type"].min()
            row_index = group.iloc[0].name

            # update the first in the group with the highest priority
            group.at[row_index, "client_type"] = highest_priority
            # add the reduced (one instead of four)
            priority_calc_list.append(group.iloc[0])

    # verify that batched task are executed in accordance to the priority
    # all tasks were enabled at the same moment of time
    prioritised_df = pd.concat(priority_calc_list, axis=1).T
    _verify_column_values_increase(
        prioritised_df,
        "client_type",
        "Activities outside the batch",
        "by priority level of cases",
    )

    # events are not impacted by the prioritisation
    # thus the only requirement to them is:
    # start_time should increase with the following row
    events_only = df[df["activity"] == "15m"]
    _verify_column_values_increase(events_only, "start_time", "Events", "by start time")


def _setup_and_write_arrival_distr_case_attr_priority_rules(
    json_path: str,
    new_arrival_dist,
    case_attributes,
    new_priority_rules,
    batch_processing,
):
    with open(json_path, "r") as f:
        json_dict = json.load(f)

    _setup_arrival_distribution(json_dict, new_arrival_dist)
    json_dict[CASE_ATTRIBUTES_SECTION] = case_attributes
    json_dict[PRIORITISATION_RULES_SECTION] = new_priority_rules
    json_dict[BATCH_PROCESSING_SECTION] = batch_processing

    with open(json_path, "w+") as json_file:
        json.dump(json_dict, json_file)


def _verify_column_values_increase(
    df, column_name: str, error_entity_name: str, sort_by_name: str
):
    "Verifying all values in the mentioned column are increasing"
    assert df[
        column_name
    ].is_monotonic_increasing, (
        f"{error_entity_name} are not sorted ascendingly {sort_by_name}"
    )


def _run_simulation_until_all_client_types_present(
    assets_path, total_cases
) -> pd.DataFrame:
    """
    Runs simulation till the moment we have all three client types present in the log
    Returns the resulted simulation log as a DataFrame
    """
    model_path = assets_path / "timer_with_task.bpmn"
    json_path = assets_path / "timer_with_task.json"
    sim_stats = assets_path / "timer_with_task_stats.csv"
    sim_logs = assets_path / "timer_with_task_logs.csv"

    start_string = "2022-06-21 13:22:30.035185+03:00"

    while True:
        _ = run_diff_res_simulation(
            start_string, total_cases, model_path, json_path, sim_stats, sim_logs, True
        )

        df = pd.read_csv(sim_logs)

        unique_client_types = df["client_type"].unique()
        if unique_client_types.size == 3:
            break

    return df
