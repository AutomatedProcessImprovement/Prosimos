import json

import pandas as pd
from prosimos.simulation_properties_parser import (
    ARRIVAL_TIME_CALENDAR,
    BATCH_PROCESSING_SECTION,
    CASE_ATTRIBUTES_SECTION,
    PRIORITISATION_RULES_SECTION,
    RESOURCE_CALENDARS,
)

from test_discovery import assets_path
from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation
from testing_scripts.test_batching import _setup_arrival_distribution
from testing_scripts.test_case_priority_is_true import BUSINESS, REGULAR
from testing_scripts.test_simulation import (
    DEFAULT_ARRIVAL_CALENDAR_TIMER_WITH_TASK,
    DEFAULT_RESOURCE_CALENDARS_TIMER_WITH_TASK,
)

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
    json_dict[ARRIVAL_TIME_CALENDAR] = DEFAULT_ARRIVAL_CALENDAR_TIMER_WITH_TASK
    json_dict[RESOURCE_CALENDARS] = DEFAULT_RESOURCE_CALENDARS_TIMER_WITH_TASK
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
