from datetime import datetime, time
import json
import pandas as pd
import pytest

from bpdfr_simulation_engine.batching_processing import (
    AndFiringRule,
    FiringSubRule,
    OrFiringRule,
)
from bpdfr_simulation_engine.resource_calendar import parse_datetime
from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation
from testing_scripts.test_batching import (
    _get_start_time_and_count,
    _setup_arrival_distribution,
    _setup_sim_scenario_file,
    _verify_logs_ordered_asc,
    _verify_same_resource_for_batch,
    assets_path,
)

data_one_week_day = [
    # Rule: daily_hour > 15
    # Current state: 5 tasks waiting for the batch execution: three of them enabled before 15.
    # Expected result: firing rule is enabled because of those three tasks.
    (
        "19/09/22 15:05:26",
        [
            "19/09/22 06:05:26",
            "19/09/22 09:05:26",
            "19/09/22 12:05:26",
            "19/09/22 15:05:26",
        ],
        ">",
        "15",
        True,
        [3],
        "19/09/2022 15:00:00",
    ),
    # Rule: daily_hour = 15
    # Current state: 5 tasks waiting for the batch execution: three of them enabled before 15.
    # Expected result: firing rule is enabled because of those three tasks.
    (
        "19/09/22 15:05:26",
        [
            "19/09/22 06:05:26",
            "19/09/22 09:05:26",
            "19/09/22 12:05:26",
            "19/09/22 15:05:26",
        ],
        "=",
        "15",
        True,
        [3],
        "19/09/2022 15:00:00",
    ),
    # Rule: daily_hour < 15
    # Current state: 5 tasks waiting for the batch execution: three of them enabled before 15.
    # Expected result:  Firing rule is enabled at 14:59 because of those three tasks.
    #                   Only two tasks will be enabled cause they arrived when the rule was enabled.
    #                   Other two tasks will be enabled at midnight (not covered by this test). 
    (
        "19/09/22 15:05:26",
        [
            "19/09/22 06:05:26",
            "19/09/22 09:05:26",
            "19/09/22 12:05:26",
            "19/09/22 15:05:26",
        ],
        "<",
        "15",
        True,
        [2],
        "19/09/2022 09:05:26",
    ),
    (
        "19/09/22 20:17:00",
        [
            "17/09/22 14:30:00",
            "17/09/22 20:00:00",
            "18/09/22 9:00:00",
            "18/09/22 10:00:00",
            "18/09/22 20:00:00",
        ],
        ">",
        "14",
        True,
        [2, 2],
        "17/09/2022 20:00:00",
    ),
]


@pytest.mark.parametrize(
    "curr_enabled_at_str, enabled_datetimes, sign_daily_hour_1, daily_hour_1, expected_is_true, expected_batch_size, expected_start_time_from_rule",
    data_one_week_day,
)
def test_daily_hour_rule_correct_enabled_and_batch_size(
    curr_enabled_at_str,
    enabled_datetimes,
    sign_daily_hour_1,
    daily_hour_1,
    expected_is_true,
    expected_batch_size,
    expected_start_time_from_rule,
):

    # ====== ARRANGE ======
    firing_sub_rule_1 = FiringSubRule(
        "daily_hour", sign_daily_hour_1, time(int(daily_hour_1), 0, 0)
    )

    firing_rule_1 = AndFiringRule([firing_sub_rule_1])
    rule = OrFiringRule([firing_rule_1])

    curr_enabled_at = datetime.strptime(curr_enabled_at_str, "%d/%m/%y %H:%M:%S")
    enabled_datetimes = [
        datetime.strptime(item, "%d/%m/%y %H:%M:%S") for item in enabled_datetimes
    ]
    waiting_time_arr = [curr_enabled_at - item for item in enabled_datetimes]

    current_exec_status = {
        "size": len(waiting_time_arr),
        "waiting_times": waiting_time_arr,
        "enabled_datetimes": enabled_datetimes,
        "curr_enabled_at": curr_enabled_at,
        "is_triggered_by_batch": False,
    }

    # ====== ACT & ASSERT ======
    (is_true, batch_spec, start_time_from_rule) = rule.is_true(current_exec_status)
    assert expected_is_true == is_true
    assert expected_batch_size == batch_spec

    if expected_start_time_from_rule == None:
        assert expected_start_time_from_rule == start_time_from_rule
    else:
        start_dt = start_time_from_rule.strftime("%d/%m/%Y %H:%M:%S")
        assert expected_start_time_from_rule == start_dt


def test_daily_hour_and_week_day_and_size_rule_correct_enabled_and_batch_size(assets_path):
    """
    Input:      Firing rules of daily_hour < 12 AND size >= 4 AND week_day IN ["Friday", "Monday"]. 
                29 process cases are being generated. A new case arrive every 3 hours.
                Batched task are executed in parallel.
    Expected:   Batched task are executed only in the range from 00:00 - 11:59.
                If batched tasks came after 12:00 (from 00:00 - 23:59),
                then they wait for the next enabled day (Monday or Friday) to be executed.
    Verified:   The start_time of the appropriate grouped D task.
                The number of tasks in every executed batch.
                The resource which executed the batch is the same for all tasks in the batch.
                The start_time of all logs files is being sorted by ASC.
    """

    # ====== ARRANGE & ACT ======
    firing_rules = [
        [
            {"attribute": "daily_hour", "comparison": "<", "value": "12"},
            {"attribute": "size", "comparison": ">=", "value": 4},
            {"attribute": "week_day", "comparison": "=", "value": "Friday"},
        ],
        [
            {"attribute": "daily_hour", "comparison": "<", "value": "12"},
            {"attribute": "size", "comparison": ">=", "value": 4},
            {"attribute": "week_day", "comparison": "=", "value": "Monday"},
        ]
    ]

    sim_logs = assets_path / "batch_logs.csv"

    start_string = "2022-09-29 23:45:30.035185+03:00"
    start_date = parse_datetime(start_string, True)

    _arrange_and_act(assets_path, firing_rules, start_date, 29, 10800)

    # ====== ASSERT ======
    df = pd.read_csv(sim_logs)
    logs_d_task = df[df["activity"] == "D"]
    grouped_by_start = logs_d_task.groupby(by="start_time")

    expected_start_time_keys = [
        ("2022-09-30 08:49:30.035185+03:00", 4),
        ("2022-10-03 00:00:00.035185+03:00", 21),
        ("2022-10-03 11:49:30.035185+03:00", 4)
    ]
    grouped_by_start_items = list(map(_get_start_time_and_count, list(grouped_by_start.groups.items())))
    assert (
        grouped_by_start_items == expected_start_time_keys
    ), f"The start_time for batched D tasks differs. Expected: {expected_start_time_keys}, but was {grouped_by_start_items}"

    # verify that the same resource execute the whole batch
    for _, group in grouped_by_start:
        _verify_same_resource_for_batch(group["resource"])

    # verify that column 'start_time' is ordered ascendingly
    _verify_logs_ordered_asc(df, start_date.tzinfo)


def test_daily_hour_every_day_correct_firing(assets_path):
    """
    Input:      Firing rule of daily_hour > 15. 
                11 process cases are being generated. A new case arrive every 4 hours.
                Batched task are executed in parallel.
    Expected:   Batched task are executed only in the range from 15:00 - 23:59.
                If batched tasks came before 15:00 (from 00:00 - 23:59), then they wait for 15:00 to be executed.
    Verified:   The start_time of the appropriate grouped D task.
                The number of tasks in every executed batch.
                The resource which executed the batch is the same for all tasks in the batch.
                The start_time of all logs files is being sorted by ASC.
    """

    # ====== ARRANGE & ACT ======
    sim_logs = assets_path / "batch_logs.csv"

    start_string = "2022-09-26 3:10:30.035185+03:00"
    start_date = parse_datetime(start_string, True)

    firing_rules = [[{"attribute": "daily_hour", "comparison": ">", "value": "15"}]]

    # 14400 seconds = 4 hours
    _arrange_and_act(assets_path, firing_rules, start_date, 11, 14400)

    # ====== ASSERT ======
    df = pd.read_csv(sim_logs)
    logs_d_task = df[df["activity"] == "D"]
    grouped_by_start = logs_d_task.groupby(by="start_time")

    expected_start_time_keys = [
        ("2022-09-26 15:00:00.000000+03:00", 3),
        ("2022-09-26 19:14:30.035185+03:00", 2),
        ("2022-09-27 15:00:00.000000+03:00", 4),
        ("2022-09-27 19:14:30.035185+03:00", 2)
    ]
    grouped_by_start_items = list(map(_get_start_time_and_count, list(grouped_by_start.groups.items())))
    assert (
        grouped_by_start_items == expected_start_time_keys
    ), f"The start_time for batched D tasks differs. Expected: {expected_start_time_keys}, but was {grouped_by_start_items}"

    # verify that column 'start_time' is ordered ascendingly
    _verify_logs_ordered_asc(df, start_date.tzinfo)

    # verify that the same resource execute the whole batch
    for _, group in grouped_by_start:
        _verify_same_resource_for_batch(group["resource"])


data_hour_every_day_and_size = [
    (
        "assets_path", 
        ">=", 
        [
            ("2022-09-26 15:00:00.000000+03:00", 3),
            ("2022-09-26 23:14:30.035185+03:00", 3),
            ("2022-09-27 15:00:00.000000+03:00", 3),
            ("2022-09-27 23:14:30.035185+03:00", 3)
        ]
    ),
    (
        "assets_path",
        "<=",
        [
            ("2022-09-26 15:00:00.000000+03:00", 3),
            ("2022-09-26 19:14:30.035185+03:00", 2),
            ("2022-09-27 15:00:00.000000+03:00", 3),
            ("2022-09-27 15:14:30.035185+03:00", 2),
            ("2022-09-27 23:14:30.035185+03:00", 2)
        ]
    ),
]

@pytest.mark.parametrize(
    "assets_path_fixture, size_rule_sign, expected_start_time_keys",
    data_hour_every_day_and_size,
)
def test_daily_hour_every_day_and_size_correct_firing(assets_path_fixture, size_rule_sign, expected_start_time_keys, request):
    """
    Input:      Firing rule of daily_hour > 15 and size >= 3. 
                12 process cases are being generated. A new case arrive every 4 hours.
                Batched task are executed in parallel.
    Expected:   Batched task are executed only in the range from 15:00 - 23:59 AND
                when there are at least 3 items waiting for the batching processing.
                If batched tasks came before 15:00 (from 00:00 - 23:59),
                then they wait for 15:00 to be executed.
    Verified:   The start_time of the appropriate grouped D task.
                The number of tasks in every executed batch.
                The resource which executed the batch is the same for all tasks in the batch.
                The start_time of all logs files is being sorted by ASC.
    """

    # ====== ARRANGE & ACT ======
    assets_path = request.getfixturevalue(assets_path_fixture)
    sim_logs = assets_path / "batch_logs.csv"

    start_string = "2022-09-26 3:10:30.035185+03:00"
    start_date = parse_datetime(start_string, True)
    
    firing_rules = [
        [
            {"attribute": "daily_hour", "comparison": ">", "value": "15"},
            {"attribute": "size", "comparison": size_rule_sign, "value": 3}
        ]
    ]

    # 14400 seconds = 4 hours
    _arrange_and_act(assets_path, firing_rules, start_date, 12, 14400)

    # ====== ASSERT ======
    df = pd.read_csv(sim_logs)
    logs_d_task = df[df["activity"] == "D"]
    grouped_by_start = logs_d_task.groupby(by="start_time")

    grouped_by_start_items = list(map(_get_start_time_and_count, list(grouped_by_start.groups.items())))
    assert (
        grouped_by_start_items == expected_start_time_keys
    ), f"The start_time for batched D tasks differs. Expected: {expected_start_time_keys}, but was {grouped_by_start_items}"

    # verify that column 'start_time' is ordered ascendingly
    _verify_logs_ordered_asc(df, start_date.tzinfo)


def _arrange_and_act(assets_path, firing_rules, start_date, num_cases, cases_arrival_rate):
    # ====== ARRANGE ======
    model_path = assets_path / "batch-example-end-task.bpmn"
    basic_json_path = assets_path / "batch-example-with-batch.json"
    json_path = assets_path / "batch-example-nearest-coef.json"
    sim_stats = assets_path / "batch_stats.csv"
    sim_logs = assets_path / "batch_logs.csv"

    with open(basic_json_path, "r") as f:
        json_dict = json.load(f)

    # case arrives every 14400 seconds (= 4 hours)
    # e.g., 14400 seconds = 4 hours
    arrival_distr = {
        "distribution_name": "fix",
        "distribution_params": [{"value": cases_arrival_rate}, {"value": 0}, {"value": 1}],
    }

    _setup_sim_scenario_file(json_dict, None, None, "Parallel", firing_rules)
    _setup_arrival_distribution(json_dict, arrival_distr)

    with open(json_path, "w+") as json_file:
        json.dump(json_dict, json_file)

    # ====== ACT ======
    _, diff_sim_result = run_diff_res_simulation(
        start_date, num_cases, model_path, json_path, sim_stats, sim_logs
    )