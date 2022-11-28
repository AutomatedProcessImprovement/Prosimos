from datetime import timedelta

import pandas as pd
import pytest
from bpdfr_simulation_engine.batch_processing import AndFiringRule, FiringSubRule, OrFiringRule
from bpdfr_simulation_engine.resource_calendar import parse_datetime

from testing_scripts.test_batching import (
    SIM_LOGS_FILENAME, 
    _verify_logs_ordered_asc, _verify_same_resource_for_batch, assets_path
)

from testing_scripts.test_batching_daily_hour import _get_current_exec_status
from testing_scripts.test_batching_ready_wt import _arrange_and_act_exp, _test_range_basic

HALF_AN_HOUR_SEC = 1800
ONE_HOUR_IN_SEC = 3600
TWO_HOURS_IN_SEC = 7200

def test_size_eq_wt_lt_correct():
    # ====== ARRANGE ======
    firing_sub_rule_1 = FiringSubRule("large_wt", "<", ONE_HOUR_IN_SEC)
    firing_sub_rule_2 = FiringSubRule("size", "=", 3)

    firing_rules = AndFiringRule([ firing_sub_rule_1, firing_sub_rule_2 ])
    firing_rules.init_boundaries()

    current_point_in_time = "17/09/22 22:00:00"
    enabled_datetimes = [
        "17/09/22 19:00:00",
        "17/09/22 19:30:00",
        "17/09/22 20:00:00",
    ]
    current_exec_status = _get_current_exec_status(current_point_in_time, enabled_datetimes)

    # ====== ACT & ASSERT ======
    is_true, _, _ = firing_rules.is_true(current_exec_status)
    assert True == is_true

    current_size = current_exec_status["size"]
    batch_size, _ = firing_rules.get_firing_batch_size(current_size, current_exec_status)
    assert batch_size == 3


def test_size_eq_wt_gt_correct():
    # ====== ARRANGE ======
    firing_sub_rule_1 = FiringSubRule("large_wt", "<", ONE_HOUR_IN_SEC)
    firing_sub_rule_2 = FiringSubRule("large_wt", ">", TWO_HOURS_IN_SEC)
    firing_sub_rule_3 = FiringSubRule("size", "=", 3)

    firing_rules = AndFiringRule([ firing_sub_rule_1, firing_sub_rule_2, firing_sub_rule_3 ])
    firing_rules.init_boundaries()
    
    curr_enabled_at = "17/09/22 19:45:00"
    enabled_times = [
        "17/09/22 19:00:00",
        "17/09/22 19:30:00",
    ]
    current_exec_status = _get_current_exec_status(curr_enabled_at, enabled_times)

    # ====== ACT & ASSERT ======
    (is_true, _, _) = firing_rules.is_true(current_exec_status)
    assert False == is_true


data_only_waiting_time = [
    # Rule: waiting time >= 3600, 10 tasks waiting for the batch execution.
    # Current state: waiting time is 3600 sec for two oldest tasks waiting for the execution.
    # Expected result: firing rule is enabled. Num of elements in the batch to be executed: 10.
    (
        "18/09/22 21:01:00",
        [
            "17/09/22 19:00:00",
            "17/09/22 19:30:00",
            "18/09/22 20:00:00",
            "18/09/22 20:30:00",
            "18/09/22 20:45:00",
        ],
        "<=",
        True,
        [2, 2],
        "17/09/2022 19:30:00",
    ),
    # Rule: waiting time > 3600, 10 tasks waiting for the batch execution.
    # Current state: waiting time is 3600 sec for two oldest tasks waiting for the execution.
    # Expected result: firing rule is not enabled (3600 > 3600 doesn't hold).
    (
        "18/09/22 21:01:00",
        [
            "18/09/22 19:00:00",
        ],
        "<=",
        True,
        [1],
        "18/09/2022 20:00:01",
    ),
    # Rule: waiting time > 3600, 10 tasks waiting for the batch execution.
    # Current state: waiting time is 3600 sec for two oldest tasks waiting for the execution.
    # Expected result: firing rule is not enabled (3600 > 3600 doesn't hold).
    (
        "18/09/22 21:01:00",
        [
            "18/09/22 19:00:00",
        ],
        "<",
        True,
        [1],
        "18/09/2022 20:00:00",
    ),
]


@pytest.mark.parametrize(
    "curr_enabled_at_str, enabled_datetimes, rule_sign, expected_is_true, expected_batch_size, expected_batch_start_time", 
    data_only_waiting_time
)
def test_only_large_wt_rule_correct_enabled_and_batch_size(
    curr_enabled_at_str, enabled_datetimes, rule_sign, expected_is_true, expected_batch_size, expected_batch_start_time):

    # ====== ARRANGE ======
    firing_sub_rule_1 = FiringSubRule("large_wt", rule_sign, ONE_HOUR_IN_SEC)
    firing_rules = AndFiringRule([ firing_sub_rule_1 ])
    firing_rules.init_boundaries()

    current_exec_status = _get_current_exec_status(curr_enabled_at_str, enabled_datetimes)

    # ====== ACT & ASSERT ======
    (is_true, batch_spec, batch_start_time) = firing_rules.is_true(current_exec_status)
    actual_batch_start_time_str = batch_start_time.strftime("%d/%m/%Y %H:%M:%S")
    assert expected_is_true             == is_true
    assert expected_batch_size          == batch_spec
    assert expected_batch_start_time    == actual_batch_start_time_str
    

data_range_large_wt = [
    # Rule: waiting time >= 3600, 10 tasks waiting for the batch execution.
    # Current state: waiting time is 3600 sec for two oldest tasks waiting for the execution.
    # Expected result: firing rule is enabled. Num of elements in the batch to be executed: 10.
    (
        "17/09/22 21:01:00",
        [
            "17/09/22 19:00:00",
            "17/09/22 19:15:00",
            "17/09/22 19:30:00",
            "17/09/22 20:15:00",
            "17/09/22 20:45:00",
        ],
        (">", ONE_HOUR_IN_SEC),
        ("<=", TWO_HOURS_IN_SEC),
        True,
        [3],
        "17/09/2022 20:00:01",
    ),
    (
        "17/09/22 23:01:00",
        [
            "17/09/22 19:00:00",
        ],
        (">", ONE_HOUR_IN_SEC),
        ("<=", TWO_HOURS_IN_SEC),
        True,
        [1],
        "17/09/2022 21:00:01",
    ),
    (
        "17/09/22 19:35:00",
        [
            "17/09/22 19:00:00",
            "17/09/22 19:30:00",
        ],
        (">", ONE_HOUR_IN_SEC),
        ("<=", TWO_HOURS_IN_SEC),
        False,
        None,
        None,
    ),
]


@pytest.mark.parametrize(
    "curr_enabled_at_str, enabled_datetimes, first_rule, second_rule, expected_is_true, expected_batch_size, expected_batch_start_time", 
    data_range_large_wt
)
def test_range_large_wt_rule_correct_enabled_and_batch_size(
    curr_enabled_at_str, enabled_datetimes, first_rule, second_rule, expected_is_true, expected_batch_size, expected_batch_start_time):

    # ====== ARRANGE ======
    fr_sign, fr_value = first_rule
    sr_sign, sr_value = second_rule

    firing_sub_rule_1 = FiringSubRule(
        "large_wt", fr_sign, fr_value
    )
    firing_sub_rule_2 = FiringSubRule(
        "large_wt", sr_sign, sr_value
    )
    firing_rule_1 = AndFiringRule([firing_sub_rule_1, firing_sub_rule_2])
    firing_rule_1.init_boundaries()
    rule = OrFiringRule([firing_rule_1])

    _test_range_basic(rule, curr_enabled_at_str, enabled_datetimes, expected_is_true,
        expected_batch_size, expected_batch_start_time)


@pytest.mark.parametrize('execution_number', range(5))
def test_range_large_wt_rule_correct_log_distances(execution_number, assets_path):
    """
    Input: firing rule of waiting time > 200 seconds.
    Expected: batch of tasks will be executed once the oldest task in the batch pull will be > 200 seconds.
    This happens during the 3rd case, so that's when the batch execution is enabled.
    Verified the appropriate start_time and end_time (tasks are executed in parallel).
    """

    # ====== ARRANGE ======
    sim_logs = assets_path / SIM_LOGS_FILENAME

    start_string = "2022-06-21 13:22:30.035185+03:00"

    firing_rules = [
        [
            {"attribute": "large_wt", "comparison": ">", "value": ONE_HOUR_IN_SEC},
            {"attribute": "large_wt", "comparison": "<", "value": TWO_HOURS_IN_SEC}
        ]
    ]
    total_num_cases = 20
    _arrange_and_act_exp(assets_path, firing_rules, start_string, total_num_cases)

    # ====== ASSERT ======

    df = pd.read_csv(sim_logs)

    df["enable_time"] = pd.to_datetime(df["enable_time"], errors="coerce")
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")

    logs_d_tasks = df[df["activity"] == "D"]
    
    # verify total number of tasks executed in batch
    # should be equal to the total number of simulated cases
    actual_count = len(df[df['activity'] == 'D'])
    expected_count = total_num_cases
    assert (
        expected_count == actual_count
    ), f"Total number of activities in batches should equal {expected_count}\
        but was {actual_count}"

    # verify that the same resource execute the whole batch
    grouped_by_start = logs_d_tasks.groupby(by=["start_time"])
    for _, group in grouped_by_start:
        _verify_same_resource_for_batch(group["resource"])

    start_date = parse_datetime(start_string, True)
    _verify_logs_ordered_asc(df, start_date.tzinfo)

    _verify_diff_start_and_first_enable_time(grouped_by_start, ONE_HOUR_IN_SEC, TWO_HOURS_IN_SEC)
    _verify_start_time_batch_one_task(grouped_by_start, TWO_HOURS_IN_SEC)

data_ready_and_large = [
    (
        "17/09/22 19:35:00",
        [
            "17/09/22 19:00:00",
            "17/09/22 19:30:00",
        ],
        AndFiringRule(array_of_subrules=[
            FiringSubRule("large_wt", "<=", ONE_HOUR_IN_SEC),
            FiringSubRule("ready_wt", ">", HALF_AN_HOUR_SEC),
            FiringSubRule("ready_wt", "<", TWO_HOURS_IN_SEC),
        ]),
        False,
        None,
        None,
    ),
    (
        "17/09/22 20:05:00",
        [
            "17/09/22 19:00:00",
            "17/09/22 19:30:00",
        ],
        AndFiringRule(array_of_subrules=[
            FiringSubRule("large_wt", "<=", ONE_HOUR_IN_SEC),
            FiringSubRule("ready_wt", ">", HALF_AN_HOUR_SEC),
            FiringSubRule("ready_wt", "<", TWO_HOURS_IN_SEC),
        ]),
        True,
        [2],
        "17/09/2022 20:00:01",
    ),
    # TODO: check expected result
    # (
    #     "17/09/22 21:05:00",
    #     [
    #         "17/09/22 19:00:00",
    #         "17/09/22 19:30:00",
    #         "17/09/22 20:00:00",
    #         "17/09/22 20:30:00",
    #     ],
    #     AndFiringRule(array_of_subrules=[
    #         FiringSubRule("large_wt", "<=", ONE_HOUR_IN_SEC),
    #         FiringSubRule("ready_wt", ">", HALF_AN_HOUR_SEC),
    #         FiringSubRule("ready_wt", "<", TWO_HOURS_IN_SEC),
    #     ]),
    #     True,
    #     [2, 2],
    #     "17/09/2022 20:00:01",
    # ),
]


@pytest.mark.parametrize(
    "curr_enabled_at_str, enabled_datetimes, and_rule, expected_is_true, expected_batch_size, expected_batch_start_time", 
    data_ready_and_large
)
def test_large_and_ready_together_is_true_correct(
    curr_enabled_at_str, enabled_datetimes, and_rule, expected_is_true, expected_batch_size, expected_batch_start_time):

    # ====== ARRANGE ======
    and_rule.init_boundaries()
    rule = OrFiringRule([and_rule])

    _test_range_basic(rule, curr_enabled_at_str, enabled_datetimes, expected_is_true,
        expected_batch_size, expected_batch_start_time)


def _verify_diff_start_and_first_enable_time(grouped_by_start, min_time_distance_sec, max_time_distance_sec):
    """
    Verify that each first item in the batch follow the rule:
    diff between start and enable time should be between [lower boundary of the rule, higher boundary]
    """
    first_items = grouped_by_start.first().reset_index()
    for _, item in first_items.iterrows():
        actual_diff_start_and_first_enabled = (item['start_time'] - item['enable_time']).seconds
        expected_low_boundary = min_time_distance_sec
        
        assert (
            expected_low_boundary <= actual_diff_start_and_first_enabled <= max_time_distance_sec
        ), f"Expected the difference to be less than {expected_low_boundary},\
            but it was {actual_diff_start_and_first_enabled}"


def _verify_start_time_batch_one_task(grouped_by_start, high_boundary_rule):
    """
    We expect that batch with one task was started at:
    tasks' enabled_time + high_boundary
    """
    batches_with_one_task = grouped_by_start.filter(lambda x: len(x) == 1)
    for _, item in batches_with_one_task.iterrows():
        expected_start_time = item['enable_time'] + timedelta(seconds=high_boundary_rule)
        actual_start_time = item['start_time']

        assert (
            expected_start_time == actual_start_time
        ), f"Expected start_time of {expected_start_time} but was {actual_start_time}."
