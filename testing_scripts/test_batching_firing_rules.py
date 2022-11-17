import pytest
from datetime import datetime, timedelta 

from bpdfr_simulation_engine.batching_processing import AndFiringRule, FiringSubRule, OrFiringRule
from testing_scripts.test_batching_daily_hour import _get_current_exec_status


def test_only_size_eq_correct():
    # ====== ARRANGE ======

    firing_sub_rule = FiringSubRule("size", "=", 3)
    firing_rules = AndFiringRule([firing_sub_rule])

    # note: enabled_datetimes and curr_enabled_at does not reflect the real situation
    # not being used for this special use case
    current_exec_status = {
        "size": 3,
        "waiting_times": [1000],
        "enabled_datetimes": [],
        "curr_enabled_at": datetime.now(),
        "is_triggered_by_batch": False
    }

    # ====== ACT & ASSERT ======
    (is_true, _, _) = firing_rules.is_true(current_exec_status)
    assert True == is_true

    current_size = current_exec_status["size"]
    batch_size, _ = firing_rules.get_firing_batch_size(current_size, current_exec_status)
    assert batch_size == 3


data_wt_and_size_rules = [
    # Rule: waiting time >= 3600, 10 tasks waiting for the batch execution and size < 3.
    # Current state: waiting time is 3600 sec for two oldest tasks waiting for the execution.
    # Expected result: firing rule is enabled.
    # Num of elements in the batch to be executed: 2. Total num of batches: 1 (others don't comply with wt rule).
    # TODO: verify this test case
    # (
    #     # [ 3600, 3600, 0, 0, 0, 0, 0, 0, 0, 0 ],
    #     "17/09/22 21:10:00",
    #     [
    #         "17/09/22 20:00:00",
    #         "17/09/22 20:30:00",
    #         "17/09/22 21:00:00",
    #         "17/09/22 21:15:00",
    #         "17/09/22 21:20:00",
    #     ],
    #     "<",
    #     ">=",
    #     "<",
    #     True,
    #     [2]
    # ),
    # Rule: waiting time >= 3600, 10 tasks waiting for the batch execution and size > 3.
    # Current state: waiting time is 3600 sec for all tasks waiting for the execution.
    # Expected result: firing rule is enabled. Num of elements in the batch to be executed: 10.
    # Total num of batches: 1. All waiting tasks will be executed as one batch.
    (
        "17/09/22 21:00:00",
        [
            "17/09/22 20:00:00",
            "17/09/22 20:00:00",
            "17/09/22 20:00:00",
            "17/09/22 20:00:00",
            "17/09/22 20:00:00",
            "17/09/22 20:00:00",
            "17/09/22 20:00:00",
            "17/09/22 20:00:00",
        ],
        ">",
        ">=",
        "<",
        True,
        [8]
    ),
]


@pytest.mark.parametrize(
    "curr_enabled_at_str, enabled_datetimes, size_rule_sign, wt_rule_sign_1, wt_rule_sign_2, expected_is_true, expected_batch_size", 
    data_wt_and_size_rules
)
def test_wt_and_size_rules_correct_enabled_and_batch_size(
    curr_enabled_at_str, enabled_datetimes, size_rule_sign, wt_rule_sign_1, wt_rule_sign_2, expected_is_true, expected_batch_size):

    # ====== ARRANGE ======
    # NB: rules should be ordered so that size rule is the last one
    firing_sub_rule_1 = FiringSubRule("large_wt", wt_rule_sign_1, 3600) # 1 hour
    firing_sub_rule_2 = FiringSubRule("large_wt", wt_rule_sign_2, 7200) # 2 hour
    firing_sub_rule_3 = FiringSubRule("size", size_rule_sign, 3)

    firing_rules = AndFiringRule([ firing_sub_rule_1, firing_sub_rule_2, firing_sub_rule_3 ])
    firing_rules.init_boundaries()

    current_exec_status = _get_current_exec_status(curr_enabled_at_str, enabled_datetimes)

    # ====== ACT & ASSERT ======
    (is_true, batch_spec, _) = firing_rules.is_true(current_exec_status)
    assert expected_is_true == is_true
    assert expected_batch_size == batch_spec


data_wt_or_size_rules = [
    # Rule: waiting time <= 3600, 5 tasks waiting for the batch execution or size > 3.
    # Current state: waiting time is 3600 sec for two oldest tasks waiting for the execution.
    # Expected result: firing rule is enabled by the waiting_time part.
    # Batch execution flow: 5. Executing all together since 'waiting time <= 3600' rule enabled the execution.
    (
        "17/09/22 15:00:00",
        [
            "17/09/22 14:00:00",
            "17/09/22 14:50:00",
            "17/09/22 14:40:00",
            "17/09/22 15:00:00",
            "17/09/22 15:00:00",
        ],
        ">",
        "<=",
        True,
        [5]
    ),
    # Rule: waiting time <= 3600, 5 tasks waiting for the batch execution and size < 3.
    # Current state: waiting time is 3601 sec for every task waiting for the execution.
    # Expected result: firing rule is enabled by the size part.
    # Batch execution flow: 2, 2. Cannot execute all together due to limiting size rule (size < 3).
    (
        "17/09/22 15:10:00",
        [
            "17/09/22 14:00:00",
            "17/09/22 14:00:00",
            "17/09/22 14:00:00",
            "17/09/22 14:00:00",
            "17/09/22 14:00:00",
        ],
        "<",
        "<=",
        True,
        [2, 2, 1]
    ),
]

@pytest.mark.parametrize(
    "curr_enabled_at_str, enabled_datetimes, size_rule_sign, wt_rule_sign, expected_is_true, expected_batch_size", 
    data_wt_or_size_rules
)
def test_wt_or_size_rule_correct_enabled_and_batch_size(
    curr_enabled_at_str, enabled_datetimes, size_rule_sign, wt_rule_sign, expected_is_true, expected_batch_size):

    # ====== ARRANGE ======
    # NB: rules should be ordered so that size rule is the last one
    firing_sub_rule_1 = FiringSubRule("large_wt", wt_rule_sign, 3600) # 1 hour
    firing_sub_rule_2 = FiringSubRule("size", size_rule_sign, 3) 

    firing_rule_1 = AndFiringRule([ firing_sub_rule_1, firing_sub_rule_2 ])
    firing_rule_1.init_boundaries()

    rule = OrFiringRule([ firing_rule_1 ])

    current_exec_status = _get_current_exec_status(curr_enabled_at_str, enabled_datetimes)

    # ====== ACT & ASSERT ======
    (is_true, batch_spec, _) = rule.is_true(current_exec_status)
    assert expected_is_true == is_true
    assert expected_batch_size == batch_spec


data_day_week_rules = [
    # Rule: week_day = Monday.
    # Current state: 5 tasks waiting for the batch execution.
    # Expected result: firing rule is NOT enabled because current time is Sunday.
    (
        '24/09/22 13:55:26',
        [ 3600, 3600, 1200, 600, 0 ],
        ("=", "Monday"),
        False,
        None,
        None
    ),
    # Rule: week_day = Monday.
    # Current state: 5 tasks waiting for the batch execution: two of them enabled before midnight, another 3 - after.
    # Expected result: firing rule is enabled because current time is Monday.
    # Batch execution flow: two tasks (enabled before midnight) is being selected for the execution with start time of Monday midnight.
    (
        '19/09/22 00:55:26',
        [ 3600, 3600, 1200, 600, 0 ],
        ("=", "Monday"),
        True,
        [2],
        '19/09/2022 00:00:00'
    ),
    # Rule: week_day = Monday.
    # Current state: 5 tasks waiting for the batch execution: two of them enabled before midnight, another 3 - after.
    # Expected result: firing rule is enabled because current time is Monday.
    # Batch execution flow: two tasks (enabled before midnight) is being selected for the execution with start time of Monday midnight.
    (
        '19/09/22 00:15:00',
        [ 1200 ],
        ("=", "Monday"),
        False,
        None,
        None
    ),
]

@pytest.mark.parametrize(
    "curr_enabled_at_str, waiting_time_arr, week_day_rule, expected_is_true, expected_batch_size, expected_start_time_from_rule", 
    data_day_week_rules
)
def test_only_week_day_rule_correct_enabled_and_batch_size(
    curr_enabled_at_str, waiting_time_arr, week_day_rule, expected_is_true, expected_batch_size, expected_start_time_from_rule):

    # ====== ARRANGE ======
    (week_day_rule_sign, week_day) = week_day_rule
    firing_sub_rule_1 = FiringSubRule("week_day", week_day_rule_sign, week_day) 
    firing_rule_1 = AndFiringRule([ firing_sub_rule_1 ])
    rule = OrFiringRule([ firing_rule_1 ])

    curr_enabled_at = datetime.strptime(curr_enabled_at_str, '%d/%m/%y %H:%M:%S')
    enabled_datetimes = [curr_enabled_at - timedelta(seconds=item) for item in waiting_time_arr ]

    current_exec_status = {
        "size": len(waiting_time_arr),
        "waiting_times": waiting_time_arr,
        "enabled_datetimes": enabled_datetimes,
        "curr_enabled_at": curr_enabled_at,
        "is_triggered_by_batch": False
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


data_multiple_day_week_rules = [
    # Rule: week_day = Tuesday or  week_day = Sunday.
    # Current state: 5 tasks waiting for the batch execution.
    # Expected result: firing rule is NOT enabled because current time is Saturday.
    (
        '24/09/22 13:55:26',
        [ 3600, 3600, 1200, 600, 0 ],
        "Tuesday",
        "Sunday",
        False,
        None,
        None
    ),
    # Rule: week_day = Monday or week_day = Friday.
    # Current state: 5 tasks waiting for the batch execution.
    # Expected result: firing rule is enabled because current time is Monday.
    # Batch execution flow: two tasks (enabled before midnight) is being selected for the execution with start time of Monday midnight.
    (
        '19/09/22 00:55:26',
        [ 3600, 3600, 1200, 600, 0 ],
        "Monday",
        "Friday",
        True,
        [2],
        '19/09/2022 00:00:00'
    )
]

@pytest.mark.parametrize(
    "curr_enabled_at_str, waiting_time_arr, week_day_1, week_day_2, expected_is_true, expected_batch_size, expected_start_time_from_rule", 
    data_multiple_day_week_rules
)
def test_multiple_week_day_rule_correct_enabled_and_batch_size(
    curr_enabled_at_str, waiting_time_arr, week_day_1, week_day_2, expected_is_true, expected_batch_size, expected_start_time_from_rule):

    # ====== ARRANGE ======
    firing_sub_rule_1 = FiringSubRule("week_day", "=", week_day_1)
    firing_sub_rule_2 = FiringSubRule("week_day", "=", week_day_2) 

    firing_rule_1 = AndFiringRule([ firing_sub_rule_1 ])
    firing_rule_2 = AndFiringRule([ firing_sub_rule_2 ])
    rule = OrFiringRule([ firing_rule_1, firing_rule_2 ])

    curr_enabled_at = datetime.strptime(curr_enabled_at_str, '%d/%m/%y %H:%M:%S')
    enabled_datetimes = [curr_enabled_at - timedelta(seconds=item) for item in waiting_time_arr ]

    current_exec_status = {
        "size": len(waiting_time_arr),
        "waiting_times": waiting_time_arr,
        "enabled_datetimes": enabled_datetimes,
        "curr_enabled_at": curr_enabled_at,
        "is_triggered_by_batch": False
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
