from datetime import datetime, time
import pytest

from bpdfr_simulation_engine.batching_processing import AndFiringRule, FiringSubRule, OrFiringRule

data_one_week_day = [
    # Rule: daily_hour > 15
    # Current state: 5 tasks waiting for the batch execution: three of them enabled before 15.
    # Expected result: firing rule is enabled because of those three tasks.
    (
        '19/09/22 15:05:26',
        ["19/09/22 06:05:26", "19/09/22 09:05:26", "19/09/22 12:05:26", "19/09/22 15:05:26"],
        ">",
        "15",
        True,
        [3],
        '19/09/2022 15:00:00'
    ),
    # Rule: daily_hour = 15
    # Current state: 5 tasks waiting for the batch execution: three of them enabled before 15.
    # Expected result: firing rule is enabled because of those three tasks.
    (
        '19/09/22 15:05:26',
        ["19/09/22 06:05:26", "19/09/22 09:05:26", "19/09/22 12:05:26", "19/09/22 15:05:26"],
        "=",
        "15",
        True,
        [3],
        '19/09/2022 15:00:00'
    ),
    # Rule: daily_hour < 15
    # Current state: 5 tasks waiting for the batch execution: three of them enabled before 15.
    # Expected result: firing rule is enabled at 14:59 because of those three tasks.
    (
        '19/09/22 15:05:26',
        ["19/09/22 06:05:26", "19/09/22 09:05:26", "19/09/22 12:05:26", "19/09/22 15:05:26"],
        "<",
        "15",
        True,
        [2, 2],
        '19/09/2022 09:05:26',
    ),
    (
        '19/09/22 20:17:00',
        ["17/09/22 14:30:00", "17/09/22 20:00:00", "18/09/22 9:00:00", "18/09/22 10:00:00", "18/09/22 20:00:00"],
        ">",
        "14",
        True,
        [2, 2],
        '17/09/2022 20:00:00',
    ),
]

@pytest.mark.parametrize(
    "curr_enabled_at_str, enabled_datetimes, sign_daily_hour_1, daily_hour_1, expected_is_true, expected_batch_size, expected_start_time_from_rule", 
    data_one_week_day
)
def test_daily_hour_rule_correct_enabled_and_batch_size(
    curr_enabled_at_str, enabled_datetimes, sign_daily_hour_1, daily_hour_1, expected_is_true, expected_batch_size, expected_start_time_from_rule):

    # ====== ARRANGE ======
    firing_sub_rule_1 = FiringSubRule("daily_hour", sign_daily_hour_1, time(int(daily_hour_1), 0, 0))

    firing_rule_1 = AndFiringRule([ firing_sub_rule_1 ])
    rule = OrFiringRule([ firing_rule_1 ])

    curr_enabled_at = datetime.strptime(curr_enabled_at_str, '%d/%m/%y %H:%M:%S')
    enabled_datetimes = [datetime.strptime(item, '%d/%m/%y %H:%M:%S') for item in enabled_datetimes ]
    waiting_time_arr = [curr_enabled_at - item for item in enabled_datetimes ]

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