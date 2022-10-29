import pytest
from datetime import datetime, time

from bpdfr_simulation_engine.batching_processing import AndFiringRule, FiringSubRule, OrFiringRule

data_one_week_day = [
    # Rule:             ready_wt < 3600 (3600 seconds = 1 hour)
    # Current state:    4 tasks waiting for the batch execution.
    # Expected result:  firing rule is enabled for the first pair of waiting items.
    #                   enabled time of the batch equals to the enabled time of the second item in the pair
    #                   (since that enabled time satisfies the rule)
    (
        "19/09/22 14:05:26",
        [
            "19/09/22 12:00:26",
            "19/09/22 12:30:26",
            "19/09/22 13:00:26",
            "19/09/22 13:30:26",
        ],
        "<",
        3600, # one hour
        True,
        [2, 2],
        "19/09/2022 12:30:26",
    ),
    # Rule:             ready_wt > 3600 (3600 seconds = 1 hour)
    # Current state:    4 tasks waiting for the batch execution.
    # Expected result:  Firing rule is enabled for the all items and this equals to two batches to be executed.
    #                   Enabled time of the batch equals to the enabled time of the second item in the first pair
    #                   (meaning, to the minimum datetime from all enabled batches).
    #                   Verify that enabled_time of the batch is one second after the datetime forced by the rule.
    (
        "19/09/22 14:05:26",
        [
            "19/09/22 11:00:26",
            "19/09/22 11:30:26",
            "19/09/22 12:00:26",
            "19/09/22 12:30:26",
        ],
        ">",
        3600, # one hour
        True,
        [2, 2],
        "19/09/2022 12:30:27",
    ),
    # Rule:             ready_wt >= 3600 (3600 seconds = 1 hour)
    # Current state:    4 tasks waiting for the batch execution.
    # Expected result:  Firing rule is enabled for the all items and this equals to two batches to be executed.
    #                   Enabled time of the batch equals to the enabled time of the second item in the first pair
    #                   (meaning, to the minimum datetime from all enabled batches).
    #                   Verify that enabled_time of the batch equals exactly to the datetime forced by the rule.
    (
        "19/09/22 14:05:26",
        [
            "19/09/22 11:00:26",
            "19/09/22 11:30:26",
            "19/09/22 12:00:26",
            "19/09/22 12:30:26",
        ],
        ">=",
        3600, # one hour
        True,
        [2, 2],
        "19/09/2022 12:30:26",
    ),
    # Rule:             ready_wt > 3600 (3600 seconds = 1 hour)
    # Current state:    3 tasks waiting for the batch execution.
    # Expected result:  Firing rule is not enabled since no items
    #                   waiting for batch execution satisfies the rule.
    (
        "19/09/22 13:00:26",
        [
            "19/09/22 12:00:26",
            "19/09/22 12:15:26",
            "19/09/22 12:30:26",
        ],
        ">",
        3600, # one hour
        False,
        None,
        None,
    ),
]

@pytest.mark.parametrize(
    "curr_enabled_at_str, enabled_datetimes, sign_ready_wt, ready_wt_value_sec, expected_is_true, expected_batch_size, expected_start_time_from_rule",
    data_one_week_day,
)
def test_ready_wt_rule_correct_enabled_and_batch_size(
    curr_enabled_at_str,
    enabled_datetimes,
    sign_ready_wt,
    ready_wt_value_sec,
    expected_is_true,
    expected_batch_size,
    expected_start_time_from_rule,
):

    # ====== ARRANGE ======
    firing_sub_rule_1 = FiringSubRule(
        "ready_wt", sign_ready_wt, ready_wt_value_sec
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