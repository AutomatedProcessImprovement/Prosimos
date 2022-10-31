import pytest
from datetime import datetime, time
import pandas as pd

from bpdfr_simulation_engine.batching_processing import AndFiringRule, FiringSubRule, OrFiringRule
from bpdfr_simulation_engine.resource_calendar import parse_datetime
from testing_scripts.test_batching import (
    _get_start_time_and_count,
    _verify_logs_ordered_asc,
    _verify_same_resource_for_batch,
    assets_path
)
from testing_scripts.test_batching_daily_hour import _arrange_and_act
from testing_scripts.test_batching import (
    _get_start_time_and_count,
    _setup_arrival_distribution,
    _setup_sim_scenario_file,
    _verify_logs_ordered_asc,
    _verify_same_resource_for_batch,
    assets_path,
)
data_one_week_day = [
    # Rule:             ready_wt < 3600 (3600 seconds = 1 hour)
    #                   (it's being parsed to ready_wt > 3598, cause it should be fired last_item_en_time + 3599)
    # Current state:    4 tasks waiting for the batch execution.
    #                   difference between each of them is not > 3598
    # Expected result:  firing rule is enabled at the time we check for enabled time 
    #                   enabled time of the batch equals to the enabled time of the last item in the batch + 3598 
    #                   (value dictated by rule)
    (
        "19/09/22 14:35:26",
        [
            "19/09/22 12:00:26",
            "19/09/22 12:30:26",
            "19/09/22 13:00:26",
            "19/09/22 13:30:26",
        ],
        ">",    # "<"   - in the json file
        3598,   # 3600  - in the json file
        True,
        [4],
        "19/09/2022 14:30:25",
    ),
    # Rule:             ready_wt > 3600 (3600 seconds = 1 hour)
    # Current state:    5 tasks waiting for the batch execution.
    # Expected result:  Firing rule is enabled for the all items and this equals to two batches to be executed.
    #                   Enabled time of the batch equals to the enabled time of the last item in each batch
    #                   (meaning, to the maximum datetime from all enabled batches).
    #                   There is difference between two activities (3d and 4th) which exceeds one hour limit,
    #                   so that's what triggers the first batch to be enabled. 
    #                   Verify that enabled_time of the batch is one second after the datetime forced by the rule.
    (
        "19/09/22 14:05:26",
        [
            "19/09/22 10:00:26",
            "19/09/22 10:00:26",
            "19/09/22 10:30:26",
            "19/09/22 12:00:26",
            "19/09/22 12:30:26",
        ],
        ">",
        3600, # one hour
        True,
        [3, 2],
        "19/09/2022 11:30:27",
    ),
    # Rule:             ready_wt >= 3600 (3600 seconds = 1 hour)
    # Current state:    4 tasks waiting for the batch execution.
    # Expected result:  Firing rule is enabled for the all items and this equals to one batch enabled.
    #                   All activities have difference of less than one hour,
    #                   that's why the rule were not satisfied at that point somewhere.
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
        [4],
        "19/09/2022 13:30:26",
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
def test_ready_wt_rule_correct_is_true(
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