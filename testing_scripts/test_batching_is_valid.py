import pytest
import pandas as pd

from bpdfr_simulation_engine.batching_processing import AndFiringRule, FiringSubRule, OrFiringRule
from bpdfr_simulation_engine.exceptions import InvalidRuleDefinition
from bpdfr_simulation_engine.resource_calendar import parse_datetime
from testing_scripts.test_batching import SIM_LOGS_FILENAME, _verify_logs_ordered_asc, _verify_same_resource_for_batch, assets_path
from testing_scripts.test_batching_daily_hour import _arrange_and_act
from testing_scripts.test_batching_ready_wt import _test_range_basic

HALF_AN_HOUR_SEC = 1800
FIVE_HOURS_SEC = 18000
SIX_HOURS_SEC = 21600

data_case_is_valid = [
    # Simple "large_wt" rule makes this whole AND rule invalid
    # We already surpass the high limit of 30 min
    # (19:55:00 - 19:20:00 > 30 min)
    # As a result, the batch is triggered at the same time we discovered it is invalid
    (
        "17/09/22 19:55:00",
        [
            "17/09/22 19:00:00",
            "17/09/22 19:05:00",
            "17/09/22 19:10:00",
            "17/09/22 19:20:00",
        ],
        ('size', '>', 5),
        ('large_wt', '<=', HALF_AN_HOUR_SEC),
        None,
        True,
        [4],
        "17/09/2022 19:55:00",
    ),
]

@pytest.mark.parametrize(
    "curr_enabled_at_str, enabled_datetimes, first_rule, second_rule, third_rule, expected_is_true, expected_batch_size, expected_start_time_from_rule",
    data_case_is_valid,
)
def test_is_true_returned_invalid_case(curr_enabled_at_str, enabled_datetimes, first_rule, second_rule, third_rule,
    expected_is_true, expected_batch_size, expected_start_time_from_rule):
    # ====== ARRANGE ======
    fr_name, fr_sign, fr_value = first_rule
    sc_name, sc_sign, sc_value = second_rule

    firing_sub_rule_1 = FiringSubRule(fr_name, fr_sign, fr_value)
    firing_sub_rule_2 = FiringSubRule(sc_name, sc_sign, sc_value)
    and_firing_rules_arr = [firing_sub_rule_1, firing_sub_rule_2]
    
    if third_rule != None:
        th_name, th_sign, th_value = third_rule
        firing_sub_rule_3 = FiringSubRule(th_name, th_sign, th_value)
        and_firing_rules_arr.append(firing_sub_rule_3)

    firing_rule_1 = AndFiringRule(and_firing_rules_arr)
    firing_rule_1.init_boundaries()
    rule = OrFiringRule([firing_rule_1])

    _test_range_basic(rule, curr_enabled_at_str, enabled_datetimes, expected_is_true,
        expected_batch_size, expected_start_time_from_rule)

def test_conflict_rule_correct_sim_log(assets_path):
    """
    Rule:       We need to collect at least 20 items in first half an hour when
                first task in the batch arrived.
    Expected:   We simulate 10 cases which arrive every 5 min. 
                In the first half an hour, we will be able to collect only 6 items
                which is not enough to trigger the batch execution.
                At the very moment when we surpass the limit,
                we execute the batch with all items we have right now
                (this number equals to six).
                Next 4 items will not satisfy the rule, as well. It will be triggered
                at the very end once all simulated activities will be executed.
    Verified:   Start time of both batches.
                All log items in the log file are ordered ascendingly.
                Resource which executes one batch is the same.
    """

    # ====== ARRANGE ======
    sim_logs = assets_path / SIM_LOGS_FILENAME

    start_string = "2022-06-21 13:22:30.035185+03:00"

    firing_rules = [
        [
            {"attribute": "size", "comparison": ">", "value": 20},
            {"attribute": "large_wt", "comparison": "<=", "value": HALF_AN_HOUR_SEC}
        ]
    ]
    total_num_cases = 10
    _arrange_and_act(assets_path, firing_rules, start_string, total_num_cases, 5 * 60) # every 5 minutes

    # ====== ASSERT ======
    df = pd.read_csv(sim_logs)
    logs_d_task = df[df["activity"] == "D"]
    grouped_by_start = logs_d_task.groupby(by="start_time")

    # verify the start_time of batches
    expected_start_time_keys = [
        "2022-06-21 13:57:30.035185+03:00",
        "2022-06-21 14:11:30.035185+03:00",
    ]
    actual_start_time_keys = list(grouped_by_start.groups.keys())
    assert (
        actual_start_time_keys == expected_start_time_keys
    ), f"The start_time for batched D tasks differs. Expected: {expected_start_time_keys}, but was {actual_start_time_keys}"

    start_date = parse_datetime(start_string, True)
    _verify_logs_ordered_asc(df, start_date.tzinfo)
    
    # verify that the same resource execute the whole batch
    for _, group in grouped_by_start:
        _verify_same_resource_for_batch(group["resource"])


data_case_invalid_end = [
    (
        OrFiringRule(or_firing_rule_arr=[
            AndFiringRule(array_of_subrules=[
                FiringSubRule("size", ">", 20),
                FiringSubRule("large_wt", "<", HALF_AN_HOUR_SEC)
            ])
        ]),
        15,
        FIVE_HOURS_SEC,
        HALF_AN_HOUR_SEC + HALF_AN_HOUR_SEC,
        True
    ),
    (
        OrFiringRule(or_firing_rule_arr=[
            AndFiringRule(array_of_subrules=[
                FiringSubRule("size", ">", 20),
                FiringSubRule("ready_wt", "<", HALF_AN_HOUR_SEC)
            ])
        ]),
        15,
        FIVE_HOURS_SEC,
        HALF_AN_HOUR_SEC + HALF_AN_HOUR_SEC,
        True
    ),
    (
        OrFiringRule(or_firing_rule_arr=[
            AndFiringRule(array_of_subrules=[
                FiringSubRule("size", ">", 20)
            ])
        ]),
        15,
        0,
        0,
        True
    ),
]

@pytest.mark.parametrize(
    "or_rule, num_tasks, first_wt_sec, last_wt_sec, expected_result",
    data_case_invalid_end,
)
def test_invalid_end_correct(or_rule: OrFiringRule, num_tasks, first_wt_sec, last_wt_sec, expected_result):
    # ====== ARRANGE ======
    for and_rule in or_rule.rules:
        and_rule.init_boundaries()

    actual_result = or_rule.is_invalid_end(num_tasks, first_wt_sec, last_wt_sec)
   
    # ====== ASSERT ======
    assert expected_result == actual_result

invalid_setup = [
    # Verify:   Defining two simple rules of the type "WEEK_DAY" invalidates the rule.
    #           There is no case when this condition could be satisfied.
    (
        [
            [
                {"attribute": "week_day", "comparison": "=", "value": "Monday"},
                {"attribute": "week_day", "comparison": "=", "value": "Tuesday"}
            ]
        ],
        "Only one WEEK_DAY subrule is allowed inside AND rule."
    ),
    # Verify:   Defining three simple rules of the type "DAILY_HOUR" invalidates the rule.
    (
        [
            [
                {"attribute": "daily_hour", "comparison": ">=", "value": "12"},
                {"attribute": "daily_hour", "comparison": "<=", "value": "18"},
                {"attribute": "daily_hour", "comparison": "<=", "value": "20"},
            ]
        ],
        "Only one or two subrules of DAILY_HOUR type is allowed inside AND rule."
    ),
    # Verify:   The only allowed operator to be used with "week_day" rule is =.
    #           Exception is being thrown in case some other operator is being used.
    (
        [
            [
                {"attribute": "week_day", "comparison": ">", "value": "Monday"},
            ]
        ],
        "'>' is not allowed operator for the week_day type of rule."
    ),
]


@pytest.mark.parametrize(
    "firing_rules, exception_match",
    invalid_setup,
)
def test_rule_setup_invalid(firing_rules, exception_match, assets_path):
    """
    Verify:     Defining two simple rules of the type "WEEK_DAY" invalidates the rule.
                There is no case when this condition could be satisfied.
    """
    start_string = "2022-06-21 13:22:30.035185+03:00"
    total_num_cases = 10

    with pytest.raises(InvalidRuleDefinition, match=exception_match):
        _arrange_and_act(assets_path, firing_rules, start_string, total_num_cases, 5 * 60) # every 5 minutes

