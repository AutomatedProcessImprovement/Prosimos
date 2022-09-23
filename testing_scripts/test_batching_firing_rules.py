import pytest
from bpdfr_simulation_engine.batching_processing import FiringRule, FiringSubRule


def test_only_size_eq_correct():
    # ====== ARRANGE ======

    firing_sub_rule = FiringSubRule("size", "=", 3)
    firing_rules = FiringRule([firing_sub_rule])

    current_exec_status = {
        "size": 3,
        "waiting_time": 1000
    }

    # ====== ACT & ASSERT ======
    (is_true, _) = firing_rules.is_true(current_exec_status)
    assert True == is_true

    current_size = current_exec_status["size"]
    batch_size = firing_rules.get_firing_batch_size(current_size)
    assert batch_size == 3

def test_size_eq_wt_lt_correct():
    # ====== ARRANGE ======
    firing_sub_rule_1 = FiringSubRule("size", "=", 3)
    firing_sub_rule_2 = FiringSubRule("waiting_time", "<", 3600) # 1 hour
    firing_rules = FiringRule([ firing_sub_rule_1, firing_sub_rule_2 ])

    current_exec_status = {
        "size": 3,
        "waiting_time": [
            120,
            60,
            0
        ]
    }

    # ====== ACT & ASSERT ======
    (is_true, _) = firing_rules.is_true(current_exec_status)
    assert True == is_true

    current_size = current_exec_status["size"]
    batch_size = firing_rules.get_firing_batch_size(current_size)
    assert batch_size == 3


def test_size_eq_and_wt_gt_correct():
    # ====== ARRANGE ======
    firing_sub_rule_1 = FiringSubRule("size", "=", 3)
    firing_sub_rule_2 = FiringSubRule("waiting_time", ">", 3600) # 1 hour
    firing_rules = FiringRule([ firing_sub_rule_1, firing_sub_rule_2 ])

    current_exec_status = {
        "size": 3,
        "waiting_time": [
            120,
            60,
            0
        ]
    }

    # ====== ACT & ASSERT ======
    (is_true, _) = firing_rules.is_true(current_exec_status)
    assert False == is_true


data_only_waiting_time = [
    # Rule: waiting time >= 3600, 10 tasks waiting for the batch execution.
    # Current state: waiting time is 3600 sec for two oldest tasks waiting for the execution.
    # Expected result: firing rule is enabled. Num of elements in the batch to be executed: 10.
    (
        [ 3600, 3600, 0, 0, 0, 0, 0, 0, 0, 0 ],
        ">=",
        True,
        (10, 1)
    ),
    # Rule: waiting time > 3600, 10 tasks waiting for the batch execution.
    # Current state: waiting time is 3600 sec for two oldest tasks waiting for the execution.
    # Expected result: firing rule is not enabled (3600 > 3600 doesn't hold).
    (
        [ 3600, 3600, 0, 0, 0, 0, 0, 0, 0, 0 ],
        ">",
        False,
        (None, None)
    ),
    # Rule: waiting time >= 3600.
    # Current state: waiting time is 120 or less for all tasks waiting for batch execution.
    # Expected result: firing rule is not enabled.
    (
        [ 120, 120, 0, 0, 0, 0, 0, 0, 0, 0 ],
        ">",
        False,
        (None, None)
    ),
    # Rule: waiting time <= 3600, one task waiting for the batch execution.
    # Current state: waiting time is 3600 sec for this one task in the queue.
    # Expected result: firing rule is not enabled (not enough items to proceed with execution).
    (
        [ 3600 ],
        "<=",
        False,
        (None, None)
    ),
    # Rule: waiting time <= 3600, 2 tasks waiting for the batch execution.
    # Current state: waiting time is 3600 sec and 1200 sec appropriately.
    # Expected result: firing rule is enabled. Num of elements in the batch to be executed: 2.
    (
        [ 3600, 1200],
        "<=",
        True,
        (2, 1)
    ),
    # Rule: waiting time <= 3600, one task waiting for the batch execution.
    # Current state: waiting time is 3601 sec and 1200 sec appropriately.
    # Expected result: firing rule is not enabled (3601 <= 3600 doesn't hold).
    (
        [ 3601, 1200],
        "<=",
        False,
        (None, None)
    )
]


@pytest.mark.parametrize(
    "waiting_time_arr, rule_sign, expected_is_true, expected_batch_size", 
    data_only_waiting_time
)
def test_only_waiting_time_rule_correct_enabled_and_batch_size(
    waiting_time_arr, rule_sign, expected_is_true, expected_batch_size):

    # ====== ARRANGE ======
    firing_sub_rule_1 = FiringSubRule("waiting_time", rule_sign, 3600) # 1 hour
    firing_rules = FiringRule([ firing_sub_rule_1 ])

    current_exec_status = {
        "size": len(waiting_time_arr),
        "waiting_time": waiting_time_arr
    }

    # ====== ACT & ASSERT ======
    (is_true, batch_spec) = firing_rules.is_true(current_exec_status)
    assert expected_is_true == is_true
    assert expected_batch_size == batch_spec


data_wt_and_size_rules = [
    # Rule: waiting time >= 3600, 10 tasks waiting for the batch execution and size < 3.
    # Current state: waiting time is 3600 sec for two oldest tasks waiting for the execution.
    # Expected result: firing rule is enabled.
    # Num of elements in the batch to be executed: 2. Total num of batches: 1 (others don't comply with wt rule).
    (
        [ 3600, 3600, 0, 0, 0, 0, 0, 0, 0, 0 ],
        "<",
        ">=",
        True,
        (2, 1)
    ),
    # Rule: waiting time >= 3600, 10 tasks waiting for the batch execution and size < 3.
    # Current state: waiting time is 3600 sec for two oldest tasks waiting for the execution.
    # Expected result: firing rule is enabled.
    # Num of elements in the batch to be executed: 2. Total num of batches: 1 (others don't comply with wt rule).
    (
        [ 3600, 0, 3600, 0, 3600, 0],
        "<",
        ">=",
        True,
        (2, 3)
    ),
    # Rule: waiting time >= 3600, 10 tasks waiting for the batch execution and size > 3.
    # Current state: waiting time is 3600 sec for all tasks waiting for the execution.
    # Expected result: firing rule is enabled. Num of elements in the batch to be executed: 10.
    # Total num of batches: 1. All waiting tasks will be executed as one batch.
    (
        [ 3600 ] * 10,
        ">",
        ">=",
        True,
        (10, 1)
    ),
]


@pytest.mark.parametrize(
    "waiting_time_arr, size_rule_sign, wt_rule_sign, expected_is_true, expected_batch_size", 
    data_wt_and_size_rules
)
def test_wt_and_size_rules_correct_enabled_and_batch_size(
    waiting_time_arr, size_rule_sign, wt_rule_sign, expected_is_true, expected_batch_size):

    # ====== ARRANGE ======
    firing_sub_rule_1 = FiringSubRule("size", size_rule_sign, 3) 
    firing_sub_rule_2 = FiringSubRule("waiting_time", wt_rule_sign, 3600) # 1 hour
    firing_rules = FiringRule([ firing_sub_rule_1, firing_sub_rule_2 ])

    current_exec_status = {
        "size": len(waiting_time_arr),
        "waiting_time": waiting_time_arr
    }

    # ====== ACT & ASSERT ======
    (is_true, batch_spec) = firing_rules.is_true(current_exec_status)
    assert expected_is_true == is_true
    assert expected_batch_size == batch_spec
