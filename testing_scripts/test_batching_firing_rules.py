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
    is_true = firing_rules.is_true(current_exec_status)
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
    is_true = firing_rules.is_true(current_exec_status)
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
    is_true = firing_rules.is_true(current_exec_status)
    assert False == is_true

    current_size = current_exec_status["size"]
    batch_size = firing_rules.get_firing_batch_size(current_size)
    assert batch_size == 3
