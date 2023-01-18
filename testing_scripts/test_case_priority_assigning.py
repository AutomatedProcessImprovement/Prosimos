import sys

import pytest
from bpdfr_simulation_engine.prioritisation import (
    AllPriorityRules,
    AndPrioritisationRule,
    OrPrioritisationRule,
    PrioritisationRule,
    PriorityWithRule,
)

NOT_INTERSECTED_RANGE = [[10, 100], [200, 500]]
INTERSECTED_RANGE = [[10, 300], [200, 500]]

priority_assignment_cases = [
    (INTERSECTED_RANGE, 278, 1),
    (NOT_INTERSECTED_RANGE, 250, 2),
    (NOT_INTERSECTED_RANGE, 90, 1),
    (NOT_INTERSECTED_RANGE, 6, sys.maxsize),
]


@pytest.mark.parametrize(
    "rules_ranges, test_value, expected_priority",
    priority_assignment_cases,
)
def test_priority_assignment_correct(rules_ranges, test_value, expected_priority):
    # ====== ARRANGE ======
    value_range1, value_range2 = rules_ranges[0], rules_ranges[1]
    priority_rule_1 = _create_priority_rule(value_range1, "1")
    priority_rule_2 = _create_priority_rule(value_range2, "2")

    all_priorities = AllPriorityRules([priority_rule_1, priority_rule_2])

    # ====== ACT ======
    actual_priority = all_priorities.get_priority({"loan_amount": test_value})

    # ====== ASSERT ======
    assert actual_priority == expected_priority


def _create_priority_rule(value_range, priority):
    or_rule = _create_or_rule_with_one_simple_rule(value_range)
    priority_rule = PriorityWithRule(or_rule, priority)

    return priority_rule


def _create_or_rule_with_one_simple_rule(value_range):
    rule = PrioritisationRule("loan_amount", "in", value_range)
    and_rule = AndPrioritisationRule([rule])
    or_rule = OrPrioritisationRule([and_rule])

    return or_rule
