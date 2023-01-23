import pytest
from bpdfr_simulation_engine.prioritisation_rules import (
    AndPrioritisationRule,
    OrPrioritisationRule,
    PrioritisationRule,
)

BUSINESS = "Business"
REGULAR = "Regular"
INF = "inf"

NOT_INTERSECTED_RANGE = [[10, 100], [200, 500]]
INTERSECTED_RANGE = [[10, 300], [200, 500]]

discrete_test_cases = [
    ([0, 100], 50, True),
    ([0, 100], 150, False),
    ([200, 500], 239, True),
    ([200, 500], 128, False),
    ([200, 500], 600, False),
    ([100, INF], 890, True),
    ([100, INF], 90, False),
]


@pytest.mark.parametrize(
    "rule_range, test_value, expected_result",
    discrete_test_cases,
)
def test_discrete_rule_correct(rule_range, test_value, expected_result):
    # ====== ARRANGE ======
    rule = PrioritisationRule("loan_amount", "in", rule_range)

    # ====== ACT ======
    actual_result = rule.is_rule_true({"loan_amount": test_value})

    # ====== ASSERT ======
    assert actual_result == expected_result


continuous_test_cases = [
    (BUSINESS, BUSINESS, True),
    (BUSINESS, REGULAR, False),
]


@pytest.mark.parametrize(
    "rule_value, test_value, expected_result",
    continuous_test_cases,
)
def test_continuous_rule_correct(rule_value, test_value, expected_result):
    # ====== ARRANGE ======
    rule = PrioritisationRule("category", "=", rule_value)

    # ====== ACT ======
    actual_result = rule.is_rule_true({"category": test_value})

    # ====== ASSERT ======
    assert actual_result == expected_result


and_test_cases = [
    (90, BUSINESS, True),
    (190, BUSINESS, False),
    (79, REGULAR, False),
    (190, REGULAR, False),
]


@pytest.mark.parametrize(
    "loan_amount_value, category_value, expected_result",
    and_test_cases,
)
def test_and_combination_correct(loan_amount_value, category_value, expected_result):
    # ====== ARRANGE ======
    and_rule = _create_and_rule_with_two_subrule([10, 100], BUSINESS)

    # ====== ACT ======
    actual_result = and_rule.is_and_rule_true(
        {
            "loan_amount": loan_amount_value,
            "category": category_value,
        }
    )

    # ====== ASSERT ======
    assert actual_result == expected_result


or_test_cases = [
    (INTERSECTED_RANGE, 280, BUSINESS, True),
    (NOT_INTERSECTED_RANGE, 239, BUSINESS, True),
    (NOT_INTERSECTED_RANGE, 73, BUSINESS, True),
    (NOT_INTERSECTED_RANGE, 130, REGULAR, False),
]


@pytest.mark.parametrize(
    "range_rules, loan_amount_value, category_value, expected_result",
    or_test_cases,
)
def test_or_combination(
    range_rules, loan_amount_value, category_value, expected_result
):
    range_rule1, range_rule2 = range_rules[0], range_rules[1]
    # ====== ARRANGE ======
    or_rule = _create_or_rule_with_two_subrule(
        [range_rule1, BUSINESS], [range_rule2, BUSINESS]
    )

    # ====== ACT ======
    actual_result = or_rule.is_or_rule_true(
        {"loan_amount": loan_amount_value, "category": category_value}
    )

    # ====== ASSERT ======
    assert actual_result == expected_result


def _create_and_rule_with_two_subrule(discrete_range, continuous_value):
    rule1 = PrioritisationRule("loan_amount", "in", discrete_range)
    rule2 = PrioritisationRule("category", "=", continuous_value)

    and_rule = AndPrioritisationRule([rule1, rule2])

    return and_rule


def _create_or_rule_with_two_subrule(values_rule1, values_rule2):
    and_rule1 = _create_and_rule_with_two_subrule(values_rule1[0], values_rule1[1])
    and_rule2 = _create_and_rule_with_two_subrule(values_rule2[0], values_rule2[1])

    or_rule = OrPrioritisationRule([and_rule1, and_rule2])

    return or_rule
