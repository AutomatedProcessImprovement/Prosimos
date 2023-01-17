import pytest
from bpdfr_simulation_engine.prioritisation import PrioritisationRule

discrete_test_cases = [
    ([0, 100], 50, True),
    ([0, 100], 150, False),
    ([200, 500], 239, True),
    ([200, 500], 128, False),
    ([200, 500], 600, False),
    ([100, "inf"], 890, True),
    ([100, "inf"], 90, False),
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
    ("Business", "Business", True),
    ("Business", "Regular", False),
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
