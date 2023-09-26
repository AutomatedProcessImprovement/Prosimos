import sys
from typing import List, Tuple


class InOperatorEvaluator:
    def __init__(self, range: Tuple[float, float], case_value: float):
        self.min, self.max = range  # including the edges
        self.value = case_value

    def eval(self):
        if self.min == self.max:
            # equal operator
            return self.min == self.value
        else:
            # check whether in the range
            return self.min <= self.value <= self.max


class BranchConditionRule:
    def __init__(self, attribute, condition, value):
        self.attribute = attribute
        self.condition = condition
        self.value = self._parse_value(value)

    def _parse_value(self, value):
        if self.condition == "in":
            min_boundary = float(value[0])
            max_boundary = sys.maxsize if value[1] == "inf" else float(value[1])

            return min_boundary, max_boundary
        else:
            return value

    def is_rule_true(self, all_case_values):
        if self.attribute not in all_case_values:
            return False

        case_value = all_case_values[self.attribute]
        if self.condition == "in":
            evaluator = InOperatorEvaluator(self.value, case_value)
            return evaluator.eval()
        else:
            return self.value == case_value


class AndBranchConditionRule:
    def __init__(self, and_rules: List[BranchConditionRule]):
        self.and_rules = and_rules

    def is_and_rule_true(self, all_case_values):
        init_val = True
        for item in self.and_rules:
            init_val = init_val and item.is_rule_true(all_case_values)
        return init_val


class OrBranchConditionRule:
    def __init__(self, or_rules: List[AndBranchConditionRule]):
        self.or_rules = or_rules

    def is_or_rule_true(self, all_case_values):
        init_val = False
        for item in self.or_rules:
            init_val = init_val or item.is_and_rule_true(all_case_values)
        return init_val


class BranchConditionWithRule:
    def __init__(self, or_rule: OrBranchConditionRule, id: str):
        self.or_rule = or_rule
        self.id = id

    def is_true(self, all_case_values):
        return self.or_rule.is_or_rule_true(all_case_values)


class AllBranchConditionRules:
    def __init__(self, rules_array: List[BranchConditionWithRule]):
        self.all_rules = rules_array

    def get_branch_condition_by_id(self, branch_condition_id):
        for rule in self.all_rules:
            if rule.id == branch_condition_id:
                return rule
        return None
