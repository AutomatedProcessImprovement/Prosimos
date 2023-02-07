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


class PrioritisationRule:
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
        case_value = all_case_values[self.attribute]
        if self.condition == "in":
            evaluator = InOperatorEvaluator(self.value, case_value)
            return evaluator.eval()
        else:
            # TODO: clarify whether == is the only possible option
            return self.value == case_value


class AndPrioritisationRule:
    def __init__(self, and_rules: List[PrioritisationRule]):
        self.and_rules = and_rules

    def is_and_rule_true(self, all_case_values):
        init_val = True
        for item in self.and_rules:
            init_val = init_val and item.is_rule_true(all_case_values)

        return init_val


class OrPrioritisationRule:
    def __init__(self, or_rules: List[AndPrioritisationRule]):
        self.or_rules = or_rules

    def is_or_rule_true(self, all_case_values):
        init_val = False
        for item in self.or_rules:
            init_val = init_val or item.is_and_rule_true(all_case_values)

        return init_val


class PriorityWithRule:
    def __init__(self, or_rule: OrPrioritisationRule, priority: str):
        self.or_rule = or_rule
        self.priority = int(priority)

    def is_true(self, all_case_values):
        return self.or_rule.is_or_rule_true(all_case_values)


class AllPriorityRules:
    def __init__(self, rules_array: List[PriorityWithRule]):
        self.all_rules = self._sort_rule_by_priority_level(rules_array)

    def _sort_rule_by_priority_level(self, rules_array: List[PriorityWithRule]):
        "Order rules from the highest priority to the lowest"
        return sorted(rules_array, key=lambda rule: rule.priority)

    def get_priority(self, all_case_values):
        # the lower number - the higher priority
        # so, by default, the highest integer value will guarantee the lowest priority
        init_priority = sys.maxsize
        for rule in self.all_rules:
            if rule.is_true(all_case_values):
                init_priority = rule.priority
                break

        return init_priority
