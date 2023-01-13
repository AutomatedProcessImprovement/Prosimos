import sys
from typing import List


class InOperatorEvaluator:
    def __init__(self, range: List[str], case_value: float):
        self.range = range          # including the edges
        self.value = case_value
        self.min, self.max = self._parse_boundaries()

    def _parse_boundaries(self):
        min_boundary = float(self.range[0])
        max_boundary = sys.maxsize if self.range[1] == "inf" else \
            float(self.range[1])

        return min_boundary, max_boundary

    def eval(self):
        if (self.min == self.max):
            # equal operator
            return self.min == self.value
        else:
            # check whether in the range
            return self.min <= self.value <= self.max


class PrioritisationRule:
    def __init__(self, attribute, condition, value):
        self.attribute = attribute
        self.condition = condition
        self.value = value

    def is_rule_true(self, case_value):
        if self.condition == 'in':
            evaluator = InOperatorEvaluator(self.value, case_value)
            return evaluator.eval()
        else:
            # TODO: clarify whether == is the only possible option
            return case_value == self.value


class AndPrioritisationRule:
    def __init__(self, and_rules: List[PrioritisationRule]):
        self.and_rules = and_rules

    def is_and_rule_true(self):
        init_val = True
        for item in self.and_rules:
            init_val = init_val and item.is_rule_true()

        return init_val


class OrPrioritisationRule:
    def __init__(self, or_rules: List[AndPrioritisationRule]):
        self.or_rules = or_rules

    def is_or_rule_true(self):
        init_val = False
        for item in self.or_rules:
            init_val = init_val and item.is_and_rule_true()

        return init_val