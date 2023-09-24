from enum import Enum
from functools import reduce
from random import choices
from typing import Dict
import ast
import operator as op

from pix_framework.statistics.distribution import DurationDistribution

from prosimos.exceptions import InvalidEventAttributeException


class EVENT_ATTR_TYPE(Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    EXPRESSION = "expression"


operators = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.USub: op.neg, ast.Eq: op.eq, ast.NotEq: op.ne, ast.Lt: op.lt,
    ast.LtE: op.le, ast.Gt: op.gt, ast.GtE: op.ge, ast.Not: op.not_,
    ast.And: op.and_, ast.Or: op.or_, ast.Mod: op.mod, ast.Pow: op.pow,
    ast.FloorDiv: op.floordiv
}


def parse_discrete_value(value_info_arr):
    prob_arr = []
    options_arr = []
    for item in value_info_arr:
        options_arr.append(item["key"])
        prob_arr.append(float(item["value"]))

    return {
        "options": options_arr,
        "probabilities": prob_arr
    }


def parse_continuous_value(value_info) -> "DurationDistribution":
    return DurationDistribution.from_dict(value_info)


def eval_expr(expr, vars_dict):
    try:
        tree = ast.parse(expr, mode='eval')
    except (SyntaxError, ZeroDivisionError, TypeError, KeyError):
        return None

    def _eval(node):
        try:
            if isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.BinOp):
                return operators[type(node.op)](_eval(node.left), _eval(node.right))
            elif isinstance(node, ast.UnaryOp):
                return operators[type(node.op)](_eval(node.operand))
            elif isinstance(node, ast.Name):
                return vars_dict[node.id]
            elif isinstance(node, ast.Str):
                return node.s
            elif isinstance(node, ast.Compare):
                return operators[type(node.ops[0])](_eval(node.left), _eval(node.comparators[0]))
            elif isinstance(node, ast.BoolOp):
                if type(node.op) is ast.And:
                    return all(_eval(value) for value in node.values)
                elif type(node.op) is ast.Or:
                    return any(_eval(value) for value in node.values)
            else:
                return None
        except (SyntaxError, ZeroDivisionError, TypeError, KeyError):
            return None

    return _eval(tree.body)


class EventAttribute:
    def __init__(self, event_id, name, event_attr_type, value):
        self.event_id: str = event_id
        self.name: str = name
        self.event_attr_type: EVENT_ATTR_TYPE = EVENT_ATTR_TYPE(event_attr_type)

        if self.event_attr_type == EVENT_ATTR_TYPE.DISCRETE:
            self.value = parse_discrete_value(value)
        elif self.event_attr_type == EVENT_ATTR_TYPE.CONTINUOUS:
            self.value = parse_continuous_value(value)
        elif self.event_attr_type == EVENT_ATTR_TYPE.EXPRESSION:
            self.value = value
        else:
            raise Exception(f"Not supported event attribute {type}")

        self.validate()

    def get_next_value(self, all_attributes={}):
        if self.event_attr_type == EVENT_ATTR_TYPE.DISCRETE:
            one_choice_arr = choices(self.value["options"], self.value["probabilities"])
            return one_choice_arr[0]
        elif self.event_attr_type == EVENT_ATTR_TYPE.EXPRESSION:
            return eval_expr(self.value, all_attributes)
        else:
            return self.value.generate_sample(1)[0]

    def validate(self):
        if self.event_attr_type == EVENT_ATTR_TYPE.DISCRETE:
            actual_sum_probabilities = reduce(lambda acc, item: acc + item, self.value["probabilities"], 0)

            if actual_sum_probabilities != 1:
                raise InvalidEventAttributeException(
                    f"Event attribute ${self.name}: probabilities' sum should be equal to 1")

        return True


class AllEventAttributes:
    def __init__(self, event_attr_arr: Dict[str, Dict[str, EventAttribute]]):
        self.attributes = event_attr_arr

    def get_columns_generated(self):
        return list({attr.name for event_id in self.attributes for attr in self.attributes[event_id].values()})

    def get_values_calculated(self):
        return {attr.name: attr.get_next_value() for event_id in self.attributes for attr in
                self.attributes[event_id].values()}
