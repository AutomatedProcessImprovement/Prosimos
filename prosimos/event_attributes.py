import pprint
from enum import Enum
from random import choices
from typing import Dict
import ast
import operator as op
import numpy as np

from pix_framework.statistics.distribution import DurationDistribution

from prosimos.exceptions import InvalidEventAttributeException
import math


class EVENT_ATTR_TYPE(Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    EXPRESSION = "expression"
    DTREE = "dtree"


operators = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.USub: op.neg, ast.Eq: op.eq, ast.NotEq: op.ne, ast.Lt: op.lt,
    ast.LtE: op.le, ast.Gt: op.gt, ast.GtE: op.ge, ast.Not: op.not_,
    ast.And: op.and_, ast.Or: op.or_, ast.Mod: op.mod, ast.Pow: op.pow,
    ast.FloorDiv: op.floordiv
}

def fix(mean):
    return DurationDistribution(name="fix", mean=mean, var=0.0, std=0.0, minimum=mean, maximum=mean).generate_sample(1)[0]


def uniform(minimum, maximum):
    return DurationDistribution(name="uniform", minimum=minimum, maximum=maximum).generate_sample(1)[0]


def norm(mean, std, minimum=None, maximum=None):
    return DurationDistribution(name="norm", mean=mean, std=std, minimum=minimum, maximum=maximum).generate_sample(1)[0]


def triang(c, minimum, maximum):
    return DurationDistribution(name="triang", mean=c, minimum=minimum, maximum=maximum).generate_sample(1)[0]


def expon(mean, minimum=None, maximum=None):
    return DurationDistribution(name="expon", mean=mean, minimum=minimum, maximum=maximum).generate_sample(1)[0]


def lognorm(mean, var, minimum=None, maximum=None):
    return DurationDistribution(name="lognorm", mean=mean, var=var, minimum=minimum, maximum=maximum).generate_sample(1)[0]


def gamma(mean, var, minimum=None, maximum=None):
    return DurationDistribution(name="gamma", mean=mean, var=var, minimum=minimum, maximum=maximum).generate_sample(1)[0]


distributions = {
    'fix': fix,
    'uniform': uniform,
    'norm': norm,
    'triang': triang,
    'expon': expon,
    'lognorm': lognorm,
    'gamma': gamma
}

math_functions = {name: getattr(math, name) for name in dir(math) if callable(getattr(math, name))}


def parse_discrete_value(value_info):
    if isinstance(value_info, list):
        prob_arr = []
        options_arr = []
        for item in value_info:
            options_arr.append(item["key"])
            prob_arr.append(float(item["value"]))

        return {
            "type": "discrete",
            "options": options_arr,
            "probabilities": prob_arr
        }
    elif isinstance(value_info, dict):
        return {
            "type": "markov",
            "transitions": value_info
        }
    else:
        raise ValueError("Unsupported value_info format for discrete value")


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
                if node.id in vars_dict:
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
            elif isinstance(node, ast.Call):
                func_name = node.func.id
                args = [_eval(arg) for arg in node.args]
                if func_name in distributions:
                    return distributions[func_name](*args)
                elif func_name in math_functions:
                    args = [_eval(arg) for arg in node.args]
                    try:
                        return math_functions[func_name ](*args)
                    except OverflowError:
                        return np.finfo(np.float32).max
                    except ValueError as e:
                        return 0
            else:
                return 0
        except (SyntaxError, ZeroDivisionError, TypeError, KeyError):
            return 0

    return _eval(tree.body)


def evaluate_dtree(dtree, vars_dict):
    for conditions, formula in dtree:
        if conditions is True or all(eval_expr(cond, vars_dict) for cond in conditions):
            return eval_expr(formula, vars_dict)
    return None

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
        elif self.event_attr_type == EVENT_ATTR_TYPE.DTREE:
            self.value = value
        else:
            raise Exception(f"Not supported event attribute {type}")

        self.validate()

    def get_next_value(self, all_attributes):
        if self.event_attr_type == EVENT_ATTR_TYPE.DISCRETE:
            if self.value["type"] == "markov":
                current_value = all_attributes.get(self.name, None)
                next_state = self.get_next_markov_state(current_value)
                if next_state is not None:
                    all_attributes[self.name] = next_state
                    return next_state
                return current_value
            else:
                one_choice_arr = choices(self.value["options"], self.value["probabilities"])
                return one_choice_arr[0]

        elif self.event_attr_type == EVENT_ATTR_TYPE.EXPRESSION:
            result = eval_expr(self.value, all_attributes)
            if isinstance(result, (int, float, np.number)) and not isinstance(result, bool):
                if result == 0:  # Specifically handle zero without adjusting to tiny (in case of any errors in eval)
                    return 0
                elif result == float('inf'):
                    return np.finfo(np.float32).max
                elif result == -float('inf'):
                    return np.finfo(np.float32).min
                elif abs(result) < np.finfo(np.float32).tiny:
                    return np.finfo(np.float32).tiny
                elif abs(result) > np.finfo(np.float32).max:
                    return np.finfo(np.float32).max if result > 0 else np.finfo(np.float32).min
                else:
                    return result
            else:
                return result
        elif self.event_attr_type == EVENT_ATTR_TYPE.DTREE:
            result = evaluate_dtree(self.value, all_attributes)
            if result is not None:
                return result
            return 0
        else:
            return self.value.generate_sample(1)[0]

    def get_next_markov_state(self, current_value):
        transitions = self.value["transitions"]
        if current_value in transitions:
            current_transitions = transitions[current_value]
            options, probabilities = zip(*current_transitions.items())
            return choices(options, probabilities)[0]
        return current_value

    def validate(self):
        epsilon = 1e-6

        if self.event_attr_type == EVENT_ATTR_TYPE.DISCRETE:
            if self.value["type"] == "discrete":
                actual_sum_probabilities = sum(self.value["probabilities"])

                if not (1 - epsilon <= actual_sum_probabilities <= 1 + epsilon):
                    raise InvalidEventAttributeException(
                        f"Event attribute {self.name}: probabilities' sum should be equal to 1")
            elif self.value["type"] == "markov":
                for state, transitions in self.value["transitions"].items():
                    actual_sum_probabilities = sum(transitions.values())
                    if not (1 - epsilon <= actual_sum_probabilities <= 1 + epsilon):
                        raise InvalidEventAttributeException(
                            f"Event attribute {self.name}, state {state}: "
                            f"probabilities' sum should be equal to 1")

        return True


class AllEventAttributes:
    def __init__(self, event_attr_arr: Dict[str, Dict[str, EventAttribute]]):
        self.attributes = event_attr_arr

    def get_columns_generated(self):
        return list({attr.name for event_id in self.attributes for attr in self.attributes[event_id].values()})

    def get_values_calculated(self):
        return {attr.name: attr.get_next_value() for event_id in self.attributes for attr in
                self.attributes[event_id].values()}
