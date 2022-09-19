from enum import Enum
import operator
import sys

OPERATOR_SYMBOLS = {
    '<': operator.lt,
    '<=': operator.le,
    '=': operator.eq,
    '!=': operator.ne,
    '>': operator.gt,
    '>=': operator.ge
}

class BATCH_TYPE(Enum):
    SEQUENTIAL = 'Sequential'   # one after another
    CONCURRENT = 'Concurrent'   # tasks are in progress simultaneously 
                                # (executor changes the context between different tasks)
    PARALLEL = 'Parallel'       # tasks are being executed simultaneously

class FiringSubRule():
    def __init__(self, variable1: str, operator: str, value2: str):
        self.variable1 = variable1
        self.operator = operator
        self.value2 = value2

    def is_true(self, element):
        value1 = element[self.variable1]
        return OPERATOR_SYMBOLS[self.operator](value1, self.value2)

    def is_batch_size(self):
        return self.variable1 == "size"


class FiringRule():
    def __init__(self, array_of_subrules):
        self.rules = array_of_subrules

    def is_true(self, element):
        result = True

        for rule in self.rules:
            result = result and rule.is_true(element)

        return result

class BatchConfigPerTask():
    def __init__(self, type, duration_distribution, firing_rules):
        self.type = type
        self.duration_distribution = duration_distribution
        self.sorted_duration_distribution = sorted(duration_distribution)
        self.firing_rules = firing_rules

    def calculate_ideal_duration(self, initial_duration, num_tasks_in_batch):
        curr_coef = self.duration_distribution.get(num_tasks_in_batch, None)

        if curr_coef is None:
            # find the nearest and lowest key in case of not matching
            min_distance = sys.maxsize
            for item in self.sorted_duration_distribution:
                curr_distance = abs(num_tasks_in_batch - item)
                if curr_distance < min_distance:
                    min_distance = curr_distance
                    min_key = item
                else:
                    break

            nearest_coef = self.duration_distribution[min_key]
            return initial_duration * nearest_coef

        return initial_duration * curr_coef
