from enum import Enum
import operator

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
        

class FiringRule():
    def __init__(self, array_of_subrules):
        self.rules = array_of_subrules

    def is_true(self, element):
        result = True

        for rule in self.rules:
            result = result and rule.is_true(element)

        return result
