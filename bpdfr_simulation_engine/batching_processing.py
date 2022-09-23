from enum import Enum
import operator
import sys
from typing import List

OPERATOR_SYMBOLS = {
    '<': operator.lt,
    '<=': operator.le,
    '=': operator.ge,
    # '!=': operator.ne,
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
        if self.variable1 == "waiting_time":
            value1_list = element[self.variable1]

            if len(value1_list) < 2:
                # not enough to be executed in batch, at least 2 tasks required
                return False

            oldest_in_batch = value1_list[0]
            
            return OPERATOR_SYMBOLS[self.operator](oldest_in_batch, self.value2)
        else:
            value1 = element[self.variable1]

            if value1 < 2:
                # not enough to be executed in batch, at least 2 tasks required
                return False

            if self.variable1 == "size":
                if self.operator == "<" and value1 >= self.value2 \
                or (self.operator == "<=" and value1 > self.value2):
                    # edge case: we can break waiting tasks for the batch execution into multiple batches
                    return True
                
            return OPERATOR_SYMBOLS[self.operator](value1, self.value2)

    def is_batch_size(self):
        return self.variable1 == "size"


class AndFiringRule():
    def __init__(self, array_of_subrules):
        self.rules = array_of_subrules

    def is_true(self, element):
        """
        :param element - object with the next structure { "size": number , "waiting_time": array }. waiting_time is array
        showing the waiting time of each task in the queue ordered by time insertion (same as ordered by case_id)
        :return: is_true_result, batch_spec where is_true_result shows whether at least one batch is enabled for the execution
        and batch_spec shows the specification for the batch execution (array where the length shows the num of batches to be executed
        and the item in array refers to num of tasks to be executed in this i-batch)
        """
        is_true_result = True

        for rule in self.rules:
            is_true_result = is_true_result and rule.is_true(element)

        if is_true_result:
            num_tasks_in_queue = element["size"]
            num_tasks_in_batch = self.get_firing_batch_size(num_tasks_in_queue)
            
            batch_spec = [num_tasks_in_batch]

            if num_tasks_in_queue > num_tasks_in_batch:
                # shift to the next tasks and validate the rule there
                new_num_tasks = num_tasks_in_queue - num_tasks_in_batch

                # adjust the processed batch passed to the 'is_true' method
                element["size"] = new_num_tasks
                element['waiting_time'] = element['waiting_time'][num_tasks_in_batch:]

                is_true_iter, total_batch_count_iter = self.is_true(element)
                if is_true_iter:
                    return is_true_result, batch_spec + total_batch_count_iter
                else:
                    # the next batch of tasks is not enabled for execution
                    return is_true_result, batch_spec
            
            return True, batch_spec
        
        return is_true_result, None

    def _get_batch_size_subrule(self):
        for rule in self.rules:
            if rule.is_batch_size():
                return rule
        
        return None

    def get_firing_batch_size(self, current_batch_size):
        batch_size_subrule = self._get_batch_size_subrule()
        if batch_size_subrule == None:
            print("WARNING: Not a size subrule")
            return current_batch_size

        value2 = batch_size_subrule.value2
        switcher = {
            '<': min(current_batch_size, value2 - 1),
            '<=': min(current_batch_size, value2),
            '=': value2,
            '>': current_batch_size if current_batch_size > value2 else 0,
            '>=': current_batch_size if current_batch_size >= value2 else 0
        }

        return switcher.get(batch_size_subrule.operator)


class OrFiringRule():
    def __init__(self, or_firing_rule_arr: List[AndFiringRule]):
        self.rules = or_firing_rule_arr

    def is_true(self, spec):
        is_batched_task_enabled = False
        for rule in self.rules:
            is_batched_task_enabled, batch_spec = rule.is_true(spec)

            # fast exit if one of the rule is true
            if is_batched_task_enabled:
                return is_batched_task_enabled, batch_spec

        return is_batched_task_enabled, None


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
