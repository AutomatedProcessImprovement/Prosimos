from enum import Enum
import operator as operator
import sys
from typing import List
from datetime import datetime, time, timedelta
from bpdfr_simulation_engine.resource_calendar import str_week_days
from bpdfr_simulation_engine.weekday_helper import CustomDatetimeAndSeconds, get_nearest_abs_day, get_nearest_past_day


def _get_operator_symbols(operator_str: str, eq_operator: operator):
    OPERATOR_SYMBOLS = {
        '<': operator.lt,
        '<=': operator.le,
        '>': operator.gt,
        '>=': operator.ge,
        '=': eq_operator
    }

    return OPERATOR_SYMBOLS[operator_str]

def _get_operator_symbols_eq(operator_str: str):
    return _get_operator_symbols(operator_str, operator.eq)

def _get_operator_symbols_ge(operator_str: str):
    return _get_operator_symbols(operator_str, operator.ge)

def _get_operator_symbols_lt(operator_str: str):
    return _get_operator_symbols(operator_str, operator.lt)

def _is_greater(op: operator):
    return op in [operator.ge, operator.gt]

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
        queue_size = element["size"]

        if queue_size < 2:
            # not enough to be executed in batch, at least 2 tasks required
            return False

        if self.variable1 == "waiting_times":
            value1_list = element[self.variable1]

            if len(value1_list) < 2:
                # not enough to be executed in batch, at least 2 tasks required
                return False

            oldest_in_batch = value1_list[0]

            op = _get_operator_symbols_eq(self.operator)
            return op(oldest_in_batch, self.value2)

        elif self.variable1 == "size":
            value1 = element[self.variable1]

            if value1 < 2:
                # not enough to be executed in batch, at least 2 tasks required
                return False

            if self.operator == "<" and value1 >= self.value2 \
            or (self.operator == "<=" and value1 > self.value2):
                # edge case: we can break waiting tasks for the batch execution into multiple batches
                return True

            op = _get_operator_symbols_ge(self.operator)
            return op(value1, self.value2)

        elif self.variable1 == "week_day": # week_day
            value1 = element["curr_enabled_at"]
            curr_week_day_int = value1.weekday()
            rule_value_str = self.value2.upper()
            rule_value_int = str_week_days[rule_value_str]

            op = _get_operator_symbols_eq(self.operator)
            is_rule_true = op(curr_week_day_int, rule_value_int)
            if is_rule_true:
                # current enabled time is at the specified range
                return is_rule_true
            else:
                spec = {
                    "size": element["size"],
                    "waiting_times": element["waiting_times"].copy(),
                    "enabled_datetimes": element["enabled_datetimes"].copy(),
                    "curr_enabled_at": element["curr_enabled_at"],
                    "is_triggered_by_batch": False
                }
                enabled_items, _ = self.get_batch_size_relative_time(spec)
                return enabled_items > 0

        elif self.variable1 == "daily_hour":
            time1 = element["curr_enabled_at"].time()
            # in case of equal sign, we use greater or equal to sign
            # cause we compare it with the enabled time of the task next in the queue
            # it might be a case that we need to insert this current batch before execution of the next task
            op = _get_operator_symbols_ge(self.operator)
            is_rule_true = op(time1, self.value2)
            if is_rule_true:
                return True
            else:
                # if not true, validate whether there are some cases
                # that satisfies the rule from the previous day
                batch_size, _ = self.get_batch_size_by_daily_hour(element)

                return batch_size > 1
                # boundary_day = datetime.combine(enabled_time, time1, enabled_time.tzinfo)
                # op = _get_operator_symbols_lt(self.operator)
                # follow_rule = []
                # for time in batch_enabled_datetimes:
                #     if op(time, boundary_day):
                #         follow_rule.append(time)
            # return is_rule_true


    def is_batch_size(self):
        return self.variable1 == "size"

    def _get_min_enabled_time_waiting_time(self, waiting_times, last_task_start_time: CustomDatetimeAndSeconds) -> tuple([int, datetime]):
        """
        Get the enabled time for the batch
        based on the maximum waiting time of the task in the batch 
        (meaning, for the 'waiting_time' kind of rule)
        """
        case_id, oldest_waiting = waiting_times[0]
        oldest_waiting_time = oldest_waiting.datetime
        rule_value = self.value2

        min_waiting_time = {
            '<': min(oldest_waiting_time, rule_value - 1),
            '<=': min(oldest_waiting_time, rule_value),
            '=': oldest_waiting_time,
            '>': oldest_waiting_time if oldest_waiting_time > rule_value else rule_value + 1,
            '>=': oldest_waiting_time if oldest_waiting_time >= rule_value else rule_value
        }

        # find the minimum waiting time (either from the rule specification or the last waiting_time of the batched task)
        needed_time_seconds = min_waiting_time.get(self.operator)
        return case_id, last_task_start_time.datetime + timedelta(needed_time_seconds)

    def _get_min_enabled_time_week_day(self, waiting_times, last_task_start_time) -> tuple([int, datetime]):
        # find the nearest day of the week (specified in the rule) in the future
        nearest_abs_day = get_nearest_abs_day(self.value2.upper(), last_task_start_time.datetime)

        # the case_id where batch should start should be equal 
        # to the oldest batched task added to the queue (= the first element in the array)
        case_id, _ = waiting_times[0]

        return case_id, nearest_abs_day


    def get_batch_size_relative_time(self, element):
        curr_enabled_at = element["curr_enabled_at"]
        rule_value_str = self.value2.upper()
  
        enabled_times = element["enabled_datetimes"].copy()
        rule_nearest_past_day = get_nearest_past_day(rule_value_str, curr_enabled_at)
            
        boundary_day = datetime(rule_nearest_past_day.year, rule_nearest_past_day.month, rule_nearest_past_day.day, \
                        0, 0, 0, rule_nearest_past_day.microsecond, rule_nearest_past_day.tzinfo)

        if element["is_triggered_by_batch"] and \
            curr_enabled_at.date() == boundary_day.date():
                # the first executing task after the boundary date is a batched one
                # we either found tasks enabled before midnight and execute them
                # or we execute enabled batched tasks (if any) today
                op = operator.lt
                follow_rule = []
                for time in enabled_times:
                    if op(time, boundary_day):
                        follow_rule.append(time)

                if len(follow_rule) > 1:
                    return len(follow_rule), boundary_day
                    
        if element["is_triggered_by_batch"]:
            boundary_day = curr_enabled_at
            op = operator.le
        else:
            rule_nearest_past_day = get_nearest_past_day(rule_value_str, curr_enabled_at)
                
            boundary_day = datetime(rule_nearest_past_day.year, rule_nearest_past_day.month, rule_nearest_past_day.day, \
                            0, 0, 0, rule_nearest_past_day.microsecond, rule_nearest_past_day.tzinfo)
            op = operator.lt

        enabled_times = element["enabled_datetimes"].copy()

        follow_rule = []
        for time in enabled_times:
            if op(time, boundary_day):
                follow_rule.append(time)

        if len(follow_rule) > 0:
            return len(follow_rule), boundary_day 
        
        return 0, None

    def get_batch_size_by_daily_hour(self, element):
        curr_enabled_at = element["curr_enabled_at"]
        enabled_times = element["enabled_datetimes"].copy()

        one_day_delta = timedelta(days=1)
        oldest_in_batch_datetime = enabled_times[0]

        op = _get_operator_symbols_eq(self.operator)
        prev_day_to_oldest_rule_time = datetime.combine(
            oldest_in_batch_datetime, 
            self.value2, 
            oldest_in_batch_datetime.tzinfo)

        final_enabled_time = prev_day_to_oldest_rule_time
        while (prev_day_to_oldest_rule_time <= curr_enabled_at):
            follow_rule = []
            start_day = datetime.combine(prev_day_to_oldest_rule_time.date(), time(0, 0, 0, 0), oldest_in_batch_datetime.tzinfo)
            end_day = datetime.combine(prev_day_to_oldest_rule_time.date(), time(23, 59, 59, 0), oldest_in_batch_datetime.tzinfo)

            for en_time in enabled_times:
                if op == operator.eq:
                    if en_time <= prev_day_to_oldest_rule_time:
                        follow_rule.append(en_time)
                        final_enabled_time = prev_day_to_oldest_rule_time

                elif op(en_time, prev_day_to_oldest_rule_time):
                    # check whether we have something to execute before moving forward
                    if len(follow_rule) > 1:
                        return len(follow_rule), prev_day_to_oldest_rule_time

                    # satisy the condition
                    # execute as pair of two tasks
                    if en_time > end_day:
                        # already in the next day execution
                        continue

                    if op in [operator.gt, operator.ge] and en_time < start_day:
                        continue
                        
                    follow_rule.append(en_time)
                    if len(follow_rule) == 2:
                        if op == operator.eq:
                            final_enabled_time = prev_day_to_oldest_rule_time
                        else:
                            final_enabled_time = en_time
                        break
                else:
                    # do not satisfy the condition
                    # execute when reached the time or midnight
                    follow_rule.append(en_time)
                    if op in [operator.lt, operator.le]:
                        final_enabled_time = datetime.combine(
                            prev_day_to_oldest_rule_time.date(),
                            time(0, 0, 0),
                            prev_day_to_oldest_rule_time.tzinfo)
                    elif op in [operator.gt, operator.ge, operator.eq]:
                        final_enabled_time = prev_day_to_oldest_rule_time

            if len(follow_rule) > 1:
                # the first found batch of activities
                # satisfies the rule and of the size of at least 2 items inside
                return len(follow_rule), final_enabled_time 

            prev_day_to_oldest_rule_time += one_day_delta
        
        return 0, None


class AndFiringRule():
    def __init__(self, array_of_subrules: List[FiringSubRule]):
        self.rules = array_of_subrules

    def is_true(self, element):
        """
        :param element - dictionary with the next structure:
            - size (type: number) is current numbers of tasks waiting in the queue for the batch execution
            - waiting_time (type: array) shows the waiting time of each task in the queue 
                ordered by time insertion (same as ordered by case_id)
            - enabled_datetimes (type: array) shows the datetime of enabling for every task in the queue
            - curr_enabled_at (type: datetime) is the datetime of enabling for the very last task in the queue
        :return: is_true_result, batch_spec where:
            - is_true_result shows whether at least one batch is enabled for the execution
            - batch_spec shows the specification for the batch execution (array where the length shows the num of batches to be executed
                and the item in array refers to num of tasks to be executed in this i-batch)
        """
        is_true_result = True

        for rule in self.rules:
            is_true_result = is_true_result and rule.is_true(element)

        if is_true_result:
            num_tasks_in_queue = element["size"]
            num_tasks_in_batch, start_time_from_rule = self.get_firing_batch_size(num_tasks_in_queue, element)

            if num_tasks_in_batch < 2:
                print("WARNING: Getting batch size for the execution returned to be 0. Verify provided rules.")
                return False, None, None
            
            batch_spec = [num_tasks_in_batch]

            if num_tasks_in_queue > num_tasks_in_batch:
                # shift to the next tasks and validate the rule there
                new_num_tasks = num_tasks_in_queue - num_tasks_in_batch

                # adjust the processed batch passed to the 'is_true' method
                element["size"] = new_num_tasks
                element['waiting_times'] = element['waiting_times'][num_tasks_in_batch:]
                element['enabled_datetimes'] = element['enabled_datetimes'][num_tasks_in_batch:]

                is_true_iter, total_batch_count_iter, _ = self.is_true(element)
                if is_true_iter:
                    if total_batch_count_iter != None:
                        return is_true_result, batch_spec + total_batch_count_iter, start_time_from_rule
                    
                    return is_true_result, batch_spec, start_time_from_rule
                else:
                    # the next batch of tasks is not enabled for execution
                    return is_true_result, batch_spec, start_time_from_rule
            
            return True, batch_spec, start_time_from_rule
        
        return is_true_result, None, None

    def _get_batch_size_subrule(self):
        for rule in self.rules:
            if rule.is_batch_size():
                return rule
        
        return None

    def _get_week_day_subrule(self):
        for rule in self.rules:
            if rule.variable1 == "week_day":
                return rule
        
        return None

    def get_firing_batch_size(self, current_batch_size, element):
        batch_size = sys.maxsize
        enabled_time = element["curr_enabled_at"]

        for subrule in self.rules:
            curr_size = 0

            if subrule.is_batch_size():
                value2 = subrule.value2
                switcher = {
                    '<': min(current_batch_size, value2 - 1),
                    '<=': min(current_batch_size, value2),
                    '=': value2,
                    '>': current_batch_size if current_batch_size > value2 else 0,
                    '>=': current_batch_size if current_batch_size >= value2 else 0
                }

                batch_size_rule = switcher.get(subrule.operator)
                
                if batch_size == 0 or batch_size_rule == 0:
                    continue
                
                rule_operator: operator = _get_operator_symbols_eq(subrule.operator)
                if rule_operator(batch_size, value2):
                    # calculated batch_size for previous rules satisfies the size rule
                    # that means we return the size and datetime of that previosuly rule
                    # e.g., daily hour rule returned 3 items and 15:00 time, size rule also returned 3 items
                    # that means the first (daily hour rule) will be returned as a resulted one
                    return batch_size, enabled_time
                else:
                    # calculated batch_size for previous rules does not satisfy the size rule
                    # that means we return the size and datetime of the size rule which overrides all others
                    return batch_size_rule, element["curr_enabled_at"]
            
            elif subrule.variable1 == "week_day":
                curr_size, enabled_time = subrule.get_batch_size_relative_time(element)
            elif subrule.variable1 == "waiting_times":
                curr_size = current_batch_size
            elif subrule.variable1 == "daily_hour":
                curr_size, enabled_time = subrule.get_batch_size_by_daily_hour(element)

            if curr_size < batch_size:
                batch_size = curr_size

        return batch_size, enabled_time


    def get_enabled_time(self, waiting_times, last_task_enabled_time: CustomDatetimeAndSeconds) -> tuple([int, datetime]):
        expected_enabled_time = []
        for subrule in self.rules:
            if subrule.variable1 == "size":
                # size does not tell us anything about expected time of batch execution
                continue
            elif subrule.variable1 == "waiting_times":
                expected_enabled_time.append(subrule._get_min_enabled_time_waiting_time(waiting_times, last_task_enabled_time))
            elif subrule.variable1 == "week_day":
                expected_enabled_time.append(subrule._get_min_enabled_time_week_day(waiting_times, last_task_enabled_time)) 
            else:
                # no other rule types are being supported
                continue

        return expected_enabled_time



class OrFiringRule():
    def __init__(self, or_firing_rule_arr):
        self.rules = or_firing_rule_arr

    def is_true(self, spec):
        """
        Verifies whether the rule is true and thus batching is enabled
        :return: (is_batched_task_enabled, batch_spec, start_time) where:
            is_batched_task_enabled: shows whether one of the set of rules is true
            batch_spec: array where values tell how many items should be taken 
                into the batch at i-position / for i-batch
            start_time: returns datetime if that was dictated by the rule (e.g., Monday),
                otherwise, enabled_time is being returned
        """
        is_batched_task_enabled = False
        for rule in self.rules:
            is_batched_task_enabled, batch_spec, start_time = rule.is_true(spec)

            # fast exit if one of the rule is true
            if is_batched_task_enabled:
                return is_batched_task_enabled, batch_spec, start_time

        return is_batched_task_enabled, None, None
    
    def get_enabled_time(self, waiting_times, last_task_enabled_time: CustomDatetimeAndSeconds):
        """
        Method is being used to find the enabled time for the batch execution
        when all activities in the flow were already executed and we need to 
        execute the batch in the near future.
        Example:    The last task from all cases was executed on Friday. 
                    But there are still some activities waiting for the execution on Monday.
                    This function should return the nearest Monday, midnight time.
                    So that this batched tasks are being enabled and executed.
        """
        
        if len(waiting_times) < 2:
            # not a batch, less than two items in the queue
            return None

        enabled_times_per_or_rule = []
        for rule in self.rules:
            enabled_times_per_or_rule = enabled_times_per_or_rule + rule.get_enabled_time(waiting_times, last_task_enabled_time)

        # find and return the tuple (case_id, enabled_time) by the minimum of enabled_time
        return min(enabled_times_per_or_rule, key = lambda et: et[1])


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
