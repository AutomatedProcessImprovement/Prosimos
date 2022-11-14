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

    def is_batch_size_enough(self, batch_size):
        if self.variable1 == "ready_wt":
            return batch_size > 0
        else:
            return batch_size > 1

    def is_true(self, element):
        queue_size = element["size"]

        if not self.is_batch_size_enough(queue_size):
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
            return is_rule_true

        elif self.variable1 == "daily_hour":
            time1 = element["curr_enabled_at"].time()
            # in case of equal sign, we use greater or equal to sign
            # cause we compare it with the enabled time of the task next in the queue
            # it might be a case that we need to insert this current batch before execution of the next task
            op = _get_operator_symbols_ge(self.operator)
            is_rule_true = op(time1, self.value2)
            return is_rule_true

        elif self.variable1 == "ready_wt":
            last_enabled_datetime = element["enabled_datetimes"][-1]
            curr_enabled_datetime = element["curr_enabled_at"]
            op = _get_operator_symbols_ge(self.operator)

            ready_wt_sec = (curr_enabled_datetime - last_enabled_datetime).seconds
            is_rule_true = op(ready_wt_sec, self.value2)
           
            if is_rule_true == False:
                # edge case
                # check if the last waiting time does not exceed the limit
                # if yes, we can rollback its enabled time so the batch is activated in that case
                if op in [operator.le, operator.lt] and ready_wt_sec >= self.value2:
                    return True

            return is_rule_true


    def is_batch_size(self):
        return self.variable1 == "size"

    def _get_min_enabled_time_waiting_time(self, waiting_times, last_task_start_time: CustomDatetimeAndSeconds) -> tuple([int, datetime]):
        """
        Get the enabled time for the batch
        based on the maximum waiting time of the task in the batch 
        (meaning, for the 'waiting_time' kind of rule)
        """
        _, oldest_waiting = waiting_times[0]
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
        return last_task_start_time.datetime + timedelta(needed_time_seconds)

    def _get_min_enabled_time_week_day(self, last_task_start_time: CustomDatetimeAndSeconds) -> tuple([int, datetime]):
        if self._is_current_date_enabled(last_task_start_time.datetime):
            # if current day satisfies the rule - return it
            nearest_abs_day = last_task_start_time.datetime
        else:
            # find the nearest day of the week (specified in the rule) in the future
            nearest_abs_day = get_nearest_abs_day(self.value2.upper(), last_task_start_time.datetime)
        
        nearest_day_midnight = datetime.combine(
            nearest_abs_day.date(),
            time(0,0,0),
            nearest_abs_day.tzinfo
        )

        return nearest_day_midnight

    def _is_current_date_enabled(self, current_date):
        """ Verify whether current date satisfies the rule """
        completed_datetime_weekday = current_date.weekday()
        timer_weekday = str_week_days.get(self.value2.upper())
        return completed_datetime_weekday == timer_weekday

    def _get_min_enabled_time_daily_hour(self, last_task_start_time: CustomDatetimeAndSeconds) -> tuple([int, datetime]):
        last_task_start_datetime = last_task_start_time.datetime
        if self.operator in [operator.lt, operator.le, operator.eq]:
            boundary_time = time(0, 0, 0)
            boundary_date = last_task_start_datetime.date()
        else: 
            boundary_time = self.value2
            boundary_date = (last_task_start_datetime + timedelta(days=1)).date()

        nearest_enabled_datetime = datetime.combine(
            boundary_date,
            boundary_time,
            last_task_start_datetime.tzinfo
        )

        return nearest_enabled_datetime


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

    def get_batch_size_by_daily_hour(self, element, only_one_date = False):
        """
        only_one_date: no loops, you just check whether some batch could be enabled at this current datetime
        """
        curr_enabled_at = element["curr_enabled_at"]
        enabled_times = element["enabled_datetimes"]

        one_day_delta = timedelta(days=1)
        oldest_in_batch_datetime = enabled_times[0]

        op = _get_operator_symbols_eq(self.operator)
        prev_day_to_oldest_rule_time = datetime.combine(
            element["curr_enabled_at"].date(),
            self.value2,
            element["curr_enabled_at"].tzinfo
            ) if only_one_date else datetime.combine(
            oldest_in_batch_datetime, 
            self.value2, 
            oldest_in_batch_datetime.tzinfo) 

        final_enabled_time = curr_enabled_at
        while True:
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
                        break

                    if op in [operator.gt, operator.ge] and en_time < start_day:
                        # previous day, does not satisfy the rule
                        break
                        
                    follow_rule.append(en_time)
                    if len(follow_rule) == 2:
                        final_enabled_time = en_time
                        break
                else:
                    # do not satisfy the condition
                    # execute when reached the time or midnight
                    follow_rule.append(en_time)
                    if op in [operator.lt, operator.le]:
                        final_enabled_time = datetime.combine(
                            prev_day_to_oldest_rule_time.date() + timedelta(days=1),
                            time(0, 0, 0),
                            prev_day_to_oldest_rule_time.tzinfo)
                    elif op in [operator.gt, operator.ge, operator.eq]:
                        final_enabled_time = prev_day_to_oldest_rule_time

                if final_enabled_time > curr_enabled_at and not only_one_date:
                    # final enabled time of the batch cannot be larger then the starting enabled time
                    # (due to ordering in the log file)
                    # the future batch enablement will be handled later
                    return 0, None

            res_len, res_time = self._return_with_validation_arr(follow_rule, final_enabled_time)
            if res_len != 0:
                return res_len, res_time

            if only_one_date:
                # if we would have something to return,
                # it returned back in the prev statement
                return 0, None

            # go to the next iteration
            prev_day_to_oldest_rule_time += one_day_delta
            if prev_day_to_oldest_rule_time > curr_enabled_at:
                # return batches of at least size of 2
                return self._return_with_validation_arr(follow_rule, final_enabled_time)


    def _return_with_validation_arr(self, arr_follow_rule, enabled_time):
        " Validate whether the length of the arr_follow_rule (batch) is at least of 2"
        return self._return_with_validation(len(arr_follow_rule), enabled_time)


    def _return_with_validation(self, length, enabled_time):
        """
        Validate whether the length is at least of 2.
        Exception: ready_wt rule allows to return one item.
        """
        if self.is_batch_size_enough(length):
            # the first found batch of activities
            # satisfies the rule and of the size of at least 2 items inside
            return length, enabled_time 
        else:
            return 0, None


class AndFiringRule():
    def __init__(self, array_of_subrules: List[FiringSubRule]):
        self.rules = array_of_subrules
        self.ready_wt_boundaries = None

    def init_ready_wt_boundaries_if_any(self):
        low_boundary = None 
        high_boundary = None
        for rule in self.rules:
            if rule.variable1 == "ready_wt":
                if rule.operator == '=':
                    high_boundary = rule.value2
                    low_boundary = rule.value2
                elif rule.operator == '<':
                    high_boundary = rule.value2 - 1
                elif rule.operator == '<=':
                    high_boundary = rule.value2
                elif rule.operator == '>':
                    low_boundary = rule.value2 + 1
                elif rule.operator == '>=':
                    low_boundary = rule.value2

        if low_boundary == None:
            # case when we have only '<' sign
            low_boundary = high_boundary

        self.ready_wt_boundaries = low_boundary, high_boundary
        # TODO: clarify which cases are invalid: with only < or only > sign


    def _has_ready_wt_rule(self):
        for rule in self.rules:
            if rule.variable1 == "ready_wt":
                return True
        
        return False


    def get_ready_wt(self, element):
        if not self._has_ready_wt_rule():
            # no ready_wt rules
            return None

        low_boundary, high_boundary = self.ready_wt_boundaries

        curr_enabled_datetime = element["curr_enabled_at"]
        enabled_datetimes = element["enabled_datetimes"]

        prev_item = enabled_datetimes[0]
        enabled_dt_with_curr_enabled = enabled_datetimes.copy()
        enabled_dt_with_curr_enabled.append(curr_enabled_datetime)

        for en_time_index, item in enumerate(enabled_dt_with_curr_enabled[1:], 1):
            if item < prev_item and len(enabled_datetimes) == 1:
                # one item in the future
                # happens when no new cases will arrive 
                return en_time_index, _get_enabled_time_for_ready_wt_rule(prev_item, operator.gt, high_boundary)

            diff = (item - prev_item).seconds
            is_batch_enabled_low = diff < low_boundary

            if is_batch_enabled_low:
                prev_item = item
                continue

            if not is_batch_enabled_low:
                if en_time_index > 1:
                    # lower boundary is not fulfilled anymore
                    # we have at least two items in the batch, so enable the batch
                    # else we wait for the new task till the high boundary is reached
                    return en_time_index, _get_enabled_time_for_ready_wt_rule(prev_item, operator.ge, low_boundary)
            
            is_batch_enabled_high = diff > high_boundary
            if is_batch_enabled_high:
                # waiting time exceeds the higher boundary
                # stop waiting, enable the batch with one task
                return en_time_index, _get_enabled_time_for_ready_wt_rule(prev_item, operator.le, high_boundary)

            prev_item = item

        return 0, None


    def _get_min_enabled_time_ready_wt(self, case_id_and_enabled_times,
        last_task_start_time: CustomDatetimeAndSeconds) -> tuple([int, datetime]):
        
        last_task_start_datetime = last_task_start_time.datetime
        waiting_times = [ (last_task_start_datetime - v.datetime).total_seconds() for (_, v) in case_id_and_enabled_times ] 

        draft_element = {
            "size": len(case_id_and_enabled_times),
            "waiting_times": waiting_times ,
            "enabled_datetimes": [ v.datetime for (_, v) in case_id_and_enabled_times ],
            "curr_enabled_at": last_task_start_datetime,
            "is_triggered_by_batch": False,
        }

        ready_wt_res = self.get_ready_wt(draft_element)

        if ready_wt_res == None:
            return None
        else:
            _, en_time = ready_wt_res
            return en_time


    def is_batch_size_enough_for_exec(self, batch_size_res: int):
        return True if batch_size_res > 0 and self._has_ready_wt_rule() \
            else batch_size_res > 1


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

        if not is_true_result:
            # try finding whether we have batch waiting
            # meaning, the one that should have been executed previously in the timeline
            draft_element = {
                "size": element["size"],
                "waiting_times": element["waiting_times"].copy(),
                "enabled_datetimes": element["enabled_datetimes"].copy(),
                "curr_enabled_at": element["curr_enabled_at"],
                "is_triggered_by_batch": False,
            }
            batch_size_res, _ = self.get_firing_batch_size(draft_element["size"], draft_element)
            is_true_result = self.is_batch_size_enough_for_exec(batch_size_res)

        if is_true_result:
            num_tasks_in_queue = element["size"]
            num_tasks_in_batch, start_time_from_rule = self.get_firing_batch_size(num_tasks_in_queue, element)

            if not self.is_batch_size_enough_for_exec(num_tasks_in_batch): # num_tasks_in_batch < 2:
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

    def get_firing_batch_size(self, current_batch_size, element):
        batch_size = sys.maxsize
        initial_curr_enabled_at = element["curr_enabled_at"]
        enabled_time = initial_curr_enabled_at

        ready_wt_result = self.get_ready_wt(element)
        if ready_wt_result != None:
            batch_size, enabled_time = ready_wt_result

        is_time_forced = False

        for subrule in self.rules:
            curr_size = 0

            if subrule.is_batch_size():
                value2 = subrule.value2
                switcher = {
                    '<': min(current_batch_size, value2 - 1),
                    '<=': min(current_batch_size, value2),
                    '=': value2 if current_batch_size >= value2 else 0,
                    '>': current_batch_size if current_batch_size > value2 else 0,
                    '>=': current_batch_size if current_batch_size >= value2 else 0
                }

                batch_size_rule = switcher.get(subrule.operator)
                
                if batch_size == 0 or batch_size_rule == 0:
                    batch_size = 0
                    break
                
                rule_operator: operator = _get_operator_symbols_eq(subrule.operator)
                
                # check whether enabled_time was forced by some prev rule (e.g., daily_hour > 15)
                # use that previous time in case it additionally satisfies the size rule
                # otherwise, use the datetime when the batched task was enabled
                enabled_time = enabled_time if is_time_forced else element["curr_enabled_at"]
                
                # initial_batch_size = batch_size
                # if previously batch_size satisfy the size rule - return it
                # if not - return the one we've just calculated with the size rule limitations
                batch_size = batch_size if rule_operator(batch_size, value2) else batch_size_rule
                
                if batch_size > current_batch_size or \
                    enabled_time > initial_curr_enabled_at:
                    # 1) we do not have enought items in the batch to satisfy the rule
                    # or
                    # 2) enabled_time is in the future and will be handled later
                    return 0, None

                return batch_size, enabled_time
            
            elif subrule.variable1 == "week_day":
                curr_size, enabled_time = subrule.get_batch_size_relative_time(element)
                if curr_size < 2:
                    batch_size = 0
                    break
                is_time_forced = True
            elif subrule.variable1 == "waiting_times":
                oldest_element = element["waiting_times"][0]
                rule_operator: operator = _get_operator_symbols_eq(subrule.operator)
                value2 = subrule.value2
                if rule_operator(oldest_element, value2):
                    curr_size = current_batch_size
                else:
                    curr_size = 0
            elif subrule.variable1 == "daily_hour":
                only_one_date = False
                if is_time_forced:
                    element["curr_enabled_at"] = enabled_time
                    only_one_date = True
                
                curr_size, enabled_time = subrule.get_batch_size_by_daily_hour(element, only_one_date)
                if curr_size < 2:
                    batch_size = 0
                    break

                if is_time_forced:
                    # week_day and hour together
                    # hour overrides cause it takes into account the enabled datetime returned by week_day rule
                    batch_size = curr_size

                if enabled_time != None and enabled_time.time() == subrule.value2:
                    is_time_forced = True
                else:
                    is_time_forced = False
            elif subrule.variable1 == "ready_wt":
                continue

            if curr_size < batch_size:
                batch_size = curr_size

        if batch_size in [sys.maxsize, 0] or \
            enabled_time == None or \
            enabled_time > initial_curr_enabled_at:
            # 1) no iterations being made or 0 as a resulted size of batch
            # 2) enabled_time is not defined
            # 3) enabled_time is in the future (will be handled later)
            return 0, None

        return batch_size, enabled_time


    def get_enabled_time(self, waiting_times, last_task_enabled_time: CustomDatetimeAndSeconds, is_in_future: bool = False) -> tuple([int, datetime]):
        expected_enabled_time = []
        week_day_date = None

        en_time = self._get_min_enabled_time_ready_wt(waiting_times, last_task_enabled_time)
        if en_time != None:
            expected_enabled_time.append(en_time)
        
        for subrule in self.rules:
            if subrule.variable1 == "size":
                # size does not tell us anything about expected time of batch execution
                continue
            elif subrule.variable1 == "waiting_times":
                en_time = subrule._get_min_enabled_time_waiting_time(waiting_times, last_task_enabled_time)
                expected_enabled_time.append(en_time)
            elif subrule.variable1 == "week_day":
                en_time = subrule._get_min_enabled_time_week_day(last_task_enabled_time)
                expected_enabled_time.append(en_time)
                week_day_date = en_time.date()
            elif subrule.variable1 == "daily_hour":
                en_time = subrule._get_min_enabled_time_daily_hour(last_task_enabled_time)
                if week_day_date != None:
                    en_time = datetime.combine(
                        week_day_date,
                        en_time.time(),
                        en_time.tzinfo
                    )
                expected_enabled_time.append(en_time)
            elif subrule.variable1 == "ready_wt":
                # was calculated previously
                continue
            else:
                # no other rule types are being supported
                continue

        return expected_enabled_time



class OrFiringRule():
    def __init__(self, or_firing_rule_arr):
        self.rules = or_firing_rule_arr

    def is_ready_wt_rule_present(self):
        for orRule in self.rules:
            is_present = orRule._has_ready_wt_rule()
            if is_present:
                return is_present
        
        return False

    def is_batch_size_enough_for_exec(self, batch_size):
        return True if batch_size > 0 and self.is_ready_wt_rule_present \
            else batch_size > 1

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
    
    def get_enabled_time(self, waiting_times, last_task_enabled_time: CustomDatetimeAndSeconds, is_in_future: bool = False):
        """
        :param: is_in_future: whether the returned datetime could be in the future
        Method is being used to find the enabled time for the batch execution
        when all activities in the flow were already executed and we need to 
        execute the batch in the near future.
        Example:    The last task from all cases was executed on Friday. 
                    But there are still some activities waiting for the execution on Monday.
                    This function should return the nearest Monday, midnight time.
                    So that this batched tasks are being enabled and executed.
        """
        
        if not self.is_batch_size_enough_for_exec(len(waiting_times)):
            # not a batch, less than two items in the queue
            return None

        enabled_times_per_or_rule = []
        for rule in self.rules:
            per_rule = rule.get_enabled_time(waiting_times, last_task_enabled_time, is_in_future)
            if len(per_rule) > 0:
                # find the max time 
                # cause that's the one that satisfies the set of rules
                selected_enabled_time = max(per_rule)
                enabled_times_per_or_rule = enabled_times_per_or_rule + [selected_enabled_time]

        if len(enabled_times_per_or_rule) == 0:
            # e.g., size_rule = 4 but we have only 2 tasks waiting
            return None

        # find and return the tuple (case_id, enabled_time) by the minimum of enabled_time
        # meaning the fastest time when one of the rule matches
        return min(enabled_times_per_or_rule)


def _get_enabled_time_for_ready_wt_rule(last_item_in_batch, op, boundary_value):
    if op in [operator.lt]:
        batch_enabled_time = last_item_in_batch + timedelta(seconds=boundary_value - 1)
    elif op in [operator.eq, operator.ge, operator.le]:
        batch_enabled_time = last_item_in_batch + timedelta(seconds=boundary_value)
    elif op == operator.gt:
        batch_enabled_time = last_item_in_batch + timedelta(seconds=boundary_value + 1)

    return batch_enabled_time


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
