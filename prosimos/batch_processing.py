import operator as operator
import sys
from datetime import date, datetime, time, timedelta
from enum import Enum
from random import choices
from typing import List

from pix_framework.discovery.resource_calendar_and_performance.crisp.resource_calendar import str_week_days

from prosimos.exceptions import InvalidRuleDefinitionException
from prosimos.weekday_helper import CustomDatetimeAndSeconds, get_nearest_abs_day, get_nearest_past_day


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


class RULE_TYPE(Enum):
    READY_WT = "ready_wt"
    LARGE_WT = "large_wt"
    DAILY_HOUR = "daily_hour"
    WEEK_DAY = "week_day"
    SIZE = "size"


class FiringSubRule():
    def __init__(self, variable1: str, operator: str, value2: str):
        self.variable1 = variable1
        self.operator = operator
        self.value2 = value2

    def is_batch_size_enough(self, batch_size):
        if RULE_TYPE(self.variable1) in [RULE_TYPE.LARGE_WT, RULE_TYPE.READY_WT]:
            return batch_size > 0
        else:
            return batch_size > 1

    def _get_new_time(self, diff_timedelta, op: operator):
        curr_datetime = datetime.combine(date.today(), self.value2)
        return op(curr_datetime, diff_timedelta).time() 

    def is_true(self, element):
        queue_size = element["size"]

        if not self.is_batch_size_enough(queue_size):
            # not enough to be executed in batch, at least 2 tasks required
            return False

        if self.variable1 == "large_wt":
            value1_list = element["waiting_times"]

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

    def get_batch_size_by_daily_hour(self, element, only_one_date, range):
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
            start_time, end_time = range
            start_day = datetime.combine(prev_day_to_oldest_rule_time.date(), start_time, oldest_in_batch_datetime.tzinfo)
            end_day = datetime.combine(prev_day_to_oldest_rule_time.date(), end_time, oldest_in_batch_datetime.tzinfo)

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


    def is_invalid_end(self, curr_batch_size, last_wt):
        is_invalid = False

        if RULE_TYPE(self.variable1) == RULE_TYPE.SIZE:
            # there is no way we will increment the number of tasks waiting for the batch exec
            # cause we are at the end of simulated cases
            if self.operator in ['=', '>=']:
                is_invalid = curr_batch_size < self.value2
            elif self.operator == '>':
                is_invalid = curr_batch_size <= self.value2
        elif RULE_TYPE(self.variable1) == RULE_TYPE.READY_WT:
            # if we surpass the upper limit of waiting_time,
            # this will not be changed in the future cause we will not receive a new item
            # and thus ready_wt value will not be reset
            _, high_boundary = self.ready_wt_boundaries
            is_invalid = last_wt > high_boundary

        return is_invalid


class AndFiringRule():
    def __init__(self, array_of_subrules: List[FiringSubRule]):
        self.rules = array_of_subrules
        self.ready_wt_boundaries = None
        self.large_wt_boundaries = None
        self.daily_hour_range = (time(0,0,0,0), time(23,59,59,0))

    
    def _has_rule(self, rule_name: List[RULE_TYPE]):
        rule_values = list(map(lambda x: x.value, rule_name))
        for rule in self.rules:
            if rule.variable1 in rule_values:
                return True
        
        return False


    def init_boundaries(self):
        self._init_ready_wt_boundaries_if_any()
        self._init_large_wt_boundaries_if_any()
        self._init_daily_hour_range_if_any()


    def _init_ready_wt_boundaries_if_any(self):
        rule_type: RULE_TYPE = RULE_TYPE.READY_WT
        if not self._has_rule([rule_type]):
            return 

        self.ready_wt_boundaries = self._get_low_and_high_boundaries(rule_type, 1)


    def _init_large_wt_boundaries_if_any(self):
        rule_type: RULE_TYPE = RULE_TYPE.LARGE_WT
        if not self._has_rule([rule_type]):
            return 

        self.large_wt_boundaries = self._get_low_and_high_boundaries(rule_type, 1)

    def _init_daily_hour_range_if_any(self):
        rule_type: RULE_TYPE = RULE_TYPE.DAILY_HOUR
        if not self._has_rule([rule_type]):
            return 

        self.daily_hour_range = self._get_low_and_high_boundaries(rule_type, timedelta(seconds=1))


    def validate(self):
        week_day_count = 0
        daily_hour_count = 0
        for rule in self.rules:
            if rule.variable1 == RULE_TYPE.WEEK_DAY.value:
                week_day_count += 1
            elif rule.variable1 == RULE_TYPE.DAILY_HOUR.value:
                daily_hour_count += 1

        if week_day_count > 1:
            raise InvalidRuleDefinitionException("Only one WEEK_DAY subrule is allowed inside AND rule.")
        elif daily_hour_count > 2:
            raise InvalidRuleDefinitionException("Only one or two subrules of DAILY_HOUR type is allowed inside AND rule.")


    def _get_low_and_high_boundaries(self, rule_name: RULE_TYPE, diff_units):
        if isinstance(diff_units, timedelta):
            low_boundary, high_boundary = self.daily_hour_range
        else:
            low_boundary, high_boundary = (None, None)

        for rule in self.rules:
            if rule.variable1 == rule_name.value:
                if rule.operator == '=':
                    high_boundary = rule.value2
                    low_boundary = rule.value2
                elif rule.operator == '<':
                    low_boundary = 0 if low_boundary == None else low_boundary
                    if isinstance(diff_units, timedelta):
                        # subtract 1 second
                        high_boundary = rule._get_new_time(diff_units, operator.sub)
                    else:
                        high_boundary = rule.value2 - diff_units
                elif rule.operator == '<=':
                    low_boundary = 0 if low_boundary == None else low_boundary
                    high_boundary = rule.value2
                elif rule.operator == '>':
                    if isinstance(diff_units, timedelta):
                        # add 1 second
                        low_boundary = rule._get_new_time(diff_units, operator.add) 
                    else:
                        low_boundary = rule.value2 + diff_units
                elif rule.operator == '>=':
                    low_boundary = rule.value2

        if low_boundary == None or high_boundary == None:
            raise Exception(f"Invalid range of {rule_name.value} rule")

        return low_boundary, high_boundary


    def get_ready_wt(self, element):
        if not self._has_rule([RULE_TYPE.READY_WT]):
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
                return en_time_index, _get_enabled_time_for_wt_rule(prev_item, operator.gt, high_boundary)

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
                    return en_time_index, _get_enabled_time_for_wt_rule(prev_item, operator.ge, low_boundary)
            
            is_batch_enabled_high = diff > high_boundary
            if is_batch_enabled_high:
                # waiting time exceeds the higher boundary
                # stop waiting, enable the batch with one task
                return en_time_index, _get_enabled_time_for_wt_rule(prev_item, operator.le, high_boundary)

            prev_item = item

        return 0, None

    def get_large_wt(self, element):
        if not self._has_rule([RULE_TYPE.LARGE_WT]):
            # no large_wt rules
            return None

        low_boundary, high_boundary = self.large_wt_boundaries

        curr_enabled_datetime = element["curr_enabled_at"]
        enabled_datetimes = element["enabled_datetimes"]

        first_item = enabled_datetimes[0]
        enabled_dt_with_curr_enabled = enabled_datetimes.copy()
        enabled_dt_with_curr_enabled.append(curr_enabled_datetime)

        result = 0, None
        prev_item = first_item
        list_len = len(enabled_dt_with_curr_enabled)
        for en_time_index, item in enumerate(enabled_dt_with_curr_enabled[1:], 1):
            diff = (item - first_item).seconds
            is_batch_enabled_low = diff > low_boundary
            
            if not is_batch_enabled_low:
                if en_time_index == list_len - 1:
                    # last item of the evaluation
                    # this will enable the batch in the future
                    result = en_time_index, _get_enabled_time_for_wt_rule(first_item, operator.ge, low_boundary)
                    break

                prev_item = item
                continue

            is_batch_enabled_high = diff > high_boundary

            if not is_batch_enabled_high and en_time_index < 2:
                # wait for the second item in the batch
                # or reaching the high_boundary
                prev_item = item
                continue

            if is_batch_enabled_high and en_time_index == 1:
                return en_time_index, _get_enabled_time_for_wt_rule(first_item, operator.gt, high_boundary)

            low_boundary_date = first_item + timedelta(seconds=low_boundary)
            high_boundary_date = first_item + timedelta(seconds=high_boundary)

            if prev_item <= low_boundary_date:
                # wait for at least the lowest possible date
                return en_time_index, _get_enabled_time_for_wt_rule(first_item, operator.ge, low_boundary)
            elif low_boundary_date <= prev_item <= high_boundary_date:
                # enabled time of the task = enabled time of the batch
                return en_time_index, prev_item
            else: # higher than the highest boundary
                return en_time_index, _get_enabled_time_for_wt_rule(first_item, operator.gt, high_boundary)
        
        return result # 0, None


    def _get_min_enabled_time_waiting_time(self, case_id_and_enabled_times,
        last_task_start_time: CustomDatetimeAndSeconds, rule_type: RULE_TYPE) -> tuple([int, datetime]):
        
        last_task_start_datetime = last_task_start_time.datetime
        waiting_times = [ (last_task_start_datetime - v.datetime).total_seconds() for (_, v) in case_id_and_enabled_times ] 

        draft_element = {
            "size": len(case_id_and_enabled_times),
            "waiting_times": waiting_times ,
            "enabled_datetimes": [ v.datetime for (_, v) in case_id_and_enabled_times ],
            "curr_enabled_at": last_task_start_datetime,
            "is_triggered_by_batch": False,
            "is_only_one_batch_return": False
        }

        wt_res = self.get_ready_wt(draft_element) \
            if rule_type == RULE_TYPE.READY_WT else self.get_large_wt(draft_element)

        if wt_res == None:
            return None
        else:
            _, en_time = wt_res
            return en_time

    def is_size_one(self):
        for rule in self.rules:
            if rule.variable1 == RULE_TYPE.SIZE.value:
                return rule.value2 == 1

        return False

    def is_batch_size_enough_for_exec(self, batch_size_res: int):
        return True if (self._has_rule([RULE_TYPE.READY_WT, RULE_TYPE.LARGE_WT]) or \
                self.is_size_one()) and \
                batch_size_res > 0 \
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
                "is_only_one_batch_return": False
            }
            batch_size_res, _ = self.get_firing_batch_size(draft_element["size"], draft_element)
            is_true_result = self.is_batch_size_enough_for_exec(batch_size_res)

        if is_true_result:
            num_tasks_in_queue = element["size"]
            num_tasks_in_batch, start_time_from_rule = self.get_firing_batch_size(num_tasks_in_queue, element)

            if not self.is_batch_size_enough_for_exec(num_tasks_in_batch):
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

    def get_firing_batch_size_ready_and_large(self, element, init_batch_size, init_enabled_time):
        batch_size, enabled_time = (init_batch_size, init_enabled_time)

        ready_wt_result = self.get_ready_wt(element)
        large_wt_result = self.get_large_wt(element)

        if ready_wt_result != None and large_wt_result != None:
            r_batch_size, r_enabled_time = ready_wt_result
            l_batch_size, l_enabled_time = large_wt_result
            if r_batch_size == 0 or l_batch_size == 0:
                return 0, None

            batch_size = min(value for value in [r_batch_size, l_batch_size] if value is not None)
            if batch_size == 1:
                # when batch_size == 1, surpassing the lowest enabled time will make the rule invalid 
                # there is no way the rule can turn to true once it surpass the upper boundary of one of the rule
                enabled_time = min(value for value in [r_enabled_time, l_enabled_time] if value is not None)
            else:
                # when we have multiple tasks in the batch, we wait for the full fulfillment of the rules
                # so we wait to the latest
                enabled_time = max(value for value in [r_enabled_time, l_enabled_time] if value is not None)

        elif ready_wt_result != None:
            batch_size, enabled_time = ready_wt_result

        elif large_wt_result != None:
            batch_size, enabled_time = large_wt_result

        return batch_size, enabled_time

    def get_firing_batch_size(self, current_batch_size, element):
        batch_size = sys.maxsize
        initial_curr_enabled_at = element["curr_enabled_at"]
        enabled_time = initial_curr_enabled_at

        batch_size, enabled_time = self.get_firing_batch_size_ready_and_large(element, batch_size, enabled_time)

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
                    # 1) we do not have enough items in the batch to satisfy the rule
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
            elif subrule.variable1 == "daily_hour":
                only_one_date = False
                if is_time_forced:
                    element["curr_enabled_at"] = enabled_time
                    only_one_date = True
                
                curr_size, enabled_time = subrule.get_batch_size_by_daily_hour(element, only_one_date, self.daily_hour_range)
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
            elif subrule.variable1 == "large_wt":
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


    def get_enabled_time(self, waiting_times, last_task_enabled_time: CustomDatetimeAndSeconds) -> tuple([int, datetime]):
        expected_enabled_time = []
        week_day_date = None

        for rule in [RULE_TYPE.READY_WT, RULE_TYPE.LARGE_WT]:
            if not self._has_rule([rule]):
                continue

            en_time = self._get_min_enabled_time_waiting_time(waiting_times, last_task_enabled_time, rule)
            if en_time != None:
                expected_enabled_time.append(en_time)
        
        for subrule in self.rules:
            if subrule.variable1 == "size":
                expected_enabled_time.append(last_task_enabled_time.datetime)
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
            elif subrule.variable1 == "large_wt":
                # was calculated previously
                continue
            elif subrule.variable1 == "ready_wt":
                # was calculated previously
                continue
            else:
                # no other rule types are being supported
                continue

        return expected_enabled_time


    def is_invalid(self, first_item_wt):
        """
        Verify whether the rule is invalid in the middle of simulation
        invalid rule = one that will never become satisfied in the future
        """
        is_simple_rule_invalid = False

        for rule in self.rules:
            if RULE_TYPE(rule.variable1) == RULE_TYPE.LARGE_WT:
                # the rule is invalid when we surpass the higher boundary of the large waiting time
                # cause the difference will only increase in the future
                _, high_boundary = self.large_wt_boundaries
                is_simple_rule_invalid = first_item_wt > high_boundary

            if is_simple_rule_invalid:
                # if at least one of the simple rule is invalid,
                # all complex rule is invalid
                return True


    def is_invalid_end(self, curr_batch_size, first_wt, last_wt):
        is_invalid = False
        
        for simple_rule in self.rules:
            is_invalid = simple_rule.is_invalid_end(curr_batch_size, last_wt)

            if is_invalid:
                break

        if not is_invalid:
            # if there are no specific end violations
            # we check whether there are normal violations
            # "normal" = those which can happen in the middle of simulation
            is_invalid = self.is_invalid(first_wt)

        return is_invalid


class OrFiringRule():
    def __init__(self, or_firing_rule_arr):
        self.rules = or_firing_rule_arr

    def is_ready_wt_rule_present(self):
        for or_rule in self.rules:
            is_present = or_rule._has_ready_wt_rule()
            if is_present:
                return is_present
        
        return False

    def is_size_rule_present(self):
        for and_rule in self.rules:
            is_present = and_rule._has_rule([RULE_TYPE.SIZE])
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

        # no enabled batches, check whether rule is invalid
        # if yes, batch is triggered straight away
        first_wt = spec["waiting_times"][0]
        for rule in self.rules:
            if rule.is_invalid(first_wt):
                # in case rule is invalid,
                # we trigger the batch execution straight away with all tasks waiting for the batch exec
                # and "current" moment of time we have
                return True, [spec["size"]], spec["curr_enabled_at"]
        
        return is_batched_task_enabled, None, None


    def is_invalid_end(self, num_tasks, first_wt, ready_wt):
        """
        Check whether items waiting for batch execution might be satisfied in the future (valid for further processing)
        or they are invalid (one part of the AND rule could not be satisfied in the future at all)
        :param num_tasks: number of tasks waiting for batch execution
        :param first_wt: waiting time of the first item (current_point_in_time - first_item.enable_time).seconds
        :param ready_wt: waiting time of the last task in the batch queue
        :return: whether the rule is invalid
        :rtype: boolean
        """
        is_invalid = False
        
        for and_rule in self.rules:
            is_invalid = and_rule.is_invalid_end(num_tasks, first_wt, ready_wt)

            if is_invalid:
                return True

        return is_invalid
    

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
        
        if not self.is_batch_size_enough_for_exec(len(waiting_times)):
            # not a batch, less than two items in the queue
            return None

        enabled_times_per_or_rule = []
        for rule in self.rules:
            per_rule = rule.get_enabled_time(waiting_times, last_task_enabled_time)
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


def _get_enabled_time_for_wt_rule(last_item_in_batch, op, boundary_value):
    if op in [operator.lt]:
        batch_enabled_time = last_item_in_batch + timedelta(seconds=boundary_value - 1)
    elif op in [operator.eq, operator.ge, operator.le]:
        batch_enabled_time = last_item_in_batch + timedelta(seconds=boundary_value)
    elif op == operator.gt:
        batch_enabled_time = last_item_in_batch + timedelta(seconds=boundary_value + 1)

    return batch_enabled_time


class BatchConfigPerTask():
    def __init__(self, type, duration_distribution, firing_rules, possible_options, probabilities):
        self.type = type
        self.duration_distribution = duration_distribution
        self.sorted_duration_distribution = sorted(duration_distribution)
        self.firing_rules = firing_rules
        self.possible_options = possible_options
        self.probabilities = probabilities
        self.are_rules_discovered = len(self.firing_rules.rules) > 0

        if not self.are_rules_discovered:
            # define the initial rule based on size_distr
            self.update_firing_rules_from_distr()
        else:
            # redefine the probabilities
            # to be used to get the probabiity of batch to be executed alone
            prob_for_one = self.probabilities[0]
            other_than_one_prob = 1 - prob_for_one
            
            self.possible_options = [1,2]
            self.probabilities = [prob_for_one, other_than_one_prob]


    def is_batch_exec_alone(self):
        """
        Based on size_distr param, we calculate
        whether the batch is executed either alone as a task or 
        in the batch following the defined rules
        """
        # could return 1 or 2
        prob_num = self.get_batch_size()

        # if we returned 1, then we execute batch alone
        # otherwise, 2 refers to all cases other than 1 and 
        # means that batch should start creating following the rule
        return prob_num == 1

    def update_firing_rules_from_distr(self):
        if len(self.possible_options) == 0:
            return

        self.firing_rules = OrFiringRule(or_firing_rule_arr=[
            AndFiringRule(array_of_subrules=[
                self.get_new_subrule_rule()
            ])
        ])

    def get_new_subrule_rule(self):
        return FiringSubRule("size", "=", self.get_batch_size())

    def get_batch_size(self):
        one_item_list = choices(self.possible_options, self.probabilities)
        return one_item_list[0]

    def calculate_ideal_duration(self, initial_duration, num_tasks_in_batch):
        if num_tasks_in_batch == 1:
            # one task in the batch is executed with the initial_duration
            return initial_duration

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
            
            # assign the nearest coeff as the current one
            curr_coef = self.duration_distribution[min_key]

        # calculate the duration for executing one of the batched task
        duration_per_task = initial_duration * curr_coef
        if self.type == BATCH_TYPE.PARALLEL:
            # the total duration of every item in the batch
            # should reflect the whole time spent for the executing of the batch
            duration = duration_per_task * num_tasks_in_batch
            return duration

        return duration_per_task
