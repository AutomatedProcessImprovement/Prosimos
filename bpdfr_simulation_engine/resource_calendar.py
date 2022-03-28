import datetime
import math
from datetime import timedelta
from dateutil import parser

import pytz

str_week_days = {"MONDAY": 0, "TUESDAY": 1, "WEDNESDAY": 2, "THURSDAY": 3, "FRIDAY": 4, "SATURDAY": 5, "SUNDAY": 6}
int_week_days = {0: "MONDAY", 1: "TUESDAY", 2: "WEDNESDAY", 3: "THURSDAY", 4: "FRIDAY", 5: "SATURDAY", 6: "SUNDAY"}

convertion_table = {'WEEKS': 604800,
                    'DAYS': 86400,
                    'HOURS': 3600,
                    'MINUTES': 60,
                    'SECONDS': 1}


class CalendarItem:
    def __init__(self, from_day, to_day, begin_time, end_time):
        self.from_day = from_day
        self.to_day = to_day
        self.begin_time = begin_time
        self.end_time = end_time


class Interval:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.duration = (end - start).total_seconds()

    def merge_interval(self, n_interval):
        self.start = min(n_interval.start, self.start)
        self.end = max(n_interval.end, self.end)
        self.duration = (self.end - self.start).total_seconds()

    def is_before(self, c_date):
        return self.end <= c_date

    def contains(self, c_date):
        return self.start < c_date < self.end

    def contains_inclusive(self, c_date):
        return self.start <= c_date <= self.end

    def is_after(self, c_date):
        return c_date <= self.start

    def intersection(self, interval):
        if interval is None:
            return None
        [first_i, second_i] = [self, interval] if self.start <= interval.start else [interval, self]
        if second_i.start < first_i.end:
            return Interval(max(first_i.start, second_i.start), min(first_i.end, second_i.end))
        return None


class CalendarIterator:
    def __init__(self, start_date: datetime, calendar_info):
        self.start_date = start_date

        self.calendar = calendar_info

        self.c_day = start_date.date().weekday()

        c_date = datetime.datetime.combine(calendar_info.default_date, start_date.time())
        c_interval = calendar_info.work_intervals[self.c_day][0]
        self.c_index = -1
        while c_interval.end < c_date and self.c_index < len(calendar_info.work_intervals[self.c_day]) - 1:
            self.c_index += 1
            c_interval = calendar_info.work_intervals[self.c_day][self.c_index]

        self.c_interval = Interval(self.start_date,
                                   self.start_date + timedelta(seconds=(c_interval.end - c_date).total_seconds()))

    def next_working_interval(self):
        res_interval = self.c_interval
        day_intervals = self.calendar.work_intervals[self.c_day]
        p_duration = 0

        self.c_index += 1
        if self.c_index >= len(day_intervals):
            p_duration += 86400 - (day_intervals[self.c_index - 1].end - self.calendar.new_day).total_seconds()
            while True:
                self.c_day = (self.c_day + 1) % 7
                day_intervals = self.calendar.work_intervals[self.c_day]
                if len(day_intervals) > 0:
                    p_duration += (day_intervals[0].start - self.calendar.new_day).total_seconds()
                    break
                else:
                    p_duration += 86400
            self.c_index = 0
        elif self.c_index > 0:
            p_duration += (day_intervals[self.c_index].start - day_intervals[self.c_index - 1].end).total_seconds()
        self.c_interval = Interval(res_interval.end + timedelta(seconds=p_duration),
                                   res_interval.end + timedelta(
                                       seconds=p_duration + day_intervals[self.c_index].duration))
        return res_interval


class IntervalPoint:
    def __init__(self, date_time, week_day, index, to_start_dist, to_end_dist):
        self.date_time = date_time
        self.week_day = week_day
        self.index = index
        self.to_start_dist = to_start_dist
        self.to_end_dist = to_end_dist

    def in_same_interval(self, another_point):
        return self.week_day == another_point.week_day and self.index == another_point.index


class RCalendar:
    def __init__(self, calendar_id):
        self.calendar_id = calendar_id
        self.default_date = None
        self.new_day = None
        self.work_intervals = dict()
        self.cumulative_work_durations = dict()
        self.work_rest_count = dict()
        self.total_weekly_work = 0
        self.total_weekly_rest = to_seconds(1, 'WEEKS')
        for i in range(0, 7):
            self.work_intervals[i] = list()
            self.cumulative_work_durations[i] = list()
            self.work_rest_count[i] = [0, to_seconds(1, 'DAYS')]

    def print_calendar_info(self):
        print('Calendar ID: %s' % self.calendar_id)
        print('Total Weekly Work: %.2f Hours' % (self.total_weekly_work / 3600))
        for i in range(0, 7):
            if len(self.work_intervals[i]) > 0:
                print(int_week_days[i])
                for interval in self.work_intervals[i]:
                    print('    from %02d:%02d - to %02d:%02d' % (interval.start.hour, interval.start.minute,
                                                                 interval.end.hour, interval.end.minute))
        print('-----------------------------------------------------------')

    def to_json(self):
        items = []
        for i in range(0, 7):
            if len(self.work_intervals[i]) > 0:
                for interval in self.work_intervals[i]:
                    items.append({
                        'from': int_week_days[i],
                        'to': int_week_days[i],
                        "beginTime": str(interval.start.time()),
                        "endTime": str(interval.end.time())
                    })
        return items

    def is_working_datetime(self, date_time):
        c_day = date_time.date().weekday()
        c_date = datetime.datetime.combine(self.default_date, date_time.time())
        i_index = 0
        for interval in self.work_intervals[c_day]:
            if interval.contains_inclusive(c_date):
                return True, IntervalPoint(date_time, i_index, c_day, (c_date - interval.start).total_seconds(),
                                           (interval.end - c_date).total_seconds())
            i_index += 1
        return False, None

    def combine_calendar(self, new_calendar):
        for i in range(0, 7):
            if len(new_calendar.work_intervals[i]) > 0:
                for interval in new_calendar.work_intervals[i]:
                    self.add_calendar_item(int_week_days[i], int_week_days[i],
                                           str(interval.start.time()), str(interval.end.time()))

    def add_calendar_item(self, from_day, to_day, begin_time, end_time):
        if from_day.upper() in str_week_days and to_day.upper() in str_week_days:
            try:
                t_interval = Interval(parse_datetime(begin_time, False), parse_datetime(end_time, False))
                if self.default_date is None:
                    self.default_date = t_interval.start.date()
                    self.new_day = datetime.datetime.combine(self.default_date, datetime.time())
                d_s = str_week_days[from_day]
                d_e = str_week_days[to_day]
                while True:
                    self._add_interval(d_s % 7, t_interval)
                    if d_s % 7 == d_e:
                        break
                    d_s += 1
            except ValueError:
                return

    def compute_cumulative_durations(self):
        for w_day in self.work_intervals:
            cumulative = 0
            for interval in self.work_intervals[w_day]:
                cumulative += interval.duration
                self.cumulative_work_durations[w_day].append(cumulative)

    def _add_interval(self, w_day, interval):
        i = 0
        for to_merge in self.work_intervals[w_day]:
            if to_merge.end < interval.start:
                i += 1
                continue
            if interval.end < to_merge.start:
                break
            merged_duration = to_merge.duration
            to_merge.merge_interval(interval)
            merged_duration = to_merge.duration - merged_duration
            i += 1
            while i < len(self.work_intervals[w_day]):
                next_i = self.work_intervals[w_day][i]
                if to_merge.end < next_i.start:
                    break
                if next_i.end <= to_merge.end:
                    merged_duration -= next_i.duration
                elif next_i.start <= to_merge.end:
                    merged_duration -= (to_merge.end - next_i.start).total_seconds()
                    to_merge.merge_interval(next_i)
                del self.work_intervals[w_day][i]
            if merged_duration < 0:
                print('HOOOOLA')
            if merged_duration > 0:
                self._update_calendar_durations(w_day, merged_duration)
            return
        self.work_intervals[w_day].insert(i, interval)
        self._update_calendar_durations(w_day, interval.duration)

    def _update_calendar_durations(self, w_day, duration):
        self.work_rest_count[w_day][0] += duration
        self.work_rest_count[w_day][1] -= duration
        self.total_weekly_work += duration
        self.total_weekly_rest -= duration

    def remove_idle_times(self, from_date, to_date, out_intervals: list):
        calendar_it = CalendarIterator(from_date, self)
        while True:
            c_interval = calendar_it.next_working_interval()
            if c_interval.end < to_date:
                out_intervals.append(c_interval)
            else:
                if c_interval.start <= to_date <= c_interval.end:
                    out_intervals.append(Interval(c_interval.start, to_date))
                break

    def find_idle_time(self, requested_date, duration):
        if duration == 0:
            return 0
        real_duration = 0
        pending_duration = duration
        if duration > self.total_weekly_work:
            real_duration += to_seconds(int(duration / self.total_weekly_work), 'WEEKS')
            pending_duration %= self.total_weekly_work
        # Addressing the first day as an special case
        c_day = requested_date.date().weekday()
        c_date = datetime.datetime.combine(self.default_date, requested_date.time())

        worked_time, total_time = self._find_time_starting(pending_duration, c_day, c_date)
        if worked_time > total_time and worked_time - total_time < 0.001:
            total_time = worked_time
        pending_duration -= worked_time
        real_duration += total_time
        c_date = self.new_day
        while pending_duration > 0:
            c_day += 1
            r_d = c_day % 7
            if pending_duration > self.work_rest_count[r_d][0]:
                pending_duration -= self.work_rest_count[r_d][0]
                real_duration += 86400
            else:
                real_duration += self._find_time_completion(pending_duration, self.work_rest_count[r_d][0], r_d, c_date)
                break
        return real_duration

    def next_available_time(self, requested_date):
        c_day = requested_date.date().weekday()
        c_date = datetime.datetime.combine(self.default_date, requested_date.time())

        for interval in self.work_intervals[c_day]:
            if interval.end == c_day:
                continue
            if interval.is_after(c_date):
                return (interval.start - c_date).total_seconds()
            if interval.contains(c_date):
                return 0
        duration = 86400 - (c_date - self.new_day).total_seconds()
        for i in range(c_day + 1, c_day + 8):
            r_day = i % 7
            if self.work_rest_count[r_day][0] > 0:
                return duration + (self.work_intervals[r_day][0].start - self.new_day).total_seconds()
            duration += 86400
        return duration

    def find_working_time(self, start_date, end_date):
        # print("%s -- %s" % (str(start_date), str(end_date)))
        pending_duration = (end_date - start_date).total_seconds()
        worked_hours = 0

        c_day = start_date.date().weekday()
        c_date = datetime.datetime.combine(self.default_date, start_date.time())

        to_complete_day = 86400 - (c_date - self.new_day).total_seconds()
        available_work = self._calculate_available_duration(c_day, c_date)

        previous_date = c_date
        while pending_duration > to_complete_day:
            pending_duration -= to_complete_day
            worked_hours += available_work
            c_day = (c_day + 1) % 7
            available_work = self.work_rest_count[c_day][0]
            to_complete_day = 86400
            previous_date = self.new_day

        for interval in self.work_intervals[c_day]:
            if interval.is_before(previous_date):
                continue
            interval_duration = interval.duration
            if interval.contains(previous_date):
                interval_duration -= (previous_date - interval.start).total_seconds()
            else:
                pending_duration -= (interval.start - previous_date).total_seconds()
            if pending_duration >= interval_duration:
                worked_hours += interval_duration
            elif pending_duration > 0:
                worked_hours += pending_duration
            pending_duration -= interval_duration
            if pending_duration <= 0:
                break
            previous_date = interval.end
        # print("Worked-hours: %s" % worked_hours)
        # print('-----------------------------------------------------')
        return worked_hours

    def _find_time_starting(self, pending_duration, c_day, from_date):
        available_duration = self._calculate_available_duration(c_day, from_date)
        if available_duration <= pending_duration:
            return available_duration, 86400 - (from_date - self.new_day).total_seconds()
        else:
            return pending_duration, self._find_time_completion(pending_duration, available_duration, c_day, from_date)

    def _calculate_available_duration(self, c_day, from_date):
        i = -1
        passed_duration = 0
        for t_interval in self.work_intervals[c_day]:
            i += 1
            if t_interval.is_before(from_date):
                passed_duration += t_interval.duration
                continue
            if t_interval.is_after(from_date):
                break
            if t_interval.contains(from_date):
                passed_duration += (from_date - self.work_intervals[c_day][i].start).total_seconds()
                break

        return self.work_rest_count[c_day][0] - passed_duration

    def _find_time_completion(self, pending_duration, total_duration, c_day, from_datetime):
        i = len(self.work_intervals[c_day]) - 1
        while total_duration > pending_duration:
            total_duration -= self.work_intervals[c_day][i].duration
            i -= 1
        if total_duration < pending_duration:
            to_datetime = self.work_intervals[c_day][i + 1].start + timedelta(
                seconds=(pending_duration - total_duration))
            return (to_datetime - from_datetime).total_seconds()
        else:
            return (self.work_intervals[c_day][i].end - from_datetime).total_seconds()


def parse_datetime(time, has_date):
    time_formats = ['%H:%M:%S.%f', '%H:%M', '%I:%M%p', '%H:%M:%S', '%I:%M:%S%p'] if not has_date \
        else ['%Y-%m-%dT%H:%M:%S.%f%z', '%b %d %Y %I:%M%p', '%b %d %Y at %I:%M%p',
              '%B %d, %Y, %H:%M:%S', '%a,%d/%m/%y,%I:%M%p', '%a, %d %B, %Y', '%Y-%m-%dT%H:%M:%SZ']
    try:
        return parser.parse(time)
    except:
        print(time)
        for time_format in time_formats:
            try:
                return datetime.datetime.strptime(time, time_format)
            except ValueError:
                pass
    raise ValueError


def to_seconds(value, from_unit):
    u_from = from_unit.upper()
    return value * convertion_table[u_from] if u_from in convertion_table else value


def seconds_from_day_beginning(from_date):
    return from_date.hour * 3600 + from_date.minute * 60 + from_date.second


def convert_time_unit_from_to(value, from_unit, to_unit):
    u_from = from_unit.upper()
    u_to = to_unit.upper()
    return value * convertion_table[u_from] / convertion_table[
        u_to] if u_from in convertion_table and u_to in convertion_table else value


def update_montly_calendars(calendar_tree, date_time, minutes_in_granule):
    month = date_time.month
    week_day = date_time.weekday()
    minute_granule = (date_time.hour * 60 + date_time.minute) / minutes_in_granule

    if month not in calendar_tree:
        calendar_tree[month] = dict()
    if week_day not in calendar_tree[month]:
        calendar_tree[month][week_day] = dict()
    if minute_granule not in calendar_tree[month][week_day]:
        calendar_tree[month][week_day][minute_granule] = 0
    calendar_tree[month][week_day][minute_granule] += 1


def update_weekly_calendar(r_calendar, date_time, minutes_x_granule=15):
    week_day = int_week_days[date_time.weekday()]
    from_minute = (date_time.minute // minutes_x_granule) * minutes_x_granule
    to_minute = from_minute + minutes_x_granule
    if to_minute >= 60:
        if date_time.hour == 23:
            r_calendar.add_calendar_item(week_day, week_day, "%d:%d:%d" % (date_time.hour, from_minute, 0),
                                         "23:59:59.999")
        else:
            r_calendar.add_calendar_item(week_day, week_day, "%d:%d:%d" % (date_time.hour, from_minute, 0),
                                         "%d:%d:%d" % (date_time.hour + 1, 0, 0))
    else:
        r_calendar.add_calendar_item(week_day, week_day, "%d:%d:%d" % (date_time.hour, from_minute, 0),
                                     "%d:%d:%d" % (date_time.hour, to_minute, 0))


class CalendarNode:
    def __init__(self):
        self.frequency = 0
        self.children_nodes = dict()


class CalendarTree:
    def __init__(self):
        self.value_to_children = dict()

    def insert_date(self, date_prefix: list):
        c_level = self.value_to_children
        for c_value in date_prefix:
            if c_value not in c_level:
                c_level[c_value] = CalendarNode()
            c_level[c_value].frequency += 1
            c_level = c_level[c_value].children_nodes


class CalendarKPIInfoFactory:
    def __init__(self, minutes_x_granule=15):
        self.minutes_x_granule = minutes_x_granule
        self.total_granules = 1440 % self.minutes_x_granule

        self.g_discarded = dict()

        # Fields to calculate Confidence and Support
        self.res_active_weekdays = dict()
        self.res_active_granules_weekdays = dict()
        self.res_enabled_task_granules = None
        self.res_granules_frequency = dict()
        self.active_res_task_weekdays = dict()
        self.active_res_task_weekdays_granules = dict()
        self.res_task_weekdays_granules_freq = dict()
        self.shared_task_granules = dict()
        self.is_joint_resource = dict()
        self.joint_to_task = dict()
        self.observed_weekdays = dict()

        # Fields to compute resource frequencies (needed for participation ratio)
        self.resource_freq = dict()
        self.resource_task_freq = dict()
        self.max_resource_freq = 0
        self.max_resource_task_freq = dict()

        self.task_events_count = dict()
        self.task_events_in_calendar = dict()
        self.total_events_in_log = 0
        self.total_events_in_calendar = 0

        self.res_count_events_in_calendar = dict()
        self.res_count_events_in_log = dict()
        self.active_granules_in_calendar = dict()
        self.active_weekdays_in_calendar = dict()
        self.confidence_numerator_sum = dict()
        self.confidence_denominator_sum = dict()

        self.task_enabled_in_granule = dict()

    def register_resource_timestamp(self, r_name, t_name, date_time, is_joint=False):
        str_date, g_index, weekday = self.split_datetime(date_time)

        if r_name not in self.resource_freq:
            self.resource_freq[r_name] = 0
            self.resource_task_freq[r_name] = dict()
            self.res_active_weekdays[r_name] = dict()
            self.res_active_granules_weekdays[r_name] = dict()
            self.res_granules_frequency[r_name] = dict()
            self.active_res_task_weekdays[r_name] = dict()
            self.active_res_task_weekdays_granules[r_name] = dict()
            self.shared_task_granules[r_name] = dict()
            self.res_task_weekdays_granules_freq[r_name] = dict()

            if is_joint:
                self.joint_to_task[r_name] = t_name

            self.g_discarded[r_name] = list()

            self.res_count_events_in_calendar[r_name] = 0
            self.res_count_events_in_log[r_name] = 0
            self.active_granules_in_calendar[r_name] = set()
            self.active_weekdays_in_calendar[r_name] = set()
            self.confidence_numerator_sum[r_name] = 0
            self.confidence_denominator_sum[r_name] = 0
            self.is_joint_resource[r_name] = is_joint

        if t_name not in self.task_events_count:
            self.task_events_count[t_name] = 0
            self.task_events_in_calendar[t_name] = 0
            self.max_resource_task_freq[t_name] = 0
        if t_name not in self.resource_task_freq[r_name]:
            self.resource_task_freq[r_name][t_name] = 0
            self.active_res_task_weekdays[r_name][t_name] = dict()
            self.active_res_task_weekdays_granules[r_name][t_name] = dict()
            self.res_task_weekdays_granules_freq[r_name][t_name] = dict()
        if weekday not in self.active_res_task_weekdays[r_name][t_name]:
            self.active_res_task_weekdays[r_name][t_name][weekday] = set()
            self.active_res_task_weekdays_granules[r_name][t_name][weekday] = dict()
        if g_index not in self.active_res_task_weekdays_granules[r_name][t_name][weekday]:
            self.active_res_task_weekdays_granules[r_name][t_name][weekday][g_index] = set()
        if g_index not in self.res_task_weekdays_granules_freq[r_name][t_name]:
            self.res_task_weekdays_granules_freq[r_name][t_name][g_index] = dict()
        if weekday not in self.res_task_weekdays_granules_freq[r_name][t_name][g_index]:
            self.res_task_weekdays_granules_freq[r_name][t_name][g_index][weekday] = 0
        if weekday not in self.res_active_weekdays[r_name]:
            self.res_active_weekdays[r_name][weekday] = set()
        if g_index not in self.res_active_granules_weekdays[r_name]:
            self.res_active_granules_weekdays[r_name][g_index] = dict()
            self.res_granules_frequency[r_name][g_index] = dict()
            self.shared_task_granules[r_name][g_index] = dict()
        if weekday not in self.res_active_granules_weekdays[r_name][g_index]:
            self.res_active_granules_weekdays[r_name][g_index][weekday] = set()
            self.res_granules_frequency[r_name][g_index][weekday] = 0
            self.shared_task_granules[r_name][g_index][weekday] = set()
        if weekday not in self.observed_weekdays:
            self.observed_weekdays[weekday] = set()

        # Updating the weekdays and granules the resource was observed working
        self.res_active_weekdays[r_name][weekday].add(str_date)
        self.res_active_granules_weekdays[r_name][g_index][weekday].add(str_date)
        self.res_granules_frequency[r_name][g_index][weekday] += 1

        self.active_res_task_weekdays_granules[r_name][t_name][weekday][g_index].add(str_date)
        self.active_res_task_weekdays[r_name][t_name][weekday].add(str_date)

        self.resource_freq[r_name] += 1
        self.resource_task_freq[r_name][t_name] += 1
        self.res_count_events_in_log[r_name] += 1
        self.shared_task_granules[r_name][g_index][weekday].add(t_name)
        self.res_task_weekdays_granules_freq[r_name][t_name][g_index][weekday] += 1

        if not is_joint:
            self.max_resource_task_freq[t_name] = max(self.max_resource_task_freq[t_name],
                                                      self.resource_task_freq[r_name][t_name])
            self.observed_weekdays[weekday].add(str_date)
            self.max_resource_freq = max(self.max_resource_freq, self.resource_freq[r_name])
            self.task_events_count[t_name] += 1
            self.total_events_in_log += 1

    def register_task_enablement(self, trace_events):
        self.res_enabled_task_granules = None
        for e_info in trace_events:
            t_name = e_info.task_name
            if t_name not in self.task_enabled_in_granule:
                self.task_enabled_in_granule[t_name] = dict()
            current_date = e_info.enabled_at
            str_date, g_index, weekday = self.split_datetime(current_date)
            while current_date < e_info.completed_at:
                if g_index not in self.task_enabled_in_granule[t_name]:
                    self.task_enabled_in_granule[t_name][g_index] = dict()
                if weekday not in self.task_enabled_in_granule[t_name][g_index]:
                    self.task_enabled_in_granule[t_name][g_index][weekday] = set()
                self.task_enabled_in_granule[t_name][g_index][weekday].add(str_date)
                current_date += timedelta(minutes=self.minutes_x_granule)
                if g_index >= self.total_granules - 1:
                    str_date, g_index, weekday = self.split_datetime(current_date)
                else:
                    g_index += 1

    def compute_resource_task_granule_enablement(self):
        self.res_enabled_task_granules = dict()
        for r_name in self.resource_task_freq:
            self.res_enabled_task_granules[r_name] = dict()
            joint_granules = dict()
            for t_name in self.resource_task_freq[r_name]:
                for g_index in self.task_enabled_in_granule[t_name]:
                    if g_index not in self.res_enabled_task_granules[r_name]:
                        joint_granules[g_index] = dict()
                        self.res_enabled_task_granules[r_name][g_index] = dict()
                    for weekday in self.task_enabled_in_granule[t_name][g_index]:
                        if weekday not in self.res_enabled_task_granules[r_name][g_index]:
                            self.res_enabled_task_granules[r_name][g_index][weekday] = set()
                            joint_granules[g_index][weekday] = set()
                        joint_granules[g_index][weekday] |= self.task_enabled_in_granule[t_name][g_index][weekday]
            for g_index in joint_granules:
                for weekday in joint_granules[g_index]:
                    self.res_enabled_task_granules[r_name][g_index][weekday] = len(joint_granules[g_index][weekday])

    def enablement_confidence(self, r_name, weekday, g_index):
        if self.res_enabled_task_granules is None:
            self.compute_resource_task_granule_enablement()
        return len(self.res_active_granules_weekdays[r_name][g_index][weekday]) / \
               self.res_enabled_task_granules[r_name][g_index][weekday]

    def task_cond_confidence(self, r_name, weekday, g_index):
        best_task = None
        max_conf_val = 0
        task_confidences = dict()
        for t_name in self.shared_task_granules[r_name][g_index][weekday]:
            task_confidences[t_name] = len(self.active_res_task_weekdays_granules[r_name][t_name][weekday][g_index]) \
                                       / len(self.active_res_task_weekdays[r_name][t_name][weekday])
            if max_conf_val < task_confidences[t_name]:
                best_task = t_name
                max_conf_val = task_confidences[t_name]
        return best_task, task_confidences

    def resource_participation_ratio(self, r_name):
        total_res = 0
        total_max = 0
        for t_name in self.resource_task_freq[r_name]:
            total_res += self.resource_task_freq[r_name][t_name]
            total_max += self.max_resource_task_freq[t_name]
        return total_res / total_max if total_max > 0 else 0

    def resource_task_participation_ratio(self, r_name, t_name):
        if self.max_resource_task_freq[t_name] > 0:
            return self.resource_task_freq[r_name][t_name] / self.max_resource_task_freq[t_name]
        return 0

    # From all the WeekDays the resource was active, in which ration they were in the given granule
    def confidence(self, r_name, weekday, g_index):
        return len(self.res_active_granules_weekdays[r_name][g_index][weekday]) / len(
            self.res_active_weekdays[r_name][weekday])

    def support(self, r_name, weekday, g_index):
        return len(self.res_active_granules_weekdays[r_name][g_index][weekday]) / len(self.observed_weekdays[weekday])

    def weekday_support(self, r_name, weekday):
        return len(self.res_active_weekdays[r_name][weekday]) / len(self.observed_weekdays[weekday])

    def task_coverage(self, t_name):
        return self.task_events_in_calendar[t_name] / self.task_events_count[t_name]

    def can_improve_support(self, r_name, weekday, g_index, min_confidence):
        best_task, confidence_values = self.task_cond_confidence(r_name, weekday, g_index)
        if r_name == 'arrival':
            return best_task

        # new_granules = 0
        # new_weekdays = 0

        # for str_date in self.active_res_task_weekdays_granules[r_name][best_task][weekday][g_index]:
        #     if str_date not in self.active_granules_in_calendar[r_name]:
        #         new_granules += 1
        # for str_date in self.active_res_task_weekdays[r_name][best_task][weekday]:
        #     if str_date not in self.active_weekdays_in_calendar[r_name]:
        #         new_weekdays += 1

        # if min_confidence <= (len(self.active_granules_in_calendar[r_name]) + new_granules) \
        #         / (len(self.active_weekdays_in_calendar[r_name]) + new_weekdays):

        # new_granules = len(self.active_res_task_weekdays_granules[r_name][best_task][weekday][g_index])
        # new_weekdays = len(self.active_res_task_weekdays[r_name][best_task][weekday])
        # if min_confidence <= (self.confidence_numerator_sum[r_name] + new_granules) \
        #         / (self.confidence_denominator_sum[r_name] + new_weekdays):
        return best_task

    def reset_calendar_info(self):
        self.total_events_in_calendar = 0
        for t_name in self.task_events_count:
            self.task_events_in_calendar[t_name] = 0
        for r_name in self.shared_task_granules:
            self.res_count_events_in_calendar[r_name] = 0
            self.active_granules_in_calendar[r_name] = set()
            self.active_weekdays_in_calendar[r_name] = set()
            self.g_discarded[r_name] = list()
            self.confidence_numerator_sum[r_name] = 0
            self.confidence_denominator_sum[r_name] = 0

    def check_accepted_granule(self, r_name, weekday, g_index, best_task):
        self.res_count_events_in_calendar[r_name] += self.res_granules_frequency[r_name][g_index][weekday]
        self.total_events_in_calendar += self.res_granules_frequency[r_name][g_index][weekday]
        self.confidence_numerator_sum[r_name] += len(
            self.active_res_task_weekdays_granules[r_name][best_task][weekday][g_index])
        self.confidence_denominator_sum[r_name] += len(self.active_res_task_weekdays[r_name][best_task][weekday])
        self.active_granules_in_calendar[r_name] |= \
            self.active_res_task_weekdays_granules[r_name][best_task][weekday][g_index]
        self.active_weekdays_in_calendar[r_name] |= self.active_res_task_weekdays[r_name][best_task][weekday]
        for t_name in self.shared_task_granules[r_name][g_index][weekday]:
            self.task_events_in_calendar[t_name] += self.res_task_weekdays_granules_freq[r_name][t_name][g_index][
                weekday]

    def check_discarded_granule(self, r_name, weekday, g_index):
        if r_name not in self.g_discarded:
            self.g_discarded[r_name] = list()
        self.g_discarded[r_name].append(GranuleInfo(weekday, g_index))

    def update_discarded_granules_list(self, r_name, accepted_indexes):
        if len(accepted_indexes) == 0:
            return
        new_discarded = list()
        c_j = 0
        for i in range(0, len(self.g_discarded[r_name])):
            if c_j < len(accepted_indexes) and i == accepted_indexes[c_j]:
                c_j += 1
            else:
                new_discarded.append(self.g_discarded[r_name][i])
        self.g_discarded[r_name] = new_discarded

    def split_datetime(self, date_time):
        str_date = str(date_time.date())
        in_minutes = date_time.hour * 60 + date_time.minute

        g_index = in_minutes // self.minutes_x_granule
        week_day = date_time.weekday()
        return str_date, g_index, week_day

    def compute_confidence_support(self, r_name):
        if r_name not in self.active_weekdays_in_calendar \
                or len(self.active_weekdays_in_calendar[r_name]) == 0 or self.res_count_events_in_log[r_name] == 0:
            return 0, 0
        return self.confidence_numerator_sum[r_name] / self.confidence_denominator_sum[r_name], \
               self.res_count_events_in_calendar[r_name] / self.res_count_events_in_log[r_name]
        # return len(self.active_granules_in_calendar[r_name]) / len(self.active_weekdays_in_calendar[r_name]), \
        #        self.res_count_events_in_calendar[r_name] / self.res_count_events_in_log[r_name]


class GranuleInfo:
    def __init__(self, week_day, g_index):
        self.week_day = week_day
        self.g_index = g_index


class CalendarFactory:
    def __init__(self, minutes_x_granule=15):
        if 1440 % minutes_x_granule != 0:
            raise ValueError(
                "The number of minutes per granule must be a divisor of the total minutes in one day (1440).")

        self.kpi_calendar = CalendarKPIInfoFactory(minutes_x_granule)

        self.minutes_x_granule = minutes_x_granule

        self.from_datetime = datetime.datetime(9999, 12, 31, tzinfo=pytz.UTC)
        self.to_datetime = datetime.datetime(1, 1, 1, tzinfo=pytz.UTC)

    def _init_calendar_tree(self):
        granules_count = 1440 // self.minutes_x_granule
        return [[0] * granules_count for _i in range(0, 7)]

    def register_task_enablement(self, trace_events):
        self.kpi_calendar.register_task_enablement(trace_events)

    def check_date_time(self, r_name, t_name, date_time, is_joint=False):
        self.kpi_calendar.register_resource_timestamp(r_name, t_name, date_time, is_joint)

        self.from_datetime = min(self.from_datetime, date_time)
        self.to_datetime = max(self.to_datetime, date_time)

    def build_weekly_calendars(self, min_confidence, desired_support, min_participation):
        r_calendars = dict()
        self.kpi_calendar.reset_calendar_info()
        for r_name in self.kpi_calendar.shared_task_granules:
            if self.kpi_calendar.resource_participation_ratio(r_name) >= min_participation:
                r_calendars[r_name] = self.build_resource_calendar(r_name, min_confidence, desired_support)
            else:
                r_calendars[r_name] = None
        return r_calendars

    def build_resource_calendar(self, r_name, min_confidence, desired_support):
        r_calendar = RCalendar("%s_Schedule" % r_name)
        kpi_c = self.kpi_calendar
        to_print = dict()

        count = 0
        for g_index in kpi_c.shared_task_granules[r_name]:
            for weekday in kpi_c.shared_task_granules[r_name][g_index]:
                best_task, conf_values = kpi_c.task_cond_confidence(r_name, weekday, g_index)
                if min_confidence <= conf_values[best_task]:
                    kpi_c.check_accepted_granule(r_name, weekday, g_index, best_task)
                    self._add_calendar_item(weekday, g_index, r_calendar)
                else:
                    count += 1
                    if weekday not in to_print:
                        to_print[weekday] = list()
                    to_print[weekday].append(g_index)
                    kpi_c.g_discarded[r_name].append(GranuleInfo(weekday, g_index))

        confidence, support = kpi_c.compute_confidence_support(r_name)

        if confidence > 0 and support < desired_support:
            kpi_c.g_discarded[r_name].sort(key=lambda x: kpi_c.res_granules_frequency[r_name][x.g_index][x.week_day],
                                           reverse=True)
            accepted_indexes = list()
            i = 0

            for g_info in kpi_c.g_discarded[r_name]:
                best_task = kpi_c.can_improve_support(r_name, g_info.week_day, g_info.g_index, min_confidence)
                if best_task is not None:
                    self._add_calendar_item(g_info.week_day, g_info.g_index, r_calendar)
                    kpi_c.check_accepted_granule(r_name, g_info.week_day, g_info.g_index, best_task)
                    accepted_indexes.append(i)
                _, support = kpi_c.compute_confidence_support(r_name)
                if support >= desired_support:
                    break
                i += 1
            kpi_c.update_discarded_granules_list(r_name, accepted_indexes)
        return r_calendar

    def task_coverage(self, t_name):
        return self.kpi_calendar.task_coverage(t_name)

    def build_unrestricted_resource_calendar(self, r_name, t_name):
        r_calendar = RCalendar("%s_Schedule" % r_name)

        r_kpi = self.kpi_calendar
        for g_index in r_kpi.res_active_granules_weekdays[r_name]:
            for week_day in r_kpi.res_active_granules_weekdays[r_name][g_index]:
                r_kpi.check_accepted_granule(r_name, week_day, g_index, t_name)
                self._add_calendar_item(week_day, g_index, r_calendar)
        return r_calendar

    def _add_calendar_item(self, week_day, g_index, r_calendar):
        str_wday = int_week_days[week_day]
        hour = (g_index * self.minutes_x_granule) // 60
        from_min = (g_index * self.minutes_x_granule) % 60
        to_min = from_min + self.minutes_x_granule
        if to_min >= 60:
            if hour == 23:
                r_calendar.add_calendar_item(str_wday, str_wday, "%d:%d:%d" % (hour, from_min, 0), "23:59:59.999")
            else:
                r_calendar.add_calendar_item(str_wday, str_wday, "%d:%d:%d" % (hour, from_min, 0),
                                             "%d:%d:%d" % (hour + 1, 0, 0))
        else:
            r_calendar.add_calendar_item(str_wday, str_wday, "%d:%d:%d" % (hour, from_min, 0),
                                         "%d:%d:%d" % (hour, to_min, 0))


def build_full_time_calendar(calendar_id):
    r_calendar = RCalendar(calendar_id)
    for i in range(0, 7):
        str_weekday = int_week_days[i]
        r_calendar.add_calendar_item(str_weekday, str_weekday, "00:00:00.000", "23:59:59.999")
    return r_calendar


def update_calendar_from_log(r_calendar, date_time, is_start, month_freq, min_eps=15):
    if date_time.month not in month_freq:
        month_freq[date_time.month] = [0, set()]
    month_freq[date_time.month][0] += 1
    month_freq[date_time.month][1].add(date_time.year)

    from_date = date_time
    to_date = date_time
    if is_start:
        to_date = date_time + timedelta(minutes=min_eps)
    else:
        from_date = date_time - timedelta(minutes=min_eps)

    from_day = int_week_days[from_date.weekday()]
    to_day = int_week_days[to_date.weekday()]

    if from_day != to_day:
        r_calendar.add_calendar_item(from_day, from_day,
                                     "%d:%d:%d" % (from_date.hour, from_date.minute, from_date.second), "23:59:59.999")
        if to_date.hour != 0 or to_date.minute != 0 or to_date.second != 0:
            r_calendar.add_calendar_item(to_day, to_day, "00:00:00",
                                         "%d:%d:%d" % (to_date.hour, to_date.minute, to_date.second))
    else:
        r_calendar.add_calendar_item(from_day, to_day,
                                     "%d:%d:%d" % (from_date.hour, from_date.minute, from_date.second),
                                     "%d:%d:%d" % (to_date.hour, to_date.minute, to_date.second))


def _worked_days_count(from_date, to_date):
    total_days = (to_date.date() - from_date.date()).days + 1
    total_days_count = dict()
    for i in range(0, 7):
        total_days_count[i] = int(total_days / 7)
    total_days %= 7
    for i in range(from_date.weekday(), from_date.weekday() + total_days):
        total_days_count[i % 7] += 1
    return total_days_count


def compute_total_weekdays(from_datetime, to_datetime):
    total_days = (to_datetime.date() - from_datetime.date()).days + 2
    same_days = total_days // 7
    rem_days = total_days % 7
    to_weekday = to_datetime.weekday()
    total_weekdays = {key: same_days for key in range(0, 7)}
    while rem_days > 0:
        total_weekdays[to_weekday] += 1
        to_weekday = to_weekday - 1 if to_weekday > 0 else 6
        rem_days -= 1
    return total_weekdays
