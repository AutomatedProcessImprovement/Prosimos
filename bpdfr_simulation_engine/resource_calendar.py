import datetime
from datetime import timedelta

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
            if self.c_day < 0 or self.c_day >= len(calendar_info.work_intervals):
                print('hola')
            if self.c_index < 0 or self.c_index >= len(calendar_info.work_intervals[self.c_day]):
                print('hola')

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
                if next_i.end < to_merge.end:
                    merged_duration -= next_i.duration
                elif next_i.start < to_merge.end:
                    merged_duration -= (to_merge.end - next_i.start).total_seconds()
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


class CalendarFactory:
    def __init__(self, minutes_x_granule=15):
        if 1440 % minutes_x_granule != 0:
            raise ValueError(
                "The number of minutes per granule must be a divisor of the total minutes in one day (1440).")
        self.minutes_x_granule = minutes_x_granule
        self.resource_calendars = dict()
        self.log_granules = self._init_calendar_tree()

        self.resource_calendar_tree = dict()
        self.r_weekdays_set = dict()
        self.r_granules_set = dict()
        self.total_weekdays = dict()

        self.from_datetime = datetime.datetime(9999, 12, 31, tzinfo=pytz.UTC)
        self.to_datetime = datetime.datetime(1, 1, 1, tzinfo=pytz.UTC)

    def _init_calendar_tree(self):
        granules_count = 1440 // self.minutes_x_granule
        return [[0] * granules_count for _i in range(0, 7)]

    def check_date_time(self, r_name, date_time):
        str_date = str(date_time.date())
        g_index = 0 if (date_time.hour + date_time.minute) % self.minutes_x_granule == 0 else 1
        g_index += (date_time.hour + date_time.minute) // self.minutes_x_granule
        week_day = date_time.weekday()
        if r_name not in self.resource_calendar_tree:
            self.r_weekdays_set[r_name] = dict()
            self.r_granules_set[r_name] = dict()
            self.resource_calendar_tree[r_name] = CalendarTree()
        if week_day not in self.r_weekdays_set[r_name]:
            self.r_weekdays_set[r_name][week_day] = set()
        if g_index not in self.r_granules_set[r_name]:
            self.r_granules_set[r_name][g_index] = set()
        self.resource_calendar_tree[r_name].insert_date([date_time.year, date_time.month, date_time.weekday(), g_index])
        self.r_weekdays_set[r_name][week_day].add(str_date)
        self.r_granules_set[r_name][g_index].add(str_date)
        self.from_datetime = min(self.from_datetime, date_time)
        self.to_datetime = max(self.to_datetime, date_time)

    def compute_total_weekdays(self):
        total_days = (self.to_datetime.date() - self.from_datetime.date()).days + 2
        same_days = total_days // 7
        rem_days = total_days % 7
        to_weekday = self.to_datetime.weekday()
        self.total_weekdays = {key: same_days for key in range(0, 7)}
        while rem_days > 0:
            self.total_weekdays[to_weekday] += 1
            to_weekday = to_weekday - 1 if to_weekday > 0 else 6
            rem_days -= 1

    def build_weekly_calendars(self, min_confidence, min_support):
        r_count = dict()
        self.compute_total_weekdays()
        resource_calendars = dict()
        for r_name in self.resource_calendar_tree:
            r_count[r_name] = [0, 0]
            resource_calendars[r_name] = self.build_resource_calendar(r_count, r_name, min_confidence, min_support)
            # print('R Name : %s' % r_name)
            # print("Added  : %d" % r_count[r_name][0])
            # print("Removed: %d" % r_count[r_name][1])
            # print('------------------------------------------')

        return resource_calendars

    def build_resource_calendar(self, r_count, r_name, min_confidence, min_support):
        r_calendar = RCalendar("%s_Schedule" % r_name)
        calendar_tree = self.resource_calendar_tree[r_name].value_to_children
        for year in calendar_tree:
            for month in calendar_tree[year].children_nodes:
                week_days = calendar_tree[year].children_nodes
                for weekday in week_days[month].children_nodes:
                    granules = week_days[month].children_nodes
                    for granule_index in granules[weekday].children_nodes:

                        g_supp = len(self.r_granules_set[r_name][granule_index]) / len(self.r_weekdays_set[r_name][weekday])
                        g_conf = len(self.r_granules_set[r_name][granule_index]) / self.total_weekdays[weekday]

                        if g_conf >= min_confidence and g_supp >= min_support:
                            r_count[r_name][0] += 1
                            str_wday = int_week_days[weekday]
                            hour = (granule_index * self.minutes_x_granule) // 60
                            from_min = (granule_index * self.minutes_x_granule) % 60
                            to_min = from_min + self.minutes_x_granule
                            if to_min >= 60:
                                if hour == 23:
                                    r_calendar.add_calendar_item(str_wday, str_wday, "%d:%d:%d" % (hour, from_min, 0),
                                                                 "23:59:59.999")
                                else:
                                    r_calendar.add_calendar_item(str_wday, str_wday, "%d:%d:%d" % (hour, from_min, 0),
                                                                 "%d:%d:%d" % (hour + 1, 0, 0))
                            else:
                                r_calendar.add_calendar_item(str_wday, str_wday, "%d:%d:%d" % (hour, from_min, 0),
                                                             "%d:%d:%d" % (hour, to_min, 0))
                        else:
                            r_count[r_name][1] += 1
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
