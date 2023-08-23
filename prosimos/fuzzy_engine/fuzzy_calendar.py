from datetime import datetime, timedelta
from enum import Enum
from numpy import random

from pix_framework.discovery.resource_calendar_and_performance.crisp.resource_calendar import IntervalPoint, Interval


class FSpan(Enum):
    AVAILABILITY = 1
    WORKLOAD = 2
    TIME = 3


class WeeklyFuzzyCalendar:
    def __init__(self, granule_size):
        self.granule_size = granule_size
        self.i_count = 1440 // granule_size

        self.probability_intervals = dict()
        self.next_shift = dict()
        for i in range(0, 7):
            self.probability_intervals[i] = [0.0] * self.i_count
            self.next_shift[i] = [0] * self.i_count

    def interval_index(self, current_date: datetime):
        return (current_date.hour * 60 + current_date.minute) // self.granule_size

    def duration_from_interval_start(self, current_date: datetime):
        current_minute = current_date.hour * 60 + current_date.minute
        start_minute = self.interval_index(current_date) * self.granule_size
        return (current_minute - start_minute) * 60 + current_date.second

    def duration_to_interval_end(self, current_date: datetime):
        return self.granule_size * 60 - self.duration_from_interval_start(current_date)

    def add_weekday_intervals(self, weekday: int, from_time: datetime, to_time: datetime, probability: float):
        from_i = self.interval_index(from_time)
        to_i = self.interval_index(to_time) - 1
        if to_i < 0:
            to_i = len(self.probability_intervals[weekday]) - 1
        while from_i <= to_i:
            self.probability_intervals[weekday][from_i] = probability
            from_i += 1

    def index_consecutive_boundaries(self):
        for weekday in self.probability_intervals:
            i = 0
            probs = self.probability_intervals[weekday]
            while i < self.i_count - 1:
                j = i + 1
                while j < self.i_count:
                    if probs[i] != probs[j] or probs[i] not in [0.0, 1.0]:
                        break
                    j += 1
                for k in range(i, j):
                    self.next_shift[weekday][k] = j - 1
                i = j


class FuzzyModel:
    def __init__(self, calendar_id):
        self.calendar_id = calendar_id
        self.availability_calendar = None
        self.workload_ratio = None
        self.allocation_criteria = {FSpan.AVAILABILITY: True, FSpan.WORKLOAD: True, FSpan.TIME: True}
        self.default_calendar = FSpan.AVAILABILITY

    def update_model(self, key_str, value: WeeklyFuzzyCalendar):
        if key_str == 'availability_probabilities' or key_str == 'time_periods':
            self.availability_calendar = value
        elif key_str == "workload_ratio":
            self.workload_ratio = value

    def update_allocation_criteria(self, criteria_type: FSpan, new_value: bool):
        self.allocation_criteria[criteria_type] = new_value

    def is_available(self, week_day, i):
        abs_prob = self.availability_calendar.probability_intervals[week_day][i]
        rel_prob = self.workload_ratio.probability_intervals[week_day][i]
        a_p = random.choice([True, False], 1, p=[abs_prob, 1 - abs_prob])[0]
        r_p = random.choice([True, False], 1, p=[rel_prob, 1 - rel_prob])[0]

        if a_p:
            self.default_calendar = FSpan.AVAILABILITY
        elif r_p:
            self.default_calendar = FSpan.WORKLOAD
        return a_p or r_p

        # return random.choice([True, False], 1, p=[abs_prob, 1 - abs_prob])[0] or \
        #        random.choice([True, False], 1, p=[rel_prob, 1 - rel_prob])[0]

    def _extract_date_info(self, current_date: datetime):
        avail = self.availability_calendar if self.default_calendar is FSpan.AVAILABILITY else self.workload_ratio
        return avail, avail.granule_size * 60, current_date.date().weekday(), avail.interval_index(current_date)

    def next_available_time(self, current_date: datetime):
        avail_info, size_seconds, week_day, from_i = self._extract_date_info(current_date)

        if self.is_available(week_day, from_i):
            return 0

        duration = avail_info.duration_to_interval_end(current_date)
        i = from_i + 1
        while True:
            if i >= avail_info.i_count:
                week_day = (week_day + 1) % 7
                i = 0
            if avail_info.probability_intervals[week_day][i] == 1.0:
                return duration
            else:
                if self.is_available(week_day, i):
                    return duration
                duration += size_seconds
                i += 1

    def find_idle_time(self, current_date: datetime, ideal_duration, worked_intervals):
        it_date = current_date
        if ideal_duration == 0:
            return 0

        avail_info, size_seconds, week_day, i = self._extract_date_info(current_date)

        real_duration = min(avail_info.duration_to_interval_end(current_date), ideal_duration)
        if self.is_available(week_day, i):
            to_date = it_date + timedelta(seconds=real_duration)
            worked_intervals.append(Interval(it_date, to_date))
            it_date = to_date
            ideal_duration -= real_duration
        i += 1
        while ideal_duration > 0:
            if i >= avail_info.i_count:
                week_day = (week_day + 1) % 7
                i = 0
            if self.is_available(week_day, i):
                i_duration = min(size_seconds, ideal_duration)
                to_date = it_date + timedelta(seconds=i_duration)
                worked_intervals.append(Interval(it_date, to_date))
                it_date += timedelta(seconds=i_duration)
                real_duration += i_duration
                ideal_duration -= size_seconds
            else:
                real_duration += size_seconds
                it_date += timedelta(seconds=size_seconds)
            i += 1
        return real_duration

    def is_working_datetime(self, current_date: datetime):
        avail_info, i_size_seconds, week_day, i = self._extract_date_info(current_date)
        if self.is_available(week_day, i):
            return True, IntervalPoint(current_date, i, week_day, avail_info.duration_from_interval_start(current_date),
                                       avail_info.duration_to_interval_end(current_date))
        return False, None

    def estimate_available_time(self, from_date: datetime, to_date: datetime):
        fuzzy_c, size_seconds, week_day, i = self._extract_date_info(from_date)
        last_duration = fuzzy_c.duration_from_interval_start(to_date)
        first_duration = fuzzy_c.duration_to_interval_end(from_date)

        duration = fuzzy_c.probability_intervals[week_day][i] * first_duration
        duration += fuzzy_c.probability_intervals[to_date.weekday()][fuzzy_c.interval_index(to_date)] * last_duration

        to_date = to_date - timedelta(seconds=last_duration)
        c_date = from_date + timedelta(seconds=first_duration)
        while c_date < to_date:
            duration += fuzzy_c.probability_intervals[c_date.weekday()][fuzzy_c.interval_index(c_date)] * size_seconds
            c_date += timedelta(seconds=size_seconds)
        return duration
