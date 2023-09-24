from datetime import timedelta

from pix_framework.discovery.resource_calendar_and_performance.crisp.resource_calendar import str_week_days


class CustomDatetimeAndSeconds:
    def __init__(self, seconds_from_start, datetime):
        self.seconds_from_start = seconds_from_start
        self.datetime = datetime


def get_nearest_abs_day(weekday, from_datetime):
    """
    Finds nearest day only in the future calendar
    Example:    from_datetime.weekday = Tuesday,
                weekday = Monday,
                new_datetime.weekday = Monday of the next week (not the previous)
    """

    completed_datetime_weekday = from_datetime.weekday()
    timer_weekday = str_week_days.get(weekday)
    if timer_weekday > completed_datetime_weekday:
        add_days = timer_weekday - completed_datetime_weekday
    else:
        diff_days = completed_datetime_weekday - timer_weekday
        add_days = 7 - diff_days

    new_datetime = from_datetime + timedelta(days=add_days)
    return new_datetime


def get_nearest_past_day(weekday, from_datetime):
    """
    Finds nearest day either in the past
    Example:    from_datetime.weekday = Tuesday,
                weekday = Monday,
                new_datetime.weekday = Monday (the day before) of this week
    """

    completed_datetime_weekday = from_datetime.weekday()
    timer_weekday = str_week_days.get(weekday)
    if timer_weekday > completed_datetime_weekday:
        diff_days = timer_weekday - completed_datetime_weekday
        sub_days = 7 - diff_days
        new_datetime = from_datetime - timedelta(days=sub_days)
    else:
        diff_days = completed_datetime_weekday - timer_weekday
        new_datetime = from_datetime - timedelta(days=diff_days)

    return new_datetime
