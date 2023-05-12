from prosimos.resource_calendar import parse_datetime
from prosimos.weekday_helper import get_nearest_abs_day, get_nearest_past_day
import pytest

data_abs_day = {
    # provided datetime's week day > rule week day (Friday < Sunday)
    ("2022-09-30 13:22:30.035185+03:00", "SUNDAY", "2022-10-02 13:22:30"),
    # provided datetime's week day > rule week day (Friday > Tuesday)
    ("2022-09-30 13:22:30.035185+03:00", "TUESDAY", "2022-10-04 13:22:30"),
    # provided datetime's week day == rule week day (Friday = Friday)
    ("2022-09-30 13:22:30.035185+03:00", "FRIDAY", "2022-10-07 13:22:30"),
}

@pytest.mark.parametrize(
    "current_datetime, rule_week_day, expected_datetime",
    data_abs_day,
)
def test_get_nearest_abs_day_correct(current_datetime, rule_week_day, expected_datetime):
    start_string = current_datetime
    start_date = parse_datetime(start_string, True)

    nearest_day = get_nearest_abs_day(rule_week_day, start_date)

    nearest_day_str = nearest_day.strftime("%Y-%m-%d %H:%M:%S")

    assert nearest_day_str == expected_datetime, \
        f"Expected: {expected_datetime}, but was {nearest_day_str}"


data_day = {
    # provided datetime's week day > rule week day (Friday < Sunday)
    ("2022-09-30 13:22:30.035185+03:00", "SUNDAY", "2022-09-25 13:22:30"),
    # provided datetime's week day > rule week day (Friday > Tuesday)
    ("2022-09-30 13:22:30.035185+03:00", "TUESDAY", "2022-09-27 13:22:30"),
    # provided datetime's week day == rule week day (Friday = Friday)
    ("2022-09-30 13:22:30.035185+03:00", "FRIDAY", "2022-09-30 13:22:30"),
}

@pytest.mark.parametrize(
    "current_datetime, rule_week_day, expected_datetime",
    data_day,
)
def test_get_nearest_past_day_correct(current_datetime, rule_week_day, expected_datetime):
    start_string = current_datetime
    start_date = parse_datetime(start_string, True)

    nearest_day = get_nearest_past_day(rule_week_day, start_date)

    nearest_day_str = nearest_day.strftime("%Y-%m-%d %H:%M:%S")

    assert nearest_day_str == expected_datetime, \
        f"Expected: {expected_datetime}, but was {nearest_day_str}"