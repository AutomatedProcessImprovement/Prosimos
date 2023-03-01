from datetime import datetime, time

import pandas as pd
import pytest

from test_simulation import assets_path
from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation
from testing_scripts.test_simulation import _setup_and_write_sim_scenario

data_start_times = {
    # start datetime provided by the user is not satisfying the arrival time calendar
    # expected: the simulation will start at the earliest possible datetime which satisfies arrival time calendar
    ("2023-02-28T13:22:30.035185+03:00", False),
    # provided start datetime satisfy the arrival calendar (it's Wednesday)
    # expected: the first activity of the first case starts at this provided datetime
    ("2023-02-22T10:22:30.035185+03:00", True),
}


@pytest.mark.parametrize(
    "start_string, is_equal_to_first_case_start_time", data_start_times
)
def test__start_datetime__in_range_of_arrival_time_calendar(
    assets_path,
    start_string,
    is_equal_to_first_case_start_time,
):
    """
    Input:      Resource works 24/7.
                Arrival time calendar includes first half (9:00 am - 2:00 pm) on Wednesday.

    Expected:   Verifying that first activity of all cases starts only in the allowed period following the arrival time calendar.
                In case start time provided in the simulation scenario is IN the range of arrival time calendar:
                    - verify that this provided start time is the start time of the first activity of the firstly simulated case.
                In case start time provided in the simulation scenario is OUT OF the range of arrival time calendar:
                    - verify that the first activity of the firstly simulated case is not equal to the provided start datetime
    """

    # ====== ARRANGE ======

    model_path = assets_path / "timer_with_task.bpmn"

    json_path = assets_path / "timer_with_task.json"
    new_arrival_time_calendar = [
        {
            "from": "WEDNESDAY",
            "to": "WEDNESDAY",
            "beginTime": "09:00:00",
            "endTime": "14:00:00",
        }
    ]
    new_resource_calendars = [
        {
            "id": "Default resource-000001timetable",
            "name": "Default resource-000001timetable",
            "time_periods": [
                {
                    "from": "MONDAY",
                    "to": "SUNDAY",
                    "beginTime": "00:00:00",
                    "endTime": "23:59:59",
                }
            ],
        }
    ]
    _setup_and_write_sim_scenario(
        json_path,
        arrival_time_calendar=new_arrival_time_calendar,
        resource_calendars=new_resource_calendars,
    )

    sim_stats = assets_path / "timer_with_task_stats.csv"
    sim_logs = assets_path / "timer_with_task_logs.csv"

    # ====== ACT ======
    _, _ = run_diff_res_simulation(
        start_string, 5, model_path, json_path, sim_stats, sim_logs, False
    )

    # ====== ASSERT ======
    df = pd.read_csv(sim_logs)

    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    first_activity_in_case = df[df["activity"] == "Task 1"]["start_time"]

    # range for the time arrival of cases
    low_time_boundary = time(9, 0, 0)
    high_time_boundray = time(14, 0, 0)

    # verify that all start times of the first activity in a case:
    # 1) is on the same day (WEDNESDAY)
    # this is according to the provided arrival_time_calendar
    # 2) lies within a range of [9, 14] time
    verify_complying_with_arrival_calendar(
        first_activity_in_case, 2, low_time_boundary, high_time_boundray
    )

    # verify the start time of the first activity in the first simulated case
    # equal / not equal to the provided start time of the simulation scenario
    sim_start_datetime = datetime.strptime(start_string, "%Y-%m-%dT%H:%M:%S.%f%z")
    first_case_start_time = first_activity_in_case[0]

    if is_equal_to_first_case_start_time:
        assert (
            sim_start_datetime == first_case_start_time
        ), f"Expected start time of the first case ({first_case_start_time}) to be equal to start of the whole simulation ({sim_start_datetime})"
    else:
        assert (
            sim_start_datetime != first_case_start_time
        ), f"Expected start time of the first case ({first_case_start_time}) NOT to be equal to start of the whole simulation ({sim_start_datetime})"


def verify_complying_with_arrival_calendar(
    datetime_list, expected_weekday, expected_low_boundary, expected_high_boundary
):
    "Verify that list of datetimes is on the same weekday and lies within a range of [low_boundary, high_boundary]"

    for curr_datetime in datetime_list:
        is_daytime_inside_arr_time_calendar = (
            curr_datetime.weekday() == expected_weekday
        )

        is_time_inside_arr_time_calendar = (
            expected_low_boundary <= curr_datetime.time() <= expected_high_boundary
        )

        assert (
            is_daytime_inside_arr_time_calendar
            and is_time_inside_arr_time_calendar == True
        ), f"First activity of the case does not lay within a range of arrival time calendar"
