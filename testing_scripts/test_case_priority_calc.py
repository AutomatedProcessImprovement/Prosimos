import datetime
import sys

import pytest
import pytz
from prosimos.prioritisation import CasePrioritisation
from prosimos.weekday_helper import CustomDatetimeAndSeconds

defaultCustomDatetimeAndSeconds = CustomDatetimeAndSeconds(
    0, datetime.datetime.now(pytz.utc)
)


class TestingCasePrioritisation(CasePrioritisation):
    def __init__(self, all_case_priorities):
        self.all_case_priorities = all_case_priorities


SAME_PRIORITY = {
    "0": sys.maxsize,
    "1": sys.maxsize,
    "2": sys.maxsize,
}

ASC_PRIORITY = {
    "0": 1,
    "1": 2,
    "2": sys.maxsize,
}

DESC_PRIORITY = {
    "0": sys.maxsize,
    "1": 2,
    "2": 1,
}

order_case_ids_test_case = [
    ([], ["2", "1"], ["2", "1"]),
    (
        SAME_PRIORITY,
        ["2", "1"],
        ["2", "1"],
    ),
    (
        ASC_PRIORITY,
        ["2", "1"],
        ["1", "2"],
    ),
    (
        DESC_PRIORITY,
        ["2", "1"],
        ["2", "1"],
    ),
]


@pytest.mark.parametrize(
    "all_case_priorities, case_ids, expected_result",
    order_case_ids_test_case,
)
def test__multiple_case_ids__correct_priority(
    all_case_priorities, case_ids, expected_result
):
    testing_obj = TestingCasePrioritisation(all_case_priorities)
    input = [(case_id, defaultCustomDatetimeAndSeconds) for case_id in case_ids]
    actual_result = testing_obj.get_ordered_case_ids_by_priority(input)

    actual_result_ordered_case_ids = [case_id for (case_id, datetime) in actual_result]

    assert actual_result_ordered_case_ids == expected_result
