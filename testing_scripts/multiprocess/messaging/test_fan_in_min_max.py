"""
Fan-in min and max (docs/messaging.md, "Fan-in"): a group is sent at once when it has max parts, at its deadline
when it has at least min, and else the moment its min-th part comes in after the deadline. As a truck that waits
for 3 parcels, or leaves after an hour once it has 2.
"""
from datetime import timedelta

import pytest

from testing_scripts.multiprocess.messaging.test_fan_in import _order_picked, _picked, _run
from testing_scripts.multiprocess.messaging.test_fan_in_deadline import (MICROSECOND, at, expected_messages,
                                                                          released)

TRUCK = {"min": 2, "max": 3}
WITHIN_AN_HOUR = {"hours": 1}


def _sent(report):
    return [(time, attributes["item_ids"], attributes["picked"]) for time, attributes in _order_picked(report)]


def test_a_group_with_max_parts_is_sent_at_once(tmp_path):
    report = released(tmp_path, TRUCK, WITHIN_AN_HOUR, ["09:40", "09:50", "10:00"])

    assert _sent(report) == [(at("10:00"), ["item-0", "item-1", "item-2"], 3)]
    assert report.engine_warnings == []


def test_a_group_with_min_parts_is_sent_at_its_deadline(tmp_path):
    report = released(tmp_path, TRUCK, WITHIN_AN_HOUR, ["09:40", "10:00", "11:00"])

    # the receiver tells a short group from a full one by counting its parts
    assert _sent(report) == [(at("10:40") + MICROSECOND, ["item-0", "item-1"], 2)]
    # item-2 opens a new group, which never gets its min
    assert [warning for _, warning in report.engine_warnings] == [
        "fan-in OrderPicked for order_id order-0: 1 of 3 parts (min 2), never sent"]


def test_a_group_short_of_min_at_its_deadline_is_sent_at_its_min_th_part(tmp_path):
    report = released(tmp_path, TRUCK, WITHIN_AN_HOUR, ["09:40", "11:00", "11:30"])

    # at 10:40 the group has 1 part, so it stays open until item-1 comes in at 11:00
    assert _sent(report) == [(at("11:00"), ["item-0", "item-1"], 2)]
    assert [warning for _, warning in report.engine_warnings] == [
        "fan-in OrderPicked for order_id order-0: 1 of 3 parts (min 2), never sent"]


def test_a_max_read_from_an_attribute(tmp_path):
    report = released(tmp_path, {"min": 1, "max": {"object_attribute": "items"}}, WITHIN_AN_HOUR,
                      ["09:40", "09:50", "10:00"])

    assert _sent(report) == [(at("10:00"), ["item-0", "item-1", "item-2"], 3)]


def test_a_min_read_from_an_attribute(tmp_path):
    report = released(tmp_path, {"min": {"object_attribute": "needed"}, "max": 3}, WITHIN_AN_HOUR,
                      ["09:40", "11:00", "11:30"], needed=2)

    # with a min of 1 the group would go out alone at 10:40; it needs 2
    assert _sent(report) == [(at("11:00"), ["item-0", "item-1"], 2)]


def test_an_invalid_min_counts_as_1_with_one_warning(tmp_path):
    report = released(tmp_path, {"min": {"object_attribute": "needed"}, "max": 3}, WITHIN_AN_HOUR,
                      ["09:40", "11:00", "11:30"], needed=2.5)

    assert _sent(report) == [(at("10:40") + MICROSECOND, ["item-0"], 1),
                             (at("12:00") + MICROSECOND, ["item-1", "item-2"], 2)]
    assert report.engine_warnings == [("Warehouse", "case 0 passes Throw_OrderPicked with no valid needed (got 2.5); "
                                                    "its group needs 1 part at its deadline")]


def test_a_max_below_a_min_from_the_case_counts_as_min_with_one_warning(tmp_path):
    report = released(tmp_path, {"min": {"object_attribute": "needed"}, "max": 2}, WITHIN_AN_HOUR,
                      ["09:40", "09:50", "10:00"], needed=3)

    # a max of 2 would send item-0 and item-1 at 09:50
    assert _sent(report) == [(at("10:00"), ["item-0", "item-1", "item-2"], 3)]
    assert report.engine_warnings == [("Warehouse", "case 0 passes Throw_OrderPicked with max 2 below needed 3; its "
                                                    "group's max counts as 3")]


def test_a_later_part_with_another_min_joins_its_group_with_one_warning(tmp_path):
    # item-0 gives a min of 1, item-1 of 2 and item-2 of 3; item-2 comes in second
    report = released(tmp_path, {"min": {"object_attribute": "needed"}, "max": 3}, WITHIN_AN_HOUR,
                      ["09:40", "11:00", "10:00"], needed="index")

    # the group keeps min 1, so it goes out at its deadline with 2 parts; with item-2's min of 3 it would wait
    assert _sent(report) == [(at("10:40") + MICROSECOND, ["item-0", "item-2"], 2)]
    assert [warning for _, warning in report.engine_warnings] == [
        "case 2 passes Throw_OrderPicked with needed 3, but its fan-in group has 1 from its first part; the group "
        "keeps min 1",
        "fan-in OrderPicked for order_id order-0: 1 of 3 parts (min 2), never sent"]


@pytest.fixture(scope="module")
def truck_run(tmp_path_factory):
    def truck(settings):
        settings["messages"]["publish"][0].update(fan_in=TRUCK, deadline={"hours": 0.5})

    return _run(tmp_path_factory.mktemp("fan_in_min_max"), truck)


def test_every_message_follows_the_three_sending_rules(truck_run):
    report, log = truck_run

    expected, rules, never_sent = expected_messages(_picked(log), fewest=2, deadline=timedelta(hours=0.5))
    assert sorted((time, attributes["order_id"], attributes["item_ids"])
                  for time, attributes in _order_picked(report)) == expected
    assert set(rules) == {"max", "deadline", "min"}
    assert sorted(warning for _, warning in report.engine_warnings) == sorted(
        f"fan-in OrderPicked for order_id {order}: 1 of 3 parts (min 2), never sent" for order in never_sent)
