"""
The slot rule (docs/orchestrator.md): each line of a quote takes the earliest unused slot across the members;
identical lines share one slot list. A pure function of the lines and the members' estimates.
"""
from datetime import datetime, timedelta

import pytz

from prosimos.orchestration.protocol import Estimate, Line
from prosimos.orchestration.quotes import Plan, plan_lines

START = pytz.utc.localize(datetime(2024, 1, 1, 9))


def hours(n):
    return START + timedelta(hours=n)


def _lines(*products):
    return [Line({"product": product}, f"l{number}") for number, product in enumerate(products, 1)]


def test_one_member():
    lines = _lines("iPhone", "iPad")
    tartu = Estimate({"l1": [hours(2), hours(5)], "l2": [hours(1)]})

    assert plan_lines(lines, {"Tartu": tartu}) == Plan({"l1": ("Tartu", hours(2)), "l2": ("Tartu", hours(1))},
                                                       ready_by=hours(2))


def test_a_split_over_two_members():
    # Tallinn has the iPhone sooner, Tartu the iPad
    lines = _lines("iPhone", "iPad")
    tartu = Estimate({"l1": [hours(30)], "l2": [hours(3)]})
    tallinn = Estimate({"l1": [hours(4)], "l2": []})

    assert plan_lines(lines, {"Tartu": tartu, "Tallinn": tallinn}) == Plan(
        {"l1": ("Tallinn", hours(4)), "l2": ("Tartu", hours(3))}, ready_by=hours(4))


def test_identical_lines_take_the_1st_2nd_3rd_slot_of_their_shared_list_across_the_members():
    lines = _lines("iPad", "iPad", "iPad", "iPhone")
    tartu = Estimate({line: [hours(1), hours(6)] for line in ("l1", "l2", "l3")} | {"l4": [hours(2)]})
    tallinn = Estimate({line: [hours(3), hours(9)] for line in ("l1", "l2", "l3")} | {"l4": []})

    assert plan_lines(lines, {"Tartu": tartu, "Tallinn": tallinn}) == Plan(
        {"l1": ("Tartu", hours(1)), "l2": ("Tallinn", hours(3)), "l3": ("Tartu", hours(6)), "l4": ("Tartu", hours(2))},
        ready_by=hours(6))


def test_a_line_with_no_slot_makes_the_plan_impossible():
    lines = _lines("iPhone", "MacBook")

    assert plan_lines(lines, {"Tartu": Estimate({"l1": [hours(1)], "l2": []}), "Tallinn": Estimate()}) is None


def test_identical_lines_with_fewer_slots_than_lines_make_it_impossible():
    lines = _lines("iPad", "iPad", "iPad")
    shelf = [hours(1), hours(2)]

    assert plan_lines(lines, {"Tartu": Estimate({line.id: shelf for line in lines})}) is None


def test_at_the_same_time_the_member_asked_first_gives_the_slot():
    lines = _lines("iPhone")

    assert plan_lines(lines, {"Tallinn": Estimate({"l1": [hours(1)]}), "Tartu": Estimate({"l1": [hours(1)]})}) == \
        Plan({"l1": ("Tallinn", hours(1))}, ready_by=hours(1))


def test_no_lines_make_an_empty_plan():
    assert plan_lines([], {"Tartu": Estimate()}) == Plan({}, ready_by=None)
