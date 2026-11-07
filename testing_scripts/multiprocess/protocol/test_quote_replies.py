"""
Deciding a quote (docs/orchestrator.md): the limit becomes a date once, the quote's time plus within; with a plan
ready by then (or without a limit), the orchestrator reserves the lines at the members and publishes the accepted
reply, else the declined one. Replies come from the orchestrator and are routed like any message, in the same step.
"""
from datetime import datetime, timedelta

import pytest
import pytz

from prosimos.orchestration.config import SimulationConfig
from prosimos.orchestration.orchestrator import orchestrate
from prosimos.orchestration.protocol import Line, Message, Quote, QuoteKind, Verdict
from testing_scripts.multiprocess.scripted_engine import ScriptedEngine

START = pytz.utc.localize(datetime(2024, 1, 1, 9))
DAY = timedelta(days=1)


def _run(shelves, within=timedelta(days=3), products=("iPhone", "iPad", "iPad"), accepted="QuoteAccepted"):
    """Sales asks a quote for products at 09:00 and takes its reply; each warehouse in shelves answers a line with
    the times of its product."""
    sales = ScriptedEngine("Sales")
    sales.at(START, lambda now: [Message("ItemQuote", {"object_id": "order-7"}, quote=Quote(
        QuoteKind.QUOTE, "ItemQuote", "order-7", [Line({"product": product}) for product in products], within,
        accepted, "QuoteDeclined"))])
    for reply in ("QuoteAccepted", "QuoteDeclined"):
        sales.consume(reply, lambda message, now: Verdict.CLAIMED)
    engines = {"Sales": sales}
    for name, shelf in shelves.items():
        engines[name] = ScriptedEngine(name)
        engines[name].slots("ItemQuote", lambda line, now, shelf=shelf: shelf.get(line.attributes["product"], []))
    report = orchestrate(engines, {"Sales": ["Sales"], "Warehouses": list(shelves)}, 1)
    return report, engines


def _replies(report, engines):
    """(type, source, time, attributes) of each reply, all of them taken by Sales at once."""
    replies = [message for message in report.published if message.source == "orchestrator"]
    assert [(message.id, now) for message, now in engines["Sales"].claimed] == [(m.id, m.time) for m in replies]
    return [(message.type, message.source, message.time, message.attributes) for message in replies]


def test_accepted_with_its_ready_by_and_the_number_of_members_delivering():
    # Tartu has the iPhone in a day and one iPad in 2 hours; Tallinn has the other iPad in a day
    report, engines = _run({"Tartu": {"iPhone": [START + DAY], "iPad": [START + timedelta(hours=2), START + 2 * DAY]},
                            "Tallinn": {"iPad": [START + DAY]}})

    assert _replies(report, engines) == [("QuoteAccepted", "orchestrator", START, {
        "request": "order-7", "ready_by": START + DAY, "deliveries": 2})]


def test_the_lines_are_reserved_at_each_member_before_the_reply():
    report, engines = _run({"Tartu": {"iPhone": [START + DAY], "iPad": [START + timedelta(hours=2), START + 2 * DAY]},
                            "Tallinn": {"iPad": [START + DAY]}})

    quote = report.published[0]
    assert engines["Tartu"].reserved == [(quote, ["l1", "l2"], START)]
    assert engines["Tallinn"].reserved == [(quote, ["l3"], START)]


def test_declined_with_the_earliest_time_it_could_be_ready_and_nothing_reserved():
    report, engines = _run({"Tartu": {"iPhone": [START + 4 * DAY], "iPad": [START, START]}})

    assert _replies(report, engines) == [("QuoteDeclined", "orchestrator", START,
                                          {"request": "order-7", "earliest": START + 4 * DAY})]
    assert engines["Tartu"].reserved == []


def test_declined_with_no_earliest_when_a_line_has_no_slot():
    report, engines = _run({"Tartu": {"iPhone": [START], "iPad": [START]}})

    assert _replies(report, engines) == [("QuoteDeclined", "orchestrator", START,
                                          {"request": "order-7", "earliest": None})]


def test_ready_exactly_at_the_limit_is_accepted():
    report, engines = _run({"Tartu": {"iPhone": [START + 3 * DAY]}}, products=("iPhone",))

    assert _replies(report, engines)[0][0] == "QuoteAccepted"


def test_without_a_limit_any_plan_is_accepted():
    report, engines = _run({"Tartu": {"iPhone": [START + 300 * DAY]}}, within=None, products=("iPhone",))

    assert _replies(report, engines) == [("QuoteAccepted", "orchestrator", START, {
        "request": "order-7", "ready_by": START + 300 * DAY, "deliveries": 1})]


@pytest.mark.parametrize("within", [timedelta(days=3), None])
def test_a_quote_without_lines_is_accepted_at_once_with_a_warning(within):
    # its quantity attribute is 0, or its list empty
    report, engines = _run({"Tartu": {"iPhone": [START]}}, within=within, products=())

    assert _replies(report, engines) == [("QuoteAccepted", "orchestrator", START,
                                          {"request": "order-7", "ready_by": START, "deliveries": 0})]
    assert engines["Tartu"].estimated == [] and engines["Tartu"].reserved == []
    assert report.warnings == ["ItemQuote m1 from Sales has no lines"]


def test_a_quote_without_the_types_of_its_replies_stops_the_run():
    with pytest.raises(ValueError, match=r"^ItemQuote m1 from Sales asks a quote without the types of its replies: a "
                                         r"quote needs accepted and declined$"):
        _run({"Tartu": {"iPhone": [START]}}, products=("iPhone",), accepted=None)


def test_a_process_named_orchestrator_is_a_load_error():
    message = r"^A process can't be named orchestrator: the replies to quotes come from the orchestrator under that " \
              r"name$"
    with pytest.raises(ValueError, match=message):
        SimulationConfig([], START, extra_processes=["Sales", "orchestrator"])
    with pytest.raises(ValueError, match=message):
        orchestrate({"orchestrator": ScriptedEngine("orchestrator")}, None, 1)
