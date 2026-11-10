"""
The Stock Reservation example (docs/stock-reservation.md): Sales quotes the products of each order to the two
warehouses, which make items and ship each order's portion as one Shipment. The stories of the base run, seed 1.
"""
import csv
from datetime import datetime, timedelta

import pytest
import pytz

from prosimos.orchestration.config import SimulationConfig
from prosimos.orchestration.simulation import run_multiprocess_simulation

CONFIG = "testing_scripts/assets/stock_reservation/simulation.json"  # seed 1, 20 orders


def at(day, clock):
    """2024-01-<day> at clock, UTC."""
    return pytz.utc.localize(datetime.fromisoformat(f"2024-01-{day:02d}T{clock}"))


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """The report of the base run, and its log rows."""
    path = tmp_path_factory.mktemp("stock_reservation") / "log.csv"
    report = run_multiprocess_simulation(SimulationConfig.from_json(CONFIG), path)
    with open(path) as file:
        return report, list(csv.DictReader(file))


def _asked(report, order):
    """The quote and modify messages of the order, by id: (time, products)."""
    return {message.id: (message.time, [line.attributes["product"] for line in message.quote.lines])
            for message in report.published
            if message.source == "Sales" and message.type in ("ItemQuote", "QuoteModified")
            and message.attributes["object_id"] == order}


def _replies(report, order):
    """The orchestrator's replies to the order: (time, type, ready_by or earliest, portions)."""
    return [(message.time, message.type, message.attributes.get("ready_by", message.attributes.get("earliest")),
             message.attributes.get("portions")) for message in report.published
            if message.source == "orchestrator" and message.attributes["request"] == order]


def _taken(report, message_ids):
    """Which items took the lines the messages reserved, and when: (warehouse, item, time), in time order."""
    return sorted(((claimer.process, claimer.object_id, claimer.at) for record in report.message_records
                   if record.message_id in message_ids for claimer in record.claimers), key=lambda taken: taken[2])


def _shipments(report, order):
    """The order's Shipments: (time, warehouse, items)."""
    return [(message.time, message.source, message.attributes["item_ids"]) for message in report.published
            if message.type == "Shipment" and message.attributes["order_id"] == order]


def _activities(rows, process, case):
    """The case's tasks: (activity, end time)."""
    return [(row["activity"], datetime.fromisoformat(row["end_time"])) for row in rows
            if row["process"] == process and row["case_id"] == case]


def test_the_run_quotes_twenty_orders_with_every_outcome(run):
    report, _ = run

    assert report.quotes == {"ItemQuote": {"accepted": 16, "declined": 4, "canceled": 0, "cancel_too_late": 1,
                                           "modified_accepted": 1, "modified_declined": 1, "declined_as_second": 0}}
    assert report.engine_warnings == [] and report.warnings == []


def test_a_quote_split_over_two_warehouses_ships_one_shipment_from_each(run):
    # order-0 asks two MacBooks and an iPad at 09:10: Tartu promises the iPad, Tallinn the MacBooks
    report, rows = run
    asked = _asked(report, "order-0")

    assert list(asked.values()) == [(at(1, "09:10"), ["MacBook", "MacBook", "iPad"])]
    assert _replies(report, "order-0") == [(at(1, "09:10"), "QuoteAccepted", at(1, "19:00"), 2)]
    assert _taken(report, asked) == [("TartuWarehouse", "item-0", at(1, "10:00")),
                                     ("TallinnWarehouse", "item-0", at(1, "13:00")),
                                     ("TallinnWarehouse", "item-1", at(1, "13:00"))]
    # each portion is one Shipment, picked in 15 minutes; Sales closes the order once both came, before 19:00
    assert _shipments(report, "order-0") == [(at(1, "10:15"), "TartuWarehouse", ["item-0"]),
                                             (at(1, "13:15"), "TallinnWarehouse", ["item-0", "item-1"])]
    assert ("Close order", at(1, "13:20")) in _activities(rows, "Sales", "0")


def _declined(report):
    """order -> its first reply, of the orders whose quote was declined."""
    return {order: replies[0] for order, replies in ((f"order-{case}", _replies(report, f"order-{case}"))
                                                     for case in range(20)) if replies[0][1] == "QuoteDeclined"}


def test_the_impatient_order_is_declined_with_an_earliest(run):
    # order-8, patient for 1 day, asks two MacBooks and an iPad at 01:36 on day 2: they could be ready at 14:00 the
    # next day, past its limit; nothing is reserved, and the customer is told
    report, rows = run
    [(asked, products)] = _asked(report, "order-8").values()

    assert products == ["MacBook", "MacBook", "iPad"]
    assert _declined(report)["order-8"] == (asked, "QuoteDeclined", at(3, "14:00"), None)
    assert asked == at(2, "01:36:19.644694") and at(3, "14:00") > asked + timedelta(days=1)
    assert ("Tell the customer", asked + timedelta(minutes=5)) in _activities(rows, "Sales", "8")
    assert _taken(report, _asked(report, "order-8")) == [] and _shipments(report, "order-8") == []


def test_quotes_no_item_is_left_for_are_declined_without_an_earliest(run):
    # every item of some product of theirs that the warehouses will make is promised already
    report, rows = run
    declined = _declined(report)

    assert sorted(declined, key=lambda order: int(order[6:])) == ["order-8", "order-13", "order-14", "order-18"]
    for order in ("order-13", "order-14", "order-18"):
        [(asked, _)] = _asked(report, order).values()
        assert declined[order] == (asked, "QuoteDeclined", None, None)
        assert ("Tell the customer", asked + timedelta(minutes=5)) in _activities(rows, "Sales", order[6:])
        assert _shipments(report, order) == []


def test_an_accepted_change_keeps_the_ipad_in_place_and_ships_it_with_the_added_macbook(run):
    # order-6 asks an iPad at 15:17 and adds a MacBook 5 minutes later: accepted, for 12:00 on day 3
    report, rows = run
    [(quote, (asked, _)), (modify, (changed, products))] = _asked(report, "order-6").items()

    assert asked == at(1, "15:17:02.524453") and changed == asked + timedelta(minutes=5)
    assert products == ["iPad", "MacBook"]
    assert _replies(report, "order-6") == [(asked, "QuoteAccepted", at(2, "00:00:01"), 1),
                                           (changed, "QuoteAccepted", at(3, "12:00"), 1)]
    # the iPad, on Tartu's shelf from 20:00 on day 1, keeps the first quote's line and waits for the MacBook's
    assert ("Make item", at(1, "20:00")) in _activities(rows, "TartuWarehouse", "5")
    assert _taken(report, [quote]) == [("TartuWarehouse", "item-5", at(3, "08:00"))]
    assert _taken(report, [modify]) == [("TartuWarehouse", "item-23", at(3, "08:00"))]
    assert _shipments(report, "order-6") == [(at(3, "08:15"), "TartuWarehouse", ["item-5", "item-23"])]


def test_a_change_after_the_portion_moved_on_is_declined_and_the_ipad_still_ships(run):
    # order-9's iPad is on Tartu's shelf when it asks: the item takes the line at once and moves on, so the quote
    # is locked when the order adds a MacBook 5 minutes later
    report, _ = run
    [(quote, (asked, _)), (modify, (changed, products))] = _asked(report, "order-9").items()

    assert products == ["iPad", "MacBook"] and changed == asked + timedelta(minutes=5)
    assert _taken(report, [quote]) == [("TartuWarehouse", "item-7", asked)]
    assert _replies(report, "order-9") == [(asked, "QuoteAccepted", asked, 1), (changed, "QuoteDeclined", None, None)]
    assert _taken(report, [modify]) == []
    assert _shipments(report, "order-9") == [(asked + timedelta(minutes=15), "TartuWarehouse", ["item-7"])]


def test_an_order_that_gives_up_cancels_too_late_and_its_other_portion_is_undelivered(run):
    # order-10 is accepted for a MacBook and an iPad, split: Tartu's iPad moves on at once; the MacBook Tallinn
    # promised for 13:00 on day 3 is scrapped, and Tallinn makes no other. After 7 days the order gives up
    report, rows = run
    [(asked, products)] = _asked(report, "order-10").values()

    assert asked == at(2, "04:15:29.893559") and products == ["MacBook", "iPad"]
    assert _replies(report, "order-10") == [(asked, "QuoteAccepted", at(3, "13:00"), 2)]
    assert _shipments(report, "order-10") == [(asked + timedelta(minutes=15), "TartuWarehouse", ["item-9"])]
    assert ("Scrap item", at(3, "13:10")) in _activities(rows, "TallinnWarehouse", "17")
    [cancel] = [message for message in report.published
                if message.type == "QuoteCanceled" and message.attributes["object_id"] == "order-10"]
    assert cancel.time == asked + timedelta(days=7, microseconds=1)  # a race timer's branch goes on 1 µs late
    # the give-up is the run's last event, so Tallinn's portion is overdue from 13:00 on day 3 until then
    assert report.executed[-1] == (cancel.time, "Sales")
    assert report.broken_promises == {"ItemQuote": {"count": 0, "average_late": None, "most_late": None,
                                                    "undelivered": 1, "most_overdue": cancel.time - at(3, "13:00")}}
