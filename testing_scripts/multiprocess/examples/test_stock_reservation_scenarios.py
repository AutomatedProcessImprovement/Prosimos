"""
Cancels and modifies on the Stock Reservation example (docs/stock-reservation.md): each scenario is the example's
configuration with a few settings changed, so that a handful of orders and items tell one story, with its times.
Unless a scenario says otherwise, there is one order, of one MacBook, which never changes and is patient for 5 days;
no item is scrapped; TartuWarehouse makes three MacBooks, one every 2 hours from 09:00, each done an hour later, and
TallinnWarehouse makes one iPhone, which nobody asks.
"""
import csv
import json
import shutil
from datetime import datetime, timedelta

import pytz

from prosimos.orchestration.config import SimulationConfig
from prosimos.orchestration.simulation import run_multiprocess_simulation

EXAMPLE = "testing_scripts/assets/stock_reservation"
MINUTE, HOUR = timedelta(minutes=1), timedelta(hours=1)
AFTER_TIES = timedelta(microseconds=1)  # a race timer's branch goes on 1 µs late


def at(clock):
    """09:00 on the run's first day, 2024-01-01, at clock, UTC."""
    return pytz.utc.localize(datetime.fromisoformat(f"2024-01-01T{clock}"))


def _run(tmp_path, change, seed=1):
    """The report and log rows of the example with its settings changed: change(settings) changes the files'
    contents, by name (simulation, sales, tartu_warehouse, tallinn_warehouse), after the defaults above."""
    folder = tmp_path / "stock_reservation"
    shutil.copytree(EXAMPLE, folder)
    settings = {name: json.loads((folder / f"{name}.json").read_text())
                for name in ("simulation", "sales", "tartu_warehouse", "tallinn_warehouse")}
    _defaults(settings)
    change(settings)
    settings["simulation"]["seed"] = seed
    for name, content in settings.items():
        (folder / f"{name}.json").write_text(json.dumps(content))
    report = run_multiprocess_simulation(SimulationConfig.from_json(folder / "simulation.json"), folder / "log.csv")
    with open(folder / "log.csv") as file:
        return report, list(csv.DictReader(file))


def _defaults(settings):
    _orders(settings, 1, "MacBook")
    _always(settings["sales"], "patience", 5)
    _branches(settings["sales"], "Change", Keep=1.0, Add=0.0)
    for warehouse in ("tartu_warehouse", "tallinn_warehouse"):
        _branches(settings[warehouse], "Check", Good=1.0, Bad=0.0)
    _makes(settings, "TartuWarehouse", "MacBook", 3)
    _makes(settings, "TallinnWarehouse", "iPhone", 1)


def _process(settings, name):
    return next(spec for spec in settings["simulation"]["organizations"][0]["processes"] if spec["name"] == name)


def _orders(settings, count, product, items=1):
    """count orders, each of items of the product."""
    sales = _process(settings, "Sales")
    sales["arrivals"] = count
    sales["references"][0]["pick"] = {"weights": {product: 1}}
    _always(settings["sales"], "items", items)


def _makes(settings, warehouse, product, count):
    _process(settings, warehouse)["arrivals"] = count
    _process(settings, warehouse)["references"][0]["pick"] = {"weights": {product: 1}}


def _always(process_settings, attribute, value):
    next(entry for entry in process_settings["object_attributes"] if entry["name"] == attribute)["values"] = [
        {"key": value, "value": 1.0}]


def _branches(process_settings, gateway, **shares):
    next(entry for entry in process_settings["gateway_branching_probabilities"] if entry["gateway_id"] == gateway)[
        "probabilities"] = [{"path_id": flow, "value": share} for flow, share in shares.items()]


def _every(process_settings, interval):
    process_settings["arrival_time_distribution"] = {"distribution_name": "fix",
                                                     "distribution_params": [{"value": interval.total_seconds()}]}


def _takes(process_settings, task_id, duration):
    next(entry for entry in process_settings["task_resource_distribution"] if entry["task_id"] == task_id)[
        "resources"][0].update(distribution_name="fix", distribution_params=[{"value": duration.total_seconds()}])


def _gives_up_after(sales, delay):
    sales["event_distribution"][0]["distribution_params"] = [{"value": delay.total_seconds()}]


def _messages(report, order):
    """The order's messages, in time order: (time, type, what): the products of a quote or modify, the ready_by or
    earliest of a reply, the warehouse and items of a Shipment, nothing for a cancel."""
    story = []
    for message in report.published:
        if message.type == "Shipment" and message.attributes["order_id"] == order:
            story.append((message.time, "Shipment", (message.source, message.attributes["item_ids"])))
        elif message.source == "orchestrator" and message.attributes["request"] == order:
            story.append((message.time, message.type,
                          message.attributes.get("ready_by", message.attributes.get("earliest"))))
        elif message.source == "Sales" and message.attributes.get("object_id") == order:
            products = [line.attributes["product"] for line in message.quote.lines] if message.quote.lines else None
            story.append((message.time, message.type, products))
    return story


def _taken(report):
    """Each line an item took: (time, warehouse, item, the quote or modify message's type, its order), in time
    order."""
    sent = {message.id: message for message in report.published}
    return sorted(((claimer.at, claimer.process, claimer.object_id, sent[record.message_id].type,
                    sent[record.message_id].attributes["object_id"])
                   for record in report.message_records if sent[record.message_id].quote is not None
                   for claimer in record.claimers), key=lambda taken: taken[0])


def _ended(rows, process, case, activity):
    return next(datetime.fromisoformat(row["end_time"]) for row in rows
                if row["process"] == process and row["case_id"] == case and row["activity"] == activity)


def _outcomes(report, **counts):
    outcomes = dict.fromkeys(("accepted", "declined", "canceled", "cancel_too_late", "modified_accepted",
                              "modified_declined", "declined_as_second"), 0)
    return report.quotes == {"ItemQuote": {**outcomes, **counts}}


ON_TIME = {"count": 0, "average_late": None, "most_late": None, "undelivered": 0, "most_overdue": None}


def test_a_cancel_frees_its_line_for_the_next_order(tmp_path):
    # an order an hour, each giving up after 30 minutes: order-0 is promised the MacBook done at 10:00, but gives up
    # at 09:40; order-1, asking at 10:10, gets that MacBook at once instead of the next one, done at 12:00
    def change(settings):
        _orders(settings, 2, "MacBook")
        _every(settings["sales"], HOUR)
        _gives_up_after(settings["sales"], 30 * MINUTE)

    report, rows = _run(tmp_path, change)

    assert _messages(report, "order-0") == [(at("09:10"), "ItemQuote", ["MacBook"]),
                                            (at("09:10"), "QuoteAccepted", at("10:00")),
                                            (at("09:40") + AFTER_TIES, "QuoteCanceled", None)]
    assert _messages(report, "order-1")[:3] == [(at("10:10"), "ItemQuote", ["MacBook"]),
                                                (at("10:10"), "QuoteAccepted", at("10:10")),
                                                (at("10:25"), "Shipment", ("TartuWarehouse", ["item-0"]))]
    assert _taken(report) == [(at("10:10"), "TartuWarehouse", "item-0", "ItemQuote", "order-1")]
    assert _ended(rows, "Sales", "1", "Close order") == at("10:30")
    assert _outcomes(report, accepted=2, canceled=1)


def test_an_order_that_gives_up_frees_its_lines_and_the_held_item_goes_back_on_the_shelf(tmp_path):
    # the order asks two MacBooks: one is done at 10:00 and held for it, the other only at 20:00; it gives up after
    # 2 hours, so the held MacBook goes back on the shelf, and neither ever ships
    def change(settings):
        _orders(settings, 1, "MacBook", items=2)
        _makes(settings, "TartuWarehouse", "MacBook", 2)
        _every(settings["tartu_warehouse"], 10 * HOUR)
        _gives_up_after(settings["sales"], 2 * HOUR)

    report, _ = _run(tmp_path, change)

    assert _messages(report, "order-0") == [(at("09:10"), "ItemQuote", ["MacBook", "MacBook"]),
                                            (at("09:10"), "QuoteAccepted", at("20:00")),
                                            (at("11:10") + AFTER_TIES, "QuoteCanceled", None)]
    assert _taken(report) == []
    assert [(process, case.object_id, case.waiting_since) for process, case in report.stalled
            if process == "TartuWarehouse"] == [("TartuWarehouse", "item-0", at("10:00")),
                                                ("TartuWarehouse", "item-1", at("20:00"))]
    # a canceled quote breaks no promise, though its ready_by passed before the run ended
    assert _outcomes(report, accepted=1, canceled=1) and report.broken_promises == {"ItemQuote": ON_TIME}


def test_a_modify_after_a_portion_moved_on_is_declined_and_the_other_portion_still_ships(tmp_path):
    # the order asks two iPads at 11:00, after a 2-hour Place order: Tartu's is on the shelf since 10:00 and moves
    # on at once; Tallinn's takes 4 hours to make, done at 13:00. The MacBook the order adds at 11:05 comes too late
    def change(settings):
        _orders(settings, 1, "iPad", items=2)
        _takes(settings["sales"], "Place_Order", 2 * HOUR)
        _branches(settings["sales"], "Change", Keep=0.0, Add=1.0)
        _makes(settings, "TartuWarehouse", "iPad", 1)
        _makes(settings, "TallinnWarehouse", "iPad", 1)
        _takes(settings["tallinn_warehouse"], "Make_Item", 4 * HOUR)

    report, rows = _run(tmp_path, change)

    assert _messages(report, "order-0")[:6] == [(at("11:00"), "ItemQuote", ["iPad", "iPad"]),
                                                (at("11:00"), "QuoteAccepted", at("13:00")),
                                                (at("11:05"), "QuoteModified", ["iPad", "iPad", "MacBook"]),
                                                (at("11:05"), "QuoteDeclined", None),
                                                (at("11:15"), "Shipment", ("TartuWarehouse", ["item-0"])),
                                                (at("13:15"), "Shipment", ("TallinnWarehouse", ["item-0"]))]
    assert _taken(report) == [(at("11:00"), "TartuWarehouse", "item-0", "ItemQuote", "order-0"),
                              (at("13:00"), "TallinnWarehouse", "item-0", "ItemQuote", "order-0")]
    assert _ended(rows, "Sales", "0", "Close order") == at("13:20")
    assert _outcomes(report, accepted=1, modified_declined=1)



def test_a_scrapped_item_makes_a_portion_late_and_the_report_says_how_late(tmp_path):
    # the example's 10% of MacBooks are scrapped, each still counting 0.9 until its check, so the order of three is
    # promised the fourth, done at 16:00 (an expected count of 3.6). Items 2 and 3 are both scrapped, and the third
    # good one is item-4, done at 18:00: the portion moves on 2 hours late. Seed 13 is the first seed that gives a
    # late portion for this setting
    def change(settings):
        _orders(settings, 1, "MacBook", items=3)
        _makes(settings, "TartuWarehouse", "MacBook", 12)
        _branches(settings["tartu_warehouse"], "Check", Good=0.9, Bad=0.1)

    report, rows = _run(tmp_path, change, seed=13)

    assert _messages(report, "order-0")[:3] == [
        (at("09:10"), "ItemQuote", ["MacBook"] * 3), (at("09:10"), "QuoteAccepted", at("16:00")),
        (at("18:30"), "Shipment", ("TartuWarehouse", ["item-0", "item-1", "item-4"]))]
    assert [_ended(rows, "TartuWarehouse", case, "Scrap item") for case in "23"] == [at("14:10"), at("16:10")]
    assert [(time, item) for time, _, item, _, _ in _taken(report)] == [
        (at("18:00"), "item-0"), (at("18:00"), "item-1"), (at("18:00"), "item-4")]
    assert report.broken_promises == {"ItemQuote": {**ON_TIME, "count": 1, "average_late": 2 * HOUR,
                                                    "most_late": 2 * HOUR}}
