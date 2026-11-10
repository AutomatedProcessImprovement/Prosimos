"""
Cancels and modifies on the Stock Reservation example (docs/stock-reservation.md): each scenario is the example's
configuration with a few settings changed, so that a handful of orders and items tell one story, with its times.
Unless a scenario says otherwise, there is one order, of one MacBook, which never changes and is patient for 5 days;
no item is scrapped; TartuWarehouse makes three MacBooks, one every 2 hours from 09:00, each done an hour later, and
TallinnWarehouse makes one iPhone, which nobody asks.
"""
from testing_scripts.multiprocess.examples.stock_reservation_runs import (
    AFTER_TIES, HOUR, MINUTE, ON_TIME, at, branches, ended, every, gives_up_after, makes, messages, orders,
    outcomes, run, taken, takes)


def test_a_cancel_frees_its_line_for_the_next_order(tmp_path):
    # an order an hour, each giving up after 30 minutes: order-0 is promised the MacBook done at 10:00, but gives up
    # at 09:40; order-1, asking at 10:10, gets that MacBook at once instead of the next one, done at 12:00
    def change(settings):
        orders(settings, 2, "MacBook")
        every(settings["sales"], HOUR)
        gives_up_after(settings["sales"], 30 * MINUTE)

    report, rows = run(tmp_path, change)

    assert messages(report, "order-0") == [(at("09:10"), "ItemQuote", ["MacBook"]),
                                            (at("09:10"), "QuoteAccepted", at("10:00")),
                                            (at("09:40") + AFTER_TIES, "QuoteCanceled", None)]
    assert messages(report, "order-1")[:3] == [(at("10:10"), "ItemQuote", ["MacBook"]),
                                                (at("10:10"), "QuoteAccepted", at("10:10")),
                                                (at("10:25"), "Shipment", ("TartuWarehouse", ["item-0"]))]
    assert taken(report) == [(at("10:10"), "TartuWarehouse", "item-0", "ItemQuote", "order-1")]
    assert ended(rows, "Sales", "1", "Close order") == at("10:30")
    assert outcomes(report, accepted=2, canceled=1)


def test_an_order_that_gives_up_frees_its_lines_and_the_held_item_goes_back_on_the_shelf(tmp_path):
    # the order asks two MacBooks: one is done at 10:00 and held for it, the other only at 20:00; it gives up after
    # 2 hours, so the held MacBook goes back on the shelf, and neither ever ships
    def change(settings):
        orders(settings, 1, "MacBook", items=2)
        makes(settings, "TartuWarehouse", "MacBook", 2)
        every(settings["tartu_warehouse"], 10 * HOUR)
        gives_up_after(settings["sales"], 2 * HOUR)

    report, _ = run(tmp_path, change)

    assert messages(report, "order-0") == [(at("09:10"), "ItemQuote", ["MacBook", "MacBook"]),
                                            (at("09:10"), "QuoteAccepted", at("20:00")),
                                            (at("11:10") + AFTER_TIES, "QuoteCanceled", None)]
    assert taken(report) == []
    assert [(process, case.object_id, case.waiting_since) for process, case in report.stalled
            if process == "TartuWarehouse"] == [("TartuWarehouse", "item-0", at("10:00")),
                                                ("TartuWarehouse", "item-1", at("20:00"))]
    # a canceled quote breaks no promise, though its ready_by passed before the run ended
    assert outcomes(report, accepted=1, canceled=1) and report.broken_promises == {"ItemQuote": ON_TIME}


def test_a_modify_after_a_portion_moved_on_is_declined_and_the_other_portion_still_ships(tmp_path):
    # the order asks two iPads at 11:00, after a 2-hour Place order: Tartu's is on the shelf since 10:00 and moves
    # on at once; Tallinn's takes 4 hours to make, done at 13:00. The MacBook the order adds at 11:05 comes too late
    def change(settings):
        orders(settings, 1, "iPad", items=2)
        takes(settings["sales"], "Place_Order", 2 * HOUR)
        branches(settings["sales"], "Change", Keep=0.0, Add=1.0)
        makes(settings, "TartuWarehouse", "iPad", 1)
        makes(settings, "TallinnWarehouse", "iPad", 1)
        takes(settings["tallinn_warehouse"], "Make_Item", 4 * HOUR)

    report, rows = run(tmp_path, change)

    assert messages(report, "order-0")[:6] == [(at("11:00"), "ItemQuote", ["iPad", "iPad"]),
                                                (at("11:00"), "QuoteAccepted", at("13:00")),
                                                (at("11:05"), "QuoteModified", ["iPad", "iPad", "MacBook"]),
                                                (at("11:05"), "QuoteDeclined", None),
                                                (at("11:15"), "Shipment", ("TartuWarehouse", ["item-0"])),
                                                (at("13:15"), "Shipment", ("TallinnWarehouse", ["item-0"]))]
    assert taken(report) == [(at("11:00"), "TartuWarehouse", "item-0", "ItemQuote", "order-0"),
                              (at("13:00"), "TallinnWarehouse", "item-0", "ItemQuote", "order-0")]
    assert ended(rows, "Sales", "0", "Close order") == at("13:20")
    assert outcomes(report, accepted=1, modified_declined=1)



def test_a_scrapped_item_makes_a_portion_late_and_the_report_says_how_late(tmp_path):
    # the example's 10% of MacBooks are scrapped, each still counting 0.9 until its check, so the order of three is
    # promised the fourth, done at 16:00 (an expected count of 3.6). Items 2 and 3 are both scrapped, and the third
    # good one is item-4, done at 18:00: the portion moves on 2 hours late. Seed 13 is the first seed that gives a
    # late portion for this setting
    def change(settings):
        orders(settings, 1, "MacBook", items=3)
        makes(settings, "TartuWarehouse", "MacBook", 12)
        branches(settings["tartu_warehouse"], "Check", Good=0.9, Bad=0.1)

    report, rows = run(tmp_path, change, seed=13)

    assert messages(report, "order-0")[:3] == [
        (at("09:10"), "ItemQuote", ["MacBook"] * 3), (at("09:10"), "QuoteAccepted", at("16:00")),
        (at("18:30"), "Shipment", ("TartuWarehouse", ["item-0", "item-1", "item-4"]))]
    assert [ended(rows, "TartuWarehouse", case, "Scrap item") for case in "23"] == [at("14:10"), at("16:10")]
    assert [(time, item) for time, _, item, _, _ in taken(report)] == [
        (at("18:00"), "item-0"), (at("18:00"), "item-1"), (at("18:00"), "item-4")]
    assert report.broken_promises == {"ItemQuote": {**ON_TIME, "count": 1, "average_late": 2 * HOUR,
                                                    "most_late": 2 * HOUR}}
