"""
Variants of the Stock Reservation example (docs/stock-reservation.md): each scenario changes its Sales model, adds a
process, or changes a setting, so that a handful of orders and items tell one story, with its times. By default there
is one order, of one MacBook, which never changes and is patient for 5 days; no item is scrapped; TartuWarehouse makes
three MacBooks, one every 2 hours from 09:00, each done an hour later, and TallinnWarehouse makes one iPhone, which
nobody asks.
"""
import os
from datetime import timedelta

import pytest

from testing_scripts.multiprocess.examples.stock_reservation_runs import (
    AFTER_TIES, HOUR, MINUTE, ON_TIME, Model, always, at, ended, every, gives_up_after, makes, messages, orders,
    outcomes, run, taken, takes)

VARIANTS = os.path.abspath("testing_scripts/assets/stock_reservation_variants")
DAY = timedelta(days=1)


def _told_then_asks_again(model):
    """The declined order, once told, asks again with the same list (Change asked) instead of ending."""
    return Model(model).retarget("Flow_7", "Throw_Modify").remove("End_Declined").text()


def test_a_modify_after_a_decline_is_accepted_against_the_original_date(tmp_path):
    # orders every 30 minutes, patient for 3 hours, which give up after 40; the MacBooks are done at 10:00 and 13:00.
    # order-1, asking at 09:40, is declined: the 13:00 MacBook is past its limit, 12:40. Told 15 minutes later, it
    # asks again; order-0 gave up at 09:50, so the 10:00 MacBook is free, within the original limit
    def change(settings):
        orders(settings, 2, "MacBook")
        every(settings["sales"], 30 * MINUTE)
        always(settings["sales"], "patience", 3)
        settings["sales"]["messages"]["publish"][0]["quote"]["within"] = {"hours": {"object_attribute": "patience"}}
        takes(settings["sales"], "Tell_Customer", 15 * MINUTE)
        gives_up_after(settings["sales"], 40 * MINUTE)
        makes(settings, "TartuWarehouse", "MacBook", 2)
        every(settings["tartu_warehouse"], 3 * HOUR)

    report, _ = run(tmp_path, change, model=_told_then_asks_again)

    assert messages(report, "order-0")[2] == (at("09:50") + AFTER_TIES, "QuoteCanceled", None)
    limit = at("09:40") + 3 * HOUR
    assert messages(report, "order-1")[:5] == [(at("09:40"), "ItemQuote", ["MacBook"]),
                                               (at("09:40"), "QuoteDeclined", at("13:00")),
                                               (at("09:55"), "QuoteModified", ["MacBook"]),
                                               (at("09:55"), "QuoteAccepted", at("10:00")),
                                               (at("10:15"), "Shipment", ("TartuWarehouse", ["item-0"]))]
    assert at("13:00") > limit >= at("10:00")
    assert taken(report) == [(at("10:00"), "TartuWarehouse", "item-0", "QuoteModified", "order-1")]
    assert outcomes(report, accepted=1, declined=1, canceled=1, modified_accepted=1)


def _withdraws(model):
    """From the start, a parallel branch waits for the customer to withdraw (a timer) and ends at a terminate end
    event."""
    return (Model(model).add("parallelGateway", "Fork", "")
            .add("intermediateCatchEvent", "Withdraws", "Customer withdraws", "timer")
            .add("endEvent", "Withdrawn", "Order withdrawn", "terminate")
            .retarget("Flow_1", "Fork").flow("Fork_Order", "Fork", "Place_Order")
            .flow("Fork_Withdraw", "Fork", "Withdraws").flow("Withdraw", "Withdraws", "Withdrawn").text())


def test_a_case_ended_by_a_terminate_end_event_cancels_its_quote_automatically(tmp_path):
    # the order asks two MacBooks: one is done at 10:00 and held for it, the other only at 20:00. The customer
    # withdraws 2 hours after the order came, at 11:00: the case ends there, and its quote is canceled then, from the
    # terminate end event, so the held MacBook goes back on the shelf
    def change(settings):
        orders(settings, 1, "MacBook", items=2)
        makes(settings, "TartuWarehouse", "MacBook", 2)
        every(settings["tartu_warehouse"], 10 * HOUR)
        settings["sales"]["event_distribution"].append(
            {"event_id": "Withdraws", "distribution_name": "fix", "distribution_params": [{"value": 7200}]})

    report, _ = run(tmp_path, change, model=_withdraws)

    assert messages(report, "order-0") == [(at("09:10"), "ItemQuote", ["MacBook", "MacBook"]),
                                           (at("09:10"), "QuoteAccepted", at("20:00")),
                                           (at("11:00"), "QuoteCanceled", None)]
    [cancel] = [record for record in report.message_records if record.type == "QuoteCanceled"]
    assert [publisher.element_id for publisher in cancel.publishers] == ["Withdrawn"]
    assert taken(report) == []
    assert [(case.object_id, case.waiting_since) for process, case in report.stalled
            if process == "TartuWarehouse"] == [("item-0", at("10:00")), ("item-1", at("20:00"))]
    assert outcomes(report, accepted=1, canceled=1) and report.broken_promises == {"ItemQuote": ON_TIME}


def _orders_again(model):
    """After Close order, an exclusive split may send the order back to Place order."""
    return (Model(model).add("exclusiveGateway", "Again", "Order again?").retarget("Flow_19", "Again")
            .flow("Again_Yes", "Again", "Place_Order").flow("Again_No", "Again", "End_Delivered").text())


def test_an_order_that_quotes_again_after_delivery_replaces_its_quote_and_the_first_late_portion_stays(tmp_path):
    # Tartu's MacBooks come every 30 minutes, but its one maker takes an hour each, so they queue. The order of two,
    # at 09:10, is promised for 10:30: the second MacBook isn't made yet, and no wait is known; it is made at 11:00,
    # 30 minutes late. Delivered, the order orders again, with a 30% chance, and places a second quote at 11:30, which
    # replaces the first; the first's late portion stays in the report. Seed 2 is the first seed that gives this story
    def change(settings):
        orders(settings, 1, "MacBook", items=2)
        makes(settings, "TartuWarehouse", "MacBook", 6)
        every(settings["tartu_warehouse"], 30 * MINUTE)
        settings["sales"]["gateway_branching_probabilities"].append({"gateway_id": "Again", "probabilities": [
            {"path_id": "Again_Yes", "value": 0.3}, {"path_id": "Again_No", "value": 0.7}]})

    report, rows = run(tmp_path, change, seed=2, model=_orders_again)

    assert messages(report, "order-0") == [
        (at("09:10"), "ItemQuote", ["MacBook", "MacBook"]), (at("09:10"), "QuoteAccepted", at("10:30")),
        (at("11:15"), "Shipment", ("TartuWarehouse", ["item-0", "item-1"])),
        (at("11:30"), "ItemQuote", ["MacBook", "MacBook"]), (at("11:30"), "QuoteAccepted", at("13:00")),
        (at("13:15"), "Shipment", ("TartuWarehouse", ["item-2", "item-3"])), (at("13:20"), "QuoteCanceled", None)]
    assert ended(rows, "TartuWarehouse", "1", "Make item") == at("11:00")
    assert outcomes(report, accepted=2)
    assert report.broken_promises == {"ItemQuote": {**ON_TIME, "count": 1, "average_late": 30 * MINUTE,
                                                    "most_late": 30 * MINUTE}}


def _counter(settings, walk_ins):
    """walk_ins customers of the Counter process, one an hour from 09:00, each asking a MacBook without a quote after
    browsing for 90 minutes; the warehouses' shelf takes their ItemAsked too, as a portion of one."""
    settings["simulation"]["organizations"][0]["processes"].append({
        "name": "Counter", "bpmn_path": f"{VARIANTS}/counter.bpmn", "json_path": f"{VARIANTS}/counter.json",
        "arrivals": walk_ins, "object_type": "purchase",
        "references": [{"object_attribute": "product", "catalogue": "product", "pick": {"weights": {"MacBook": 1}}}]})
    settings["simulation"]["consumer_groups"]["Counter"] = ["Counter"]
    for warehouse in ("tartu_warehouse", "tallinn_warehouse"):
        settings[warehouse]["messages"]["consume"].append({
            "event_id": "Shelf", "type": "ItemAsked",
            "condition": [[{"attribute": "product", "comparison": "=", "object_attribute": "product"}]],
            "copy": {"order_id": "object_id", "portion_size": "portion_size"}})


def test_ordinary_orders_wait_for_items_no_quote_holds(tmp_path):
    # the quote of three MacBooks, at 09:10, is promised the ones done at 10:00, 12:00 and 14:00. Walk-ins ask a
    # MacBook at 10:30 and 11:30: the 10:00 one is held, so they wait; the 12:00 one fills the quote's open line
    # first. The order gives up at 13:00, and its two held MacBooks serve the walk-ins
    def change(settings):
        orders(settings, 1, "MacBook", items=3)
        gives_up_after(settings["sales"], 3 * HOUR + 50 * MINUTE)
        makes(settings, "TartuWarehouse", "MacBook", 4)
        _counter(settings, 2)

    report, rows = run(tmp_path, change)

    assert messages(report, "order-0") == [(at("09:10"), "ItemQuote", ["MacBook"] * 3),
                                           (at("09:10"), "QuoteAccepted", at("14:00")),
                                           (at("13:00") + AFTER_TIES, "QuoteCanceled", None)]
    asked = {message.id: (message.attributes["object_id"], message.time) for message in report.published
             if message.type == "ItemAsked"}
    assert sorted(asked.values()) == [("purchase-0", at("10:30")), ("purchase-1", at("11:30"))]
    assert ended(rows, "TartuWarehouse", "1", "Make item") == at("12:00")
    assert sorted((asked[record.message_id][0], claimer.object_id) for record in report.message_records
                  if record.message_id in asked for claimer in record.claimers) == [
        ("purchase-0", "item-0"), ("purchase-1", "item-1")]
    assert taken(report) == []
    assert [(message.time, message.attributes["order_id"], message.attributes["item_ids"])
            for message in report.published if message.type == "Shipment"] == [
        (at("13:15") + AFTER_TIES, "purchase-0", ["item-0"]), (at("13:15") + AFTER_TIES, "purchase-1", ["item-1"])]
    assert outcomes(report, accepted=1, canceled=1)


@pytest.mark.parametrize("within", [True, False], ids=["within its patience", "backorder"])
def test_a_quote_without_within_is_a_backorder_accepted_however_far_its_ready_by(tmp_path, within):
    # orders an hour apart, patient for a day; Tartu makes a MacBook every two days. order-0 gets the first, at
    # 10:00; order-1's is done on day 3 at 10:00: past its patience, declined, unless its quote has no within
    def change(settings):
        orders(settings, 2, "MacBook")
        every(settings["sales"], HOUR)
        always(settings["sales"], "patience", 1)
        if not within:
            del settings["sales"]["messages"]["publish"][0]["quote"]["within"]
        makes(settings, "TartuWarehouse", "MacBook", 2)
        every(settings["tartu_warehouse"], 2 * DAY)

    report, _ = run(tmp_path, change)

    reply = "QuoteDeclined" if within else "QuoteAccepted"
    assert messages(report, "order-1")[:2] == [(at("10:10"), "ItemQuote", ["MacBook"]),
                                               (at("10:10"), reply, at("10:00") + 2 * DAY)]
    if not within:
        assert messages(report, "order-1")[2] == (at("10:15") + 2 * DAY, "Shipment", ("TartuWarehouse", ["item-1"]))
