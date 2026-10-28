"""
"same" (docs/messaging.md): a collecting case takes only messages with the same values of the listed attributes
as the first message it claimed there, e.g. the oldest waiting item and then all waiting items of the same customer.
"""
import pytest

from prosimos.exceptions import InvalidSimScenarioException
from prosimos.orchestration.orchestrator import orchestrate
from prosimos.orchestration.protocol import Message, Verdict
from testing_scripts.scripted_engine import ScriptedEngine
from testing_scripts.test_collect import _orders, _run_until_idle, _settings, at
from testing_scripts.test_race import ELEVEN, QUOTE_JSON, _quote, _run_until
from testing_scripts.test_race import _settings as _race_settings


def _packing(collect, same, change=lambda entry: None):
    """Catch_Items takes the items of any order (no condition on order_id), copying their customer."""
    def apply(settings):
        entry = settings["messages"]["consume"][0]
        del entry["condition"]
        entry.update(collect=collect, copy={"customer": "customer"})
        if same is not None:
            entry["same"] = same
        change(entry)
    return apply


def _item(item_id, customer=None, **more):
    attributes = {"item_id": item_id, **({} if customer is None else {"customer": customer}), **more}
    return Message("ItemReady", attributes)


def _run(tmp_path, published, collect="all", same=("customer",)):
    """Orders 0 and 1 (waiting at Catch_Items from 09:10 and 10:10) and a Picking engine publishing the items of
    published, (time, item), those with the same time in one step; the report, the log, and per item id the case
    that took it and when."""
    orders, log = _orders(2, _settings(tmp_path, _packing(collect, None if same is None else list(same))))
    picking = ScriptedEngine("Picking")
    for time in sorted({time for time, _ in published}):
        items = [item for item_time, item in published if item_time == time]
        picking.at(at(time), lambda now, items=items: items)
    report = orchestrate({"Orders": orders, "Picking": picking}, None, 1)
    item_of = {message.id: message.attributes["item_id"] for message in report.published}
    case_of = {record.message_id: record.claimers[0].object_id for record in report.message_records if record.claimers}
    taken = {item_of[message_id]: (case_of[message_id], time.strftime("%H:%M"))
             for message_id, _, time in report.claims}
    return report, log, taken


def test_with_anna_s_i1_i4_i7_and_bob_s_i2_waiting_a_case_takes_i1_i4_i7_and_i2_stays_for_the_next_case(tmp_path):
    published = [("08:30", _item("i1", "Anna")), ("08:35", _item("i2", "Bob")), ("08:40", _item("i4", "Anna")),
                 ("08:45", _item("i7", "Anna"))]

    _, log, taken = _run(tmp_path, published)

    assert taken == {"i1": ("Orders-0", "09:10"), "i4": ("Orders-0", "09:10"), "i7": ("Orders-0", "09:10"),
                     "i2": ("Orders-1", "10:10")}
    assert log.packed() == {0: at("09:10"), 1: at("10:10")}


def test_with_nothing_waiting_a_case_takes_the_next_item_and_that_customer_s_others_offered_then(tmp_path):
    published = [("09:30", _item("i1", "Anna")), ("09:30", _item("i2", "Bob")), ("09:30", _item("i3", "Anna")),
                 ("09:50", _item("i4", "Anna"))]

    _, log, taken = _run(tmp_path, published)

    # order 1 then takes the oldest waiting item, Bob's i2, and so not Anna's i4
    assert taken == {"i1": ("Orders-0", "09:30"), "i3": ("Orders-0", "09:30"), "i2": ("Orders-1", "10:10")}
    assert log.packed() == {0: at("09:30"), 1: at("10:10")}


def test_without_same_a_case_takes_every_customer_s_items(tmp_path):
    published = [("08:30", _item("i1", "Anna")), ("08:35", _item("i2", "Bob"))]

    _, _, taken = _run(tmp_path, published, same=None)

    assert taken == {"i1": ("Orders-0", "09:10"), "i2": ("Orders-0", "09:10")}


def test_same_works_with_min_and_max(tmp_path):
    # order 0 takes Anna's i1 and waits for a second of hers; at 10:00 it takes i4 and i5, its max, but not Bob's i3
    # or Anna's i6; order 1 takes Bob's i2 and i3 at 10:10
    published = [("08:30", _item("i1", "Anna")), ("08:35", _item("i2", "Bob"))] + [
        ("10:00", _item(item_id, customer)) for item_id, customer in (("i3", "Bob"), ("i4", "Anna"), ("i5", "Anna"),
                                                                     ("i6", "Anna"))]

    _, log, taken = _run(tmp_path, published, collect={"min": 2, "max": 3})

    assert taken == {"i1": ("Orders-0", "09:10"), "i4": ("Orders-0", "10:00"), "i5": ("Orders-0", "10:00"),
                     "i2": ("Orders-1", "10:10"), "i3": ("Orders-1", "10:10")}
    assert log.packed() == {0: at("10:00"), 1: at("10:10")}


def test_every_listed_attribute_must_have_the_same_value(tmp_path):
    published = [("08:30", _item("i1", "Anna", channel="web")), ("08:35", _item("i2", "Anna", channel="shop")),
                 ("08:40", _item("i3", "Anna", channel="web")), ("08:45", _item("i4", "Bob", channel="web"))]

    _, _, taken = _run(tmp_path, published, same=("customer", "channel"))

    assert taken == {"i1": ("Orders-0", "09:10"), "i3": ("Orders-0", "09:10"), "i2": ("Orders-1", "10:10")}


def test_the_first_message_sets_the_values_even_when_it_is_the_case_s_only_one(tmp_path):
    # with min 1 and max 1, order 0 takes Bob's i1 and nothing else; order 1 then starts over with Anna's i2
    published = [("08:30", _item("i1", "Bob")), ("08:35", _item("i2", "Anna")), ("08:40", _item("i3", "Anna"))]

    _, _, taken = _run(tmp_path, published, collect={"min": 1, "max": 1})

    assert taken == {"i1": ("Orders-0", "09:10"), "i2": ("Orders-1", "10:10")}


def test_a_message_without_a_listed_attribute_doesn_t_match_with_one_warning_per_event_and_attribute(tmp_path):
    orders, _ = _orders(1, _settings(tmp_path, _packing("all", ["customer"])))
    _run_until_idle(orders)  # order 0 waits at Catch_Items from 09:10

    verdicts = [orders.deliver(item, at("09:30")) for item in (_item("i1"), _item("i2"), _item("i3", "Anna"))]

    assert verdicts == [Verdict.DISCARDED, Verdict.DISCARDED, Verdict.CLAIMED]
    assert [warning for warning in orders.finish().warnings if "same" in warning] == [
        "ItemReady message offered at Catch_Items has no customer, which 'same' compares; it doesn't match there"]


def test_in_a_race_a_message_the_first_listed_branch_s_same_rejects_goes_to_the_other_branch(tmp_path):
    # both branches take Reply; Catch_Accepted, listed first, collects 2 with the same answer
    def both_take_replies(settings):
        accepted, rejected = settings["messages"]["consume"]
        accepted.update(type="Reply", collect=2, same=["answer"])
        rejected.update(type="Reply")

    quote, log = _quote(json_path=_race_settings(tmp_path, both_take_replies, base=QUOTE_JSON))
    _run_until(quote, ELEVEN)

    for answer in ("yes", "no"):
        assert quote.deliver(Message("Reply", {"quote_id": "Quote-0", "answer": answer}), ELEVEN) is Verdict.CLAIMED
    _run_until_idle(quote)

    assert log.times("Archive quote") == {0: ELEVEN}
    assert log.times("Confirm order") == {}


# load errors

@pytest.mark.parametrize("same, reason", [
    ("customer", r"'same' must be a non-empty list of message attribute names, got 'customer'"),
    ([], r"'same' must be a non-empty list of message attribute names, got \[\]"),
    (["customer", ""], r"'same' must be a non-empty list"),
    (["customer", 3], r"'same' must be a non-empty list"),
    (["customer", "channel", "customer"], r"'same' lists customer more than once"),
])
def test_an_invalid_same_is_rejected(tmp_path, same, reason):
    with pytest.raises(InvalidSimScenarioException, match=r"messages.consume\[0\]: " + reason):
        _orders(json_path=_settings(tmp_path, _packing("all", same)))


def test_same_without_collect_is_rejected(tmp_path):
    def no_collect(entry):
        del entry["collect"]

    with pytest.raises(InvalidSimScenarioException, match=r"'same' needs a collect"):
        _orders(json_path=_settings(tmp_path, _packing("all", ["customer"], no_collect)))
