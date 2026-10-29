"""
Collect (min, max) (docs/messaging.md): a case at a catch event waits until it has claimed min messages; at that
moment it also takes every other matching message offered then, up to max, and continues. It never waits to
fill up to max.
"""
import pytest

from prosimos.exceptions import InvalidSimScenarioException
from prosimos.messaging.parser import parse_messages
from prosimos.orchestration.orchestrator import orchestrate
from prosimos.orchestration.protocol import Message, Verdict
from testing_scripts.scripted_engine import ScriptedEngine
from testing_scripts.test_collect import _orders, _settings, at
from testing_scripts.test_race import ITEMS_RACE, ITEMS_RACE_JSON, _race_orders
from testing_scripts.test_race import _settings as _race_settings


def _collecting(collect, change=lambda settings: None):
    def apply(settings):
        settings["messages"]["consume"][0]["collect"] = collect
        change(settings)
    return apply


def _run(tmp_path, collect, published, change=lambda settings: None, more_engines=(), orders_name="Orders"):
    """Order 0 (Orders-0: Pick from 09:00, then waiting at Catch_Items from 09:10) and a Picking engine
    publishing an ItemReady for it per (time, item id) of published, those with the same time in one step; the
    report, the log and the item ids claimed with their times. The orchestrator steps engines due at the same
    time in the order of their names, the orders engine's being orders_name."""
    orders, log = _orders(json_path=_settings(tmp_path, _collecting(collect, change)))
    picking = ScriptedEngine("Picking")
    for time in sorted({time for time, _ in published}):
        items = [item_id for item_time, item_id in published if item_time == time]
        picking.at(at(time), lambda now, items=items: [_item(item_id) for item_id in items])
    report = orchestrate({orders_name: orders, "Picking": picking, **dict(more_engines)}, None, 1)
    item_of = {message.id: message.attributes["item_id"] for message in report.published}
    claimed = [(item_of[message_id], time.strftime("%H:%M")) for message_id, _, time in report.claims]
    return report, log, claimed


def _item(item_id):
    return Message("ItemReady", {"order_id": "Orders-0", "item_id": item_id})


def _discarded(report):
    """The item ids of the messages no case took: once order 0 has moved on, none ever can."""
    item_of = {message.id: message.attributes["item_id"] for message in report.published}
    return [item_of[message_id] for message_id, _, _ in report.discards]


def test_3_3_behaves_exactly_like_collect_3(tmp_path):
    # two items before the order reaches the event, then two at 10:00 at once, then one at 12:00
    published = [("08:30", "i1"), ("08:40", "i2"), ("10:00", "i3"), ("10:00", "i4"), ("12:00", "i5")]

    report, log, claimed = _run(tmp_path, {"min": 3, "max": 3}, published)
    expected_report, expected_log, expected_claimed = _run(tmp_path, 3, published)

    assert claimed == expected_claimed == [("i1", "09:10"), ("i2", "09:10"), ("i3", "10:00")]
    assert log.rows == expected_log.rows
    assert _discarded(report) == _discarded(expected_report) == ["i4", "i5"]


def test_2_5_with_1_waiting_waits_and_when_the_2nd_comes_takes_it_and_up_to_3_others_offered_then(tmp_path):
    # i1 waits before the order reaches the event; five come at 10:00 at once
    published = [("08:30", "i1")] + [("10:00", f"i{number}") for number in range(2, 7)]

    report, log, claimed = _run(tmp_path, {"min": 2, "max": 5}, published)

    assert claimed == [("i1", "09:10"), ("i2", "10:00"), ("i3", "10:00"), ("i4", "10:00"), ("i5", "10:00")]
    assert log.packed() == {0: at("10:00")}
    assert _discarded(report) == ["i6"]


def test_2_5_never_waits_to_fill_up_to_max(tmp_path):
    report, log, claimed = _run(tmp_path, {"min": 2, "max": 5}, [("10:00", "i1"), ("10:30", "i2"), ("11:00", "i3")])

    assert claimed == [("i1", "10:00"), ("i2", "10:30")]
    assert log.packed() == {0: at("10:30")}


def test_all_with_4_waiting_takes_the_4_and_continues(tmp_path):
    published = [("08:30", "i1"), ("08:35", "i2"), ("08:40", "i3"), ("08:45", "i4"), ("12:00", "i5")]

    report, log, claimed = _run(tmp_path, "all", published)

    assert claimed == [(f"i{number}", "09:10") for number in range(1, 5)]
    assert log.packed() == {0: at("09:10")}


def test_all_with_none_waiting_takes_the_next_one_and_the_others_offered_at_that_instant(tmp_path):
    published = [("10:00", "i1"), ("10:00", "i2"), ("10:00", "i3"), ("11:00", "i4")]

    report, log, claimed = _run(tmp_path, "all", published)

    assert claimed == [("i1", "10:00"), ("i2", "10:00"), ("i3", "10:00")]
    assert log.packed() == {0: at("10:00")}


def test_a_min_without_max_takes_every_message_offered_then(tmp_path):
    _, log, claimed = _run(tmp_path, {"min": 2}, [("10:00", f"i{number}") for number in range(1, 7)])

    assert claimed == [(f"i{number}", "10:00") for number in range(1, 7)]
    assert log.packed() == {0: at("10:00")}


def test_the_pool_offers_waiting_messages_oldest_first(tmp_path):
    published = [(f"08:{minute}", f"i{number}") for number, minute in enumerate(range(31, 37), start=1)]

    report, _, claimed = _run(tmp_path, {"min": 1, "max": 5}, published)

    assert claimed == [(f"i{number}", "09:10") for number in range(1, 6)]
    assert _discarded(report) == ["i6"]


def test_the_collection_closes_once_the_offers_of_the_step_that_reached_min_are_done(tmp_path):
    # at 10:00, Picking steps first, then Restock, and the orders engine (named Store here) last, when the order
    # goes on: Restock publishes at the same instant, but after the offers of Picking's step are done
    restock = ScriptedEngine("Restock")
    restock.at(at("10:00"), lambda now: [_item("r1")])

    report, log, claimed = _run(tmp_path, {"min": 2, "max": 5}, [("10:00", "i1"), ("10:00", "i2")],
                                more_engines=[("Restock", restock)], orders_name="Store")

    assert claimed == [("i1", "10:00"), ("i2", "10:00")]
    assert _discarded(report) == ["r1"]


def test_a_message_offered_at_a_later_time_finds_the_collection_closed(tmp_path):
    orders, _ = _orders(json_path=_settings(tmp_path, _collecting({"min": 2, "max": 5})))
    while orders.next_event_time() is not None:
        orders.step()  # order 0 waits at Catch_Items from 09:10

    verdicts = [orders.deliver(_item(item_id), at(time)) for item_id, time in (("i1", "10:00"), ("i2", "10:00"),
                                                                              ("i3", "10:00"), ("i4", "10:30"))]

    assert verdicts[:3] == [Verdict.CLAIMED] * 3 and verdicts[3] is not Verdict.CLAIMED


@pytest.mark.parametrize("published, claimed_items", [
    ([("08:30", "i1"), ("08:40", "i2"), ("08:50", "i3")], [("i1", "09:10"), ("i2", "09:10")]),
    ([("10:00", "i1")], []),
])
def test_min_0_takes_up_to_max_of_the_waiting_messages_and_goes_on_at_once(tmp_path, published, claimed_items):
    report, log, claimed = _run(tmp_path, {"min": 0, "max": 2}, published)

    assert claimed == claimed_items
    assert log.packed() == {0: at("09:10")}


def test_0_passes_straight_on(tmp_path):
    report, log, claimed = _run(tmp_path, 0, [("08:30", "i1")])

    assert claimed == []
    assert log.packed() == {0: at("09:10")}


def test_bounds_read_from_object_attributes(tmp_path):
    def most(settings):
        settings["object_attributes"].append({"name": "most", "type": "continuous", "values": {
            "distribution_name": "fix", "distribution_params": [{"value": 4}]}})

    published = [("10:00", f"i{number}") for number in range(1, 7)]
    _, _, claimed = _run(tmp_path, {"min": {"object_attribute": "items"}, "max": {"object_attribute": "most"}}, published,
                         most)

    assert claimed == [(f"i{number}", "10:00") for number in range(1, 5)]


def test_a_max_below_min_counts_as_min_with_one_warning(tmp_path):
    published = [("10:00", f"i{number}") for number in range(1, 7)]
    report, _, claimed = _run(tmp_path, {"min": {"object_attribute": "items"}, "max": 2}, published)

    assert claimed == [(f"i{number}", "10:00") for number in range(1, 4)]  # items is 3
    assert [warning for _, warning in report.engine_warnings] == [
        "case 0 reaches Catch_Items with no valid max (got 2, min 3); it collects no more than 3"]


def test_a_case_stalled_below_min_reports_how_many_of_min_it_collected(tmp_path):
    report, _, _ = _run(tmp_path, {"min": 2, "max": 5}, [("10:00", "i1")])

    [(_, stalled)] = report.stalled
    assert (stalled.object_id, stalled.collected, stalled.needed) == ("Orders-0", 1, 2)


def test_in_a_race_the_message_branch_wins_when_min_is_reached(tmp_path):
    # order 0 reaches the gateway at 09:10, its deadline is 11:10; one item at 09:20, three at 09:40 at once
    orders, log = _race_orders(ITEMS_RACE, _race_settings(tmp_path, _collecting({"min": 2, "max": 5}),
                                                          base=ITEMS_RACE_JSON), 1)
    shelf = ScriptedEngine("Shelf")
    shelf.at(at("09:20"), lambda now: [_item("a")])
    shelf.at(at("09:40"), lambda now: [_item(item_id) for item_id in ("b", "c", "d")])

    report = orchestrate({"Orders": orders, "Shelf": shelf}, None, 1)

    assert log.times("Pack") == {0: at("09:40")}
    assert log.times("Cancel order") == {}
    assert [time for _, _, time in report.claims] == [at("09:20")] + [at("09:40")] * 3


# load errors

@pytest.mark.parametrize("collect, reason", [
    ({"min": 5, "max": 2}, r"'collect' min 5 is above max 2"),
    ({"min": 1, "max": 0}, r"'collect' min 1 is above max 0"),
    ({"min": {"object_attribute": "items"}, "max": 0}, r"'collect' max is 0 but min is read from items"),
    ({"max": 5}, r"'collect' max needs a min"),
    ({"min": -1}, r"'collect' min must be a whole number of at least 0 or \{\"object_attribute\": <object attribute>\}, "
                  r"got -1"),
    ({"min": 1, "max": "all"}, r"'collect' max must be a whole number of at least 0"),
    ({"min": {"object_atribute": "items"}}, r"collect.min: unknown key 'object_atribute'; allowed: object_attribute"),
    ({"min": 1, "most": 3}, r"collect: unknown key 'most'; allowed: max, min"),
    ("some", r"'collect' must be either a whole number of at least 0, \"all\""),
    (-1, r"'collect' must be either"),
    (2.5, r"'collect' must be either"),
])
def test_an_invalid_collect_is_rejected(tmp_path, collect, reason):
    with pytest.raises(InvalidSimScenarioException, match=r"messages.consume\[0\]" + r"(: |\.)" + reason):
        _orders(json_path=_settings(tmp_path, _collecting(collect)))


@pytest.mark.parametrize("collect", [{"min": 2, "max": 5}, "all", 3])
def test_collect_with_capacity_is_rejected(tmp_path, collect):
    def with_capacity(settings):
        settings["messages"]["consume"][0]["capacity"] = 2

    with pytest.raises(InvalidSimScenarioException, match=r"collect and capacity can't be combined"):
        _orders(json_path=_settings(tmp_path, _collecting(collect, with_capacity)))


@pytest.mark.parametrize("collect", [{"min": 2, "max": 5}, "all", 3])
def test_collect_on_a_start_event_is_rejected(collect):
    section = {"consume": [{"event_id": "Start_Package", "type": "ItemPicked", "collect": collect}]}

    with pytest.raises(InvalidSimScenarioException, match=r"a start event can't collect"):
        parse_messages(section, "testing_scripts/assets/messaging/packages.bpmn", set())
