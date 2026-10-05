"""
Races at event-based gateways (docs/messaging.md): when a branch after the gateway waits for a message,
every branch is armed; the first to happen wins and the others are canceled.
"""
import csv
import json
from datetime import datetime, timedelta

import pytest
import pytz

from prosimos.exceptions import InvalidSimScenarioException
from prosimos.orchestrator import Message, ProcessSpec, ProsimosEngine, StalledCase, Verdict, run_engines
from prosimos.simulation_engine import run_simulation
from prosimos.simulation_setup import SimDiffSetup
from testing_scripts.scripted_engine import ScriptedEngine

ASSETS = "testing_scripts/assets/messaging"
# Take order -> throw OrderPlaced -> event-based gateway: Shipment (Close order) or 6 h timer (Cancel order)
SALES = f"{ASSETS}/sales_with_deadline.bpmn"
SALES_JSON = f"{ASSETS}/sales_with_deadline.json"
START = pytz.utc.localize(datetime(2024, 1, 1, 9))
DEADLINE = timedelta(hours=6)
# a race timer is due one microsecond after its time, so that a message at exactly that time wins; the
# case continues from then (the timer's own log row shows the real time)
AFTER_TIES = timedelta(microseconds=1)


class _Log:
    def __init__(self):
        self.rows = []

    def writerow(self, header):
        pass

    def writerows(self, rows):
        self.rows.extend(rows)

    def times(self, activity):
        """case -> enable time of the given activity."""
        return {row[0]: _time(row[2]) for row in self.rows if row[1] == activity}


def _time(logged):
    return logged if isinstance(logged, datetime) else datetime.fromisoformat(logged)


def _sales(cases=1, json_path=SALES_JSON):
    log = _Log()
    return ProsimosEngine(ProcessSpec("Sales", SALES, json_path, cases), START, log, seed=1), log


def _run_until_idle(engine):
    while engine.next_event_time() is not None:
        engine.step()


def _reached_gateway(log, case):
    """The case reaches the gateway when Take order ends."""
    return next(_time(row[4]) for row in log.rows if row[0] == case and row[1] == "Take order")


def _settings(tmp_path, change, base=SALES_JSON):
    with open(base) as file:
        settings = json.load(file)
    change(settings)
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(settings))
    return str(path)


def shipment(order_id="Sales-0"):
    return Message("Shipment", {"order_id": order_id}, source="Warehouse")


def test_a_shipment_before_the_deadline_closes_the_order_and_the_timer_never_fires():
    sales, log = _sales()
    warehouse = ScriptedEngine("Warehouse")
    warehouse.publish_at(START.replace(hour=11), "Shipment", order_id="Sales-0")

    report = run_engines({"Sales": sales, "Warehouse": warehouse}, None, 1)

    assert log.times("Close order") == {0: START.replace(hour=11)}
    assert log.times("Cancel order") == {}
    assert report.stalled == []
    # the canceled timer still came off the queue at its time, but was skipped: Sales stepped then and
    # nothing happened
    deadline = _reached_gateway(log, 0) + DEADLINE + AFTER_TIES
    assert (deadline, "Sales") in report.executed
    assert all(_time(row[2]) < deadline for row in log.rows)


def test_no_shipment_before_the_deadline_cancels_the_order_and_a_later_shipment_is_discarded():
    sales, log = _sales()
    _run_until_idle(sales)

    assert log.times("Cancel order") == {0: _reached_gateway(log, 0) + DEADLINE + AFTER_TIES}
    assert log.times("Close order") == {}
    assert sales._env._parked_events == {}  # the waiting record was dropped
    assert sales.deliver(shipment(), _reached_gateway(log, 0) + DEADLINE + timedelta(hours=1)) is Verdict.DISCARDED


def test_each_case_runs_its_own_race():
    sales, log = _sales(cases=2)
    while sales.next_event_time() < START.replace(hour=11):
        sales.step()

    assert sales.deliver(shipment("Sales-1"), START.replace(hour=11)) is Verdict.CLAIMED
    _run_until_idle(sales)

    assert log.times("Close order") == {1: START.replace(hour=11)}
    assert log.times("Cancel order") == {0: _reached_gateway(log, 0) + DEADLINE + AFTER_TIES}


def test_a_gateway_without_a_message_branch_behaves_as_before(tmp_path):
    # without the consume entry, Catch_Shipment is an ordinary catch event with a drawn delay (1 h here),
    # so the gateway is decided at once as before: the shorter branch wins, no message involved
    def no_message_branch(settings):
        settings["messages"]["consume"] = []
        settings["event_distribution"].append(
            {"event_id": "Catch_Shipment", "distribution_name": "fix", "distribution_params": [{"value": 3600}]})

    sales, log = _sales(json_path=_settings(tmp_path, no_message_branch))
    _run_until_idle(sales)

    assert sales.subscriptions() == []
    assert log.times("Close order") == {0: _reached_gateway(log, 0) + timedelta(hours=1)}
    assert log.times("Cancel order") == {}


def test_a_race_timer_of_a_case_ended_by_a_terminate_end_event_is_skipped(tmp_path):
    # after Take order the case splits: one branch races the Shipment against the 6 h timer, the other
    # withdraws the order (2 h) and reaches a terminate end event, ending the whole case; no Shipment comes
    def withdrawal(settings):
        settings["resource_profiles"][0]["resource_list"][0]["assigned_tasks"].append("Withdraw_Order")
        settings["task_resource_distribution"].append({"task_id": "Withdraw_Order", "resources": [
            {"resource_id": "Clerk", "distribution_name": "fix", "distribution_params": [{"value": 7200}]}]})

    log = _Log()
    spec = ProcessSpec("Sales", f"{ASSETS}/sales_race_terminated.bpmn", _settings(tmp_path, withdrawal), 1)
    sales = ProsimosEngine(spec, START, log, seed=1)
    _run_until_idle(sales)

    split = _reached_gateway(log, 0)
    assert [(row[1], _time(row[4])) for row in log.rows if row[1] != "Take order"] == [
        ("Withdraw order", split + timedelta(hours=2))]  # the case ended there; no Close or Cancel order
    assert not any(sales._env.all_process_states[0].tokens.values())
    assert sales.finish().stalled == []
    assert sales._env._races == {}


def _take_order_in_10_minutes(settings):
    """Case 0 then reaches the gateway at 09:10, so the deadline is exactly 15:10."""
    settings["task_resource_distribution"][0]["resources"][0].update(
        distribution_name="fix", distribution_params=[{"value": 600}])


EXACT_DEADLINE = START.replace(hour=15, minute=10)


@pytest.mark.parametrize("warehouse_name", ["Warehouse", "Atelier"])  # steps after / before Sales at a tie
def test_a_shipment_at_exactly_the_deadline_wins(tmp_path, warehouse_name):
    sales, log = _sales(json_path=_settings(tmp_path, _take_order_in_10_minutes))
    warehouse = ScriptedEngine(warehouse_name)
    warehouse.publish_at(EXACT_DEADLINE, "Shipment", order_id="Sales-0")

    report = run_engines({"Sales": sales, warehouse_name: warehouse}, None, 1)

    assert [(process, time) for _, process, time in report.claims] == [("Sales", EXACT_DEADLINE)]
    assert log.times("Close order") == {0: EXACT_DEADLINE}
    assert log.times("Cancel order") == {}


def test_the_timers_log_row_shows_its_real_time(tmp_path):
    log_path = tmp_path / "log.csv"
    run_simulation(SALES, _settings(tmp_path, _take_order_in_10_minutes), 1, None, log_path, START.isoformat(),
                   is_event_added_to_log=True)

    with open(log_path) as file:
        rows = {row["activity"]: row for row in csv.DictReader(file)}
    assert _time(rows["6 h"]["start_time"]) == START.replace(hour=9, minute=10)
    assert _time(rows["6 h"]["end_time"]) == EXACT_DEADLINE  # not a microsecond later
    assert _time(rows["Cancel order"]["enable_time"]) == EXACT_DEADLINE + AFTER_TIES


def test_the_same_seed_gives_the_same_race(tmp_path):
    def race():
        sales, _ = _sales(cases=3)
        warehouse = ScriptedEngine("Warehouse")
        warehouse.publish_at(START.replace(hour=12), "Shipment", order_id="Sales-1")
        return run_engines({"Sales": sales, "Warehouse": warehouse}, None, 1)

    assert race() == race()


def _with_branch(tmp_path, branch_xml):
    """The deadline model with the timer branch replaced by branch_xml (id Deadline)."""
    with open(SALES) as file:
        bpmn = file.read()
    timer = '<bpmn:intermediateCatchEvent id="Deadline" name="6 h"><bpmn:incoming>Flow_5</bpmn:incoming><bpmn:outgoing>Flow_7</bpmn:outgoing><bpmn:timerEventDefinition /></bpmn:intermediateCatchEvent>'
    assert timer in bpmn
    path = tmp_path / "race.bpmn"
    path.write_text(bpmn.replace(timer, branch_xml))
    return str(path)


def test_a_branch_other_than_a_message_catch_event_or_a_timer_is_rejected(tmp_path):
    signal = ('<bpmn:intermediateCatchEvent id="Deadline"><bpmn:incoming>Flow_5</bpmn:incoming>'
              '<bpmn:outgoing>Flow_7</bpmn:outgoing><bpmn:signalEventDefinition /></bpmn:intermediateCatchEvent>')

    with pytest.raises(InvalidSimScenarioException,
                       match="after the event-based gateway Race, which races a branch waiting for a message, every "
                             "branch must be a message catch event or a timer, but Deadline is a signal intermediateCatchEvent"):
        SimDiffSetup(_with_branch(tmp_path, signal), SALES_JSON, False, 1, START)


# Send quote -> event-based gateway: QuoteAccepted (Confirm order) or QuoteRejected (Archive quote); Send quote
# takes 10 minutes and a case arrives every hour, so case 0 reaches the gateway at 09:10 and case 1 at 10:10
QUOTE = f"{ASSETS}/quote_race.bpmn"
QUOTE_JSON = f"{ASSETS}/quote_race.json"
# the same with a third branch: a 3-day expiry timer (Follow up)
QUOTE_WITH_EXPIRY = f"{ASSETS}/quote_race_with_expiry.bpmn"
ELEVEN = START.replace(hour=11)


def _quote(cases=1, json_path=QUOTE_JSON, bpmn_path=QUOTE):
    log = _Log()
    return ProsimosEngine(ProcessSpec("Quote", bpmn_path, json_path, cases), START, log, seed=1), log


def _run_until(engine, time):
    while engine.next_event_time() is not None and engine.next_event_time() < time:
        engine.step()


def answer(message_type, quote_id="Quote-0"):
    return Message(message_type, {"quote_id": quote_id}, source="Customer")


@pytest.mark.parametrize("first, second, taken, not_taken", [
    ("QuoteAccepted", "QuoteRejected", "Confirm order", "Archive quote"),
    ("QuoteRejected", "QuoteAccepted", "Archive quote", "Confirm order"),
])
def test_with_two_message_branches_the_first_claim_wins_and_a_later_message_of_the_other_type_is_discarded(
        first, second, taken, not_taken):
    quote, log = _quote()
    customer = ScriptedEngine("Customer")
    customer.publish_at(ELEVEN, first, quote_id="Quote-0")

    report = run_engines({"Quote": quote, "Customer": customer}, None, 1)

    assert log.times(taken) == {0: ELEVEN}
    assert log.times(not_taken) == {}
    assert report.stalled == []
    assert quote._env._parked_events == {}  # the other branch's waiting record was dropped
    assert quote.deliver(answer(second), START.replace(hour=12)) is Verdict.DISCARDED


def test_two_answers_at_the_same_instant_go_to_the_one_the_orchestrator_offers_first():
    def race():
        quote, log = _quote()
        customer = ScriptedEngine("Customer")
        customer.publish_at(ELEVEN, "QuoteRejected", quote_id="Quote-0")
        customer.publish_at(ELEVEN, "QuoteAccepted", quote_id="Quote-0")
        return run_engines({"Quote": quote, "Customer": customer}, None, 1), log

    report, log = race()

    # Quote is the only subscriber, so the two answers are offered in the order they were published
    types = {message.id: message.type for message in report.published}
    assert [(types[message_id], process, time) for message_id, process, time in report.claims] == [
        ("QuoteRejected", "Quote", ELEVEN)]
    assert log.times("Archive quote") == {0: ELEVEN}
    assert log.times("Confirm order") == {}
    assert race()[0] == report  # the same seed gives the same order


@pytest.mark.parametrize("listed_first, taken, not_taken", [
    ("Catch_Accepted", "Confirm order", "Archive quote"),
    ("Catch_Rejected", "Archive quote", "Confirm order"),  # the JSON order decides, not the event ids
])
def test_a_message_matching_two_branches_takes_the_branch_listed_first_with_one_warning(
        tmp_path, listed_first, taken, not_taken):
    def both_accept_replies(settings):
        consume = settings["messages"]["consume"]
        for entry in consume:
            entry["type"] = "Reply"
        consume.sort(key=lambda entry: entry["event_id"] != listed_first)

    quote, log = _quote(cases=2, json_path=_settings(tmp_path, both_accept_replies, base=QUOTE_JSON))
    _run_until(quote, ELEVEN)

    assert quote.deliver(answer("Reply", "Quote-0"), ELEVEN) is Verdict.CLAIMED
    assert quote.deliver(answer("Reply", "Quote-1"), ELEVEN) is Verdict.CLAIMED
    _run_until_idle(quote)

    assert log.times(taken) == {0: ELEVEN, 1: ELEVEN}
    assert log.times(not_taken) == {}
    assert quote.finish().warnings == [  # once per gateway, though it happened for both cases
        "a Reply message matches several branches of the race at Race (Catch_Accepted, Catch_Rejected); "
        f"it goes to {listed_first}, whose 'consume' entry comes first"]


def test_a_case_waiting_in_a_race_of_two_message_branches_is_stalled_at_the_gateway_with_both_types():
    quote, _ = _quote()
    _run_until_idle(quote)

    assert quote.finish().stalled == [
        StalledCase("Quote-0", "Race", ["QuoteAccepted", "QuoteRejected"], START.replace(hour=9, minute=10))]


def test_with_a_timer_as_well_the_first_of_the_three_branches_wins(tmp_path):
    def expiry(settings):
        settings["task_resource_distribution"].append({"task_id": "Follow_Up", "resources": [
            {"resource_id": "Clerk", "distribution_name": "fix", "distribution_params": [{"value": 300}]}]})
        settings["event_distribution"].append(
            {"event_id": "Expiry", "distribution_name": "fix", "distribution_params": [{"value": 3 * 24 * 3600}]})

    quote, log = _quote(cases=2, json_path=_settings(tmp_path, expiry, base=QUOTE_JSON), bpmn_path=QUOTE_WITH_EXPIRY)
    _run_until(quote, ELEVEN)

    # case 0 is accepted at 11:00; case 1 gets no answer, so its timer fires 3 days after it reached the gateway
    assert quote.deliver(answer("QuoteAccepted", "Quote-0"), ELEVEN) is Verdict.CLAIMED
    _run_until_idle(quote)

    assert log.times("Confirm order") == {0: ELEVEN}
    assert log.times("Archive quote") == {}
    assert log.times("Follow up") == {1: START.replace(hour=10, minute=10) + timedelta(days=3) + AFTER_TIES}
    assert quote.finish().stalled == []


# Capacity and collect inside a race. Orders racing a truck (capacity 2, then Load) against a 4-hour deadline
# (Send by courier); an order arrives every 10 minutes from 09:00 and is packed in a minute, so orders 0, 1 and 2
# reach the gateway at 09:01, 09:11 and 09:21
TRUCK_RACE = f"{ASSETS}/orders_and_trucks_race.bpmn"
TRUCK_RACE_JSON = f"{ASSETS}/orders_and_trucks_race.json"
# orders racing their 3 items (ItemReady, then Pack) against a 2-hour deadline (Cancel order); an order arrives
# every hour from 09:00 and is picked in 10 minutes, so orders 0 and 1 reach the gateway at 09:10 and 10:10
ITEMS_RACE = f"{ASSETS}/orders_and_items_race.bpmn"
ITEMS_RACE_JSON = f"{ASSETS}/orders_and_items_race.json"


def _race_orders(bpmn_path, json_path, cases):
    log = _Log()
    return ProsimosEngine(ProcessSpec("Orders", bpmn_path, json_path, cases), START, log, seed=1), log


def _message_types(report):
    types = {message.id: message.type for message in report.published}
    return lambda entries: [(types[message_id], process, time) for message_id, process, time in entries]


def test_a_truck_of_capacity_2_wins_the_races_of_the_two_orders_it_picks_up_while_a_third_races_on():
    orders, log = _race_orders(TRUCK_RACE, TRUCK_RACE_JSON, 3)
    carrier = ScriptedEngine("Carrier")
    carrier.publish_at(START.replace(minute=30), "Truck", dock="Tartu", case_id="Carrier-0")

    report = run_engines({"Orders": orders, "Carrier": carrier}, None, 1)

    assert log.times("Load") == {0: START.replace(minute=30), 1: START.replace(minute=30)}
    # their timers never fire; order 2 didn't fit, so its race went on and its timer won 4 hours after 09:21
    assert log.times("Send by courier") == {2: START.replace(hour=13, minute=21) + AFTER_TIES}
    assert report.stalled == []


@pytest.fixture
def three_items_per_order():
    """Order 0 gets one item before its deadline and one after; order 1 gets all three before its deadline."""
    shelf = ScriptedEngine("Shelf")
    shelf.publish_at(START.replace(hour=9, minute=30), "ItemReady", order_id="Orders-0", item_id="a")
    for minute, item_id in [(20, "x"), (30, "y"), (40, "z")]:
        shelf.publish_at(START.replace(hour=10, minute=minute), "ItemReady", order_id="Orders-1", item_id=item_id)
    shelf.publish_at(START.replace(hour=14), "ItemReady", order_id="Orders-0", item_id="b")
    return shelf


def test_an_order_whose_timer_fires_after_1_of_3_items_takes_the_timer_branch_and_a_later_item_is_discarded(
        three_items_per_order):
    orders, log = _race_orders(ITEMS_RACE, ITEMS_RACE_JSON, 2)

    report = run_engines({"Orders": orders, "Shelf": three_items_per_order}, None, 1)

    assert log.times("Cancel order") == {0: START.replace(hour=11, minute=10) + AFTER_TIES}
    assert 0 not in log.times("Pack")
    as_types = _message_types(report)
    # the item it got stays bound to it (a claim); the one after its deadline is discarded
    assert ("ItemReady", "Orders", START.replace(hour=9, minute=30)) in as_types(report.claims)
    assert as_types(report.discards) == [("ItemReady", "Orders", START.replace(hour=14))]
    assert report.stalled == []


def test_an_order_that_gets_all_3_items_before_its_timer_takes_the_message_branch(three_items_per_order):
    orders, log = _race_orders(ITEMS_RACE, ITEMS_RACE_JSON, 2)

    report = run_engines({"Orders": orders, "Shelf": three_items_per_order}, None, 1)

    assert log.times("Pack") == {1: START.replace(hour=10, minute=40)}
    assert 1 not in log.times("Cancel order")
    assert [time for _, process, time in _message_types(report)(report.claims) if time.hour == 10] == [
        START.replace(hour=10, minute=20), START.replace(hour=10, minute=30), START.replace(hour=10, minute=40)]


def test_an_order_with_no_items_to_collect_takes_the_message_branch_at_once(tmp_path):
    def no_items(settings):
        settings["case_attributes"][0]["values"]["distribution_params"] = [{"value": 0}]

    orders, log = _race_orders(ITEMS_RACE, _settings(tmp_path, no_items, base=ITEMS_RACE_JSON), 2)
    _run_until_idle(orders)

    assert log.times("Pack") == {0: START.replace(hour=9, minute=10), 1: START.replace(hour=10, minute=10)}
    assert log.times("Cancel order") == {}  # the message branch won, so the timers never fire
    assert orders._env._races == {}


@pytest.mark.parametrize("listed_first, taken, not_taken", [
    ("Catch_Accepted", "Confirm order", "Archive quote"),
    ("Catch_Rejected", "Archive quote", "Confirm order"),  # the JSON order decides, not the event ids
])
def test_of_several_branches_with_nothing_to_collect_the_one_listed_first_wins_for_every_seed(
        tmp_path, listed_first, taken, not_taken):
    def nothing_to_collect(settings):
        # read from a case attribute that is 0 for every case (a fixed 0 is rejected at load)
        settings["case_attributes"] = [{"name": "answers_needed", "type": "continuous",
                                        "values": {"distribution_name": "fix", "distribution_params": [{"value": 0}]}}]
        consume = settings["messages"]["consume"]
        for entry in consume:
            entry["collect"] = {"case_attribute": "answers_needed"}
        consume.sort(key=lambda entry: entry["event_id"] != listed_first)

    json_path = _settings(tmp_path, nothing_to_collect, base=QUOTE_JSON)
    for seed in range(1, 21):  # the branches come off the queue in a random order, which the seed decides
        log = _Log()
        quote = ProsimosEngine(ProcessSpec("Quote", QUOTE, json_path, 2), START, log, seed=seed)
        _run_until_idle(quote)

        assert log.times(taken) == {0: START.replace(hour=9, minute=10), 1: START.replace(hour=10, minute=10)}
        assert log.times(not_taken) == {}
        assert quote.finish().warnings == [  # once per gateway, though it happened for both cases
            "several branches of the race at Race have nothing to collect (Catch_Accepted, Catch_Rejected); "
            f"{listed_first}, whose 'consume' entry comes first, wins"]


def test_a_race_branch_with_a_fixed_collect_of_0_is_rejected(tmp_path):
    def fixed_0(settings):
        settings["messages"]["consume"][1]["collect"] = {"value": 0}

    with pytest.raises(InvalidSimScenarioException,
                       match="Catch_Rejected, after the event-based gateway Race: a race branch with a fixed collect "
                             "of 0 always wins its race; remove the race or the branch"):
        SimDiffSetup(QUOTE, _settings(tmp_path, fixed_0, base=QUOTE_JSON), False, 1, START)
