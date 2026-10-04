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
from prosimos.orchestrator import Message, ProcessSpec, ProsimosEngine, Verdict, run_engines
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


def _settings(tmp_path, change):
    with open(SALES_JSON) as file:
        settings = json.load(file)
    change(settings)
    path = tmp_path / "sales.json"
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


def test_a_race_with_two_branches_waiting_for_messages_is_rejected_for_now(tmp_path):
    second_message = ('<bpmn:intermediateCatchEvent id="Deadline"><bpmn:incoming>Flow_5</bpmn:incoming>'
                      '<bpmn:outgoing>Flow_7</bpmn:outgoing><bpmn:messageEventDefinition /></bpmn:intermediateCatchEvent>')
    two_waiting = _settings(tmp_path, lambda settings: settings["messages"]["consume"].append(
        {"event_id": "Deadline", "type": "Cancellation"}))

    with pytest.raises(InvalidSimScenarioException,
                       match=r"the event-based gateway Race has several branches waiting for a message \(Catch_Shipment, "
                             r"Deadline\); a race with more than one isn't supported yet"):
        SimDiffSetup(_with_branch(tmp_path, second_message), two_waiting, False, 1, START)
