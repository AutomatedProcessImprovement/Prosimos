"""
Races at event-based gateways (docs/messaging.md): when a branch after the gateway waits for a message,
every branch is armed; the first to happen wins and the others are canceled.
"""
import json
from datetime import datetime, timedelta

import pytest
import pytz

from prosimos.exceptions import InvalidSimScenarioException
from prosimos.orchestrator import Message, ProcessSpec, ProsimosEngine, Verdict, run_engines
from prosimos.simulation_setup import SimDiffSetup
from testing_scripts.scripted_engine import ScriptedEngine

ASSETS = "testing_scripts/assets/messaging"
# Take order -> throw OrderPlaced -> event-based gateway: Shipment (Close order) or 6 h timer (Cancel order)
SALES = f"{ASSETS}/sales_with_deadline.bpmn"
SALES_JSON = f"{ASSETS}/sales_with_deadline.json"
START = pytz.utc.localize(datetime(2024, 1, 1, 9))
DEADLINE = timedelta(hours=6)


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
    deadline = _reached_gateway(log, 0) + DEADLINE
    assert (deadline, "Sales") in report.executed
    assert all(_time(row[2]) < deadline for row in log.rows)


def test_no_shipment_before_the_deadline_cancels_the_order_and_a_later_shipment_is_discarded():
    sales, log = _sales()
    _run_until_idle(sales)

    assert log.times("Cancel order") == {0: _reached_gateway(log, 0) + DEADLINE}
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
    assert log.times("Cancel order") == {0: _reached_gateway(log, 0) + DEADLINE}


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
