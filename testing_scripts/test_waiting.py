"""
Waiting (docs/messaging-model.md): a case reaching a catch event listed under 'consume' waits
there; deliver() resumes the matching case at the time of delivery.
"""
import json
import random
from datetime import datetime

import numpy as np
import pytest
import pytz

from prosimos.exceptions import InvalidSimScenarioException
from prosimos.orchestrator import Message, ProcessSpec, ProsimosEngine, Verdict, run_engines
from prosimos.simulation_setup import SimDiffSetup
from prosimos.warning_logger import warning_logger
from testing_scripts.scripted_engine import ScriptedEngine

ASSETS = "testing_scripts/assets/messaging"
SALES = f"{ASSETS}/sales.bpmn"  # Take order -> throw OrderPlaced -> catch Shipment -> Close order
SALES_JSON = f"{ASSETS}/sales.json"  # a Shipment is accepted by the case whose case_id is its order_id
START = pytz.utc.localize(datetime(2024, 1, 1, 9))


def at(hh_mm):
    hours, minutes = map(int, hh_mm.split(":"))
    return START.replace(hour=hours, minute=minutes)


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


def _sales(cases, bpmn=SALES, json_path=SALES_JSON, seed=1):
    random.seed(seed)
    np.random.seed(seed)
    log = _Log()
    return ProsimosEngine(ProcessSpec("Sales", bpmn, json_path, cases), START, log), log


def _run_until_idle(engine):
    while engine.next_event_time() is not None:
        engine.step()


def shipment(order_id):
    return Message("Shipment", {"order_id": order_id}, source="Warehouse")


def _settings(tmp_path, change):
    with open(SALES_JSON) as file:
        settings = json.load(file)
    change(settings)
    path = tmp_path / "sales.json"
    path.write_text(json.dumps(settings))
    return str(path)


def test_a_matching_message_resumes_the_right_case_at_now():
    engine, log = _sales(2)
    _run_until_idle(engine)  # both cases now wait for their shipment

    assert engine.next_event_time() is None
    assert engine.deliver(shipment("Sales-0"), at("12:00")) is Verdict.CLAIMED
    assert engine.next_event_time() == at("12:00")
    _run_until_idle(engine)

    # case 0 continued from the catch event at 12:00; case 1 is still waiting
    assert log.times("Close order") == {0: at("12:00")}


def test_a_message_for_a_case_that_has_not_reached_the_catch_event_is_pending_and_claimed_later():
    engine, log = _sales(1)
    engine.step()  # case 0 starts Take order at 09:00; it reaches the catch event when Take order ends

    assert engine.deliver(shipment("Sales-0"), START) is Verdict.PENDING
    _run_until_idle(engine)
    reached = engine._env._waiting[(0, "Catch_Shipment")].enabled_datetime
    assert engine.deliver(shipment("Sales-0"), reached) is Verdict.CLAIMED


def test_a_message_delivered_during_the_task_before_the_catch_event_is_claimed_only_when_the_case_reaches_it():
    # the warehouse ships order Sales-0 at 09:01, while case 0 is still in Take order
    sales, log = _sales(1)
    warehouse = ScriptedEngine("Warehouse")
    warehouse.publish_at(at("09:01"), "Shipment", order_id="Sales-0")

    report = run_engines({"Sales": sales, "Warehouse": warehouse}, None, 1)

    take_order_ended = next(_time(row[4]) for row in log.rows if row[1] == "Take order")
    assert take_order_ended > at("09:01")
    assert [(process, time) for _, process, time in report.claims] == [("Sales", take_order_ended)]
    assert log.times("Close order") == {0: take_order_ended}


def test_a_message_for_a_finished_case_is_discarded():
    engine, log = _sales(2)
    _run_until_idle(engine)
    engine.deliver(shipment("Sales-0"), at("12:00"))
    _run_until_idle(engine)

    assert engine.deliver(shipment("Sales-0"), at("13:00")) is Verdict.DISCARDED  # case 0 is finished
    assert engine.deliver(shipment("Sales-1"), at("13:00")) is Verdict.CLAIMED  # case 1 still waits


@pytest.mark.parametrize("message", [
    shipment("Sales-7"),  # no such case
    shipment("Billing-0"),  # a case of another process
    Message("Shipment", {}, source="Warehouse"),  # no order_id at all
])
def test_a_message_no_case_can_ever_accept_is_discarded(message):
    engine, _ = _sales(2)
    _run_until_idle(engine)

    assert engine.deliver(message, at("12:00")) is Verdict.DISCARDED


def test_a_condition_on_the_message_alone_decides_a_discard(tmp_path):
    only_from_tartu = _settings(tmp_path, lambda settings: settings["messages"]["consume"][0].update(
        condition=[[{"attribute": "source", "comparison": "=", "value": "TartuWarehouse"}]]))
    engine, _ = _sales(1, json_path=only_from_tartu)
    engine.step()  # case 0 is in Take order, not waiting yet

    assert engine.deliver(Message("Shipment", {}, source="TallinnWarehouse"), START) is Verdict.DISCARDED
    assert engine.deliver(Message("Shipment", {}, source="TartuWarehouse"), START) is Verdict.PENDING


def test_when_several_cases_match_the_one_waiting_longest_claims_it(tmp_path):
    any_shipment = _settings(tmp_path, lambda settings: settings["messages"]["consume"][0].pop("condition"))
    engine, log = _sales(3, json_path=any_shipment, seed=2)  # with this seed case 2 reaches the catch event before case 1
    _run_until_idle(engine)
    reached = {row[0]: _time(row[4]) for row in log.rows if row[1] == "Take order"}  # Take order ends = catch event reached
    longest_first = sorted(reached, key=reached.get)

    resumed = []
    for hour in ("12:00", "13:00", "14:00"):
        assert engine.deliver(shipment("anything"), at(hour)) is Verdict.CLAIMED
        _run_until_idle(engine)
        resumed.extend(case for case, time in log.times("Close order").items() if time == at(hour))
    assert longest_first == [0, 2, 1] and resumed == longest_first


def test_cases_waiting_equally_long_are_resumed_by_case_id(tmp_path):
    # all cases arrive at 09:00 and Take order always takes 10 minutes with two clerks, so cases 0
    # and 1 reach the catch event at the same moment
    def same_moment(settings):
        settings["messages"]["consume"][0].pop("condition")
        settings["arrival_time_distribution"] = {"distribution_name": "fix", "distribution_params": [{"value": 0}]}
        settings["task_resource_distribution"][0]["resources"][0].update(
            distribution_name="fix", distribution_params=[{"value": 600}])

    engine, log = _sales(2, json_path=_settings(tmp_path, same_moment))
    _run_until_idle(engine)
    assert {_time(row[4]) for row in log.rows if row[1] == "Take order"} == {at("09:10")}

    engine.deliver(shipment("anything"), at("12:00"))
    _run_until_idle(engine)
    assert log.times("Close order") == {0: at("12:00")}
    assert sorted(engine._env._waiting) == [(1, "Catch_Shipment")]  # case 1 still waits


def test_other_cases_keep_running_while_one_waits():
    engine, log = _sales(5)
    _run_until_idle(engine)
    reached_catch_event_at = {case: parked.enabled_datetime for (case, _), parked in engine._env._waiting.items()}

    assert sorted(reached_catch_event_at) == [0, 1, 2, 3, 4]
    # case 0 was already waiting when case 4 arrived (its Take order was enabled), and case 4 still
    # went through Take order to its own catch event
    assert reached_catch_event_at[0] < log.times("Take order")[4]


def test_a_shipment_for_sales_0_resumes_case_0_and_not_case_1():
    engine, log = _sales(2)
    _run_until_idle(engine)

    engine.deliver(shipment("Sales-0"), at("12:00"))
    _run_until_idle(engine)

    assert list(log.times("Close order")) == [0]
    assert sorted(engine._env._waiting) == [(1, "Catch_Shipment")]


def test_an_engine_whose_cases_all_wait_still_claims_a_message_published_later():
    sales, log = _sales(2)
    warehouse = ScriptedEngine("Warehouse")
    warehouse.publish_at(at("18:00"), "Shipment", order_id="Sales-1")

    report = run_engines({"Sales": sales, "Warehouse": warehouse}, None, 1)

    # Sales stepped only in the morning, then had no next event while both cases waited, until the
    # 18:00 shipment resumed case 1 (two steps: the catch event, then Close order)
    sales_steps = [time for time, name in report.executed if name == "Sales"]
    assert sales_steps[-2:] == [at("18:00")] * 2 and max(sales_steps[:-2]) < at("12:00")
    assert [(process, time) for _, process, time in report.claims] == [("Sales", at("18:00"))]
    assert log.times("Close order") == {1: at("18:00")}


def test_a_claim_that_moves_a_case_to_a_second_catch_event_lets_a_pending_copy_be_claimed():
    # the invoice for case 0 comes first (09:01) and stays pending: the case waits for its shipment
    # first. The shipment (12:00) resumes the case, which then reaches Catch_Invoice at 12:00 and
    # claims the pooled invoice right after that step
    sales, log = _sales(1, bpmn=f"{ASSETS}/sales_two_waits.bpmn", json_path=f"{ASSETS}/sales_two_waits.json")
    warehouse = ScriptedEngine("Warehouse")
    warehouse.publish_at(at("09:01"), "Invoice", order_id="Sales-0")
    warehouse.publish_at(at("12:00"), "Shipment", order_id="Sales-0")

    report = run_engines({"Sales": sales, "Warehouse": warehouse}, None, 1)
    published = {message.id: message.type for message in report.published}

    assert [(published[message_id], time) for message_id, _, time in report.claims] == [
        ("Shipment", at("12:00")), ("Invoice", at("12:00")),
    ]
    assert report.unclaimed == [] and report.discards == []
    assert log.times("Close order") == {0: at("12:00")}


def test_a_catch_event_not_listed_under_consume_keeps_its_drawn_delay(tmp_path):
    def delay_instead(settings):
        settings["messages"]["consume"] = []
        settings["event_distribution"] = [
            {"event_id": "Catch_Shipment", "distribution_name": "fix", "distribution_params": [{"value": 3600}]}]

    engine, log = _sales(1, json_path=_settings(tmp_path, delay_instead))
    _run_until_idle(engine)

    take_order_ended = next(_time(row[4]) for row in log.rows if row[1] == "Take order")
    assert engine.subscriptions() == []
    assert (log.times("Close order")[0] - take_order_ended).total_seconds() == 3600


def test_a_duration_given_for_a_waiting_catch_event_is_ignored_with_a_warning(tmp_path):
    with_duration = _settings(tmp_path, lambda settings: settings.update(event_distribution=[
        {"event_id": "Catch_Shipment", "distribution_name": "fix", "distribution_params": [{"value": 3600}]}]))
    warning_logger.clear_warnings()

    engine, log = _sales(1, json_path=with_duration)
    _run_until_idle(engine)

    assert warning_logger.get_all_warnings() == ["duration of Catch_Shipment is ignored: it waits for a message"]
    assert engine.next_event_time() is None and "Close order" not in [row[1] for row in log.rows]
    warning_logger.clear_warnings()


def test_a_waiting_catch_event_after_an_event_based_gateway_is_rejected(tmp_path):
    bpmn = tmp_path / "race.bpmn"
    bpmn.write_text("""<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL"><bpmn:process id="P">
      <bpmn:startEvent id="Start"/><bpmn:task id="Take_Order"/><bpmn:eventBasedGateway id="Race"/>
      <bpmn:intermediateCatchEvent id="Catch_Shipment"><bpmn:messageEventDefinition/></bpmn:intermediateCatchEvent>
      <bpmn:intermediateCatchEvent id="Timeout"><bpmn:timerEventDefinition/></bpmn:intermediateCatchEvent>
      <bpmn:intermediateThrowEvent id="Throw_OrderPlaced"><bpmn:messageEventDefinition/></bpmn:intermediateThrowEvent>
      <bpmn:endEvent id="End"/>
      <bpmn:sequenceFlow id="F1" sourceRef="Start" targetRef="Take_Order"/>
      <bpmn:sequenceFlow id="F2" sourceRef="Take_Order" targetRef="Throw_OrderPlaced"/>
      <bpmn:sequenceFlow id="F3" sourceRef="Throw_OrderPlaced" targetRef="Race"/>
      <bpmn:sequenceFlow id="F4" sourceRef="Race" targetRef="Catch_Shipment"/>
      <bpmn:sequenceFlow id="F5" sourceRef="Race" targetRef="Timeout"/>
      <bpmn:sequenceFlow id="F6" sourceRef="Catch_Shipment" targetRef="End"/>
      <bpmn:sequenceFlow id="F7" sourceRef="Timeout" targetRef="End"/>
    </bpmn:process></bpmn:definitions>""")

    with pytest.raises(InvalidSimScenarioException, match="'Catch_Shipment' follows an event-based gateway"):
        SimDiffSetup(str(bpmn), SALES_JSON, False, 1, START)
