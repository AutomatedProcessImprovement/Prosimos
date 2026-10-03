"""
Cases started by messages (docs/messaging.md): deliver() offers a message to the waiting cases
first, then to the start event, which creates a new case at the time of the claim.
"""
from datetime import datetime

import pytz

from prosimos.orchestrator import Message, ProcessSpec, ProsimosEngine, Verdict, run_engines
from testing_scripts.scripted_engine import ScriptedEngine

ASSETS = "testing_scripts/assets/messaging"
START = pytz.utc.localize(datetime(2024, 1, 1, 9))


def at(hh_mm):
    hours, minutes = map(int, hh_mm.split(":"))
    return START.replace(hour=hours, minute=minutes)


class _Log:
    def __init__(self):
        self.rows = []

    def writerow(self, header):
        self.header = header

    def writerows(self, rows):
        self.rows.extend(rows)

    def cases(self, activity):
        """case -> enable time of the given activity."""
        return {row[0]: _time(row[2]) for row in self.rows if row[1] == activity}


def _time(logged):
    return logged if isinstance(logged, datetime) else datetime.fromisoformat(logged)


def _warehouse():
    # started by every OrderPlaced for Tartu or Tapa; packs it (1 h) and announces the shipment
    log = _Log()
    spec = ProcessSpec("TartuWarehouse", f"{ASSETS}/tartu_warehouse.bpmn", f"{ASSETS}/tartu_warehouse.json")
    return ProsimosEngine(spec, START, log), log


def _updates():
    # started by an Update; after Handle (10 min) it waits for the next Update, then Close (5 min)
    log = _Log()
    spec = ProcessSpec("Updates", f"{ASSETS}/started_and_waiting.bpmn", f"{ASSETS}/started_and_waiting.json")
    return ProsimosEngine(spec, START, log), log


def _run_until_idle(engine):
    while engine.next_event_time() is not None:
        engine.step()


def order(case_id, city):
    return Message("OrderPlaced", {"case_id": case_id, "city": city}, source="Sales")


def test_a_claimed_start_message_creates_exactly_one_case_starting_at_the_claim_time():
    engine, log = _warehouse()
    assert engine.next_event_time() is None  # nothing to do before a message starts a case

    assert engine.deliver(order("Sales-0", "Tartu"), at("09:30")) is Verdict.CLAIMED
    assert engine.next_event_time() == at("09:30")
    _run_until_idle(engine)

    assert log.cases("Pack order") == {0: at("09:30")}
    assert engine.finish().stalled == []


def test_a_message_that_doesnt_match_the_start_condition_is_discarded():
    engine, log = _warehouse()

    assert engine.deliver(order("Sales-1", "Tallinn"), at("09:30")) is Verdict.DISCARDED
    assert engine.next_event_time() is None and log.rows == []


def test_a_message_both_a_waiting_case_and_the_start_event_accept_goes_to_the_waiting_case():
    engine, log = _updates()
    engine.deliver(Message("Update", source="Feed"), at("09:00"))  # starts case 0
    _run_until_idle(engine)  # case 0 handles it and waits for the next Update from 09:10

    assert engine.deliver(Message("Update", source="Feed"), at("12:00")) is Verdict.CLAIMED
    _run_until_idle(engine)

    assert log.cases("Handle") == {0: at("09:00")}  # no second case was started
    assert log.cases("Close") == {0: at("12:00")}  # case 0 was resumed instead


def test_the_next_message_after_that_starts_a_new_case_with_drawn_case_attributes():
    engine, log = _updates()
    for hh_mm in ("09:00", "12:00", "13:00"):  # start case 0, resume it, start case 1
        engine.deliver(Message("Update", source="Feed"), at(hh_mm))
        _run_until_idle(engine)

    assert log.cases("Handle") == {0: at("09:00"), 1: at("13:00")}
    assert [row[-1] for row in log.rows] == ["high"] * len(log.rows)  # the case attribute column
    assert [case.case_id for case in engine.finish().stalled] == ["Updates-1"]


def test_a_process_with_no_arrivals_runs_only_the_cases_its_messages_start():
    warehouse, log = _warehouse()
    sales = ScriptedEngine("Sales")
    for hh_mm, case_id, city in (("09:10", "Sales-0", "Tartu"), ("09:20", "Sales-1", "Tallinn"),
                                 ("09:40", "Sales-2", "Tapa")):
        sales.publish_at(at(hh_mm), "OrderPlaced", case_id=case_id, city=city)

    report = run_engines({"Sales": sales, "TartuWarehouse": warehouse}, None, 1)

    # one case per Tartu or Tapa order, started when the order was published; the Tallinn order is discarded
    assert log.cases("Pack order") == {0: at("09:10"), 1: at("09:40")}
    assert [message.time for message in report.published if message.type == "Shipment"] == [at("10:10"), at("10:40")]
    assert [process for _, process, _ in report.discards] == ["TartuWarehouse"]
