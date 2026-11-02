"""
Start events (docs/messaging.md, "Processes started by messages"): a process started by messages may have several
message start events, each with its own consume entry; a message starts its case at the start event whose entry
accepts it.
"""
import csv
from datetime import timedelta

import pytest

from prosimos.exceptions import InvalidSimScenarioException
from prosimos.orchestration.config import ProcessSpec, SimulationConfig
from prosimos.orchestration.prosimos_engine import load_process
from prosimos.orchestration.simulation import run_multiprocess_simulation
from prosimos.simulation_properties_parser import parse_simulation_model
from prosimos.simulation_setup import SimDiffSetup
from testing_scripts.multiprocess.bpmn.test_end_events import START, _model
from testing_scripts.multiprocess.orchestrator_settings import one_organization
from testing_scripts.multiprocess.scripted_engine import ScriptedEngine

# a repair started by a customer's complaint or by a warranty claim, each handled first in its own way
REPAIR = (
    {"Complaint": "startEvent:message", "Warranty": "startEvent:message", "Handle_Complaint": "task",
     "Check_Warranty": "task", "Merge": "exclusiveGateway", "Repair": "task", "End": "endEvent"},
    [("complained", "Complaint", "Handle_Complaint"), ("claimed", "Warranty", "Check_Warranty"),
     ("handled", "Handle_Complaint", "Merge"), ("checked", "Check_Warranty", "Merge"), ("to_repair", "Merge", "Repair"),
     ("repaired", "Repair", "End")])


def _repairs(tmp_path, consume, sent):
    """The Repair process with these consume entries, sent each (type, attributes) of sent by Customers, an hour
    apart from 09:00; case id -> its activities in order, and the copied 'customer' of each case."""
    bpmn, json_path = _model(tmp_path, *REPAIR, messages={"consume": consume})
    customers = ScriptedEngine("Customers")
    for hours, (message_type, attributes) in enumerate(sent):
        customers.publish_at(START + timedelta(hours=hours), message_type, **attributes)
    config = SimulationConfig(one_organization([ProcessSpec("Repairs", str(bpmn), str(json_path), None, "repair")]),
                              START, 1, extra_processes=["Customers"])
    report = run_multiprocess_simulation(config, tmp_path / "log.csv", {"Customers": customers})
    with open(tmp_path / "log.csv") as file:
        rows = sorted(csv.DictReader(file), key=lambda row: (int(row["case_id"]), row["start_time"]))
    activities = {}
    for row in rows:
        activities.setdefault(row["case_id"], []).append(row["activity"])
    customers_of = {case.object_id: dict((name, value) for name, value, _ in case.attributes).get("customer")
                    for process, case in report.objects if process == "Repairs"}
    return activities, customers_of


def test_a_process_may_have_several_start_events(tmp_path):
    bpmn, _ = _model(tmp_path, *REPAIR)

    assert parse_simulation_model(bpmn).start_events == ["Complaint", "Warranty"]


def test_each_message_starts_its_case_at_the_start_event_that_accepts_it(tmp_path):
    activities, customers_of = _repairs(
        tmp_path, [{"event_id": "Complaint", "type": "CustomerComplaint", "copy": {"customer": "customer"}},
                   {"event_id": "Warranty", "type": "WarrantyClaim", "copy": {"customer": "claimant"}}],
        [("CustomerComplaint", {"customer": "c1"}), ("WarrantyClaim", {"claimant": "c2"}),
         ("CustomerComplaint", {"customer": "c3"})])

    assert activities == {"0": ["Handle_Complaint", "Repair"], "1": ["Check_Warranty", "Repair"],
                          "2": ["Handle_Complaint", "Repair"]}
    assert customers_of == {"repair-0": "c1", "repair-1": "c2", "repair-2": "c3"}  # each entry's own copy


def test_with_one_message_type_each_start_events_condition_decides(tmp_path):
    activities, _ = _repairs(
        tmp_path, [{"event_id": "Complaint", "type": "Claim",
                    "condition": [[{"attribute": "kind", "comparison": "=", "value": "complaint"}]]},
                   {"event_id": "Warranty", "type": "Claim",
                    "condition": [[{"attribute": "kind", "comparison": "=", "value": "warranty"}]]}],
        [("Claim", {"kind": "warranty"}), ("Claim", {"kind": "complaint"})])

    assert activities == {"0": ["Check_Warranty", "Repair"], "1": ["Handle_Complaint", "Repair"]}


def test_of_several_start_events_accepting_a_message_the_entry_listed_first_starts_the_case(tmp_path):
    # Warranty's entry for Claim comes before Complaint's in the JSON, though Complaint is listed first with
    # another type
    activities, _ = _repairs(
        tmp_path, [{"event_id": "Complaint", "type": "Feedback"}, {"event_id": "Warranty", "type": "Claim"},
                   {"event_id": "Complaint", "type": "Claim"}],
        [("Claim", {}), ("Feedback", {})])

    assert activities == {"0": ["Check_Warranty", "Repair"], "1": ["Handle_Complaint", "Repair"]}


# one scheduled start next to message starts

# a kitchen opened every morning by its schedule and by each order; the order's message start is drawn last, so
# it is the last start event parsed
KITCHEN = (
    {"Morning": "startEvent", "Open_Kitchen": "task", "Cook": "task", "End": "endEvent",
     "Order": "startEvent:message"},
    [("opening", "Morning", "Open_Kitchen"), ("opened", "Open_Kitchen", "End"), ("ordered", "Order", "Cook"),
     ("cooked", "Cook", "End")])
ORDERS = {"consume": [{"event_id": "Order", "type": "Order"}]}


def test_the_scheduled_start_is_the_start_event_without_a_consume_entry(tmp_path):
    bpmn, json_path = _model(tmp_path, *KITCHEN, messages=ORDERS)

    setup = SimDiffSetup(bpmn, json_path, False, 1, START)

    assert parse_simulation_model(bpmn).start_events == ["Morning", "Order"]
    assert setup.scheduled_start == setup.bpmn_graph.starting_event == "Morning"


def test_the_schedule_and_the_messages_start_cases_at_their_own_start_events(tmp_path):
    # three mornings, a day apart from 1 January at 09:00, and two orders on the first day
    bpmn, json_path = _model(tmp_path, *KITCHEN, messages=ORDERS)
    customers = ScriptedEngine("Customers")
    for hours in (3, 4):
        customers.publish_at(START + timedelta(hours=hours), "Order")
    config = SimulationConfig(one_organization([ProcessSpec("Kitchen", str(bpmn), str(json_path), 3, "day")]),
                              START, 1, extra_processes=["Customers"])

    run_multiprocess_simulation(config, tmp_path / "log.csv", {"Customers": customers})

    with open(tmp_path / "log.csv") as file:
        rows = sorted((row["start_time"][5:16], row["activity"]) for row in csv.DictReader(file))
    assert rows == [("01-01 09:00", "Open_Kitchen"), ("01-01 12:00", "Cook"), ("01-01 13:00", "Cook"),
                    ("01-02 09:00", "Open_Kitchen"), ("01-03 09:00", "Open_Kitchen")]


def test_a_scheduled_start_next_to_message_starts_needs_arrivals(tmp_path):
    bpmn, json_path = _model(tmp_path, *KITCHEN, messages=ORDERS)

    with pytest.raises(InvalidSimScenarioException,
                       match="needs arrivals: its start event Morning is started by the arrival schedule"):
        load_process(str(bpmn), str(json_path), None, START)


def test_two_scheduled_starts_are_rejected(tmp_path):
    # without a consume entry, both the morning and the evening opening would be started by the schedule
    bpmn, json_path = _model(
        tmp_path, {"Morning": "startEvent", "Evening": "startEvent", "Open_Kitchen": "task", "End": "endEvent"},
        [("morning", "Morning", "Open_Kitchen"), ("evening", "Evening", "Open_Kitchen"),
         ("opened", "Open_Kitchen", "End")])

    with pytest.raises(InvalidSimScenarioException, match="start events Morning, Evening have no consume entry, so "
                                                          "each would be started by the arrival schedule, and a "
                                                          "process has one: model each schedule as its own process"):
        SimDiffSetup(bpmn, json_path, False, 1, START)
