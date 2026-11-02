"""
End events (docs/engine-internals.md, "Implicit merges"): an end event takes each token that reaches it, without a
hidden join in front of it, a model may have several end events, and a case is finished when it has no token left.
"""
import csv
import datetime
import json
import random

import numpy as np
import pytz

from prosimos.control_flow_manager import CustomDatetimeAndSeconds
from prosimos.simulation_engine import run_simulation
from prosimos.simulation_properties_parser import parse_simulation_model
from prosimos.orchestration.config import ProcessSpec, SimulationConfig
from prosimos.orchestration.prosimos_engine import ProsimosEngine
from prosimos.orchestration.simulation import run_multiprocess_simulation
from prosimos.simulation_setup import SimDiffSetup
from testing_scripts.multiprocess.orchestrator_settings import one_organization

START = pytz.utc.localize(datetime.datetime(2024, 1, 1, 9))
ALWAYS = [{"from": "MONDAY", "to": "SUNDAY", "beginTime": "00:00:00", "endTime": "23:59:59"}]


def _node(element, tag):
    """A BPMN element, tag "endEvent:message" for one with a message event definition."""
    tag, _, definition = tag.partition(":")
    inside = f"<{definition}EventDefinition/>" if definition else ""
    return f'<{tag} id="{element}" name="{element}">{inside}</{tag}>'


def _model(tmp_path, elements, flows, hours=None, probabilities=None, messages=None, events=None):
    """A process of these elements (id -> BPMN tag) and flows ((id, source, target)), and its JSON settings: every
    task takes hours[task] hours, 1 by default, and has a resource of its own; a catch event in events lasts
    events[event] hours."""
    nodes = "".join(_node(element, tag) for element, tag in elements.items())
    arrows = "".join(f'<sequenceFlow id="{flow}" sourceRef="{source}" targetRef="{target}"/>'
                     for flow, source, target in flows)
    (tmp_path / "model.bpmn").write_text(
        '<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL" id="d" targetNamespace="t">'
        f'<process id="p">{nodes}{arrows}</process></definitions>')
    tasks = [element for element, tag in elements.items() if tag == "task"]
    settings = {
        "resource_profiles": [{"id": "Staff", "name": "Staff", "resource_list": [
            {"id": f"R_{task}", "name": f"R_{task}", "cost_per_hour": 1, "amount": 1, "calendar": "Always",
             "assigned_tasks": [task]} for task in tasks]}],
        "arrival_time_distribution": {"distribution_name": "fix", "distribution_params": [{"value": 86400}]},
        "arrival_time_calendar": ALWAYS,
        "gateway_branching_probabilities": [
            {"gateway_id": gateway, "probabilities": [{"path_id": flow, "value": value} for flow, value in shares]}
            for gateway, shares in (probabilities or {}).items()],
        "task_resource_distribution": [{"task_id": task, "resources": [
            {"resource_id": f"R_{task}", "distribution_name": "fix",
             "distribution_params": [{"value": 3600 * (hours or {}).get(task, 1)}]}]} for task in tasks],
        "resource_calendars": [{"id": "Always", "name": "Always", "time_periods": ALWAYS}],
        "event_distribution": [{"event_id": event, "distribution_name": "fix", "distribution_params": [
            {"value": 3600 * hours}]} for event, hours in (events or {}).items()]}
    if messages is not None:
        settings["messages"] = messages
    (tmp_path / "model.json").write_text(json.dumps(settings))
    return tmp_path / "model.bpmn", tmp_path / "model.json"


# Send Invoice and Ship Package run side by side, each into the one end event
INVOICE_AND_PACKAGE = (
    {"Start": "startEvent", "Split": "parallelGateway", "Send_Invoice": "task", "Ship_Package": "task",
     "End": "endEvent"},
    [("start", "Start", "Split"), ("to_invoice", "Split", "Send_Invoice"), ("to_package", "Split", "Ship_Package"),
     ("invoice_sent", "Send_Invoice", "End"), ("package_shipped", "Ship_Package", "End")])


def test_an_end_event_with_several_arrows_gets_no_hidden_join(tmp_path):
    bpmn, _ = _model(tmp_path, *INVOICE_AND_PACKAGE)

    graph = parse_simulation_model(bpmn)

    assert not [element for element in graph.element_info if element.startswith("or_join_")]
    assert graph.element_info["End"].incoming_flows == ["invoice_sent", "package_shipped"]


def _setup(bpmn, json_path):
    setup = SimDiffSetup(bpmn, json_path, False, 1, START)
    setup.bpmn_graph.watched_elements = {"End"}  # records every pass of the end event
    return setup, setup.bpmn_graph, setup.initial_state()


def test_an_end_event_takes_each_token_and_the_case_keeps_the_others(tmp_path):
    setup, graph, state = _setup(*_model(tmp_path, *INVOICE_AND_PACKAGE))
    at = CustomDatetimeAndSeconds(0, START)
    graph.update_process_state(0, "Start", state, at)

    graph.update_process_state(0, "Send_Invoice", state, at)

    assert [element for _, element, _ in graph.passed_watched_elements] == ["End"]
    assert state.pending_tokens() == ["to_package"]  # Ship Package still runs: the case isn't finished

    graph.update_process_state(0, "Ship_Package", state, at)

    assert [element for _, element, _ in graph.passed_watched_elements] == ["End", "End"]
    assert state.pending_tokens() == []  # no token left: the case is finished


def test_tokens_arriving_at_once_pass_the_end_event_one_after_the_other(tmp_path):
    # both branches of the split lead straight into the end event
    setup, graph, state = _setup(*_model(tmp_path, {"Start": "startEvent", "Split": "parallelGateway",
                                                    "End": "endEvent"},
                                         [("start", "Start", "Split"), ("one", "Split", "End"),
                                          ("other", "Split", "End")]))

    graph.update_process_state(0, "Start", state, CustomDatetimeAndSeconds(0, START))

    assert [element for _, element, _ in graph.passed_watched_elements] == ["End", "End"]
    assert state.pending_tokens() == []


def _log(bpmn, json_path, cases, tmp_path):
    """The log of a run of cases; Prosimos draws from the global generators, so they are seeded first, to make the
    run the same every time."""
    random.seed(1)
    np.random.seed(1)
    run_simulation(bpmn, json_path, cases, log_out_path=tmp_path / "log.csv", starting_at=START.isoformat())
    with open(tmp_path / "log.csv") as file:
        return list(csv.DictReader(file))


def test_a_case_finishes_with_its_last_branch(tmp_path):
    # Send Invoice is done at 10:00, Ship Package at 14:00: the case's last row ends at 14:00
    bpmn, json_path = _model(tmp_path, *INVOICE_AND_PACKAGE, hours={"Ship_Package": 5})

    rows = _log(bpmn, json_path, 1, tmp_path)

    assert sorted((row["activity"], row["end_time"][11:16]) for row in rows) == [
        ("Send_Invoice", "10:00"), ("Ship_Package", "14:00")]


def test_each_case_of_an_exclusive_choice_ends_after_its_one_branch(tmp_path):
    bpmn, json_path = _model(
        tmp_path, {"Start": "startEvent", "Choice": "exclusiveGateway", "Approve": "task", "Reject": "task",
                   "End": "endEvent"},
        [("start", "Start", "Choice"), ("to_approve", "Choice", "Approve"), ("to_reject", "Choice", "Reject"),
         ("approved", "Approve", "End"), ("rejected", "Reject", "End")],
        probabilities={"Choice": [("to_approve", 0.5), ("to_reject", 0.5)]})

    rows = _log(bpmn, json_path, 20, tmp_path)

    by_case = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row["activity"])
    assert len(by_case) == 20 and all(len(activities) == 1 for activities in by_case.values())
    assert {activity for activities in by_case.values() for activity in activities} == {"Approve", "Reject"}


# several end events

INVOICE_SENT_AND_PACKAGE_SHIPPED = (
    {"Start": "startEvent", "Split": "parallelGateway", "Send_Invoice": "task", "Ship_Package": "task",
     "Invoice_Sent": "endEvent", "Package_Shipped": "endEvent"},
    [("start", "Start", "Split"), ("to_invoice", "Split", "Send_Invoice"), ("to_package", "Split", "Ship_Package"),
     ("invoice_sent", "Send_Invoice", "Invoice_Sent"), ("package_shipped", "Ship_Package", "Package_Shipped")])


def test_a_model_may_have_several_end_events(tmp_path):
    bpmn, _ = _model(tmp_path, *INVOICE_SENT_AND_PACKAGE_SHIPPED)

    assert parse_simulation_model(bpmn).end_events == ["Invoice_Sent", "Package_Shipped"]


def test_each_branch_ends_at_its_own_end_event_and_the_case_with_the_last(tmp_path):
    setup, graph, state = _setup(*_model(tmp_path, *INVOICE_SENT_AND_PACKAGE_SHIPPED))
    graph.watched_elements = {"Invoice_Sent", "Package_Shipped"}
    at = CustomDatetimeAndSeconds(0, START)
    graph.update_process_state(0, "Start", state, at)

    graph.update_process_state(0, "Send_Invoice", state, at)

    assert [element for _, element, _ in graph.passed_watched_elements] == ["Invoice_Sent"]
    assert state.pending_tokens() == ["to_package"]

    graph.update_process_state(0, "Ship_Package", state, at)

    assert [element for _, element, _ in graph.passed_watched_elements] == ["Invoice_Sent", "Package_Shipped"]
    assert state.pending_tokens() == []


def test_a_case_with_several_end_events_finishes_with_its_last_branch(tmp_path):
    bpmn, json_path = _model(tmp_path, *INVOICE_SENT_AND_PACKAGE_SHIPPED, hours={"Ship_Package": 5})

    rows = _log(bpmn, json_path, 1, tmp_path)

    assert sorted((row["activity"], row["end_time"][11:16]) for row in rows) == [
        ("Send_Invoice", "10:00"), ("Ship_Package", "14:00")]


def test_a_message_end_event_next_to_another_end_event_publishes_when_its_token_arrives(tmp_path):
    # an order is delivered, publishing Delivered at its message end event, or canceled, at a plain one
    bpmn, json_path = _model(
        tmp_path, {"Start": "startEvent", "Choice": "exclusiveGateway", "Deliver": "task", "Cancel": "task",
                   "Delivered": "endEvent:message", "Cancelled": "endEvent"},
        [("start", "Start", "Choice"), ("to_deliver", "Choice", "Deliver"), ("to_cancel", "Choice", "Cancel"),
         ("delivered", "Deliver", "Delivered"), ("cancelled", "Cancel", "Cancelled")],
        hours={"Deliver": 2}, probabilities={"Choice": [("to_deliver", 0.5), ("to_cancel", 0.5)]},
        messages={"publish": [{"event_id": "Delivered", "type": "Delivered", "attributes": ["object_id"]}]})
    config = SimulationConfig(one_organization([ProcessSpec("Orders", str(bpmn), str(json_path), 20, "order")]),
                              START, 1)

    report = run_multiprocess_simulation(config, tmp_path / "log.csv")

    with open(tmp_path / "log.csv") as file:
        delivered = {f"order-{row['case_id']}": datetime.datetime.fromisoformat(row["end_time"])
                     for row in csv.DictReader(file) if row["activity"] == "Deliver"}
    assert 0 < len(delivered) < 20  # some orders are canceled
    assert sorted((message.attributes["object_id"], message.time) for message in report.published) == \
        sorted(delivered.items())


# a terminate end event beside running work

def _terminating(tmp_path, after_b, hours_b, messages=None):
    """Send Invoice (1 h) reaches a terminate end event at 10:00, ending the case; Ship Package, on the other branch,
    takes hours_b hours and goes on to after_b, then Track Package (1 h)."""
    elements = {"Start": "startEvent", "Split": "parallelGateway", "Send_Invoice": "task",
                "Terminated": "endEvent:terminate", "Ship_Package": "task", after_b[0]: after_b[1],
                "Track_Package": "task", "End": "endEvent"}
    flows = [("start", "Start", "Split"), ("to_invoice", "Split", "Send_Invoice"),
             ("invoice_sent", "Send_Invoice", "Terminated"), ("to_package", "Split", "Ship_Package"),
             ("shipped", "Ship_Package", after_b[0]), ("to_tracking", after_b[0], "Track_Package"),
             ("tracked", "Track_Package", "End")]
    return _model(tmp_path, elements, flows, hours={"Ship_Package": hours_b}, messages=messages)


def test_a_terminate_end_event_ends_the_case_while_another_branch_runs(tmp_path):
    # Ship Package, started at 09:00, is logged as it ran; Track Package, due at 10:30, comes after the terminate
    bpmn, json_path = _terminating(tmp_path, ("Shipped", "intermediateThrowEvent"), 1.5)

    rows = _log(bpmn, json_path, 10, tmp_path)

    by_case = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append((row["activity"], row["end_time"][11:16]))
    assert len(by_case) == 10
    assert all(sorted(steps) == [("Send_Invoice", "10:00"), ("Ship_Package", "10:30")] for steps in by_case.values())


def test_work_the_case_starts_before_the_terminate_goes_on(tmp_path):
    # Ship Package ends at 09:30, before the terminate at 10:00, so Track Package starts then and is logged
    bpmn, json_path = _terminating(tmp_path, ("Shipped", "intermediateThrowEvent"), 0.5)

    rows = _log(bpmn, json_path, 1, tmp_path)

    assert sorted((row["activity"], row["start_time"][11:16]) for row in rows) == [
        ("Send_Invoice", "09:00"), ("Ship_Package", "09:00"), ("Track_Package", "09:30")]


def _published(tmp_path, hours_b):
    tmp_path.mkdir()
    bpmn, json_path = _terminating(
        tmp_path, ("Shipped", "intermediateThrowEvent:message"), hours_b,
        messages={"publish": [{"event_id": "Shipped", "type": "Shipped", "attributes": ["object_id"]}]})
    config = SimulationConfig(one_organization([ProcessSpec("Orders", str(bpmn), str(json_path), 1, "order")]),
                              START, 1)
    report = run_multiprocess_simulation(config, None)
    return [(message.type, message.time.strftime("%H:%M")) for message in report.published], report


def test_a_message_due_after_the_terminate_isnt_published_and_one_due_before_is(tmp_path):
    late, report = _published(tmp_path / "late", 1.5)  # Shipped would be thrown at 10:30
    early, _ = _published(tmp_path / "early", 0.5)  # at 09:30

    assert late == [] and report.stalled == []
    assert early == [("Shipped", "09:30")]


def test_a_branch_waiting_for_a_message_is_dropped_with_its_case(tmp_path):
    # Ship Package's branch waits for a Delivered message that never comes; the terminate ends the case
    bpmn, json_path = _terminating(
        tmp_path, ("Delivered", "intermediateCatchEvent:message"), 0.5,
        messages={"consume": [{"event_id": "Delivered", "type": "Delivered"}]})
    engine = ProsimosEngine(ProcessSpec("Orders", str(bpmn), str(json_path), 1, "order"), START, seed=1)
    while engine.next_event_time() is not None:
        engine.step()

    assert engine.finish().stalled == []
    assert engine._env._parked_events == {} and engine._env._collected == {}
    assert not any(engine._env.all_process_states[0].tokens.values())
