"""
Load errors for what Prosimos doesn't simulate: each element gives an error that names it and says why or what to
draw instead, rather than a crash or a run that silently means something else.
"""
import pytest

from prosimos.exceptions import InvalidBpmnModelException
from prosimos.simulation_properties_parser import parse_simulation_model

NS = 'xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL"'


def _bpmn(tmp_path, process):
    path = tmp_path / "model.bpmn"
    path.write_text(f'<definitions {NS} id="d" targetNamespace="t">{process}</definitions>')
    return path


def test_a_boundary_event_is_rejected(tmp_path):
    # a 30 min timer on the border of Approve would interrupt it and take the path to Escalate
    bpmn = _bpmn(tmp_path, '''<process id="p">
        <startEvent id="Start"/><task id="Approve"/><task id="Escalate"/><endEvent id="End"/>
        <boundaryEvent id="Too_Long" attachedToRef="Approve"><timerEventDefinition/></boundaryEvent>
        <sequenceFlow id="f1" sourceRef="Start" targetRef="Approve"/>
        <sequenceFlow id="f2" sourceRef="Approve" targetRef="End"/>
        <sequenceFlow id="f3" sourceRef="Too_Long" targetRef="Escalate"/>
        <sequenceFlow id="f4" sourceRef="Escalate" targetRef="End"/></process>''')

    with pytest.raises(InvalidBpmnModelException, match="boundary event Too_Long on task Approve is not supported: "
                                                        "Prosimos runs a task in one go, so nothing can interrupt it "
                                                        "or start a path beside it"):
        parse_simulation_model(bpmn)


@pytest.mark.parametrize("definition, kind", [("errorEventDefinition", "error"),
                                              ("escalationEventDefinition", "escalation"),
                                              ("signalEventDefinition", "signal"),
                                              ("compensateEventDefinition", "compensation")])
def test_an_end_event_of_another_kind_than_none_message_or_terminate_is_rejected(tmp_path, definition, kind):
    bpmn = _bpmn(tmp_path, f'''<process id="p">
        <startEvent id="Start"/><task id="Approve"/><endEvent id="Failed"><{definition}/></endEvent>
        <sequenceFlow id="f1" sourceRef="Start" targetRef="Approve"/>
        <sequenceFlow id="f2" sourceRef="Approve" targetRef="Failed"/></process>''')

    with pytest.raises(InvalidBpmnModelException, match=f"^end event Failed of kind {kind} is not supported: only "
                                                        f"none, message and terminate end events are$"):
        parse_simulation_model(bpmn)


@pytest.mark.parametrize("definition", ["", "<messageEventDefinition/>", "<terminateEventDefinition/>"])
def test_none_message_and_terminate_end_events_are_accepted(tmp_path, definition):
    bpmn = _bpmn(tmp_path, f'''<process id="p">
        <startEvent id="Start"/><task id="Approve"/><endEvent id="Done">{definition}</endEvent>
        <sequenceFlow id="f1" sourceRef="Start" targetRef="Approve"/>
        <sequenceFlow id="f2" sourceRef="Approve" targetRef="Done"/></process>''')

    assert parse_simulation_model(bpmn).end_events == ["Done"]
