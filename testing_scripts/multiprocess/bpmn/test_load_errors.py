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
