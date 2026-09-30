"""
Intermediate throw events: message and none (milestone) throw events pass their token straight on,
taking no time; the other kinds are rejected when the model is loaded.
"""
import csv
import random
from datetime import datetime

import numpy as np
import pytest
import pytz

from prosimos.exceptions import InvalidBpmnModelException
from prosimos.simulation_engine import run_simulation
from prosimos.simulation_setup import SimDiffSetup

ASSETS = "testing_scripts/assets/throw_events"
WITH_THROW = f"{ASSETS}/with_message_throw.bpmn"  # start -> Task A -> message throw -> Task B -> end
WITHOUT_THROW = f"{ASSETS}/without_throw.bpmn"  # start -> Task A -> Task B -> end
JSON = f"{ASSETS}/two_tasks.json"
MESSAGE_DEFINITION = '<bpmn:messageEventDefinition id="Throw_definition" />'
START = "2024-01-01T09:00:00+00:00"


def _run(bpmn, out_path, seed, events_in_log=False):
    random.seed(seed)
    np.random.seed(seed)
    run_simulation(bpmn, JSON, 20, None, out_path, START, events_in_log)
    with open(out_path) as log:
        return list(csv.DictReader(log))


def _variant(tmp_path, definition):
    """The test model with the message definition of its throw event replaced."""
    path = tmp_path / "variant.bpmn"
    with open(WITH_THROW) as model:
        path.write_text(model.read().replace(MESSAGE_DEFINITION, definition))
    return str(path)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_message_throw_event_logs_the_same_as_the_model_without_it(tmp_path, seed):
    with_throw = _run(WITH_THROW, tmp_path / "with.csv", seed)
    without_throw = _run(WITHOUT_THROW, tmp_path / "without.csv", seed)

    assert len(with_throw) == 40
    assert with_throw == without_throw


def test_a_none_throw_event_logs_the_same_as_the_model_without_it(tmp_path):
    milestone = _variant(tmp_path, "")

    assert _run(milestone, tmp_path / "with.csv", 1) == _run(WITHOUT_THROW, tmp_path / "without.csv", 1)


def test_throw_events_are_logged_like_intermediate_events_when_events_are_logged(tmp_path):
    with_throw = _run(WITH_THROW, tmp_path / "with.csv", 1, events_in_log=True)
    without_throw = _run(WITHOUT_THROW, tmp_path / "without.csv", 1, events_in_log=True)

    throws = [row for row in with_throw if row["activity"] == "Throw"]
    assert len(throws) == 20
    for row in throws:
        # takes no time, and happens when Task A of the same case ends
        assert row["enable_time"] == row["start_time"] == row["end_time"]
        assert row["resource"] == "No assigned resource"
        task_a = next(r for r in with_throw if r["case_id"] == row["case_id"] and r["activity"] == "Task A")
        assert row["start_time"] == task_a["end_time"]
    assert [row for row in with_throw if row["activity"] != "Throw"] == without_throw


@pytest.mark.parametrize("definition, kind", [
    ('<bpmn:signalEventDefinition id="Definition" />', "signal"),
    ('<bpmn:escalationEventDefinition id="Definition" />', "escalation"),
    ('<bpmn:compensateEventDefinition id="Definition" />', "compensation"),
    ('<bpmn:linkEventDefinition id="Definition" name="Jump" />', "link"),
])
def test_an_unsupported_throw_event_is_rejected_when_the_model_is_loaded(tmp_path, definition, kind):
    with pytest.raises(InvalidBpmnModelException, match=f"^throw event Throw of kind {kind} is not supported$"):
        SimDiffSetup(_variant(tmp_path, definition), JSON, False, 1, pytz.utc.localize(datetime(2024, 1, 1)))
