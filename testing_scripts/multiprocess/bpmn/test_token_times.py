"""
Token times (docs/engine-internals.md, "Implicit merges"): only a parallel or inclusive join takes the latest of its
tokens' times; every other element passes each token at that token's own time, whatever order Prosimos works the
case's tasks out in. Each test runs 20 cases of an AND split into Task A (ending at 10:00) and Task B (ending at
11:00), both leading into the element.
"""
import random
from collections import defaultdict

import numpy as np

from prosimos.simulation_engine import SimBPMEnv, execute_full_process
from prosimos.simulation_setup import SimDiffSetup
from testing_scripts.multiprocess.bpmn.test_end_events import START, _log, _model

CASES = 20


def _split_into(tmp_path, merge, merge_tag, after=("Task_C", "task")):
    """Task A (1 h) and Task B (2 h) after an AND split, both into merge, then 'after', then the end event."""
    elements = {"Start": "startEvent", "Split": "parallelGateway", "Task_A": "task", "Task_B": "task",
                merge: merge_tag, after[0]: after[1], "End": "endEvent"}
    flows = [("start", "Start", "Split"), ("to_a", "Split", "Task_A"), ("to_b", "Split", "Task_B"),
             ("a_done", "Task_A", merge), ("b_done", "Task_B", merge), ("merged", merge, after[0]),
             ("done", after[0], "End")]
    return _model(tmp_path, elements, flows, hours={"Task_B": 2})


def _starts(rows, activity):
    """Case id -> the start times (hh:mm) of activity in it, in order."""
    starts = defaultdict(list)
    for row in rows:
        if row["activity"] == activity:
            starts[row["case_id"]].append(row["start_time"][11:16])
    return {case: sorted(times) for case, times in starts.items()}


def test_an_exclusive_merge_passes_each_token_at_its_own_time(tmp_path):
    rows = _log(*_split_into(tmp_path, "Merge", "exclusiveGateway"), CASES, tmp_path)

    assert _starts(rows, "Task_C") == {str(case): ["10:00", "11:00"] for case in range(CASES)}


def test_a_task_with_two_arrows_runs_for_each_token_at_its_own_time(tmp_path):
    # Task C itself has both arrows; the parser puts a hidden xor_join in front of it
    elements = {"Start": "startEvent", "Split": "parallelGateway", "Task_A": "task", "Task_B": "task",
                "Task_C": "task", "End": "endEvent"}
    flows = [("start", "Start", "Split"), ("to_a", "Split", "Task_A"), ("to_b", "Split", "Task_B"),
             ("a_done", "Task_A", "Task_C"), ("b_done", "Task_B", "Task_C"), ("done", "Task_C", "End")]
    rows = _log(*_model(tmp_path, elements, flows, hours={"Task_B": 2}), CASES, tmp_path)

    assert _starts(rows, "Task_C") == {str(case): ["10:00", "11:00"] for case in range(CASES)}


def _passes(bpmn, json_path, element):
    """Case id -> the times (hh:mm) a run of CASES cases passes element, in order."""
    random.seed(1)
    np.random.seed(1)
    setup = SimDiffSetup(bpmn, json_path, False, CASES, START)
    engine = SimBPMEnv(setup, None, None)
    setup.bpmn_graph.watched_elements = {element}  # after the engine, which watches its publishing events
    for _ in execute_full_process(engine):
        pass
    passes = defaultdict(list)
    for case, _, at in setup.bpmn_graph.passed_watched_elements:
        passes[str(case)].append(at.datetime.strftime("%H:%M"))
    return {case: sorted(times) for case, times in passes.items()}


def test_a_throw_event_passes_each_token_at_its_own_time(tmp_path):
    bpmn, json_path = _split_into(tmp_path, "Throw", "intermediateThrowEvent")

    assert _passes(bpmn, json_path, "Throw") == {str(case): ["10:00", "11:00"] for case in range(CASES)}
    assert _starts(_log(bpmn, json_path, CASES, tmp_path), "Task_C") == {
        str(case): ["10:00", "11:00"] for case in range(CASES)}


def test_an_end_event_takes_each_token_at_its_own_time(tmp_path):
    elements = {"Start": "startEvent", "Split": "parallelGateway", "Task_A": "task", "Task_B": "task",
                "End": "endEvent"}
    flows = [("start", "Start", "Split"), ("to_a", "Split", "Task_A"), ("to_b", "Split", "Task_B"),
             ("a_done", "Task_A", "End"), ("b_done", "Task_B", "End")]

    passes = _passes(*_model(tmp_path, elements, flows, hours={"Task_B": 2}), "End")

    assert passes == {str(case): ["10:00", "11:00"] for case in range(CASES)}


def test_a_parallel_join_still_takes_the_latest_time(tmp_path):
    rows = _log(*_split_into(tmp_path, "Join", "parallelGateway"), CASES, tmp_path)

    assert _starts(rows, "Task_C") == {str(case): ["11:00"] for case in range(CASES)}
