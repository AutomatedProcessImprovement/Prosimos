"""
Catch events (docs/engine-internals.md, "Implicit merges"): a catch event Prosimos runs as a drawn delay fires once
per arriving token, each token waiting its own duration, without waiting for its other arrows.
"""
from prosimos.control_flow_manager import CustomDatetimeAndSeconds
from prosimos.simulation_setup import SimDiffSetup
from testing_scripts.multiprocess.bpmn.test_end_events import START, _log, _model


def _steps(rows):
    """Case id -> its (activity, start, end) rows, in time order, times as day and hh:mm."""
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(
            (row["activity"], row["start_time"][8:16].replace("T", " "), row["end_time"][8:16].replace("T", " ")))
    return {case: sorted(steps, key=lambda step: (step[1], step[0])) for case, steps in by_case.items()}


def test_after_an_exclusive_choice_the_timer_fires_for_the_branch_taken(tmp_path):
    # today the timer waits for both arrows, and the case hangs forever
    bpmn, json_path = _model(
        tmp_path, {"Start": "startEvent", "Choice": "exclusiveGateway", "Approve": "task", "Reject": "task",
                   "Two_Days": "intermediateCatchEvent:timer", "Ship": "task", "End": "endEvent"},
        [("start", "Start", "Choice"), ("to_approve", "Choice", "Approve"), ("to_reject", "Choice", "Reject"),
         ("approved", "Approve", "Two_Days"), ("rejected", "Reject", "Two_Days"), ("waited", "Two_Days", "Ship"),
         ("shipped", "Ship", "End")],
        probabilities={"Choice": [("to_approve", 0.5), ("to_reject", 0.5)]}, events={"Two_Days": 48})

    steps = _steps(_log(bpmn, json_path, 10, tmp_path))

    assert len(steps) == 10
    assert {tuple(activity for activity, _, _ in case) for case in steps.values()} == {
        ("Approve", "Ship"), ("Reject", "Ship")}
    # case 0 starts on 1 January at 09:00: its branch ends at 10:00, so it ships two days later, on the 3rd
    assert steps["0"][1] == ("Ship", "03 10:00", "03 11:00")


def test_after_a_parallel_split_the_timer_fires_once_per_branch(tmp_path):
    # Task A ends at 10:00, Task B at 11:00: Task C runs 1 h after each, at 11:00 and at 12:00, in each of 20 cases
    bpmn, json_path = _model(
        tmp_path, {"Start": "startEvent", "Split": "parallelGateway", "Task_A": "task", "Task_B": "task",
                   "One_Hour": "intermediateCatchEvent:timer", "Task_C": "task", "End": "endEvent"},
        [("start", "Start", "Split"), ("to_a", "Split", "Task_A"), ("to_b", "Split", "Task_B"),
         ("a_done", "Task_A", "One_Hour"), ("b_done", "Task_B", "One_Hour"), ("waited", "One_Hour", "Task_C"),
         ("c_done", "Task_C", "End")],
        hours={"Task_B": 2}, events={"One_Hour": 1})

    steps = _steps(_log(bpmn, json_path, 20, tmp_path))

    assert len(steps) == 20
    assert all([(activity, start[3:], end[3:]) for activity, start, end in case] == [
        ("Task_A", "09:00", "10:00"), ("Task_B", "09:00", "11:00"), ("Task_C", "11:00", "12:00"),
        ("Task_C", "12:00", "13:00")] for case in steps.values())


def test_each_token_at_a_catch_event_is_taken_by_its_own_firing(tmp_path):
    # two tokens wait at the timer, one on each arrow: each firing takes one of them
    bpmn, json_path = _model(
        tmp_path, {"Start": "startEvent", "Split": "parallelGateway", "One_Hour": "intermediateCatchEvent:timer",
                   "End": "endEvent"},
        [("start", "Start", "Split"), ("one", "Split", "One_Hour"), ("other", "Split", "One_Hour"),
         ("waited", "One_Hour", "End")], events={"One_Hour": 1})
    setup = SimDiffSetup(bpmn, json_path, False, 1, START)
    graph, state = setup.bpmn_graph, setup.initial_state()
    at = CustomDatetimeAndSeconds(0, START)

    enabled, _ = graph.update_process_state(0, "Start", state, at)

    assert [task.task_id for task in enabled] == ["One_Hour", "One_Hour"]
    graph.update_process_state(0, "One_Hour", state, at)
    assert sum(state.tokens.values()) == 1  # the other firing's token is still there
    graph.update_process_state(0, "One_Hour", state, at)
    assert sum(state.tokens.values()) == 0
