import csv
import io
import json
import random
from datetime import datetime, timedelta

import numpy as np
import pytz

from prosimos.simulation_engine import SimBPMEnv, execute_full_process
from prosimos.simulation_setup import SimDiffSetup
from testing_scripts.test_batching_stats import get_path

MODEL_FILENAME = "1_task-batch.bpmn"
JSON_FILENAME = "1_task-batch.json"
TOTAL_CASES = 20
SEED = 42


def _build_and_run(bpmn_path, json_path, total_cases, seed, driver):
    """Build a fresh, identically-seeded engine and drive it with `driver`,
    returning the CSV log it produced, one row per line."""
    random.seed(seed)
    np.random.seed(seed)

    sim_setup = SimDiffSetup(bpmn_path, json_path, False, total_cases,
                             pytz.utc.localize(datetime(2024, 1, 1)))

    output = io.StringIO()
    env = SimBPMEnv(sim_setup, None, csv.writer(output))

    driver(env)

    env.log_writer.force_write()
    return output.getvalue().splitlines()


def _drain_all_at_once(env):
    for _ in execute_full_process(env):
        pass


def _drain_step_by_step(env):
    while env.next_event_time() is not None:
        env.step()


def test_step_by_step_matches_normal_run():
    """
    Driving the engine via next_event_time()/step() must produce the exact
    same simulation as draining execute_full_process() in one go -- including
    the final batch of tasks, which only fires once the event queue runs dry
    (see SimBPMEnv.execute_if_any_unexecuted_batch).
    """
    assets_path = get_path()
    bpmn_path = assets_path / MODEL_FILENAME
    json_path = assets_path / JSON_FILENAME

    normal_log = _build_and_run(bpmn_path, json_path, TOTAL_CASES, SEED, _drain_all_at_once)
    stepped_log = _build_and_run(bpmn_path, json_path, TOTAL_CASES, SEED, _drain_step_by_step)

    assert len(normal_log) == len(stepped_log), \
        "Stepped run produced a different number of log rows than a normal run"

    # every column matches except the case id (last column), which is generated
    # via uuid.uuid4()/os.urandom and isn't tied to random/np.random seeding
    for normal_row, stepped_row in zip(normal_log, stepped_log):
        assert normal_row.rsplit(",", 1)[0] == stepped_row.rsplit(",", 1)[0], \
            f"Row mismatch:\n  normal:  {normal_row}\n  stepped: {stepped_row}"


def test_step_by_step_reaches_the_final_batch():
    """
    Sanity check that stepping actually drives the simulation all the way to
    completion, rather than stopping early and happening to match a truncated
    normal run.
    """
    assets_path = get_path()
    bpmn_path = assets_path / MODEL_FILENAME
    json_path = assets_path / JSON_FILENAME

    stepped_log = _build_and_run(bpmn_path, json_path, TOTAL_CASES, SEED, _drain_step_by_step)

    # header row + one row per case
    assert len(stepped_log) == TOTAL_CASES + 1


def _timer_model_engine(tmp_path, total_cases):
    # every case arrives at once, so tasks due at the start wait for the single worker while
    # each finished task starts a 15-minute timer ("15m") that is due later
    assets_path = get_path()
    with open(assets_path / "timer_with_task.json") as f:
        config = json.load(f)
    config["arrival_time_distribution"] = {"distribution_name": "fix", "distribution_params": [{"value": 0}, {"value": 0}, {"value": 1}]}
    config["arrival_time_calendar"] = [{"from": "MONDAY", "to": "SUNDAY", "beginTime": "00:00:00", "endTime": "23:59:59"}]
    json_path = tmp_path / "timer_with_task.json"
    with open(json_path, "w") as f:
        json.dump(config, f)

    sim_setup = SimDiffSetup(assets_path / "timer_with_task.bpmn", json_path, False, total_cases,
                             pytz.utc.localize(datetime(2024, 1, 1, 9, 0)))
    return SimBPMEnv(sim_setup, None, None)


def test_timer_due_later_is_not_handled_before_task_due_earlier(tmp_path):
    engine = _timer_model_engine(tmp_path, total_cases=12)
    names = engine.sim_setup.bpmn_graph.element_info

    handled = []
    while (due := engine.next_event_time()) is not None:
        event = engine.step()
        handled.append((due, names[event.task_id].name))

    tasks_due_at_nine = [i for i, (due, name) in enumerate(handled) if name == "Task 1" and due.hour == 9 and due.minute == 0]
    timer_due_at_two = [i for i, (due, name) in enumerate(handled) if name == "15m" and due.hour == 14 and due.minute == 0]
    assert len(tasks_due_at_nine) == 12 and len(timer_due_at_two) == 1
    assert timer_due_at_two[0] > max(tasks_due_at_nine)
    assert [due for due, _ in handled] == sorted(due for due, _ in handled)


def test_seconds_since_start_counts_whole_days(tmp_path):
    engine = _timer_model_engine(tmp_path, total_cases=1)
    start = engine.sim_setup.start_datetime

    assert engine.simulation_at_from_datetime(start + timedelta(days=3, hours=2)) == 3 * 86400 + 2 * 3600
