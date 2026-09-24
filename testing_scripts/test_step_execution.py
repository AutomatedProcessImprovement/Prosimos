import csv
import io
import random
from datetime import datetime

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

    sim_setup = SimDiffSetup(bpmn_path, json_path, False, total_cases)
    sim_setup.set_starting_datetime(pytz.utc.localize(datetime(2024, 1, 1)))

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
