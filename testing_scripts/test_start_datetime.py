import random
import time
from datetime import datetime, timedelta

import numpy as np
import pytz

from prosimos.simulation_engine import SimBPMEnv
from prosimos.simulation_setup import SimDiffSetup
from testing_scripts.test_batching_stats import get_path

MODEL_FILENAME = "1_task-batch.bpmn"
JSON_FILENAME = "1_task-batch.json"

# inside the model's arrival calendar (09:00-10:00 daily), so it is used as-is
# rather than being pushed forward to the next arrival window
START = pytz.utc.localize(datetime(2024, 1, 1, 9, 30))


def _build_engine(start_datetime, total_cases=1):
    assets_path = get_path()
    sim_setup = SimDiffSetup(
        assets_path / MODEL_FILENAME, assets_path / JSON_FILENAME, False, total_cases, start_datetime
    )
    return SimBPMEnv(sim_setup, None, None)


def test_engines_built_at_different_moments_share_the_same_clock():
    first = _build_engine(START)
    # stand-in for the time it takes to load a second configuration file
    time.sleep(0.05)
    second = _build_engine(START)

    assert first.sim_setup.start_datetime == second.sim_setup.start_datetime == START

    three_hundred_seconds_in = START + timedelta(seconds=300)
    assert first.simulation_datetime_from(300) == three_hundred_seconds_in
    assert second.simulation_datetime_from(300) == three_hundred_seconds_in


def _event_times(engine):
    times = []
    while (next_time := engine.next_event_time()) is not None:
        times.append(next_time)
        engine.step()
    return times


def test_engines_built_at_different_moments_report_the_same_event_times():
    random.seed(42)
    np.random.seed(42)
    first = _build_engine(START, total_cases=20)
    first_times = _event_times(first)

    time.sleep(0.05)

    random.seed(42)
    np.random.seed(42)
    second = _build_engine(START, total_cases=20)
    second_times = _event_times(second)

    # cases arrive exactly one day apart and are processed in batches of 4, so the first
    # event is the batch that fires when the 4th case arrives, 3 days after the start
    assert first_times[0] == START + timedelta(days=3)
    assert len(first_times) > 1
    assert first_times == second_times
