import pandas as pd
import os
from pathlib import Path
import pytest
import logging
from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation

LOGGER = logging.getLogger(__name__)


@pytest.fixture
def assets_path(request) -> Path:
    entry_path: Path
    if os.path.basename(os.getcwd()) == "testing_scripts":
        entry_path = Path("assets/event_attributes")
    else:
        entry_path = Path("testing_scripts/assets/event_attributes")

    def teardown():
        output_paths = [
            entry_path / "event_attributes_stats.csv",
            entry_path / "event_attributes_logs.csv",
        ]
        for output_path in output_paths:
            if output_path.exists():
                os.remove(output_path)

    request.addfinalizer(teardown)

    return entry_path


TOTAL_CASES = 250

TEST_CONFIGS = {
    "bpmn_filename": "event_attributes_model.bpmn",
    "json_filename": "event_attributes.json"
}


def contains_all(sequence, pattern):
    return all(element in sequence for element in pattern)


def test_run_simulation(assets_path):
    model_path = assets_path / "event_attributes_model.bpmn"
    json_path = assets_path / "event_attributes.json"
    sim_stats = assets_path / "event_attributes_stats.csv"
    sim_logs = assets_path / "event_attributes_logs.csv"
    start_string = "2023-06-21 13:22:30.035185+03:00"

    _, sim_results = run_diff_res_simulation(
        start_string, TOTAL_CASES, model_path, json_path, sim_stats, sim_logs
    )

    df = pd.read_csv(sim_logs)

    # Test 1: Number of cases
    assert len(df['case_id'].unique()) == TOTAL_CASES, f"The number of cases ({df['case_id'].unique()}) in the simulation doesn't match the expected {TOTAL_CASES}."

    activities_per_case = df.groupby('case_id')['activity'].apply(list)

    pattern_good = ['Generate A', 'Generate B', 'Generate C', 'Use A', 'Use B', 'Use C']
    pattern_bad = ['Generate A', 'Update A', 'Use new A']

    for case_id, activities in activities_per_case.items():
        gen_a_index = activities.index('Generate A')
        next_activity = activities[gen_a_index + 1]

        if next_activity == 'Generate B':
            assert contains_all(activities, pattern_good), f"Case {case_id} does not follow the good pattern ({pattern_good})"
        elif next_activity == 'Update A':
            assert contains_all(activities, pattern_bad), f"Case {case_id} does not follow the bad pattern ({pattern_bad})"

    assert os.path.isfile(sim_logs), "Simulation log file is not created at the specified path."
    assert os.path.isfile(sim_stats), "Simulation stats file is not created at the specified path."
