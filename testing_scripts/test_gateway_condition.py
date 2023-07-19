import pandas as pd
import os
from pathlib import Path
import pytest
import logging
from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation

LOGGER = logging.getLogger(__name__)

TEST_CONFIGS = [
    ("gateway_all_false_condition.json", "gateway_condition_or_model.bpmn", 3),
    ("gateway_all_true_condition.json", "gateway_condition_or_model.bpmn", 3),
    ("gateway_multiple_true_condition.json", "gateway_condition_or_model.bpmn", 2),
    ("gateway_one_true_condition.json", "gateway_condition_or_model.bpmn", 1),
    ("gateway_one_condition.json", "gateway_condition_or_model.bpmn", 1),
    ("gateway_no_condition.json", "gateway_condition_or_model.bpmn", 3),
    ("gateway_all_false_condition.json", "gateway_condition_xor_model.bpmn", 3),
    ("gateway_all_true_condition.json", "gateway_condition_xor_model.bpmn", 3),
    ("gateway_multiple_true_condition.json", "gateway_condition_xor_model.bpmn", 2),
    ("gateway_one_true_condition.json", "gateway_condition_xor_model.bpmn", 1),
    ("gateway_one_condition.json", "gateway_condition_xor_model.bpmn", 1),
    ("gateway_no_condition.json", "gateway_condition_xor_model.bpmn", 3)
]

activity_durations = {
    "A": 1,
    "B": 2,
    "C": 3
}

@pytest.fixture
def assets_path(request) -> Path:
    entry_path: Path
    if os.path.basename(os.getcwd()) == "testing_scripts":
        entry_path = Path("assets/gateway_conditions")
    else:
        entry_path = Path("testing_scripts/assets/gateway_conditions")

    def teardown():
        output_paths = [
            entry_path / "branch_condition_xor_stats.csv",
            entry_path / "branch_condition_xor_logs.csv",
        ]
        for output_path in output_paths:
            if output_path.exists():
                os.remove(output_path)

    request.addfinalizer(teardown)

    return entry_path


@pytest.mark.parametrize("json_filename, bpmn_filename, expected_unique_activities", TEST_CONFIGS)
def test_run_simulation(assets_path, json_filename, bpmn_filename, expected_unique_activities):
    model_path = assets_path / bpmn_filename
    json_path = assets_path / json_filename
    sim_stats = assets_path / "branch_condition_stats.csv"
    sim_logs = assets_path / "branch_condition_logs.csv"
    start_string = "2023-06-21 13:22:30.035185+03:00"

    _, sim_results = run_diff_res_simulation(
        start_string, 100, model_path, json_path, sim_stats, sim_logs
    )

    df = pd.read_csv(sim_logs)

    processing_time_total = sim_results.process_kpi_map.processing_time.total
    expected_processing_total = df["activity"].map(activity_durations).sum()

    df_string = '\t'.join(df.groupby('case_id')['activity'].apply(list).reset_index()
                          .apply(lambda row: f"[{row['case_id']} -> ({', '.join(row['activity']) })]", axis=1))

    LOGGER.info(f"Gateway type: {bpmn_filename}")
    LOGGER.info(f"Condition type: {json_filename}")
    LOGGER.info(f"All routes: {df['activity'].sum()}")
    LOGGER.info(f"By case id: {df_string}")

    assert processing_time_total == expected_processing_total
    assert df["activity"].unique().size == expected_unique_activities
