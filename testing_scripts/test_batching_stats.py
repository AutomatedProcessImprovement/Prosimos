import os
from pathlib import Path

import pandas as pd

from prosimos.simulation_engine import run_simulation
from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation


def get_path(sub_folder: str = None):
    entry_path: Path
    if os.path.basename(os.getcwd()) == "testing_scripts":
        entry_path = Path("assets")
    else:
        entry_path = Path("testing_scripts/assets")

    return entry_path / sub_folder if sub_folder is not None else entry_path


def remove_files(files_to_delete):
    for file in files_to_delete:
        if file.exists():
            os.remove(file)


def test_batching_id_saved_to_log():
    # ====== ARRANGE ======
    assets_path = get_path()

    model_path = assets_path / "1_task-batch.bpmn"
    json_path = assets_path / "1_task-batch.json"
    sim_stats = assets_path / "1_task-batch_stats.csv"
    sim_logs = assets_path / "1_task-batch_logs.csv"

    start_string = "2022-06-21 13:22:30.035185+03:00"

    # ====== ACT ======
    _, _ = run_diff_res_simulation(
        start_string, 10, model_path, json_path, sim_stats, sim_logs, True
    )

    # ====== ASSERT ======
    df = pd.read_csv(sim_logs)
    # 1. Group by "case_id" and ensure 10 cases exist
    grouped_by_case_id = df.groupby(by="case_id")["case_id"]
    assert (
            grouped_by_case_id.count().size == 10
    ), "The total number of simulated cases does not equal to the setup number"

    # 2. Check if "batch_id" column exists in output event log
    assert "batch_id" in df.columns, "Column 'batch_id' is missing from the log"

    # 3. Check if "batch_id" column has no None (NaN) values
    assert df["batch_id"].notna().all(), "Column 'batch_id' contains None or NaN values"

    # 4. Verify batch_id grouping conditions
    expected_case_groups = [
        [0, 1, 2, 3],  # Group 1
        [4, 5, 6, 7],  # Group 2
        [8, 9]  # Group 3
    ]
    batch_ids_per_group = [df[df["case_id"].isin(group)]["batch_id"].unique() for group in expected_case_groups]

    # Check that each group has exactly one unique batch_id
    for i, batch_ids in enumerate(batch_ids_per_group):
        assert len(batch_ids) == 1, f"Case group {expected_case_groups[i]} has inconsistent batch_id values"

    # Check that each group has a different batch_id from the previous one
    for i in range(len(batch_ids_per_group) - 1):
        assert batch_ids_per_group[i][0] != batch_ids_per_group[i + 1][0], \
            f"Batch ID for case group {expected_case_groups[i]} is the same as the next group"

    remove_files([sim_logs, sim_stats])


def test_batching_returned_but_not_saved_in_log():
    # ====== ARRANGE ======
    assets_path = get_path()

    model_path = assets_path / "1_task-batch.bpmn"
    json_path = assets_path / "1_task-batch.json"

    # ====== ACT ======
    _, sim_out = run_simulation(model_path, json_path, 10)

    for trace in sim_out.trace_list:
        for evt in trace.event_list:
            assert evt.batch_id is not None, "Batch ID cannot be None"


def test_no_batching_saved_to_log_if_no_batch_in_model():
    # ====== ARRANGE ======
    assets_path = get_path("attributes_interaction")

    model_path = assets_path / "attributes_interaction_model.bpmn"
    json_path = assets_path / "attributes_interaction.json"
    sim_stats = assets_path / "attributes_interaction_stats.csv"
    sim_logs = assets_path / "attributes_interaction_logs.csv"
    sim_warnings = assets_path / "simulation_warnings.txt"

    start_string = "2022-06-21 13:22:30.035185+03:00"

    # ====== ACT ======
    _, sim_out = run_diff_res_simulation(
        start_string, 10, model_path, json_path, sim_stats, sim_logs, True
    )

    # ====== ASSERT ======
    df = pd.read_csv(sim_logs)
    # 1. Group by "case_id" and ensure 10 cases exist
    grouped_by_case_id = df.groupby(by="case_id")["case_id"]
    assert (
            grouped_by_case_id.count().size == 10
    ), "The total number of simulated cases does not equal to the setup number"

    # 2. Check if "batch_id" column exists in output event log
    assert "batch_id" not in df.columns, "Column 'batch_id' is wrongly added to the log"

    remove_files([sim_logs, sim_stats, sim_warnings])
