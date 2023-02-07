import pandas as pd
import numpy as np
from test_discovery import assets_path
from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation
from testing_scripts.test_simulation import _setup_and_write_sim_scenario

basic_columns = [
    "case_id",
    "activity",
    "enable_time",
    "start_time",
    "end_time",
    "resource",
]


def test_present_case_attr_correct_output(assets_path):
    """
    Input: run simulation with writting events to the log file.
            Case attributes' setup defines two additional case_attributes and fixed value REGULAR and 240.

    Output:
    1) verify the number&naming&order of the columns
    2) verify that there are no empty columns
    3) verify the correctness of the case attributes' values
    4) verify that resource names are correctly filled for both general task and event
        (we expect 'No assigned resource' for the event row)
    """

    model_path = assets_path / "timer_with_task.bpmn"

    # provide setup for the case attributes generation
    json_path = assets_path / "timer_with_task.json"
    case_attributes = [
        {
            "name": "client_type",
            "type": "discrete",
            "values": [{"key": "REGULAR", "value": 1}],
        },
        {
            "name": "loan_amount",
            "type": "continuous",
            "values": {
                "distribution_name": "fix",
                "distribution_params": [{"value": 240}, {"value": 0}, {"value": 1}],
            },
        },
    ]
    _setup_and_write_sim_scenario(json_path, case_attributes)

    sim_stats = assets_path / "timer_with_task_stats.csv"
    sim_logs = assets_path / "timer_with_task_logs.csv"

    start_string = "2022-06-21 13:22:30.035185+03:00"

    # ====== ACT ======
    _ = run_diff_res_simulation(
        start_string, 5, model_path, json_path, sim_stats, sim_logs, True
    )

    # ====== ASSERT ======
    df = pd.read_csv(sim_logs)

    # verify the number and naming of the columns
    actual_columns = df.columns.values.tolist()
    case_attr_columns = [
        "client_type",
        "loan_amount",
    ]
    expected_columns = [*basic_columns, *case_attr_columns]
    assert (
        actual_columns == expected_columns
    ), f"Wrong columns' naming. Expected: {expected_columns} but was {actual_columns}"

    # verify that there are no empty columns
    _verify_no_missed_values(df)

    # verify the correctness of the case attributes' values
    is_client_type_regular_for_all = (df["client_type"] == "REGULAR").all()
    assert is_client_type_regular_for_all, f"Expected value for client_type: REGULAR"

    is_loan_amount_eq_to_expected = (df["loan_amount"] == 240).all()
    assert is_loan_amount_eq_to_expected, f"Expected value for loan_amount column: 240"

    # verify that resource names are correctly filled for both general task and event
    _verify_resource_name(df, "Task 1", "Default resource-000001")
    _verify_resource_name(df, "15m", "No assigned resource")


def test_no_case_attr_setup_correct_output(assets_path):
    """
    Input: run simulation with writting events to the log file.
            Case attributes' setup is empty so no additional case attributes are added to the log file.

    Output:
    1) verify the number&naming&order of the columns (only basic columns are present)
    2) verify that there are no empty columns
    """

    model_path = assets_path / "timer_with_task.bpmn"

    # provide setup for the case attributes generation
    json_path = assets_path / "timer_with_task.json"
    _setup_and_write_sim_scenario(json_path)

    sim_stats = assets_path / "timer_with_task_stats.csv"
    sim_logs = assets_path / "timer_with_task_logs.csv"

    start_string = "2022-06-21 13:22:30.035185+03:00"

    # ====== ACT ======
    _ = run_diff_res_simulation(
        start_string, 5, model_path, json_path, sim_stats, sim_logs, True
    )

    # ====== ASSERT ======
    df = pd.read_csv(sim_logs)

    # verify the number and naming of the columns
    actual_columns = df.columns.values.tolist()
    assert (
        actual_columns == basic_columns
    ), f"Wrong columns' naming. Expected: {basic_columns} but was {actual_columns}"

    # verify that there are no empty columns
    _verify_no_missed_values(df)


def _verify_resource_name(df, activity_name, expected_resource_name):
    filtered_rows = df[df["activity"] == activity_name]
    assert filtered_rows.shape[0] == 5  # because we simulate 5 process cases
    is_task_resource_correct = (
        filtered_rows["resource"] == expected_resource_name
    ).all()
    assert (
        is_task_resource_correct
    ), f"Expected value for task '{activity_name}' resource field: {expected_resource_name}"


def _verify_no_missed_values(df: pd.DataFrame):
    df.replace("", np.nan)  # replace empty columns with NaN
    rows_with_nan = df.isna().any(
        axis=1
    )  # detect whether there are rows with empty columns
    no_nan_values = (rows_with_nan == False).all()
    assert (
        no_nan_values
    ), f"The following rows {df[df.isna().any(axis=1)].index.values} have missing columns"
