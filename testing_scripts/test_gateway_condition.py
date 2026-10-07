import pandas as pd
import os
from pathlib import Path
import pytest
import logging
from prosimos.simulation_properties_parser import parse_gateway_conditions
from prosimos.warning_logger import warning_logger
from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation

LOGGER = logging.getLogger(__name__)


@pytest.fixture
def assets_path(request) -> Path:
    entry_path: Path
    if os.path.basename(os.getcwd()) == "testing_scripts":
        entry_path = Path("assets/gateway_conditions")
    else:
        entry_path = Path("testing_scripts/assets/gateway_conditions")

    def teardown():
        output_paths = [
            entry_path / "gateway_condition_stats.csv",
            entry_path / "gateway_condition_logs.csv",
            entry_path / "simulation_warnings.txt"
        ]
        for output_path in output_paths:
            if output_path.exists():
                os.remove(output_path)

    request.addfinalizer(teardown)

    return entry_path


TOTAL_CASES = 250

ACTIVITY_DURATIONS = {
    "A": 1,
    "B": 2,
    "C": 3
}


def sum_mapped_values(input_set, mapping_dict, n, reverse=False):
    mapped_values = sorted([mapping_dict[item] for item in input_set], reverse=reverse)
    return sum(mapped_values[:n])


def generate_test_config(test_name, activities, json_filename, bpmn_filename, total_cases, activities_per_case):
    min_activity_duration = sum_mapped_values(activities, ACTIVITY_DURATIONS, activities_per_case[0])
    max_activity_duration = sum_mapped_values(activities, ACTIVITY_DURATIONS, activities_per_case[1], True)

    return {
        "test_name": test_name,
        "json_filename": json_filename,
        "bpmn_filename": bpmn_filename,
        "expected_results": {
            "activities": activities,
            "activities_amount": len(activities),
            "activities_per_case": {
                "min": activities_per_case[0],
                "max": activities_per_case[1]
            },
            "case_duration": {
                "min": total_cases * min_activity_duration,
                "max": total_cases * max_activity_duration
            }
        }
    }


TEST_CONFIGS = [
    (lambda test_name="Test OR with all false conditions should use probabilities",
            bpmn_filename="gateway_condition_or_model.bpmn", json_filename="gateway_all_false_condition.json",
            activities=set(["A", "B", "C"]), activities_per_case=[1, 3], total_cases=TOTAL_CASES:
     generate_test_config(test_name, activities, json_filename, bpmn_filename, total_cases, activities_per_case))(),

    (lambda test_name="Test OR with all true conditions should use all flows",
            bpmn_filename="gateway_condition_or_model.bpmn", json_filename="gateway_all_true_condition.json",
            activities=set(["A", "B", "C"]), activities_per_case=[3, 3], total_cases=TOTAL_CASES:
     generate_test_config(test_name, activities, json_filename, bpmn_filename, total_cases, activities_per_case))(),

    (lambda test_name="Test OR with multiple true conditions should use all true flows",
            bpmn_filename="gateway_condition_or_model.bpmn", json_filename="gateway_multiple_true_condition.json",
            activities=set(["A", "B"]), activities_per_case=[2, 2], total_cases=TOTAL_CASES:
     generate_test_config(test_name, activities, json_filename, bpmn_filename, total_cases, activities_per_case))(),

    (lambda test_name="Test OR with one true condition should use only 1 true flow",
            bpmn_filename="gateway_condition_or_model.bpmn", json_filename="gateway_one_true_condition.json",
            activities=set(["B"]), activities_per_case=[1, 1], total_cases=TOTAL_CASES:
     generate_test_config(test_name, activities, json_filename, bpmn_filename, total_cases, activities_per_case))(),

    (lambda test_name="Test OR with only one condition should use only 1 true condition",
            bpmn_filename="gateway_condition_or_model.bpmn", json_filename="gateway_one_condition.json",
            activities=set(["A"]), activities_per_case=[1, 1], total_cases=TOTAL_CASES:
     generate_test_config(test_name, activities, json_filename, bpmn_filename, total_cases, activities_per_case))(),

    (lambda test_name="Test OR with no conditions should use probabilities",
            bpmn_filename="gateway_condition_or_model.bpmn", json_filename="gateway_no_condition.json",
            activities=set(["A", "B", "C"]), activities_per_case=[1, 3], total_cases=TOTAL_CASES:
     generate_test_config(test_name, activities, json_filename, bpmn_filename, total_cases, activities_per_case))(),

    (lambda test_name="Test XOR with all false conditions should use probabilities and use 1 flow",
            bpmn_filename="gateway_condition_xor_model.bpmn", json_filename="gateway_all_false_condition.json",
            activities=set(["A", "B", "C"]), activities_per_case=[1, 1], total_cases=TOTAL_CASES:
     generate_test_config(test_name, activities, json_filename, bpmn_filename, total_cases, activities_per_case))(),

    (lambda test_name="Test XOR with all true conditions should use probabilities and use 1 flow",
            bpmn_filename="gateway_condition_xor_model.bpmn", json_filename="gateway_all_true_condition.json",
            activities=set(["A", "B", "C"]), activities_per_case=[1, 1], total_cases=TOTAL_CASES:
     generate_test_config(test_name, activities, json_filename, bpmn_filename, total_cases, activities_per_case))(),

    (lambda
         test_name="Test XOR with multiple true conditions should use probabilities on true conditions and use 1 flow",
         bpmn_filename="gateway_condition_xor_model.bpmn", json_filename="gateway_multiple_true_condition.json",
         activities=set(["A", "B"]), activities_per_case=[1, 1], total_cases=TOTAL_CASES:
     generate_test_config(test_name, activities, json_filename, bpmn_filename, total_cases, activities_per_case))(),

    (lambda test_name="Test XOR with one true condition should use 1 true flow",
            bpmn_filename="gateway_condition_xor_model.bpmn", json_filename="gateway_one_true_condition.json",
            activities=set(["B"]), activities_per_case=[1, 1], total_cases=TOTAL_CASES:
     generate_test_config(test_name, activities, json_filename, bpmn_filename, total_cases, activities_per_case))(),

    (lambda test_name="Test XOR with one condition should use 1 true flow",
            bpmn_filename="gateway_condition_xor_model.bpmn", json_filename="gateway_one_condition.json",
            activities=set(["A"]), activities_per_case=[1, 1], total_cases=TOTAL_CASES:
     generate_test_config(test_name, activities, json_filename, bpmn_filename, total_cases, activities_per_case))(),

    (lambda test_name="Test XOR with no conditions should use probabilities and use 1 flow",
            bpmn_filename="gateway_condition_xor_model.bpmn", json_filename="gateway_no_condition.json",
            activities=set(["A", "B", "C"]), activities_per_case=[1, 1], total_cases=TOTAL_CASES:
     generate_test_config(test_name, activities, json_filename, bpmn_filename, total_cases, activities_per_case))()
]
TEST_NAMES = [test['test_name'] for test in TEST_CONFIGS]


@pytest.mark.parametrize("config", TEST_CONFIGS, ids=TEST_NAMES)
def test_run_simulation(assets_path, config):
    json_filename = config["json_filename"]
    bpmn_filename = config["bpmn_filename"]
    expected_results = config["expected_results"]

    model_path = assets_path / bpmn_filename
    json_path = assets_path / json_filename
    sim_stats = assets_path / "gateway_condition_stats.csv"
    sim_logs = assets_path / "gateway_condition_logs.csv"
    start_string = "2023-06-21 13:22:30.035185+03:00"

    _, sim_results = run_diff_res_simulation(
        start_string, TOTAL_CASES, model_path, json_path, sim_stats, sim_logs
    )

    df = pd.read_csv(sim_logs)

    processing_time_total = sim_results.process_kpi_map.processing_time.total
    expected_processing_total = df["activity"].map(ACTIVITY_DURATIONS).sum()

    total_duration = df['activity'].map(ACTIVITY_DURATIONS).sum()
    activities_count_by_case = df.groupby('case_id')['activity'].count()

    errors = []

    try:
        assert processing_time_total == expected_processing_total
    except AssertionError:
        errors.append(f"Processing time total mismatch: got {processing_time_total}, "
                      f"expected {expected_processing_total}")

    try:
        assert df["activity"].unique().size == expected_results["activities_amount"]
    except AssertionError:
        errors.append(f"Activities amount mismatch: got {df['activity'].unique().size}, "
                      f"expected {expected_results['activities_amount']}")

    try:
        assert expected_results["case_duration"]["min"] <= total_duration <= expected_results["case_duration"]["max"]
    except AssertionError:
        errors.append(f"Case duration is out of expected range got {total_duration}, "
                      f"expected from {expected_results['case_duration']['min']} "
                      f"to {expected_results['case_duration']['max']}")

    try:
        assert set(df['activity'].unique()) == expected_results['activities']
    except AssertionError:
        errors.append(f"Activities mismatch: got {set(df['activity'].unique())}, "
                      f"expected {expected_results['activities']}")

    try:
        assert activities_count_by_case.min() == expected_results["activities_per_case"]["min"]
    except AssertionError:
        errors.append(f"Min activities per case mismatch: got {activities_count_by_case.min()}, "
                      f"expected {expected_results['activities_per_case']['min']}")

    try:
        assert activities_count_by_case.max() == expected_results["activities_per_case"]["max"]
    except AssertionError:
        errors.append(f"Max activities per case mismatch: got {activities_count_by_case.max()}, "
                      f"expected {expected_results['activities_per_case']['max']}")

    if errors:
        errors_str = f"Test failed (json: {json_filename}, bpmn: {bpmn_filename})\n" + "\n".join(errors)
        raise AssertionError(errors_str)


class _BranchRules:
    """Stands in for the parsed branch rules: every condition id is a rule."""

    def get_branch_condition_by_id(self, condition_id):
        return condition_id


@pytest.mark.parametrize("probabilities, warned", [
    # probabilities only: no conditions to miss
    ([{"path_id": "Flow_A", "value": 0.3}, {"path_id": "Flow_B", "value": 0.7}], False),
    # some, but not all, arrows have a condition
    ([{"path_id": "Flow_A", "value": 0.3, "condition_id": "rule_a"}, {"path_id": "Flow_B", "value": 0.7}], True),
    # every arrow has a condition
    ([{"path_id": "Flow_A", "value": 0.3, "condition_id": "rule_a"},
      {"path_id": "Flow_B", "value": 0.7, "condition_id": "rule_b"}], False),
])
def test_a_gateway_is_warned_about_only_when_some_of_its_arrows_lack_a_condition(monkeypatch, probabilities, warned):
    monkeypatch.setattr(warning_logger, "warnings_queue", [])

    parse_gateway_conditions([{"gateway_id": "Gateway_1", "probabilities": probabilities}], _BranchRules())

    expected = ["Gateway Gateway_1 is using conditions, but some are missing. Flows without conditions: Flow_B"]
    assert warning_logger.get_all_warnings() == (expected if warned else [])
