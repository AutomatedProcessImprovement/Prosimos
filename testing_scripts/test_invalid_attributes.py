import pandas as pd
import json
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
            entry_path / "invalid_attributes_stats.csv",
            entry_path / "invalid_attributes_logs.csv",
        ]
        for output_path in output_paths:
            if output_path.exists():
                os.remove(output_path)

    request.addfinalizer(teardown)

    return entry_path


TOTAL_CASES = 250

TEST_CONFIGS = [
    {
        "test_name": "Test event_attribute evaluating to all false conditions should use probabilities",
        "modified_properties": {
            "case_attributes": [],
            "event_attributes": [{
                "event_id": "Activity_0xj7kbs",
                "attributes": [{
                    "name": "attribute_a",
                    "type": "discrete",
                    "values": [{
                        "key": "3",
                        "value": 1
                    }]
                }]
            }]
        }
    }, 
    {
        "test_name": "Test event_attribute defined after gateway should use probabilities",
        "modified_properties": {
            "case_attributes": [],
            "event_attributes": [{
                "event_id": "Activity_1945c0j",
                "attributes": [{
                    "name": "attribute_a",
                    "type": "discrete",
                    "values": [{
                        "key": "2",
                        "value": 1
                    }]
                }]
            }]
        }
    }, 
    {
        "test_name": "Test case_attribute evaluating to all false conditions should use probabilities",
        "modified_properties": {
            "case_attributes": [{
                "name": "attribute_a",
                "type": "discrete",
                "values": [{
                    "key": "3",
                    "value": 1
                }]
            }],
            "event_attributes": []
        }
    }
]

TEST_NAMES = [test['test_name'] for test in TEST_CONFIGS]

def _validate_activities(activities_per_case):
    second_activities = set()
    for activity_sequence in activities_per_case:
        second_activities.add(activity_sequence[1])
        if 'B' in second_activities and 'C' in second_activities:
            return True
    return False

@pytest.mark.parametrize("config", TEST_CONFIGS, ids=TEST_NAMES)
def test_run_simulation(assets_path, config):
    model_path = assets_path / "invalid_attributes_model.bpmn"
    json_path = assets_path / "invalid_attributes.json"
    sim_stats = assets_path / "invalid_attributes_stats.csv"
    sim_logs = assets_path / "invalid_attributes_logs.csv"
    start_string = "2023-06-21 13:22:30.035185+03:00"
    
    _modify_json_parameters(json_path, config["modified_properties"])

    _, sim_results = run_diff_res_simulation(
        start_string, TOTAL_CASES, model_path, json_path, sim_stats, sim_logs
    )

    df = pd.read_csv(sim_logs)
    activities_per_case = df.groupby('case_id')['activity'].apply(list)
        
    assert len(df['case_id'].unique()) == TOTAL_CASES, f"The number of cases ({df['case_id'].unique()}) in the simulation doesn't match the expected {TOTAL_CASES}."
    assert _validate_activities(activities_per_case), "Invalid sequence of activities detected. Second activity must be both 'B' and 'C' at least once."
    assert os.path.isfile(sim_logs), "Simulation log file is not created at the specified path."
    assert os.path.isfile(sim_stats), "Simulation stats file is not created at the specified path."


def _modify_json_parameters(json_path, parameters):
    with open(json_path, "r") as f:
        json_dict = json.load(f)

    for key, value in parameters.items():
        if isinstance(value, dict):
            if key in json_dict and isinstance(json_dict[key], dict):
                for inner_key, inner_value in value.items():
                    json_dict[key][inner_key] = inner_value
            else:
                json_dict[key] = value
        else:
            json_dict[key] = value
    
    with open(json_path, "w+") as json_file:
        json.dump(json_dict, json_file)