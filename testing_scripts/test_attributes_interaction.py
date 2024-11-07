import numpy as np
import pandas as pd
import os
import json
from pathlib import Path
import pytest
import logging
import random
from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 5000)

LOGGER = logging.getLogger(__name__)
TOTAL_CASES = 10


@pytest.fixture
def assets_path(request) -> Path:
    entry_path: Path
    if os.path.basename(os.getcwd()) == "testing_scripts":
        entry_path = Path("assets/attributes_interaction")
    else:
        entry_path = Path("testing_scripts/assets/attributes_interaction")

    def teardown():
        output_paths = [
            entry_path / "attributes_interaction_stats.csv",
            entry_path / "attributes_interaction_logs.csv",
            entry_path / "simulation_warnings.txt"
        ]
        for output_path in output_paths:
            if output_path.exists():
                os.remove(output_path)

    request.addfinalizer(teardown)

    return entry_path


class SoftAssertions:
    def __init__(self):
        self.errors = []

    def add_error(self, message):
        self.errors.append(message)

    def assert_all(self):
        if self.errors:
            error_message = '\n'.join(self.errors)
            raise AssertionError(error_message)


def group_attribute_values(df: pd.DataFrame, attributes: list) -> dict:
    columns_to_keep = ['case_id'] + attributes
    filtered_df = df[columns_to_keep]
    grouped = filtered_df.groupby('case_id').agg({attribute: list for attribute in attributes}).reset_index()

    result_dict = {}
    for _, row in grouped.iterrows():
        case_id = row['case_id']
        attr_values = {attribute: row[attribute] for attribute in attributes}
        result_dict[case_id] = attr_values

    return result_dict


def is_nan(val):
    return isinstance(val, float) and np.isnan(val)


def deep_dict_equal(dict1, dict2):
    for key, value in dict1.items():
        if key not in dict2:
            return False
        if isinstance(value, list) and isinstance(dict2[key], list):
            for v1, v2 in zip(value, dict2[key]):
                if is_nan(v1) and is_nan(v2):
                    continue
                elif v1 != v2:
                    return False
        elif value != dict2[key]:
            return False
    return True


def check_case_pattern(log, expected_values):
    errors = []
    attributes = list(expected_values.keys())
    grouped_attributes = group_attribute_values(log, attributes)

    for case, values in grouped_attributes.items():
        if not deep_dict_equal(values, expected_values):
            for attribute, expected_array in expected_values.items():
                if attribute in values and not np.array_equal(values[attribute], expected_array):
                    errors.append(
                        f"Failed for case {case}. Attribute '{attribute}' mismatch. "
                        f"Expected: {expected_array}, got: {values[attribute]}"
                    )
                elif attribute not in values:
                    errors.append(f"Attribute '{attribute}' not found for case {case}.")

    return errors


def check_attribute_change(log, event_attribute_pairs):
    errors = []

    for pair in event_attribute_pairs:
        event_name = pair["event"]
        attribute_name = pair["attribute"]

        for case_id, case_data in log.groupby('case_id'):
            prev_value = None

            for idx, row in case_data.iterrows():
                current_event = row['activity']
                current_value = row[attribute_name]

                if current_event == event_name and prev_value is not None:
                    if current_value == prev_value:
                        errors.append(
                            f"For case {case_id} at activity {idx}, {attribute_name} did not change after the {event_name} event."
                        )

                prev_value = current_value

    return errors


def soft_assert_all(df, test_config):
    soft_assert = SoftAssertions()

    for assertion_function in test_config["assertion_functions"]:
        errors = assertion_function(df)
        for error in errors:
            soft_assert.add_error(error)

    soft_assert.assert_all()


class ConfigBuilder:
    def __init__(self, test_name):
        self.ACTIVITY_ID = "Activity_1gpdzmu"  # attribute generator id in the model
        self.config = {
            "crisp": {
                "test_name": "[CRISP] " + test_name,
                "modified_properties": {
                    "model_type": "CRISP",
                    "granule_size": None,
                    "global_attributes": [],
                    "case_attributes": [],
                    "event_attributes": [],
                    "resource_calendars": self.get_crisp_calendars()
                },
            },
            "fuzzy": self.convert_to_fuzzy_config(test_name)
        }

        self.config["crisp"]["assertion_functions"] = []
        self.config["fuzzy"]["assertion_functions"] = []

    def convert_to_fuzzy_config(self, test_name):
        fuzzy_config = {
            "test_name": "[FUZZY] " + test_name,
            "modified_properties": {
                "model_type": "FUZZY",
                "granule_size": {
                    "time_unit": "minutes",
                    "value": 1
                },
                "global_attributes": [],
                "case_attributes": [],
                "event_attributes": [],
                "resource_calendars": self.get_fuzzy_calendars()
            }
        }
        return fuzzy_config

    def get_crisp_calendars(self):
        return [
            {
                "id": "sid-30dd3c27-2d47-41da-beaa-997a668ef5b8",
                "name": "default schedule",
                "time_periods": [
                    {
                        "from": "MONDAY",
                        "to": "FRIDAY",
                        "beginTime": "00:00:00.000",
                        "endTime": "23:59:00.000"
                    }
                ]
            }
        ]

    def get_fuzzy_calendars(self):
        calendars = self.get_crisp_calendars()
        for calendar in calendars:
            for time_period in calendar["time_periods"]:
                time_period["probability"] = random.random()

            calendar["workload_ratio"] = [
                {**period, "probability": random.random()} for period in calendar["time_periods"]
            ]
        return calendars

    @staticmethod
    def create_attribute(attr_name, values, distribution_name=None):
        if distribution_name == "discrete":
            if not isinstance(values, dict):
                raise ValueError(
                    "For discrete distributions, values must be a dictionary with label-probability pairs.")

            # Convert the label-probability dict to the required list of dicts format
            distribution_params = [{"key": key, "value": float(prob)} for key, prob in values.items()]

            return {
                "name": attr_name,
                "type": "discrete",
                "values": distribution_params
            }

        # If value is a single number, assume fixed distribution
        if isinstance(values, (int, float)):
            distribution_name = "fix"
            distribution_params = [{"value": float(values)}]
        else:
            # Otherwise, assume value is an array of distribution parameters
            if distribution_name is None:
                raise ValueError("For non-fixed distributions, distribution_name must be provided.")
            distribution_params = values

        return {
            "name": attr_name,
            "type": "continuous",
            "values": {
                "distribution_name": distribution_name,
                "distribution_params": distribution_params
            }
        }

    def add_case_attribute(self, attr_name, value, distribution_name=None):
        attr = self.create_attribute(attr_name, value, distribution_name)
        self.config["crisp"]["modified_properties"]["case_attributes"].append(attr)
        self.config["fuzzy"]["modified_properties"]["case_attributes"].append(attr)
        return self

    def add_global_attribute(self, attr_name, value, distribution_name=None):
        attr = self.create_attribute(attr_name, value, distribution_name)
        self.config["crisp"]["modified_properties"]["global_attributes"].append(attr)
        self.config["fuzzy"]["modified_properties"]["global_attributes"].append(attr)
        return self

    def add_event_attribute(self, attr_name, value, distribution_name=None):
        attr = self.create_attribute(attr_name, value, distribution_name)
        event_attr = {
            "event_id": self.ACTIVITY_ID,
            "attributes": [attr]
        }

        if not self._update_event_attributes(self.config["crisp"]["modified_properties"]["event_attributes"],
                                             event_attr):
            self.config["crisp"]["modified_properties"]["event_attributes"].append(event_attr)

        if not self._update_event_attributes(self.config["fuzzy"]["modified_properties"]["event_attributes"],
                                             event_attr):
            self.config["fuzzy"]["modified_properties"]["event_attributes"].append(event_attr)

        return self

    def _update_event_attributes(self, event_attributes, event_attr):
        for existing_event_attr in event_attributes:
            if existing_event_attr["event_id"] == event_attr["event_id"]:
                existing_event_attr["attributes"].extend(event_attr["attributes"])
                return True
        return False

    def add_global_case_attribute(self, attr_name, global_attr_value, case_attr_value, distribution_name=None):
        self.add_global_attribute(attr_name, global_attr_value, distribution_name)
        self.add_case_attribute(attr_name, case_attr_value, distribution_name)
        return self

    def add_global_event_attribute(self, attr_name, global_attr_value, event_attr_value, distribution_name=None):
        self.add_global_attribute(attr_name, global_attr_value, distribution_name)
        self.add_event_attribute(attr_name, event_attr_value, distribution_name)
        return self

    def add_assertion(self, func):
        self.config["crisp"]["assertion_functions"].append(func)
        self.config["fuzzy"]["assertion_functions"].append(func)
        return self

    def build(self):
        return [self.config["crisp"], self.config["fuzzy"]]


CONFIG = [
    *ConfigBuilder("test single discrete attribute - GLOBAL")
    .add_global_attribute("GLOBAL", {"G":1}, "discrete")
    .add_assertion(lambda log: check_case_pattern(log, {"GLOBAL": ["G", "G", "G"]}))
    .build(),

    *ConfigBuilder("test single discrete attribute - EVENT")
    .add_global_attribute("EVENT", {"E": 1}, "discrete")
    .add_assertion(lambda log: check_case_pattern(log, {"EVENT": ["E", "E", "E"]}))
    .build(),

    *ConfigBuilder("test single discrete attribute - CASE")
    .add_global_attribute("CASE", {"C": 1}, "discrete")
    .add_assertion(lambda log: check_case_pattern(log, {"CASE": ["C", "C", "C"]}))
    .build(),

    *ConfigBuilder("test single attribute creation - GLOBAL")
    .add_global_attribute("GLOBAL", 1)
    .add_assertion(lambda log: check_case_pattern(log, {"GLOBAL": [1, 1, 1]}))
    .build(),

    *ConfigBuilder("test single attribute creation - CASE")
    .add_case_attribute("CASE", 2)
    .add_assertion(lambda log: check_case_pattern(log, {"CASE": [2, 2, 2]}))
    .build(),

    *ConfigBuilder("test single attribute creation - EVENT")
    .add_event_attribute("EVENT", 3)
    .add_assertion(lambda log: check_case_pattern(log, {"EVENT": [np.nan, 3, 3]}))
    .build(),

    *ConfigBuilder("test single attribute creation - GLOBAL CASE")
    .add_global_attribute("G_CASE", 0)
    .add_case_attribute("G_CASE", [{"value": 1}, {"value": 2}, {"value": 3}, {"value": 4}, {"value": 5}], "gamma")
    .add_assertion(lambda log: check_attribute_change(log, [{"attribute": "G_CASE", "event": "START"}]))
    .build(),

    *ConfigBuilder("test single attribute creation - GLOBAL EVENT")
    .add_global_attribute("G_EVENT", 0)
    .add_event_attribute("G_EVENT", [{"value": 1}, {"value": 2}, {"value": 3}, {"value": 4}, {"value": 5}], "gamma")
    .add_assertion(lambda log: check_attribute_change(log, [{"attribute": "G_EVENT", "event": "Generate Attribute"}]))
    .build(),

    *ConfigBuilder("test multiple attributes creation - GLOBAL")
    .add_global_attribute("GLOBAL_1", 1)
    .add_global_attribute("GLOBAL_2", 2)
    .add_assertion(lambda log: check_case_pattern(log, {
        "GLOBAL_1": [1, 1, 1],
        "GLOBAL_2": [2, 2, 2]
    }))
    .build(),

    *ConfigBuilder("test multiple attributes creation - CASE")
    .add_case_attribute("CASE_1", 11)
    .add_case_attribute("CASE_2", 22)
    .add_assertion(lambda log: check_case_pattern(log, {
        "CASE_1": [11, 11, 11],
        "CASE_2": [22, 22, 22]
    }))
    .build(),

    *ConfigBuilder("test multiple attributes creation - EVENT")
    .add_event_attribute("EVENT_1", 111)
    .add_event_attribute("EVENT_2", 222)
    .add_assertion(lambda log: check_case_pattern(log, {
        "EVENT_1": [np.nan, 111, 111],
        "EVENT_2": [np.nan, 222, 222]
    }))
    .build(),

    *ConfigBuilder("test multiple attributes creation - ALL")
    .add_global_attribute("GLOBAL_1", 1)
    .add_case_attribute("CASE_1", 11)
    .add_event_attribute("EVENT_1", 111)
    .add_assertion(lambda log: check_case_pattern(log, {
        "GLOBAL_1": [1, 1, 1],
        "CASE_1": [11, 11, 11],
        "EVENT_1": [np.nan, 111, 111],
    }))
    .build(),

    *ConfigBuilder("test multiple attributes creation - GLOBAL CASE")
    .add_global_attribute("G_CASE_1", 0)
    .add_case_attribute("G_CASE_1", [{"value": 1}, {"value": 2}, {"value": 3}, {"value": 4}, {"value": 5}], "gamma")
    .add_global_attribute("G_CASE_2", 0)
    .add_case_attribute("G_CASE_2", [{"value": 1}, {"value": 2}, {"value": 3}, {"value": 4}, {"value": 5}], "gamma")
    .add_assertion(lambda log: check_attribute_change(log, [
        {"attribute": "G_CASE_1", "event": "START"},
        {"attribute": "G_CASE_2", "event": "START"}
    ]))
    .build(),

    *ConfigBuilder("test multiple attributes creation - GLOBAL EVENT")
    .add_global_attribute("G_EVENT_1", 0)
    .add_event_attribute("G_EVENT_1", [{"value": 1}, {"value": 2}, {"value": 3}, {"value": 4}, {"value": 5}], "gamma")
    .add_global_attribute("G_EVENT_2", 0)
    .add_event_attribute("G_EVENT_2", [{"value": 1}, {"value": 2}, {"value": 3}, {"value": 4}, {"value": 5}], "gamma")
    .add_assertion(lambda log: check_attribute_change(log, [
        {"attribute": "G_EVENT_1", "event": "Generate Attribute"},
        {"attribute": "G_EVENT_2", "event": "Generate Attribute"}
    ]))
    .build(),

    *ConfigBuilder("test attribute interaction - GLOBAL CASE x GLOBAL EVENT")
    .add_global_attribute("G_CASE_EVENT", 0)
    .add_case_attribute("G_CASE_EVENT", 1)
    .add_event_attribute("G_CASE_EVENT", 2)
    .add_assertion(lambda log: check_attribute_change(log, [
        {"attribute": "G_CASE_EVENT", "event": "Generate Attribute"},
        {"attribute": "G_CASE_EVENT", "event": "START"}]))
    .build()
]

TEST_NAMES = [config['test_name'] for config in CONFIG]


@pytest.mark.parametrize("test_config", CONFIG, ids=TEST_NAMES)
def test_attributes_interaction(assets_path, test_config):
    model_path = assets_path / "attributes_interaction_model.bpmn"
    json_path = assets_path / "attributes_interaction.json"
    sim_stats = assets_path / "attributes_interaction_stats.csv"
    sim_logs = assets_path / "attributes_interaction_logs.csv"
    warning_logs = assets_path / "simulation_warnings.txt"
    start_string = "2023-06-21 13:22:30.035185+03:00"

    _modify_json_parameters(json_path, test_config["modified_properties"])

    _, sim_results = run_diff_res_simulation(
        start_string, TOTAL_CASES, model_path, json_path, sim_stats, sim_logs
    )

    df = pd.read_csv(sim_logs)

    soft_assert_all(df, test_config)

    assert os.path.isfile(sim_logs), "Simulation log file is not created at the specified path."
    assert os.path.isfile(sim_stats), "Simulation stats file is not created at the specified path."
    assert os.path.isfile(warning_logs), "Simulation warnings file is not created at the specified path."


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
