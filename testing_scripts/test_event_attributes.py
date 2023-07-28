import pandas as pd
import os
from pathlib import Path
import pytest
import logging
from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation
from prosimos.event_attributes import EventAttribute, EVENT_ATTR_TYPE


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


TOTAL_CASES = 50


def contains_all(sequence, pattern):
    return all(element in sequence for element in pattern)


def test_validate_flows_based_on_event_attributes(assets_path):
    model_path = assets_path / "event_attributes_model.bpmn"
    json_path = assets_path / "event_attributes.json"
    sim_stats = assets_path / "event_attributes_stats.csv"
    sim_logs = assets_path / "event_attributes_logs.csv"
    start_string = "2023-06-21 13:22:30.035185+03:00"

    _, sim_results = run_diff_res_simulation(
        start_string, TOTAL_CASES, model_path, json_path, sim_stats, sim_logs
    )

    df = pd.read_csv(sim_logs)
    activities_per_case = df.groupby('case_id')['activity'].apply(list)

    pattern_good = ['Generate A', 'Generate B', 'Generate C', 'Use A', 'Use B', 'Use C']
    pattern_bad = ['Generate A', 'Update A', 'Use new A']

    assert len(df['case_id'].unique()) == TOTAL_CASES, f"The number of cases ({df['case_id'].unique()}) in the simulation doesn't match the expected {TOTAL_CASES}."

    for case_id, activities in activities_per_case.items():
        gen_a_index = activities.index('Generate A')
        next_activity = activities[gen_a_index + 1]

        if next_activity == 'Generate B':
            assert contains_all(activities, pattern_good), f"Case {case_id} does not follow the good pattern ({pattern_good})"
        elif next_activity == 'Update A':
            assert contains_all(activities, pattern_bad), f"Case {case_id} does not follow the bad pattern ({pattern_bad})"

    assert os.path.isfile(sim_logs), "Simulation log file is not created at the specified path."
    assert os.path.isfile(sim_stats), "Simulation stats file is not created at the specified path."


EXPRESSION_TEST_CONFIGS = [
    {
        "test_name": "Addition Test",
        "event_attributes": {
            "x": 5,
            "y": 10
        },
        "expression": "x + y",
        "expected_value": 15
    },
    {
        "test_name": "Subtraction Test",
        "event_attributes": {
            "x": 15,
            "y": 10
        },
        "expression": "x - y",
        "expected_value": 5
    },
    {
        "test_name": "Multiplication Test",
        "event_attributes": {
            "x": 5,
            "y": 10
        },
        "expression": "x * y",
        "expected_value": 50
    },
    {
        "test_name": "Division Test",
        "event_attributes": {
            "x": 10,
            "y": 5
        },
        "expression": "x / y",
        "expected_value": 2
    },
    {
        "test_name": "Unary Negation Test",
        "event_attributes": {
            "x": 5
        },
        "expression": "-x",
        "expected_value": -5
    },
    {
        "test_name": "Equality Test",
        "event_attributes": {
            "x": 5,
            "y": 5
        },
        "expression": "x == y",
        "expected_value": True
    },
    {
        "test_name": "Inequality Test",
        "event_attributes": {
            "x": 5,
            "y": 10
        },
        "expression": "x != y",
        "expected_value": True
    },
    {
        "test_name": "Less Than Test",
        "event_attributes": {
            "x": 5,
            "y": 10
        },
        "expression": "x < y",
        "expected_value": True
    },
    {
        "test_name": "Less Than or Equal Test",
        "event_attributes": {
            "x": 5,
            "y": 5
        },
        "expression": "x <= y",
        "expected_value": True
    },
    {
        "test_name": "Greater Than Test",
        "event_attributes": {
            "x": 10,
            "y": 5
        },
        "expression": "x > y",
        "expected_value": True
    },
    {
        "test_name": "Greater Than or Equal Test",
        "event_attributes": {
            "x": 10,
            "y": 10
        },
        "expression": "x >= y",
        "expected_value": True
    },
    {
        "test_name": "Not Test",
        "event_attributes": {
            "x": False
        },
        "expression": "not x",
        "expected_value": True
    },
    {
        "test_name": "And Test",
        "event_attributes": {
            "x": True,
            "y": False
        },
        "expression": "x and y",
        "expected_value": False
    },
    {
        "test_name": "Or Test",
        "event_attributes": {
            "x": True,
            "y": False
        },
        "expression": "x or y",
        "expected_value": True
    },
    {
        "test_name": "Division by Zero Test",
        "event_attributes": {
            "x": 10,
            "y": 0
        },
        "expression": "x / y",
        "expected_value": None
    },
    {
        "test_name": "Nonexistent Attribute Test",
        "event_attributes": {
            "x": 5
        },
        "expression": "x + y",
        "expected_value": None
    },
    {
        "test_name": "String Concatenation Test",
        "event_attributes": {
            "x": "Hello",
            "y": " World"
        },
        "expression": "x + y",
        "expected_value": "Hello World"
    },
    {
        "test_name": "String Multiplication Test",
        "event_attributes": {
            "x": "Hello",
            "y": 2
        },
        "expression": "x * y",
        "expected_value": "HelloHello"
    },
    {
        "test_name": "String Division Test",
        "event_attributes": {
            "x": "Hello",
            "y": 2
        },
        "expression": "x / y",
        "expected_value": None
    },
    {
        "test_name": "Boolean and String Test",
        "event_attributes": {
            "x": "Hello",
            "y": True
        },
        "expression": "x and y",
        "expected_value": True
    },
    {
        "test_name": "Boolean and Integer Test",
        "event_attributes": {
            "x": 1,
            "y": False
        },
        "expression": "x and y",
        "expected_value": False
    },
    {
        "test_name": "Logical Not Test on String",
        "event_attributes": {
            "x": ""
        },
        "expression": "not x",
        "expected_value": True
    },
    {
        "test_name": "String Equality Test",  # Name of the test
        "event_attributes": {  # Attributes used in expression
            "x": "Hello",
            "y": "Hello"
        },
        "expression": "x == y",  # Expression to be evaluated
        "expected_value": True  # Expected result of the evaluation
    },
    {
        "test_name": "String Inequality Test",
        "event_attributes": {
            "x": "Hello",
            "y": "World"
        },
        "expression": "x != y",
        "expected_value": True
    },
    {
        "test_name": "Concatenation and Multiplication Test",
        "event_attributes": {
            "x": "Hello",
            "y": "World",
            "z": 2
        },
        "expression": "(x + ' ' + y) * z",
        "expected_value": "Hello WorldHello World"
    },
    {
        "test_name": "Empty String Test",
        "event_attributes": {
            "x": ""
        },
        "expression": "x == ''",
        "expected_value": True
    },
    {
        "test_name": "Boolean Or Test with False",
        "event_attributes": {
            "x": False,
            "y": False
        },
        "expression": "x or y",
        "expected_value": False
    },
    {
        "test_name": "Boolean And Test with False",
        "event_attributes": {
            "x": False,
            "y": True
        },
        "expression": "x and y",
        "expected_value": False
    },
    {
        "test_name": "Modulus Test",
        "event_attributes": {
            "x": 10,
            "y": 3
        },
        "expression": "x % y",
        "expected_value": 1
    },
    {
        "test_name": "Exponentiation Test",
        "event_attributes": {
            "x": 2,
            "y": 3
        },
        "expression": "x ** y",
        "expected_value": 8
    },
    {
        "test_name": "True Division Test",
        "event_attributes": {
            "x": 7,
            "y": 2
        },
        "expression": "x / y",
        "expected_value": 3.5
    },
    {
        "test_name": "Floor Division Test",
        "event_attributes": {
            "x": 7,
            "y": 2
        },
        "expression": "x // y",
        "expected_value": 3
    },
    {
        "test_name": "Mixed Type Addition Test",
        "event_attributes": {
            "x": "Hello",
            "y": 5
        },
        "expression": "x + y",
        "expected_value": None
    },
    {
        "test_name": "Nested Operation Test",
        "event_attributes": {
            "x": 2,
            "y": 3,
            "z": 1
        },
        "expression": "(x + y) * z",
        "expected_value": 5
    },
    {
        "test_name": "Inequality Test With Zero",
        "event_attributes": {
            "x": 0,
            "y": 10
        },
        "expression": "x != y",
        "expected_value": True
    },
    {
        "test_name": "Greater Than Test With Negative Number",
        "event_attributes": {
            "x": -5,
            "y": -10
        },
        "expression": "x > y",
        "expected_value": True
    },
    {
        "test_name": "Less Than Test With Negative Number",
        "event_attributes": {
            "x": -5,
            "y": -1
        },
        "expression": "x < y",
        "expected_value": True
    }
]


EXPRESSION_TEST_NAMES = [test['test_name'] for test in EXPRESSION_TEST_CONFIGS]


@pytest.mark.parametrize("test_config", EXPRESSION_TEST_CONFIGS, ids=EXPRESSION_TEST_NAMES)
def test_event_attribute_expression(test_config):
    event_attribute = EventAttribute("event_id", "new_attr_id", EVENT_ATTR_TYPE.EXPRESSION, test_config["expression"])

    actual_result = event_attribute.get_next_value(test_config["event_attributes"])
    assert actual_result == test_config["expected_value"], f'Test "{test_config["test_name"]}" failed. Expected: {test_config["expected_value"]}, Got: {actual_result}'