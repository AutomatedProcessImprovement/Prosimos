from bpdfr_simulation_engine.simulation_properties_parser import parse_json_sim_parameters, parse_simulation_model
from bpdfr_simulation_engine.simulation_setup import SimDiffSetup
from test_discovery import assets_path

import pytz
import datetime


def test_not_enabled_event_empty_tasks(assets_path):
    """
    Input: e_id - event which is not enabled
    Output: no enabled tasks being returned (empty array)
    """

    # ====== ARRANGE ======
    bpmn_path = assets_path / 'test_and_or.bpmn'
    json_path = assets_path / 'test_or_xor_follow.json'
    
    _, _, element_probability, task_resource, _ \
        = parse_json_sim_parameters(json_path)

    bpmn_graph = parse_simulation_model(bpmn_path)
    bpmn_graph.set_element_probabilities(element_probability, task_resource)
    
    sim_setup = SimDiffSetup(bpmn_path, json_path)
    sim_setup.set_starting_satetime(pytz.utc.localize(datetime.datetime.now()))
    p_state = sim_setup.initial_state()

    # Task 1 A                      -> join inclusive (OR) gateway
    p_state.add_token("Flow_0mcgg0k")

    # split exclusive (XOR) gateway -> join inclusive (OR) gateway
    p_state.add_token("Flow_0urvgxh")

    # ====== ACT ======
    e_id = "Activity_1tidlw3"       # id of the 'Task 1 A' activity
    result = bpmn_graph.update_process_state(e_id, p_state)

    # ====== ASSERT ======
    assert len(result) == 0, "List with enabled tasks should not contain elements"

    all_tokens = p_state.tokens
    expected_flows_with_token = ["Flow_0mcgg0k", "Flow_0urvgxh"]
    verify_flow_tokens(all_tokens, expected_flows_with_token)


def test_enabled_first_task_enables_next_one(assets_path):
    """
    Input: activated activity 'Task 1 B', another token before 'Task 1 A'.
    XOR gateway will result in moving to 'Task 2' activity 
    (this is guaranteed by the gateway probability of 1 - 0).

    Output: 'Task 2' activity is being returned as an enabled.
    Two flows ("Flow_1sl476n", "Flow_0vgoazd") contain tokens while others do not. 
    """

    # ====== ARRANGE ======
    bpmn_path = assets_path / 'test_and_or.bpmn'
    json_path = assets_path / 'test_or_xor_follow.json'
    
    _, _, element_probability, task_resource, _ \
        = parse_json_sim_parameters(json_path)

    bpmn_graph = parse_simulation_model(bpmn_path)
    bpmn_graph.set_element_probabilities(element_probability, task_resource)
    
    sim_setup = SimDiffSetup(bpmn_path, json_path)
    sim_setup.set_starting_satetime(pytz.utc.localize(datetime.datetime.now()))
    p_state = sim_setup.initial_state()

    # split inclusive (OR) gateway      -> Task 1 B
    p_state.add_token("Flow_0wy9dja")

    # split inclusive (OR) gateway      -> Task 1 A
    p_state.add_token("Flow_1sl476n")

    # ====== ACT ======
    e_id = "Activity_1uiiyhu"           # id of the 'Task 1 B' activity
    result = bpmn_graph.update_process_state(e_id, p_state)

    # ====== ASSERT ======
    assert len(result) == 1, "List with enabled tasks should contain one element"
    assert sorted(result) == ["Activity_0mz9221"]

    all_tokens = p_state.tokens
    expected_flows_with_token = ["Flow_1sl476n", "Flow_0vgoazd"]
    verify_flow_tokens(all_tokens, expected_flows_with_token)


def test_enabled_first_task_token_wait_at_the_or_join(assets_path):
    """
    Input: activated activity 'Task 1 B', another token before 'Task 1 A'.
    XOR gateway will result in moving to join of OR gateway 
    (this is guaranteed by the gateway probability of 0 - 1).

    Output: No activities are being returned as an enabled.
    One token will change its location (from before 'Task 1 B' to before OR gateway).
    The other one will stay where it was: right before the activity 'Task 1 A'.
    """

    # ====== ARRANGE ======
    bpmn_path = assets_path / 'test_and_or.bpmn'
    json_path = assets_path / 'test_or_not_xor_follow.json'
    
    _, _, element_probability, task_resource, _ \
        = parse_json_sim_parameters(json_path)

    bpmn_graph = parse_simulation_model(bpmn_path)
    bpmn_graph.set_element_probabilities(element_probability, task_resource)
    
    sim_setup = SimDiffSetup(bpmn_path, json_path)
    sim_setup.set_starting_satetime(pytz.utc.localize(datetime.datetime.now()))
    p_state = sim_setup.initial_state()

    # split inclusive (OR) gateway      -> Task 1 B
    p_state.add_token("Flow_0wy9dja")

    # split inclusive (OR) gateway      -> Task 1 A
    p_state.add_token("Flow_1sl476n")

    # ====== ACT ======
    e_id = "Activity_1uiiyhu"           # id of the 'Task 1 B' activity
    result = bpmn_graph.update_process_state(e_id, p_state)

    # ====== ASSERT ======
    assert len(result) == 0, "List with enabled tasks should not contain elements"

    all_tokens = p_state.tokens
    expected_flows_with_token = ["Flow_1sl476n", "Flow_0urvgxh"]
    verify_flow_tokens(all_tokens, expected_flows_with_token)


def verify_flow_tokens(all_tokens, expected_flows_with_token):
    for flow in expected_flows_with_token: 
        assert all_tokens[flow] == 1, \
            f"Flow {flow} expected to contain token but it does not"

    expected_flows_without_token = { key: all_tokens[key] for key in all_tokens if key not in expected_flows_with_token }
    for flow in expected_flows_without_token:
        assert all_tokens[flow] == 0, \
            f"Flow {flow} expected not to contain token but it does"
