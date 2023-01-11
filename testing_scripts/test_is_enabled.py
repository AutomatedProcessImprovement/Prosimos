from bpdfr_simulation_engine.simulation_properties_parser import parse_json_sim_parameters, parse_simulation_model
from bpdfr_simulation_engine.simulation_setup import SimDiffSetup
from test_discovery import assets_path

import pytz
import datetime


def test_or_gateway_one_token_before_or_true(assets_path):
    """
    OR gateway has two incoming flows
    Token is right now in the one of them.
    Another incoming flow contains XOR gateway.
    And the token is in the outgoing flow
    of that XOR gateway which does not lead back to the OR gateway.
    In that case, we expect is_enabled to equal to True.
    """

    # ====== ARRANGE ======
    bpmn_path = assets_path / 'test_and_or.bpmn'
    json_path = assets_path / 'test_or_xor_follow.json'
    
    _, _, element_probability, task_resource, _, event_distribution, batch_processing, _ \
        = parse_json_sim_parameters(json_path)

    bpmn_graph = parse_simulation_model(bpmn_path)
    bpmn_graph.set_additional_fields_from_json(element_probability, task_resource, 
        event_distribution, batch_processing)
    
    sim_setup = SimDiffSetup(bpmn_path, json_path, False, 1)
    sim_setup.set_starting_datetime(pytz.utc.localize(datetime.datetime.now()))
    p_state = sim_setup.initial_state()

    # Task 1 A -> join inclusive (OR) gateway
    p_state.add_token("Flow_0mcgg0k")

    # split exclusive (XOR) gateway -> Task 2
    p_state.add_token("Flow_0vgoazd")

    # ====== ACT ======
    e_id = "Gateway_1aucmm2" # id of the inclusive (OR) join gateway
    result = bpmn_graph.is_enabled(e_id, p_state)

    # ====== ASSERT ======
    assert result == True


def test_or_gateway_both_tokens_before_or_true(assets_path):
    """
    OR gateway has two incoming flows.
    Tokens are directly before the OR gateway in both incoming flows.
    In that case, we expect is_enabled to equal to True.
    """

    # ====== ARRANGE ======
    bpmn_path = assets_path / 'test_and_or.bpmn'
    json_path = assets_path / 'test_or_xor_follow.json'
    
    _, _, element_probability, task_resource, _, event_distribution, batch_processing, _ \
        = parse_json_sim_parameters(json_path)

    bpmn_graph = parse_simulation_model(bpmn_path)
    bpmn_graph.set_additional_fields_from_json(element_probability, task_resource,
        event_distribution, batch_processing)
    
    sim_setup = SimDiffSetup(bpmn_path, json_path, False, 1)
    sim_setup.set_starting_datetime(pytz.utc.localize(datetime.datetime.now()))
    p_state = sim_setup.initial_state()

    # Task 1 A -> join inclusive (OR) gateway
    p_state.add_token("Flow_0mcgg0k")

    # split exclusive (XOR) gateway -> join inclusive (OR) gateway
    p_state.add_token("Flow_0urvgxh")

    # ====== ACT ======
    # inclusive join gateway
    e_id = "Gateway_1aucmm2"
    result = bpmn_graph.is_enabled(e_id, p_state)

    # ====== ASSERT ======
    assert result == True


def test_or_gateway_one_token_before_xor_false(assets_path):
    """
    OR gateway has two incoming flows
    Token is right now in the one of them.
    Another incoming flow contains XOR gateway.
    And the token is before that XOR gateway.
    We cannot predict yet where the token will move next.
    So we expect is_enabled to equal to True.
    """

    # ====== ARRANGE ======
    bpmn_path = assets_path / 'test_and_or.bpmn'
    json_path = assets_path / 'test_or_xor_follow.json'
    
    _, _, element_probability, task_resource, _, event_distribution, batch_processing, _ \
        = parse_json_sim_parameters(json_path)

    bpmn_graph = parse_simulation_model(bpmn_path)
    bpmn_graph.set_additional_fields_from_json(element_probability, task_resource,
        event_distribution, batch_processing)
    
    sim_setup = SimDiffSetup(bpmn_path, json_path, False, 1)
    sim_setup.set_starting_datetime(pytz.utc.localize(datetime.datetime.now()))
    p_state = sim_setup.initial_state()

    # Task 1 A -> join inclusive (OR) gateway
    p_state.add_token("Flow_0mcgg0k")

    # Task 1 B -> split exclusive (XOR) gateway
    p_state.add_token("Flow_102lwvt")

    # ====== ACT ======
    # inclusive join gateway
    e_id = "Gateway_1aucmm2"
    result = bpmn_graph.is_enabled(e_id, p_state)

    # ====== ASSERT ======
    assert result == False
