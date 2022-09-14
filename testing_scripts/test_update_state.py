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
    
    _, _, element_probability, task_resource, _, event_distribution, batch_processing \
        = parse_json_sim_parameters(json_path)

    bpmn_graph = parse_simulation_model(bpmn_path)
    bpmn_graph.set_additional_fields_from_json(element_probability, task_resource,
        event_distribution, batch_processing)
    
    sim_setup = SimDiffSetup(bpmn_path, json_path, False)
    sim_setup.set_starting_datetime(pytz.utc.localize(datetime.datetime.now()))
    p_case = 0
    p_state = sim_setup.initial_state()

    # Task 1 A                      -> join inclusive (OR) gateway
    p_state.add_token("Flow_0mcgg0k")

    # split exclusive (XOR) gateway -> join inclusive (OR) gateway
    p_state.add_token("Flow_0urvgxh")

    # ====== ACT ======
    e_id = "Activity_1tidlw3"       # id of the 'Task 1 A' activity
    prev_completed_event_time = datetime.datetime.now(pytz.utc)
    result = bpmn_graph.update_process_state(p_case, e_id, p_state, prev_completed_event_time)

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
    Flow_1sl476n:   OR split -> activity Task 1 A
    Flow_0vgoazd:   XOR split -> activity Task 2
    """

    # ====== ARRANGE ======
    bpmn_path = assets_path / 'test_and_or.bpmn'
    json_path = assets_path / 'test_or_xor_follow.json'
    
    _, _, element_probability, task_resource, _, event_distribution, batch_processing \
        = parse_json_sim_parameters(json_path)

    bpmn_graph = parse_simulation_model(bpmn_path)
    bpmn_graph.set_additional_fields_from_json(element_probability, task_resource,
        event_distribution, batch_processing)
    
    sim_setup = SimDiffSetup(bpmn_path, json_path, False)
    sim_setup.set_starting_datetime(pytz.utc.localize(datetime.datetime.now()))
    p_case = 0
    p_state = sim_setup.initial_state()

    # split inclusive (OR) gateway      -> Task 1 B
    p_state.add_token("Flow_0wy9dja")

    # split inclusive (OR) gateway      -> Task 1 A
    p_state.add_token("Flow_1sl476n")

    # ====== ACT ======
    e_id = "Activity_1uiiyhu"           # id of the 'Task 1 B' activity
    prev_completed_event_time = datetime.datetime.now(pytz.utc)
    result = bpmn_graph.update_process_state(p_case, e_id, p_state, prev_completed_event_time)

    # ====== ASSERT ======
    assert len(result) == 1, "List with enabled tasks should contain one element"
    assert result[0].task_id == "Activity_0mz9221"
    assert result[0].batch_info == None 

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
    
    _, _, element_probability, task_resource, _, event_distribution, batch_processing \
        = parse_json_sim_parameters(json_path)

    bpmn_graph = parse_simulation_model(bpmn_path)
    bpmn_graph.set_additional_fields_from_json(element_probability, task_resource,
        event_distribution, batch_processing)
    
    sim_setup = SimDiffSetup(bpmn_path, json_path, False)
    sim_setup.set_starting_datetime(pytz.utc.localize(datetime.datetime.now()))
    p_case = 0
    p_state = sim_setup.initial_state()

    # split inclusive (OR) gateway      -> Task 1 B
    p_state.add_token("Flow_0wy9dja")

    # split inclusive (OR) gateway      -> Task 1 A
    p_state.add_token("Flow_1sl476n")

    # ====== ACT ======
    e_id = "Activity_1uiiyhu"           # id of the 'Task 1 B' activity
    prev_completed_event_time = datetime.datetime.now(pytz.utc)
    result = bpmn_graph.update_process_state(p_case, e_id, p_state, prev_completed_event_time)

    # ====== ASSERT ======
    assert len(result) == 0, "List with enabled tasks should not contain elements"

    all_tokens = p_state.tokens
    expected_flows_with_token = ["Flow_1sl476n", "Flow_0urvgxh"]
    verify_flow_tokens(all_tokens, expected_flows_with_token)


def test_update_state_event_gateway_event_happened(assets_path):
    """
    Input: enabled event-based gateway due to the token before.
    
    Output: event-based gateway results in executing the first event 'Order response received'
    (this happens due to the provided event_distribution).
    Verify that token changes its position before the event and 
    returns the event as an enabled event
    """

    # ====== ARRANGE ======
    bpmn_path = assets_path / 'stock_replenishment.bpmn'
    json_path = assets_path / 'stock_replenishment_logs.json'
    
    _, _, element_probability, task_resource, _, event_distribution, batch_processing \
        = parse_json_sim_parameters(json_path)

    bpmn_graph = parse_simulation_model(bpmn_path)
    bpmn_graph.set_additional_fields_from_json(element_probability, task_resource,
        event_distribution, batch_processing)
    
    sim_setup = SimDiffSetup(bpmn_path, json_path, False)
    sim_setup.set_starting_datetime(pytz.utc.localize(datetime.datetime.now()))
    p_case = 0
    p_state = sim_setup.initial_state()

    # Parallel gateway split            -> Event-based gateway split
    p_state.add_token("Flow_0d8kgwc")

    # ====== ACT ======
    e_id = "Gateway_0ntcp3d"            # Event-based gateway split
    prev_completed_event_time = \
        datetime.datetime.fromisoformat('2022-08-10T16:05:00') # Wednesday, 16:05
    result = bpmn_graph.update_process_state(p_case, e_id, p_state, prev_completed_event_time)

    # ====== ASSERT ======
    assert len(result) == 1, "List with enabled tasks should contain one element"
    assert result[0].task_id == "Event_1qclhcl"
    assert result[0].batch_info == None

    all_tokens = p_state.tokens
    expected_flows_with_token = ["Flow_0bzfgao"]
    verify_flow_tokens(all_tokens, expected_flows_with_token)


def test_update_state_event_gateway_upper_limit(assets_path):
    """
    Input: enabled event-based gateway due to the token before.
    
    Output: event-based gateway results in executing the timer of 'Friday, 14:00'
    (this happens because the prev activity happens at 12:05 
    and we will only receive response three hours after).
    Verify that token changes its position before the event 'Friday, 14:00' and 
    returns the event as an enabled event.
    """

    # ====== ARRANGE ======
    bpmn_path = assets_path / 'stock_replenishment.bpmn'
    json_path = assets_path / 'stock_replenishment_logs.json'
    
    _, _, element_probability, task_resource, _, event_distribution, batch_processing \
        = parse_json_sim_parameters(json_path)

    bpmn_graph = parse_simulation_model(bpmn_path)
    bpmn_graph.set_additional_fields_from_json(element_probability, task_resource,
        event_distribution, batch_processing)
    
    sim_setup = SimDiffSetup(bpmn_path, json_path, False)
    sim_setup.set_starting_datetime(pytz.utc.localize(datetime.datetime.now()))
    p_case = 0
    p_state = sim_setup.initial_state()

    # Parallel gateway split            -> Event-based gateway split
    p_state.add_token("Flow_0d8kgwc")

    # ====== ACT ======
    e_id = "Gateway_0ntcp3d"            # Event-based gateway split
    prev_completed_event_time = \
        datetime.datetime.fromisoformat('2022-08-05T12:05:00')
    result = bpmn_graph.update_process_state(p_case, e_id, p_state, prev_completed_event_time)

    # ====== ASSERT ======
    assert len(result) == 1, "List with enabled tasks should contain one element"
    assert result[0].task_id == "Event_0bsdbzb"
    assert result[0].batch_info == None

    all_tokens = p_state.tokens
    expected_flows_with_token = ["Flow_0u4ip3z"]
    verify_flow_tokens(all_tokens, expected_flows_with_token)


def test_update_state_terminate_event(assets_path):
    """
    Input: two tokens executing in parallel due to the parallel gateway.
    'Handle order response' activity was executed and, as result, token is now at 'Flow_14zyrni'.
    
    Output: update_process_state of the enabled event triggers the Terminate event.
    All tokens should be nullified as this is the end of the process.
    """

    # ====== ARRANGE ======
    bpmn_path = assets_path / 'stock_replenishment.bpmn'
    json_path = assets_path / 'stock_replenishment_logs.json'
    
    _, _, element_probability, task_resource, _, event_distribution, batch_processing \
        = parse_json_sim_parameters(json_path)

    bpmn_graph = parse_simulation_model(bpmn_path)
    bpmn_graph.set_additional_fields_from_json(element_probability, task_resource,
        event_distribution, batch_processing)
    
    sim_setup = SimDiffSetup(bpmn_path, json_path, False)
    sim_setup.set_starting_datetime(pytz.utc.localize(datetime.datetime.now()))
    p_case = 0
    p_state = sim_setup.initial_state()

    # 'Order response received'         -> 'Handle order response'
    p_state.add_token("Flow_14zyrni")

    p_state.add_token("Flow_1jwj934")
    
    # ====== ACT ======
    e_id = "Event_06aw5gs"            # 'Handle order response' activity
    prev_completed_event_time = \
        datetime.datetime.fromisoformat('2022-08-05T12:05:00')
    result = bpmn_graph.update_process_state(p_case, e_id, p_state, prev_completed_event_time)

    # ====== ASSERT ======
    assert len(result) == 0, "List with enabled tasks should contain no elements"

    all_tokens = p_state.tokens
    expected_flows_with_token = []
    verify_flow_tokens(all_tokens, expected_flows_with_token)


def verify_flow_tokens(all_tokens, expected_flows_with_token):
    for flow in expected_flows_with_token: 
        assert all_tokens[flow] == 1, \
            f"Flow {flow} expected to contain token but it does not"

    expected_flows_without_token = { key: all_tokens[key] for key in all_tokens if key not in expected_flows_with_token }
    for flow in expected_flows_without_token:
        assert all_tokens[flow] == 0, \
            f"Flow {flow} expected not to contain token but it does"
