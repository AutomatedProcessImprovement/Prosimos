import os
import json
import datetime

import pytest
from pix_framework.io.event_log import read_csv_log, PROSIMOS_LOG_IDS

from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation

test_short_term_simulation_exact_cases = [
    {
        'test_id': "Simple",
        'bps_model_path': "./assets/short-term/STsim_test_simple_exact.bpmn",
        'sim_params_path': "./assets/short-term/STsim_test_simple_exact.json",
        'simulation_horizon': "2024-01-01T07:59:59.000Z",
        'ongoing_cases_path': "./assets/short-term/STsim_test_simple_exact__executed.csv",
        'ground_truth_path': "./assets/short-term/STsim_test_simple_exact__continuation.csv",
    },
    # {
    #     'test_id': "Simple with ongoing activities",
    #     'bps_model_path': "./assets/short-term/STsim_test_simple_exact.bpmn",
    #     'sim_params_path': "./assets/short-term/STsim_test_simple_exact.json",
    #     'simulation_horizon': "2024-01-01T07:59:59.000Z",
    #     'ongoing_cases_path': "./assets/short-term/STsim_test_simple_exact__executed_wOngoingAct.csv",
    #     'ground_truth_path': "./assets/short-term/STsim_test_simple_exact__continuation_wOngoingAct.csv"
    # }
]


@pytest.mark.xfail(
    reason="Broken test fixture: assets/short-term/output.json places every case's "
           "token on Flow_08i82lb (F->End) with no ongoing/enabled activities, so "
           "nothing continues, while the *__continuation.csv ground truth expects the "
           "cases executing A-F. The snapshot and ground truth are mutually inconsistent "
           "and must be regenerated. See the self-contained resume tests below for "
           "verified short-term behavior.",
    strict=False,
)
@pytest.mark.parametrize(
    "test_data",
    test_short_term_simulation_exact_cases,
    ids=[test_data['test_id'] for test_data in test_short_term_simulation_exact_cases]
)
def test_short_term_simulation_exact(test_data):
    # Discover the ongoing process state
    log_ids = PROSIMOS_LOG_IDS
    ongoing_cases = read_csv_log(test_data['ongoing_cases_path'], log_ids)
    # simulation_starting_point = max(
    #     max(ongoing_cases[log_ids.enabled_time]),
    #     max(ongoing_cases[log_ids.start_time]),
    #     max(ongoing_cases[log_ids.end_time])
    # )
    with open('./assets/short-term/output.json', 'r') as f:
        process_state = json.load(f)

    process_state = parse_process_state(process_state)
    # Run Prosimos in short-term mode
    output_path = "./assets/short-term/out/output_log.csv"
    _ = run_diff_res_simulation(
        '2024-01-01 01:59:59.000000+02:00',
        30,  # Short-term, Prosimos will simulate based on time
        test_data['bps_model_path'],
        test_data['sim_params_path'],
        None,  # No simulation stats needed
        output_path,
        process_state=process_state,
        simulation_horizon=parse_datetime(test_data['simulation_horizon']),
    )
    # Assert expected result
    simulated_continuation = read_csv_log(output_path, log_ids)
    ground_truth = read_csv_log(test_data['ground_truth_path'], log_ids)
    assert simulated_continuation.equals(ground_truth)
    # Remove intermediate files
    os.remove(output_path)

def test_short_term_simulation_preserves_case_attributes(tmp_path):
    """
    A case resumed from a process-state snapshot must keep the case-attribute
    values it already had, instead of the values freshly sampled by
    CasePrioritisation. New cases arriving after the snapshot must still be
    sampled normally.

    The snapshot injects a sentinel value ("SNAPSHOT-<case_id>") that the
    attribute distribution can never produce, so its presence after
    initialization proves the value came from the snapshot, not from sampling.
    """
    import copy
    from prosimos.simulation_engine import SimDiffSetup, SimBPMEnv

    bpmn_path = "./assets/short-term/STsim_test_simple_exact.bpmn"
    base_json_path = "./assets/short-term/STsim_test_simple_exact.json"
    total_cases = 30

    # 1) Add a discrete case attribute to the simulation parameters.
    with open(base_json_path, "r") as f:
        sim_params = json.load(f)
    sim_params["case_attributes"] = [
        {
            "name": "client_type",
            "type": "discrete",
            "values": [
                {"key": "REGULAR", "value": 0.5},
                {"key": "BUSINESS", "value": 0.5},
            ],
        }
    ]
    json_path = tmp_path / "sim_params_with_case_attr.json"
    with open(json_path, "w") as f:
        json.dump(sim_params, f)

    # 2) Inject sentinel attribute values into every snapshot case.
    with open("./assets/short-term/output.json", "r") as f:
        process_state = json.load(f)
    snapshot_case_ids = sorted(int(cid) for cid in process_state["cases"].keys())
    for cid in process_state["cases"]:
        process_state["cases"][cid]["case_attributes"] = {
            "client_type": f"SNAPSHOT-{cid}"
        }
    process_state = parse_process_state(copy.deepcopy(process_state))

    # 3) Build the simulation environment (runs initialize_from_process_state).
    simulation_horizon = parse_datetime("2024-01-01T07:59:59.000Z")
    diffsim_info = SimDiffSetup(
        bpmn_path, str(json_path), False, total_cases,
        process_state=process_state, simulation_horizon=simulation_horizon,
    )
    diffsim_info.set_starting_datetime(parse_datetime("2024-01-01 01:59:59.000000+02:00"))
    diffsim_info.setup_horizon()
    total_cases = diffsim_info.total_num_cases
    bpm_env = SimBPMEnv(
        diffsim_info, None, None,
        process_state=process_state, simulation_horizon=simulation_horizon,
    )

    all_attrs = bpm_env.sim_setup.bpmn_graph.all_attributes
    case_prio = bpm_env.case_prioritisation

    # Resumed cases keep their snapshot value in both the graph attributes and
    # the prioritisation store.
    for cid in snapshot_case_ids:
        expected = f"SNAPSHOT-{cid}"
        assert all_attrs[cid]["client_type"] == expected
        assert case_prio.all_case_attributes[cid]["client_type"] == expected

    # A case that is NOT in the snapshot was sampled normally (never a sentinel).
    sampled_id = max(snapshot_case_ids) + 1
    assert sampled_id < total_cases  # sanity: this id is a future arrival
    assert case_prio.all_case_attributes[sampled_id]["client_type"] in ("REGULAR", "BUSINESS")


def _build_resumed_env(tmp_path, params_extra, mutate_snapshot):
    """
    Build a SimBPMEnv resumed from the short-term snapshot, after merging
    `params_extra` into the simulation parameters and applying
    `mutate_snapshot(process_state)` to the snapshot. Returns the env without
    running the simulation, so tests can inspect the restored attribute state
    produced by initialize_from_process_state.
    """
    import copy
    from prosimos.simulation_engine import SimDiffSetup, SimBPMEnv

    bpmn_path = "./assets/short-term/STsim_test_simple_exact.bpmn"
    with open("./assets/short-term/STsim_test_simple_exact.json", "r") as f:
        sim_params = json.load(f)
    sim_params.update(params_extra)
    json_path = tmp_path / "sim_params.json"
    with open(json_path, "w") as f:
        json.dump(sim_params, f)

    with open("./assets/short-term/output.json", "r") as f:
        process_state = json.load(f)
    mutate_snapshot(process_state)
    process_state = parse_process_state(copy.deepcopy(process_state))

    simulation_horizon = parse_datetime("2024-01-01T07:59:59.000Z")
    diffsim_info = SimDiffSetup(
        bpmn_path, str(json_path), False, 30,
        process_state=process_state, simulation_horizon=simulation_horizon,
    )
    diffsim_info.set_starting_datetime(parse_datetime("2024-01-01 01:59:59.000000+02:00"))
    diffsim_info.setup_horizon()
    return SimBPMEnv(
        diffsim_info, None, None,
        process_state=process_state, simulation_horizon=simulation_horizon,
    )


def test_resume_restores_global_attributes(tmp_path):
    """A global attribute value captured in the snapshot must replace the
    freshly drawn initial value (sentinel 999.0 can never be sampled)."""
    params = {
        "global_attributes": [
            {
                "name": "g_val",
                "type": "continuous",
                "values": {"distribution_name": "fix", "distribution_params": [{"value": 0.0}]},
            }
        ]
    }

    def mutate(ps):
        ps["global_attributes"] = {"g_val": 999.0}

    env = _build_resumed_env(tmp_path, params, mutate)
    assert env.sim_setup.bpmn_graph.all_attributes["global"]["g_val"] == 999.0


def test_resume_restores_event_attributes(tmp_path):
    """A per-case event-attribute value captured in the snapshot must land in
    that case's attribute store."""
    def mutate(ps):
        for cid in ps["cases"]:
            ps["cases"][cid]["event_attributes"] = {"ev": 777.0}

    env = _build_resumed_env(tmp_path, {}, mutate)
    for cid in env.resumed_case_ids:
        assert env.sim_setup.bpmn_graph.all_attributes[cid]["ev"] == 777.0


def test_resume_explicit_priority_override(tmp_path):
    """An explicit priority in the snapshot wins over the rule-derived one."""
    def mutate(ps):
        for cid in ps["cases"]:
            ps["cases"][cid]["priority"] = 42

    env = _build_resumed_env(tmp_path, {}, mutate)
    assert env.resumed_case_ids
    for cid in env.resumed_case_ids:
        assert env.case_prioritisation.all_case_priorities[cid] == 42


def test_resume_warns_when_case_attributes_missing(tmp_path):
    """When the model uses case attributes but a resumed case has none in the
    snapshot, a warning is emitted (silent random fallback otherwise)."""
    from prosimos.warning_logger import warning_logger

    params = {
        "case_attributes": [
            {"name": "client_type", "type": "discrete",
             "values": [{"key": "A", "value": 1.0}]}
        ]
    }

    def mutate(ps):
        # deliberately do NOT add case_attributes to any case
        pass

    warning_logger.clear_warnings()
    env = _build_resumed_env(tmp_path, params, mutate)
    warnings = warning_logger.get_all_warnings()
    assert any("no case_attributes" in w for w in warnings)
    # one warning per resumed case
    assert sum("no case_attributes" in w for w in warnings) == len(env.resumed_case_ids)
    warning_logger.clear_warnings()


def test_resume_control_flow_follows_case_attribute(tmp_path):
    """
    End-to-end: resume three cases parked right before an attribute-conditioned
    XOR gateway, each with a different client_type set in the snapshot, run the
    continuation to completion, and assert each case took the branch its
    attribute dictates AND that the attribute value is preserved in the log.

        Gateway_004nfcw:  BUSINESS -> A (Activity_0ydef2v)
                          REGULAR  -> B (Activity_1tvjx3e)
                          NONE     -> C (Activity_0paaiex)
    """
    import csv as _csv
    import copy
    import pandas as pd
    from prosimos.simulation_engine import SimDiffSetup, run_simpy_simulation

    base = "./assets/gateway_conditions"
    bpmn_path = f"{base}/gateway_condition_xor_model.bpmn"
    json_path = f"{base}/gateway_one_true_condition.json"

    GATEWAY = "Gateway_004nfcw"
    FLOW_INTO_GW = "Flow_1p0tebp"
    start_dt_str = "2024-01-01 09:00:00.000000+00:00"  # Monday, inside arrival calendar

    plan = {"0": "BUSINESS", "1": "REGULAR", "2": "NONE"}
    expected_activity = {0: "A", 1: "B", 2: "C"}

    cases = {}
    for cid, ct in plan.items():
        cases[cid] = {
            "control_flow_state": {"flows": [FLOW_INTO_GW], "activities": []},
            "ongoing_activities": [],
            "enabled_activities": [],
            "enabled_gateways": [{"id": GATEWAY, "enabled_time": start_dt_str}],
            "enabled_events": [],
            "case_attributes": {"client_type": ct},
        }

    process_state = {"last_case_arrival": start_dt_str, "cases": cases}
    process_state = parse_process_state(copy.deepcopy(process_state))

    diffsim_info = SimDiffSetup(bpmn_path, json_path, False, 3, process_state=process_state)
    diffsim_info.total_num_cases = 3  # 3 snapshot cases, no new arrivals
    diffsim_info.set_starting_datetime(parse_datetime(start_dt_str))

    out = tmp_path / "log.csv"
    with open(out, "w", newline="") as f:
        run_simpy_simulation(diffsim_info, None, _csv.writer(f),
                             process_state=process_state, simulation_horizon=None)

    df = pd.read_csv(out)
    assert not df.empty, "continuation produced no log rows"

    for cid, expected in expected_activity.items():
        rows = df[df["case_id"] == cid]
        acts = set(rows["activity"])
        # control flow: the attribute-dictated branch was taken ...
        assert expected in acts, (
            f"case {cid} ({plan[str(cid)]}) should run {expected}, got {sorted(acts)}"
        )
        # ... and only that branch (the other two activities must not appear)
        assert acts <= {expected}, f"case {cid} took extra branches: {sorted(acts)}"
        # log: the snapshot attribute value is preserved on every row
        assert (rows["client_type"] == plan[str(cid)]).all()


def test_resume_prioritisation_orders_by_case_attribute(tmp_path):
    """
    Prioritisation: two resumed cases competing for the same single resource
    must be served in the order dictated by their attribute-derived priority,
    not by case id. The snapshot gives case 0 the LOW-priority attribute and
    case 1 the HIGH-priority one, so case 1 must start first.
    """
    import copy
    from prosimos.simulation_engine import SimDiffSetup, SimBPMEnv

    bpmn_path = "./assets/gateway_conditions/gateway_condition_xor_model.bpmn"
    base_json = "./assets/gateway_conditions/gateway_one_true_condition.json"
    start_dt_str = "2024-01-01 09:00:00.000000+00:00"

    with open(base_json, "r") as f:
        params = json.load(f)
    # client_type drives priority: BUSINESS -> level 1 (high), REGULAR -> level 2
    params["case_attributes"] = [
        {"name": "client_type", "type": "discrete",
         "values": [{"key": "BUSINESS", "value": 0.5}, {"key": "REGULAR", "value": 0.5}]}
    ]
    params["prioritisation_rules"] = [
        {"priority_level": 1, "rules": [[{"attribute": "client_type", "comparison": "=", "value": "BUSINESS"}]]},
        {"priority_level": 2, "rules": [[{"attribute": "client_type", "comparison": "=", "value": "REGULAR"}]]},
    ]
    json_path = tmp_path / "prio.json"
    with open(json_path, "w") as f:
        json.dump(params, f)

    # case 0 = REGULAR (lower priority), case 1 = BUSINESS (higher priority)
    cases = {
        "0": {"control_flow_state": {"flows": ["Flow_1p0tebp"], "activities": []},
              "ongoing_activities": [], "enabled_activities": [],
              "enabled_gateways": [{"id": "Gateway_004nfcw", "enabled_time": start_dt_str}],
              "enabled_events": [], "case_attributes": {"client_type": "REGULAR"}},
        "1": {"control_flow_state": {"flows": ["Flow_1p0tebp"], "activities": []},
              "ongoing_activities": [], "enabled_activities": [],
              "enabled_gateways": [{"id": "Gateway_004nfcw", "enabled_time": start_dt_str}],
              "enabled_events": [], "case_attributes": {"client_type": "BUSINESS"}},
    }
    process_state = parse_process_state(copy.deepcopy({"last_case_arrival": start_dt_str, "cases": cases}))

    diffsim_info = SimDiffSetup(bpmn_path, str(json_path), False, 2, process_state=process_state)
    diffsim_info.total_num_cases = 2
    diffsim_info.set_starting_datetime(parse_datetime(start_dt_str))
    env = SimBPMEnv(diffsim_info, None, None, process_state=process_state, simulation_horizon=None)

    prio = env.case_prioritisation
    # BUSINESS case (1) must outrank REGULAR case (0): lower number == higher priority
    assert prio.all_case_priorities[1] < prio.all_case_priorities[0]
    # and the priority queue must order the high-priority case ahead of the low one
    ordered = prio.get_ordered_case_ids_by_priority([0, 1])
    assert ordered[0] == 1, f"expected high-priority case 1 first, got {ordered}"


"""
Comprehensive short-term-simulation mechanics, exercised on the
attribute-conditioned XOR model (Gateway_004nfcw):

    BUSINESS -> A (Activity_0ydef2v)
    REGULAR  -> B (Activity_1tvjx3e)
    NONE     -> C (Activity_0paaiex)
    every branch -> Gateway_18j0t4m (join) -> End
"""

GW_BPMN = "./assets/gateway_conditions/gateway_condition_xor_model.bpmn"
GW_JSON = "./assets/gateway_conditions/gateway_one_true_condition.json"
GW_GATEWAY = "Gateway_004nfcw"
GW_FLOW_INTO_GW = "Flow_1p0tebp"
GW_TASK_A = "Activity_0ydef2v"
GW_RES_NAME = "Default resource profile 1"
GW_START = "2024-01-01 09:00:00.000000+00:00"


def _run_gateway_resume(tmp_path, cases, total_cases, horizon=None, params_extra=None):
    """Resume the XOR model from `cases`, run to completion, return the log DataFrame."""
    import csv as _csv
    import copy
    import pandas as pd
    from prosimos.simulation_engine import SimDiffSetup, run_simpy_simulation

    json_path = GW_JSON
    if params_extra:
        with open(GW_JSON, "r") as f:
            params = json.load(f)
        params.update(params_extra)
        json_path = str(tmp_path / "params.json")
        with open(json_path, "w") as f:
            json.dump(params, f)

    process_state = parse_process_state(copy.deepcopy({"last_case_arrival": GW_START, "cases": cases}))
    diffsim_info = SimDiffSetup(GW_BPMN, json_path, False, total_cases,
                               process_state=process_state, simulation_horizon=horizon)
    diffsim_info.total_num_cases = total_cases
    diffsim_info.set_starting_datetime(parse_datetime(GW_START))

    out = tmp_path / "log.csv"
    with open(out, "w", newline="") as f:
        run_simpy_simulation(diffsim_info, None, _csv.writer(f),
                             process_state=process_state, simulation_horizon=horizon)
    return pd.read_csv(out)


def _parked_case(client_type):
    """A case with a token parked on the flow into the XOR gateway."""
    return {
        "control_flow_state": {"flows": [GW_FLOW_INTO_GW], "activities": []},
        "ongoing_activities": [], "enabled_activities": [],
        "enabled_gateways": [{"id": GW_GATEWAY, "enabled_time": GW_START}],
        "enabled_events": [], "case_attributes": {"client_type": client_type},
    }


def test_resume_ongoing_activity_completes_and_continues(tmp_path):
    """A mid-execution (ongoing) activity resumes, completes after its remaining
    duration, and the case continues to the end of the process."""
    ongoing_start = "2024-01-01 08:59:00.000000+00:00"
    cases = {
        "0": {
            "control_flow_state": {"flows": [], "activities": []},
            "ongoing_activities": [{
                "id": GW_TASK_A, "resource": GW_RES_NAME,
                "start_time": ongoing_start, "enabled_time": ongoing_start,
                "remaining_duration": 120,
            }],
            "enabled_activities": [], "enabled_gateways": [], "enabled_events": [],
            "case_attributes": {"client_type": "BUSINESS"},
        }
    }
    df = _run_gateway_resume(tmp_path, cases, total_cases=1)
    rows = df[df["case_id"] == 0]
    assert "A" in set(rows["activity"]), f"ongoing activity not logged: {set(rows['activity'])}"
    # A started before the snapshot and completed during the continuation
    a = rows[rows["activity"] == "A"].iloc[0]
    assert str(a["start_time"]).startswith("2024-01-01 08:59")


def test_resume_enabled_activity_executes(tmp_path):
    """An enabled-but-not-started activity from the snapshot is executed."""
    cases = {
        "0": {
            "control_flow_state": {"flows": [], "activities": []},
            "ongoing_activities": [],
            "enabled_activities": [{"id": GW_TASK_A, "enabled_time": GW_START}],
            "enabled_gateways": [], "enabled_events": [],
            "case_attributes": {"client_type": "BUSINESS"},
        }
    }
    df = _run_gateway_resume(tmp_path, cases, total_cases=1)
    assert "A" in set(df[df["case_id"] == 0]["activity"])


def test_resume_generates_new_arrivals_after_snapshot(tmp_path):
    """The continuation generates new cases numbered after the snapshot ids and
    runs them through the full process."""
    cases = {"0": _parked_case("BUSINESS"), "1": _parked_case("REGULAR")}
    df = _run_gateway_resume(tmp_path, cases, total_cases=4)  # 2 resumed + 2 new
    logged = set(df["case_id"])
    assert {0, 1} <= logged, f"resumed cases missing: {logged}"
    new_ids = {cid for cid in logged if cid >= 2}
    assert new_ids, f"no new arrivals were generated/logged: {logged}"
    # new arrivals run a real activity (A/B/C)
    for cid in new_ids:
        assert set(df[df["case_id"] == cid]["activity"]) <= {"A", "B", "C"}


def test_resume_horizon_keeps_inflight_drops_late_arrivals(tmp_path):
    """With a horizon shortly after resumption, in-flight cases are logged while
    new arrivals that start after the horizon are filtered out."""
    cases = {"0": _parked_case("BUSINESS"), "1": _parked_case("REGULAR")}
    # horizon a few seconds after the resumed activities (which run at 09:00:00),
    # but before the first new arrival (>= 09:00:30 given fix 30s arrivals)
    horizon = parse_datetime("2024-01-01T09:00:05.000Z")
    df = _run_gateway_resume(tmp_path, cases, total_cases=6, horizon=horizon)
    logged = set(df["case_id"])
    assert logged == {0, 1}, f"expected only in-flight cases, got {logged}"


def test_resume_logs_inflight_cases_under_early_horizon(tmp_path):
    """
    Regression for the empty-log bug: in-flight (resumed) cases must be logged
    even when their continuation activities start at/after the horizon - they
    are the subject of the short-term run. The horizon only filters NEW cases.
    """
    import csv as _csv
    import copy
    import pandas as pd
    from prosimos.simulation_engine import SimDiffSetup, run_simpy_simulation

    bpmn_path = "./assets/gateway_conditions/gateway_condition_xor_model.bpmn"
    json_path = "./assets/gateway_conditions/gateway_one_true_condition.json"
    start_dt_str = "2024-01-01 09:00:00.000000+00:00"
    # horizon BEFORE the resumed activities run -> old code skipped everything
    early_horizon = parse_datetime("2024-01-01T08:59:59.000Z")

    cases = {
        "0": {"control_flow_state": {"flows": ["Flow_1p0tebp"], "activities": []},
              "ongoing_activities": [], "enabled_activities": [],
              "enabled_gateways": [{"id": "Gateway_004nfcw", "enabled_time": start_dt_str}],
              "enabled_events": [], "case_attributes": {"client_type": "BUSINESS"}},
        "1": {"control_flow_state": {"flows": ["Flow_1p0tebp"], "activities": []},
              "ongoing_activities": [], "enabled_activities": [],
              "enabled_gateways": [{"id": "Gateway_004nfcw", "enabled_time": start_dt_str}],
              "enabled_events": [], "case_attributes": {"client_type": "REGULAR"}},
    }
    process_state = parse_process_state(copy.deepcopy({"last_case_arrival": start_dt_str, "cases": cases}))

    diffsim_info = SimDiffSetup(bpmn_path, json_path, False, 2,
                               process_state=process_state, simulation_horizon=early_horizon)
    diffsim_info.total_num_cases = 2  # only the resumed cases, no new arrivals
    diffsim_info.set_starting_datetime(parse_datetime(start_dt_str))

    out = tmp_path / "log.csv"
    with open(out, "w", newline="") as f:
        run_simpy_simulation(diffsim_info, None, _csv.writer(f),
                             process_state=process_state, simulation_horizon=early_horizon)

    df = pd.read_csv(out)
    assert set(df["case_id"]) == {0, 1}, f"in-flight cases dropped under horizon: {set(df['case_id'])}"
    assert "A" in set(df[df["case_id"] == 0]["activity"])  # BUSINESS -> A
    assert "B" in set(df[df["case_id"] == 1]["activity"])  # REGULAR  -> B


def test_resume_event_attributes_forward_and_historical(tmp_path):
    """
    End-to-end event-attribute behavior across resumption:

    * FORWARD: a case that executes activity A during the continuation gets the
      event attribute 'stage' computed by update_attributes (point-in-time) and
      logged (value 100).
    * HISTORICAL: a case whose A is already ongoing at snapshot time, carrying a
      historical 'stage' value, logs that exact historical value (555), NOT the
      recomputed one.
    """
    import csv as _csv
    import copy
    import pandas as pd
    from prosimos.simulation_engine import SimDiffSetup, run_simpy_simulation

    A = "Activity_0ydef2v"          # task "A"
    RES_NAME = "Default resource profile 1"
    start_dt_str = "2024-01-01 09:00:00.000000+00:00"
    ongoing_start = "2024-01-01 08:59:00.000000+00:00"

    base_json = "./assets/gateway_conditions/gateway_one_true_condition.json"
    with open(base_json, "r") as f:
        params = json.load(f)
    # event attribute 'stage' produced when A executes (forward => 100)
    params["event_attributes"] = [
        {"event_id": A, "attributes": [
            {"name": "stage", "type": "continuous",
             "values": {"distribution_name": "fix", "distribution_params": [{"value": 100}]}}
        ]}
    ]
    json_path = tmp_path / "evt.json"
    with open(json_path, "w") as f:
        json.dump(params, f)

    cases = {
        # forward case: parked before the gateway, BUSINESS -> branch to A
        "0": {
            "control_flow_state": {"flows": ["Flow_1p0tebp"], "activities": []},
            "ongoing_activities": [], "enabled_activities": [],
            "enabled_gateways": [{"id": "Gateway_004nfcw", "enabled_time": start_dt_str}],
            "enabled_events": [],
            "case_attributes": {"client_type": "BUSINESS"},
        },
        # historical case: A already running, carrying its point-in-time value
        "1": {
            "control_flow_state": {"flows": [], "activities": []},
            "ongoing_activities": [{
                "id": A, "resource": RES_NAME,
                "start_time": ongoing_start, "enabled_time": ongoing_start,
                "remaining_duration": 60,
                "event_attributes": {"stage": 555},
            }],
            "enabled_activities": [], "enabled_gateways": [], "enabled_events": [],
            "case_attributes": {"client_type": "BUSINESS"},
        },
    }
    process_state = parse_process_state(copy.deepcopy({"last_case_arrival": start_dt_str, "cases": cases}))

    diffsim_info = SimDiffSetup(
        "./assets/gateway_conditions/gateway_condition_xor_model.bpmn",
        str(json_path), False, 2, process_state=process_state)
    diffsim_info.total_num_cases = 2
    diffsim_info.set_starting_datetime(parse_datetime(start_dt_str))

    out = tmp_path / "log.csv"
    with open(out, "w", newline="") as f:
        run_simpy_simulation(diffsim_info, None, _csv.writer(f),
                             process_state=process_state, simulation_horizon=None)

    df = pd.read_csv(out)
    a_rows = df[df["activity"] == "A"]
    # forward case 0 -> computed value 100
    fwd = a_rows[a_rows["case_id"] == 0]["stage"]
    assert not fwd.empty and float(fwd.iloc[0]) == 100.0, f"forward stage={list(fwd)}"
    # historical case 1 -> preserved snapshot value 555
    hist = a_rows[a_rows["case_id"] == 1]["stage"]
    assert not hist.empty and float(hist.iloc[0]) == 555.0, f"historical stage={list(hist)}"


def parse_process_state(process_state):
    process_state['last_case_arrival'] = parse_datetime(process_state['last_case_arrival'])

    for resource_id, end_time_str in process_state.get('resource_last_end_times', {}).items():
        process_state['resource_last_end_times'][resource_id] = parse_datetime(end_time_str)

    for case_id, case_data in process_state.get('cases', {}).items():
        # Convert enabled activity times
        for activity in case_data.get('enabled_activities', []):
            activity['enabled_time'] = parse_datetime(activity['enabled_time'])
        # Convert ongoing activity times
        for activity in case_data.get('ongoing_activities', []):
            activity['start_time'] = parse_datetime(activity['start_time'])
    return process_state

def parse_datetime(datetime_str):
    return datetime.datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))