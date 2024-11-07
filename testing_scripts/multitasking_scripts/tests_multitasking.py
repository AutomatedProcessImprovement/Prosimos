import json
from pathlib import Path

from pix_framework.discovery.probabilistic_multitasking.discovery import MultiType
from prosimos.simulation_engine import run_simulation
from prosimos.simulation_properties_parser import parse_multitasking_model, parse_json_sim_parameters

assets_dir = Path(__file__).parent.parent / "assets/multitasking"


# bpmn_path,
#     json_path,
#     total_cases,
#     stat_out_path=None,
#     log_out_path=None,
#     starting_at=None,
#     is_event_added_to_log=False,
#     fixed_arrival_times=None,

def test_run_prosimos_with_multitasking():
    result = run_simulation(
        bpmn_path=assets_dir / "bpmn_models" / "sequential.bpmn",
        json_path=assets_dir / "json_params" / "sequential_local.json",
        total_cases=100)
    assert result is not None


def test_parse_full_prosimos_molde():
    p_model = parse_json_sim_parameters(assets_dir / "sequential_no_multitasking.json")
    assert p_model is not None


def test_parse_global_multitasking():
    with open(assets_dir / "sequential_global.json") as json_file:
        json_data = json.load(json_file)
        mt_model = parse_multitasking_model(json_data["multitask"])
        assert mt_model is not None
        assert mt_model.mt_type is MultiType.GLOBAL  # The Model is Global
        assert len(mt_model.res_multitask_info) == 5  # The model has 5 resources


def test_parse_local_multitasking():
    with open(assets_dir / "sequential_local.json") as json_file:
        json_data = json.load(json_file)
        mt_model = parse_multitasking_model(json_data["multitask"])
        assert mt_model is not None
        assert mt_model.mt_type is MultiType.LOCAL  # The Model is Global
        assert len(mt_model.res_multitask_info) == 5  # The model has 5 resources
