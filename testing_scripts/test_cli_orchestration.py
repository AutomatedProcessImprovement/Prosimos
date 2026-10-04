"""
The start-orchestration command: a multi-process simulation from the command line, on the
configuration of the all-real running example.
"""
import dataclasses
import json

from click.testing import CliRunner

from cli.diff_res_bpsim import cli, run_summary
from prosimos.orchestrator import SimulationConfig, run_orchestrator

CONFIG = "testing_scripts/assets/running_example/simulation.json"  # its seed is 1


def _command(tmp_path, *extra):
    result = CliRunner().invoke(cli, ["start-orchestration", "--config", CONFIG,
                                      "--log_out_path", str(tmp_path / "cli_log.csv"),
                                      "--report_out_path", str(tmp_path / "cli_report.json"), *extra])
    assert result.exit_code == 0, result.output
    return result.output


def _in_python(tmp_path, seed=None):
    config = SimulationConfig.from_json(CONFIG)
    if seed is not None:
        config = dataclasses.replace(config, seed=seed)
    return run_orchestrator(config, tmp_path / "python_log.csv")


def _text(path):
    return path.read_text(encoding="utf-8")


def test_the_command_writes_the_same_merged_log_as_run_orchestrator(tmp_path):
    _command(tmp_path)
    _in_python(tmp_path)

    assert _text(tmp_path / "cli_log.csv") == _text(tmp_path / "python_log.csv")


def test_the_summary_and_the_json_report_match_the_run_report(tmp_path):
    output = _command(tmp_path)
    report = _in_python(tmp_path)

    assert output == run_summary(report) + "\n"
    assert output.startswith(f"Messages: {len(report.published)} published")
    assert "Sales: 3 (" in output  # the three Pärnu orders
    saved = json.loads(_text(tmp_path / "cli_report.json"))
    assert saved == json.loads(json.dumps(report.to_dict(), default=str))
    assert [case["case_id"] for case in saved["stalled"]] == [case.case_id for _, case in report.stalled]


def test_seed_overrides_the_configurations_seed(tmp_path):
    _command(tmp_path, "--seed", "2")
    _in_python(tmp_path, seed=2)
    assert _text(tmp_path / "cli_log.csv") == _text(tmp_path / "python_log.csv")

    _in_python(tmp_path, seed=1)
    assert _text(tmp_path / "cli_log.csv") != _text(tmp_path / "python_log.csv")


def test_without_output_paths_the_command_only_prints_the_summary(tmp_path):
    result = CliRunner().invoke(cli, ["start-orchestration", "--config", CONFIG])

    assert result.exit_code == 0, result.output
    assert result.output.startswith("Messages: ") and "Stalled cases:" in result.output
