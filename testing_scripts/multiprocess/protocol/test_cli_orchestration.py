"""
The start-orchestration command: a multi-process simulation from the command line, on the
configuration of the running example.
"""
import dataclasses
import json
import re
import shlex
from pathlib import Path

from click.testing import CliRunner

from cli.diff_res_bpsim import cli, run_summary
from prosimos.orchestration.config import SimulationConfig
from prosimos.orchestration.simulation import run_multiprocess_simulation

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
    return run_multiprocess_simulation(config, tmp_path / "python_log.csv")


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
    assert [case["object_id"] for case in saved["stalled"]] == [case.object_id for _, case in report.stalled]


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


def test_the_quick_start_runs_as_written(tmp_path):
    # docs/orchestrator.md shows the running example's configuration, the command that runs it and what it prints
    quick_start = Path("docs/orchestrator.md").read_text(encoding="utf-8").split("## Quick start")[1].split("\n## ")[0]
    config, command, summary = re.findall(r"```(?:json)?\n(.*?)```", quick_start, re.DOTALL)[:3]
    assert config == _text(Path(CONFIG))

    words = shlex.split(command)
    assert words[:3] == ["poetry", "run", "prosimos"] and words[words.index("--config") + 1] == CONFIG
    # the output files go to tmp_path instead of the current folder
    arguments = [str(tmp_path / word) if previous.endswith("_out_path") else word
                 for previous, word in zip([""] + words[3:], words[3:])]
    result = CliRunner().invoke(cli, arguments)
    assert result.exit_code == 0, result.output
    assert result.output == summary
    assert all((tmp_path / name).exists() for name in ("merged_log.csv", "report.json", "running_example.json"))
