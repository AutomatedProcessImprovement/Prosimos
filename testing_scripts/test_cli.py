import os
from pathlib import Path

import pytest
import traceback
from click.testing import CliRunner

from cli.diff_res_bpsim import cli


@pytest.fixture
def assets_path() -> Path:
    if os.path.basename(os.getcwd()) == 'testing_scripts':
        return Path('assets')
    return Path('testing_scripts/assets')

def test_start_simulation(assets_path, tmp_path):
    runner = CliRunner()
    model_path = assets_path / 'LoanApp_sequential_9-5.bpmn'
    json_path = assets_path / 'LoanApp_arrival_fix_10.json'
    output_path = tmp_path / 'LoanApp_cli_log.csv'
    
    result = runner.invoke(cli, [
        'start-simulation',
        '--bpmn_path', str(model_path),
        '--json_path', str(json_path),
        '--log_out_path', str(output_path),
        '--total_cases', '20'
        # '--granule_size', 60,
        # '--conf', 0.1,
        # '--supp', 0.7,
        # '--part', 0.3
    ])

    assert result.exit_code == 0, f'Traceback: {traceback.print_tb(result.exc_info[2])}'

    with output_path.open('r') as f:
        assert len(f.readlines()) > 1, 'Output log must have more than 1 line'
