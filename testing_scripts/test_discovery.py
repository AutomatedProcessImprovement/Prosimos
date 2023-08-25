import os
from pathlib import Path

import pytest
from bpdfr_discovery.log_parser import preprocess_xes_log
from prosimos.exceptions import InvalidBpmnModelException, InvalidLogFileException


@pytest.fixture
def assets_path(request) -> Path:
    entry_path: Path
    if os.path.basename(os.getcwd()) == 'testing_scripts':
        entry_path = Path('assets')
    else:
        entry_path = Path('testing_scripts/assets')

    def teardown():
        output_path = entry_path / 'purchasing_example.json'
        if output_path.exists():
            os.remove(output_path)
    request.addfinalizer(teardown)

    return entry_path


def test_discovery_valid_input_not_empty_json(assets_path):
    model_path = assets_path / 'purchasing_example.bpmn'
    log_path = assets_path / 'purchasing_example_log.csv'
    output_path = assets_path / 'purchasing_example.json'

    [granule, conf, supp, part, adj_calendar] = [60, 0.1, 0.9, 0.6, True]

    [diff_resource_profiles,
     arrival_time_dist,
     json_arrival_calendar,
     gateways_branching,
     task_res_dist,
     task_resources,
     diff_res_calendars,
     task_events,
     task_resource_events,
     id_from_name] = preprocess_xes_log(log_path.as_posix(),
                                        model_path.as_posix(),
                                        output_path.as_posix(), granule, conf, supp, part,
                                        adj_calendar,
                                        True)

    with output_path.open('r') as f:
        assert len(f.readlines()) == 1, 'Output log must have 1 line'

    assert len(diff_resource_profiles) != 0, 'Resource Profiles should not be empty'
    assert arrival_time_dist is not None, 'Arrival Time Distibutions should not be empty'
    assert len(json_arrival_calendar) != 0, 'Arrival Calendar should not be empty'
    assert len(gateways_branching) != 0, 'Gateway Branching section should not be empty'
    assert len(task_res_dist) != 0, 'Task Resource Distribution should not be empty'
    assert len(task_resources) != 0, "Map 'task - list of assigned resources' should not be empty"
    assert len(diff_res_calendars) != 0, 'Resource Calendars should not be empty'
    assert len(task_events) != 0, 'Task Events should not be empty'
    assert len(task_resource_events) != 0, 'Task Resource Events should not be empty'
    assert len(id_from_name) != 0, "Map 'elementId - name' should not be empty"


def test_discovery_valid_input_histogram_sampling_not_empty_json(assets_path):
    model_path = assets_path / 'LoanApp_sequential_9-5.bpmn'
    log_path = assets_path / 'LoanApp_arrival_fix_10.csv'
    output_path = assets_path / 'LoanApp_arrival_fix_10.json'

    [granule, conf, supp, part, adj_calendar] = [60, 0.1, 0.9, 0.6, True]

    [_, arrival_time_dist, _, _, _, _, _, _, _, _] = preprocess_xes_log(
        log_path=log_path.as_posix(),
        bpmn_path=model_path.as_posix(),
        out_f_path=output_path.as_posix(),
        minutes_x_granule=granule,
        min_confidence=conf,
        min_support=supp,
        min_participation=part,
        fit_calendar=adj_calendar,
        is_csv=True,
        use_observed_arrival_times=True
    )

    with output_path.open('r') as f:
        assert len(f.readlines()) == 1, 'Output JSON parameters must have 1 line'
    assert len(arrival_time_dist) != 0, 'Arrival Time Distibutions should not be empty'
    assert arrival_time_dist['distribution_name'] == 'histogram_sampling', "Arrival Time Distribution should be 'Histogram Sampling'"
    assert len(arrival_time_dist['histogram_data']['cdf']) == 20, "Arrival Time Distribution CDF should have 20 values"
    assert len(arrival_time_dist['histogram_data']['bin_midpoints']) == 20, "Arrival Time Distribution bin_midpoints should have 20 values"


def test_discovery_csv_input_error(assets_path):
    model_path = assets_path / 'financial.bpmn'
    log_path = assets_path / 'financial_log.csv'
    output_path = assets_path / 'purchasing_example.json'

    [granule, conf, supp, part, adj_calendar] = [60, 0.1, 0.9, 0.6, True]

    [diff_resource_profiles,
     arrival_time_dist,
     json_arrival_calendar,
     gateways_branching,
     task_res_dist,
     task_resources,
     diff_res_calendars,
     task_events,
     task_resource_events,
     id_from_name] = preprocess_xes_log(log_path.as_posix(),
                                            model_path.as_posix(),
                                            output_path.as_posix(), granule, conf, supp, part,
                                            adj_calendar,
                                            True)

    with output_path.open('r') as f:
        assert len(f.readlines()) == 1, 'Output log must have 1 line'

    assert len(diff_resource_profiles) != 0, 'Resource Profiles should not be empty'
    assert arrival_time_dist is not None, 'Arrival Time Distibutions should not be empty'
    assert len(json_arrival_calendar) != 0, 'Arrival Calendar should not be empty'
    assert len(gateways_branching) != 0, 'Gateway Branching section should not be empty'
    assert len(task_res_dist) != 0, 'Task Resource Distribution should not be empty'
    assert len(task_resources) != 0, "Map 'task - list of assigned resources' should not be empty"
    assert len(diff_res_calendars) != 0, 'Resource Calendars should not be empty'
    assert len(task_events) != 0, 'Task Events should not be empty'
    assert len(task_resource_events) != 0, 'Task Resource Events should not be empty'
    assert len(id_from_name) != 0, "Map 'elementId - name' should not be empty"

    
def test_discovery_two_end_events_error_message(assets_path):
    model_path = assets_path / 'purchasing_example_multiple_end_events.bpmn'
    log_path = assets_path / 'purchasing_example_multiple_events_log.xes'
    output_path = assets_path / 'purchasing_example.json'

    [granule, conf, supp, part, adj_calendar] = [60, 0.1, 0.9, 0.6, True]

    with pytest.raises(InvalidBpmnModelException):
        _ = preprocess_xes_log(log_path.as_posix(),
                                            model_path.as_posix(),
                                            output_path.as_posix(), granule, conf, supp, part,
                                            adj_calendar,
                                            False)

def test_discovery_invalid_csv_error_message(assets_path):
    model_path = assets_path / 'purchasing_example.bpmn'
    log_path = assets_path / 'purchasing_example_sim_scenario.csv'
    output_path = assets_path / 'purchasing_example.json'

    [granule, conf, supp, part, adj_calendar] = [60, 0.1, 0.9, 0.6, True]

    with pytest.raises(InvalidLogFileException, match='end column missing in the CSV file.'):
        _ = preprocess_xes_log(log_path.as_posix(),
                                            model_path.as_posix(),
                                            output_path.as_posix(), granule, conf, supp, part,
                                            adj_calendar,
                                            True # is_csv
                                )