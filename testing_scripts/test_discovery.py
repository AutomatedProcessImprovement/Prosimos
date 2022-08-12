import os
from pathlib import Path

import pytest
from bpdfr_discovery.exceptions import NotXesFormatException
from bpdfr_discovery.log_parser import preprocess_xes_log
from bpdfr_simulation_engine.exceptions import InvalidBpmnModelException


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
    log_path = assets_path / 'purchasing_example_log.xes'
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
                                        False)

    with output_path.open('r') as f:
        assert len(f.readlines()) == 1, 'Output log must have 1 line'

    assert len(diff_resource_profiles) != 0, 'Resource Profiles should not be empty'
    assert len(arrival_time_dist) != 0, 'Arrival Time Distibutions should not be empty'
    assert len(json_arrival_calendar) != 0, 'Arrival Calendar should not be empty'
    assert len(gateways_branching) != 0, 'Gateway Branching section should not be empty'
    assert len(task_res_dist) != 0, 'Task Resource Distribution should not be empty'
    assert len(task_resources) != 0, "Map 'task - list of assigned resources' should not be empty"
    assert len(diff_res_calendars) != 0, 'Resource Calendars should not be empty'
    assert len(task_events) != 0, 'Task Events should not be empty'
    assert len(task_resource_events) != 0, 'Task Resource Events should not be empty'
    assert len(id_from_name) != 0, "Map 'elementId - name' should not be empty"

def test_discovery_csv_input_error(assets_path):
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
    assert len(arrival_time_dist) != 0, 'Arrival Time Distibutions should not be empty'
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
