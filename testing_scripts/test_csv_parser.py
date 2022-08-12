from bpdfr_discovery.log_parser import parse_csv
from test_discovery import assets_path

def test_discovery_valid_input_not_empty_json(assets_path):
    log_path = assets_path / 'purchasing_example_log.csv'

    log_traces = parse_csv(log_path)

    assert len(log_traces) == 8, \
        f"Parsed log file should contain 8 cases, instead has {len(log_traces)}"

    for trace in log_traces:
        case_id = trace.attributes['concept:name']

        assert len(trace.events) == 6 * 2, \
            f"Parsed log file should contain 12 events per case, instead trace {case_id} has {len(trace.events)} events"

        for event in trace.events:
            validate_key_presence(case_id, event, 'org:resource')
            validate_key_presence(case_id, event, 'elementId')

def validate_key_presence(case_id, event, key: str):
    assert key in event, \
        f"Trace {case_id} should contain required attribute key: '{key}' but it does not"
