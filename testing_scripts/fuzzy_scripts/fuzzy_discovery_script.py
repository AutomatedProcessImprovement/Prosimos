import csv
from pathlib import Path

import pytz

from bpdfr_discovery.log_parser import event_list_from_csv
from prosimos.file_manager import FileManager
from testing_scripts.fuzzy_scripts.test_simod.fuzzy_calendars.fuzzy_discovery import build_fuzzy_calendars

from testing_scripts.fuzzy_scripts.fuzzy_test_files import FileType
from testing_scripts.fuzzy_scripts.syntetic_logs_generator import get_file_path


def discover_model_from_csv_log(proc_name, g_size, angle, c_type, even, is_fuzzy):
    build_fuzzy_calendars(
        csv_log_path=get_file_path(is_fuzzy, proc_name, FileType.TRAINING_CSV_LOG, g_size, angle, 0, c_type, even),
        bpmn_path=get_file_path(is_fuzzy, proc_name, FileType.BPMN, g_size, angle, 0, c_type, even),
        json_path=Path(get_file_path(is_fuzzy, proc_name, FileType.SIMULATION_JSON, g_size, angle, 0, c_type, even)),
        i_size_minutes=g_size,
        angle=angle)


def transform_log_datetimes_to_utc(csv_log_path):
    traces = event_list_from_csv(csv_log_path)
    for trace in traces:
        for ev in trace.event_list:
            if ev.enabled_at is not None:
                ev.enabled_at = ev.enabled_at.tz_convert("UTC")
            ev.started_at = ev.started_at.tz_convert("UTC")
            ev.completed_at = ev.completed_at.tz_convert("UTC")
    save_event_log(csv_log_path, traces)


def localize_datetimes(csv_log_path):
    traces = event_list_from_csv(csv_log_path)
    for trace in traces:
        for ev in trace.event_list:
            if ev.enabled_at is not None:
                ev.enabled_at = ev.enabled_at.replace(tzinfo=pytz.UTC)
            ev.started_at = ev.started_at.replace(tzinfo=pytz.UTC)
            ev.completed_at = ev.completed_at.replace(tzinfo=pytz.UTC)
    save_event_log(csv_log_path, traces)


def split_event_log(original_log_path, training_fpath, testing_fpath, training_ratio, f_traces=None):
    traces = event_list_from_csv(original_log_path) if f_traces is None else f_traces
    train_length = int(training_ratio * len(traces))

    train_events = _generate_and_save_log(training_fpath, traces, 0, train_length)
    test_events = _generate_and_save_log(testing_fpath, traces, train_length, len(traces))
    print("Traces in Logs -- Training: %d, Testing: %d" % (train_length, len(traces) - train_length))
    print("Events in Logs -- Training: %d, Testing: %d" % (train_events, test_events))


def _generate_and_save_log(out_log_path, in_traces, from_i, to_i):
    total_events = 0
    out_traces = list()
    for i in range(from_i, to_i):
        out_traces.append(in_traces[i])
        total_events += len(in_traces[i].event_list)

    save_event_log(out_log_path, out_traces)

    return total_events


def save_event_log(out_log_path, traces):
    with open(out_log_path, mode='w', newline='', encoding='utf-8') as log_csv_file:
        f_writer = csv.writer(log_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # add_simulation_event_log_header(f_writer)
        log_writer = FileManager(10000, f_writer)
        for trace in traces:
            for ev in trace.event_list:
                log_writer.add_csv_row([ev.p_case, ev.task_id, '', ev.started_at, ev.completed_at, ev.resource_id])
        log_writer.force_write()


# def fix_datetime_format(to_fix):


# def add_simulation_event_log_header(log_fwriter):
#     if log_fwriter:
#         log_fwriter.writerow([
#             'case_id', 'activity', 'enable_time', 'start_time', 'end_time', 'resource', ])
