import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
from pix_framework.discovery.resource_calendar_and_performance.crisp.factory import CalendarFactory
from pix_framework.discovery.resource_calendar_and_performance.crisp.resource_calendar import RCalendar
from pix_framework.statistics.distribution import DurationDistribution, get_best_fitting_distribution

from bpdfr_discovery.exceptions import InvalidInputDiscoveryParameters
from prosimos.control_flow_manager import BPMN, BPMNGraph
from prosimos.exceptions import InvalidBpmnModelException, InvalidLogFileException
from prosimos.execution_info import TaskEvent, Trace
from prosimos.file_manager import FileManager
from prosimos.simulation_properties_parser import parse_simulation_model

print_info = False


def event_list_from_xes_log(log_path):
    from pm4py.objects.log.importer.xes import importer as xes_importer

    log_traces = xes_importer.apply(log_path)
    trace_list = list()
    for trace in log_traces:
        started_events = dict()
        trace_info = Trace(trace.attributes["concept:name"])
        for event in trace:
            task_name = event["concept:name"]
            state = event["lifecycle:transition"].lower()
            if state in ["start", "assign"]:
                started_events[task_name] = trace_info.start_event(
                    task_name, task_name, event["time:timestamp"], event["org:resource"]
                )
            elif state == "complete":
                if task_name in started_events:
                    c_event = trace_info.complete_event(started_events.pop(task_name), event["time:timestamp"])
                    trace_list.append(c_event)
    return trace_list


def transform_xes_to_csv(log_path, csv_out_path):
    from pm4py.objects.log.importer.xes import importer as xes_importer

    visited_events = dict()

    with open(csv_out_path, mode="w", newline="", encoding="utf-8") as log_csv_file:
        csv_writer = csv.writer(log_csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        log_writer = FileManager(10000, csv_writer)
        log_traces = xes_importer.apply(log_path)
        for trace in log_traces:
            started_events = dict()
            trace_info = Trace(trace.attributes["concept:name"])

            for event in trace:
                task_name = event["concept:name"]
                resource = event["org:resource"] if "org:resource" in event else "D_SYSTEM"
                state = event["lifecycle:transition"].lower()
                timestamp = event["time:timestamp"]

                if is_duplicated(visited_events, trace_info.p_case, task_name, resource, timestamp, state):
                    continue

                if state in ["start", "assign"]:
                    started_events[task_name] = trace_info.start_event(task_name, task_name, timestamp, resource)
                elif state == "complete":
                    if task_name in started_events:
                        c_event = trace_info.complete_event(started_events.pop(task_name), timestamp)
                        log_writer.add_csv_row(
                            [
                                trace_info.p_case,
                                task_name,
                                "",
                                str(c_event.started_at),
                                str(c_event.completed_at),
                                resource,
                            ]
                        )

        log_writer.force_write()


def is_duplicated(visited_events, p_case, task_name, resource, timestamp, state):
    if p_case not in visited_events:
        visited_events[p_case] = []
    for evt in visited_events[p_case]:
        if task_name in evt and resource in evt and timestamp in evt and state in evt:
            return True
    visited_events[p_case].append({task_name, resource, timestamp, state})
    return False


def dataframe_from_csv(log_path, extended_out=False):
    event_log = pd.read_csv(log_path)
    event_log["start_time"] = pd.to_datetime(event_log["start_time"], utc=True)
    event_log["end_time"] = pd.to_datetime(event_log["end_time"], utc=True)
    event_log.sort_values(by=["case_id", "end_time"], inplace=True, ascending=[True, True])

    # act_freq = event_log['activity'].value_counts()
    # res_freq = event_log['resource'].value_counts()


def event_list_from_csv(log_path):
    try:
        with open(log_path, mode="r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            trace_list = list()
            trace_map = dict()
            i_map = dict()
            row_count = 0
            for row in csv_reader:
                if row_count > 0:
                    case_id = row[i_map["case_id"]]
                    event_info = TaskEvent(case_id, row[i_map["activity"]], row[i_map["resource"]])
                    if "enable_time" in i_map and len(row[i_map["enable_time"]]) > 0:
                        event_info.enabled_at = pd.to_datetime(row[i_map["enable_time"]])
                    event_info.started_at = pd.to_datetime(row[i_map["start_time"]])
                    event_info.completed_at = pd.to_datetime(row[i_map["end_time"]])
                    if case_id not in trace_map:
                        trace_map[case_id] = len(trace_list)
                        trace_list.append(Trace(case_id))
                    trace_list[trace_map[case_id]].event_list.append(event_info)
                else:
                    i_map = find_index(row)
                row_count += 1
            return trace_list
    except IOError:
        return list()


def find_index(csv_row):
    i_map = dict()
    for i in range(0, len(csv_row)):
        i_map[csv_row[i]] = i
    return i_map


def compute_kpi_times_from_csv_log(log_path, bpmn_graph):
    trace_list = event_list_from_csv(log_path)
    flow_arcs_frequency = dict()
    total_enablement = wrong_enablement = fixed_enablement = 0
    cumul_task_stats = dict()

    for trace_info in trace_list:
        task_sequence = sort_by_completion_times(trace_info)
        is_correct, fired_tasks, pending_tokens, enabling_times = bpmn_graph.reply_trace(
            task_sequence, flow_arcs_frequency, True, trace_info.event_list
        )
        for i in range(0, len(enabling_times)):
            total_enablement += 1
            if trace_info.event_list[i].started_at < enabling_times[i]:
                wrong_enablement += 1
                if (
                    fix_enablement_from_incorrect_models(i, enabling_times, trace_info.event_list)
                    and not trace_info.event_list[i].started_at < enabling_times[i]
                ):
                    fixed_enablement += 1
            ev_info = trace_info.event_list[i]
            ev_info.update_enabling_times(enabling_times[i])
            if ev_info.task_id not in cumul_task_stats:
                cumul_task_stats[ev_info.task_id] = [0, 0, 0]
            cumul_task_stats[ev_info.task_id][0] += ev_info.waiting_time
            cumul_task_stats[ev_info.task_id][1] += ev_info.processing_time
            cumul_task_stats[ev_info.task_id][2] += 1

    # print("Correct Enablement Ratio: %.2f" % ((total_enablement - wrong_enablement) / total_enablement))
    # print("Fixed   Enablement Ratio: %.2f" % (
    #             (total_enablement - wrong_enablement + fixed_enablement) / total_enablement))
    return cumul_task_stats


def parse_and_validate_input(log_path, bpmn_path, minutes_x_granule, conf, supp, part, is_csv=False):
    if minutes_x_granule < 0 or 1440 % minutes_x_granule != 0:
        raise InvalidInputDiscoveryParameters(
            "Invalid granule_size. The time interval must be a divisor of 1400, e.g., 15, 30, 60 minutes"
        )
    if conf < 0 or conf > 1:
        raise InvalidInputDiscoveryParameters(
            "Invalid confidence. The confidence index must be a value between 0 and 1, both inclusive."
        )
    if supp < 0 or supp > 1:
        raise InvalidInputDiscoveryParameters(
            "Invalid support. The support index must be a value between 0 and 1, both inclusive."
        )
    if part < 0 or part > 1:
        raise InvalidInputDiscoveryParameters(
            "Invalid resource participation ratio. It must be a value between 0 and 1, both inclusive."
        )
    try:
        bpmn_graph = parse_simulation_model(bpmn_path)
    except InvalidBpmnModelException as e:
        error_str = str(e)
        print(error_str)
        raise InvalidBpmnModelException(f"Invalid BPMN model ({error_str})")
    except InvalidLogFileException as e:
        error_str = str(e)
        print(error_str)
        raise InvalidLogFileException(error_str)
    except:
        raise InvalidBpmnModelException("Invalid BPMN model.")

    if is_csv:
        try:
            log_traces = parse_csv(log_path)
        except InvalidLogFileException as e:
            raise e
        except:
            raise InvalidLogFileException("Invalid CSV event-log.")
    else:
        try:
            from pm4py.objects.log.importer.xes import importer as xes_importer

            log_traces = xes_importer.apply(log_path)
        except:
            raise InvalidLogFileException("Invalid XES event-log.")

    return bpmn_graph, log_traces


def process_csv_header(first_row):
    i_map = dict()
    i = 0
    key_words = ["case", "activity", "start", "end", "resource"]
    for key in first_row:
        l_key = key.lower()
        for kw in key_words:
            if kw in l_key:
                i_map[kw] = i
                break
        i += 1

    for key in key_words:
        if key not in i_map:
            raise InvalidLogFileException("%s column missing in the CSV file." % key)

    return i_map


class CSVTrace:
    def __init__(self, case_id):
        self.attributes = {"concept:name": case_id}
        self.events = list()

    def add_event(self, activity, state, resource, timestamp):
        self.events.append(
            {
                "concept:name": activity.strip(),
                "elementId": activity.strip(),
                "org:resource": resource.strip(),
                "lifecycle:transition": state.strip(),
                "time:timestamp": timestamp,
            }
        )

    def __iter__(self):
        return CSVTraceIterator(self.events)


class CSVTraceIterator:
    def __init__(self, events):
        self._events = events
        self._index = -1

    def __next__(self):
        self._index += 1
        if self._index < len(self._events):
            return self._events[self._index]
        raise StopIteration


def parse_csv(log_path):
    try:
        with open(log_path, mode="r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            i_map = process_csv_header(next(csv_reader))

            log_traces = list()
            trace_map = dict()

            for row in csv_reader:
                case_id = row[i_map["case"]]
                if case_id not in trace_map:
                    trace_map[case_id] = len(log_traces)
                    log_traces.append(CSVTrace(case_id))
                log_traces[trace_map[case_id]].add_event(
                    row[i_map["activity"]],
                    "start",
                    row[i_map["resource"]],
                    pd.to_datetime(row[i_map["start"]], utc=True),
                )
                log_traces[trace_map[case_id]].add_event(
                    row[i_map["activity"]],
                    "complete",
                    row[i_map["resource"]],
                    pd.to_datetime(row[i_map["end"]], utc=True),
                )

            for trace in log_traces:
                trace.events.sort(key=lambda x: x["time:timestamp"])
            return log_traces
    except IOError as e:
        print(str(e))
        return list()


def preprocess_xes_log(
    log_path,
    bpmn_path,
    out_f_path,
    minutes_x_granule,
    min_confidence,
    min_support,
    min_participation,
    fit_calendar,
    is_csv=False,
    min_bin=50,
    use_observed_arrival_times=False,
):
    print(
        "Discovery Params: Conf: %.2f, Supp: %.2f, R. Part: %.2f, Adj. Cal: %s"
        % (min_confidence, min_support, min_participation, str(fit_calendar))
    )
    bpmn_graph, log_traces = parse_and_validate_input(
        log_path, bpmn_path, minutes_x_granule, min_confidence, min_support, min_participation, is_csv
    )

    calendar_factory = CalendarFactory(minutes_x_granule)
    completed_events = list()
    total_traces = 0

    resource_cases = dict()
    resource_freq = dict()
    max_resource_freq = 0
    task_resource_freq = dict()
    task_resource_events = dict()
    initial_events = dict()
    flow_arcs_frequency = dict()
    min_date = None
    task_events = dict()
    observed_task_resources = dict()
    min_max_task_duration = dict()
    total_events = 0
    removed_traces = 0
    removed_events = 0

    for trace in log_traces:
        caseid = trace.attributes["concept:name"]
        total_traces += 1
        started_events = dict()
        trace_info = Trace(caseid)
        initial_events[caseid] = datetime(9999, 12, 31, tzinfo=pytz.UTC)
        for event in trace:
            total_events += 1
            if is_trace_event_start_or_end(event, bpmn_graph):
                # trace event is a start or end event, we skip it for further parsing
                removed_events += 1
                continue
            if not is_event_in_bpmn_model(event, bpmn_graph):
                # event in the log does not match any task in the BPMN model
                removed_events += 1
                continue

            resource = validate_and_get_resource(event, bpmn_graph)

            state = event["lifecycle:transition"].lower()
            timestamp = event["time:timestamp"]
            if min_date is None:
                min_date = timestamp
            min_date = min(min_date, timestamp)

            initial_events[caseid] = min(initial_events[caseid], timestamp)

            if resource not in resource_freq:
                resource_cases[resource] = set()
                resource_freq[resource] = 0
            resource_cases[resource].add(caseid)
            resource_freq[resource] += 1

            max_resource_freq = max(max_resource_freq, resource_freq[resource])
            task_name = event["concept:name"]

            if task_name not in task_resource_freq:
                task_resource_events[task_name] = dict()
                task_resource_freq[task_name] = [0, dict()]
                task_events[task_name] = list()
                observed_task_resources[task_name] = set()
                min_max_task_duration[task_name] = [sys.float_info.max, 0]
            if resource not in task_resource_freq[task_name][1]:
                task_resource_freq[task_name][1][resource] = 0
                task_resource_events[task_name][resource] = list()
            task_resource_freq[task_name][1][resource] += 1
            task_resource_freq[task_name][0] = max(
                task_resource_freq[task_name][0], task_resource_freq[task_name][1][resource]
            )

            calendar_factory.check_date_time(resource, task_name, timestamp)
            observed_task_resources[task_name].add(resource)
            if state in ["start", "assign"]:
                started_events[task_name] = trace_info.start_event(task_name, task_name, timestamp, resource)
            elif state == "complete":
                if task_name in started_events:
                    c_event = trace_info.complete_event(started_events.pop(task_name), timestamp)
                    task_events[task_name].append(c_event)
                    task_resource_events[task_name][resource].append(c_event)
                    completed_events.append(c_event)
                    duration = (c_event.completed_at - c_event.started_at).total_seconds()
                    min_max_task_duration[task_name][0] = min(min_max_task_duration[task_name][0], duration)
                    min_max_task_duration[task_name][1] = max(min_max_task_duration[task_name][1], duration)

        removed_events += trace_info.filter_incomplete_events()
        if len(trace_info.event_list) == 0:
            removed_traces += 1
            continue

        task_sequence = sort_by_completion_times(trace_info)

        is_correct, fired_tasks, pending_tokens, _ = bpmn_graph.reply_trace(
            task_sequence, flow_arcs_frequency, True, trace_info.event_list
        )

    # print('Processed Traces in Log ----- %d (real) / %d (total)' % (len(log_traces) - removed_traces, len(log_traces)))
    # print('Processed Events in Log ----- %d (real) / %d (total)' % (total_events - removed_events, total_events))
    # print("Total Activities in Log - %d" % len(task_events))
    # print("Total Resources in Log -- %d" % len(resource_freq))

    if removed_traces == len(log_traces):
        raise InvalidLogFileException(
            "Invalid Log: All traces filtered due to events missing at least one of the following attributes"
            "- task_name, resource, start_datetime or end datetime"
        )

    if total_events - removed_events < total_events / 2:
        raise InvalidLogFileException(
            "Invalid Log: More than 50% of events filtered due of at least ine of the following attributes missing"
            "- task_name, resource, start_datetime or end_datetime"
        )

    resource_freq_ratio = dict()
    for r_name in resource_freq:
        resource_freq_ratio[r_name] = resource_freq[r_name] / max_resource_freq

    # # (1) Discovering Resource Calendars
    # # resource_calendars = calendar_factory.build_weekly_calendars(min_confidence, min_support)
    # # removed_resources = print_initial_resource_calendar_info(resource_calendars, resource_freq, max_resource_freq)

    res_calendars, task_resources, joint_resource_events, pools_json, coverage_map = discover_resource_calendars(
        calendar_factory, task_resource_events, min_confidence, min_support, min_participation
    )
    if print_info:
        print_joint_resource_calendar_info(
            res_calendars,
            calendar_factory.kpi_calendar,
            task_resources,
            task_resource_events,
            joint_resource_events,
            coverage_map,
        )

    res_json_calendar = []
    for r_id in res_calendars:
        res_json_calendar.append(
            {"id": "%s_timetable" % r_id, "name": "%s_timetable" % r_id, "time_periods": res_calendars[r_id].intervals_to_json()}
        )

    # # (2) Discovering Arrival Time Calendar
    arrival_calendar = discover_arrival_calendar(initial_events, 60, 0.1, 1.0)
    # arrival_calendar = discover_arrival_calendar(initial_events, minutes_x_granule, min_confidence, min_support)
    json_arrival_calendar = arrival_calendar.to_dict()

    # # (3) Discovering Arrival Time Distribution
    arrival_time_dist = discover_arrival_time_distribution(initial_events, arrival_calendar, use_observed_arrival_times)

    # # (4) Discovering Task Duration Distributions per resource
    task_resource_dist = discover_resource_task_duration_distribution(
        task_resource_events, res_calendars, task_resources, joint_resource_events, fit_calendar, min_bin
    )

    # # (5) Discovering Gateways Branching Probabilities
    # print("Discovering Branching Probabilities ...")
    # for flow_id in flow_arcs_frequency:
    #     print("%s: %d" % (flow_id, flow_arcs_frequency[flow_id]))

    gateways_branching = bpmn_graph.compute_branching_probability(flow_arcs_frequency)

    if not use_observed_arrival_times:
        arrival_time_dist = arrival_time_dist.to_prosimos_distribution()

    to_save = {
        "model_type": "CRISP",
        "resource_profiles": build_resource_profiles(task_resource_dist, bpmn_graph.from_name),
        "arrival_time_distribution": arrival_time_dist,
        "arrival_time_calendar": json_arrival_calendar,
        "gateway_branching_probabilities": gateway_branching_to_json(gateways_branching),
        # "task_resource_distribution": map_task_id_from_names(task_resource_dist, bpmn_graph.from_name),
        "task_resource_distribution": processing_times_json(task_resource_dist, task_resources, bpmn_graph),
        "resource_calendars": res_json_calendar,
    }
    # save_prosimos_json(to_save, out_f_path)
    save_json(out_f_path, to_save)
    return [
        map_task_id_from_names(pools_json, bpmn_graph.from_name),
        arrival_time_dist,
        json_arrival_calendar,
        gateways_branching,
        map_task_id_from_names(task_resource_dist, bpmn_graph.from_name),
        task_resources,
        res_calendars,
        task_events,
        task_resource_events,
        bpmn_graph.from_name,
    ]


def save_json(json_path, simulation_params):
    json_path = Path(json_path)

    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w") as f:
            json.dump(simulation_params, f)


def processing_times_json(res_task_distr, task_resources, bpmn_graph):
    distributions = []

    for t_name in task_resources:
        resources = []
        for r_id in task_resources[t_name]:
            if r_id not in res_task_distr[t_name]:
                continue

            distribution: DurationDistribution = res_task_distr[t_name][r_id]
            distribution_prosimos = distribution.to_prosimos_distribution()

            resources.append(
                {
                    "resource_id": r_id,
                    "distribution_name": distribution_prosimos["distribution_name"],
                    "distribution_params": distribution_prosimos["distribution_params"],
                }
            )

        distributions.append({"task_id": bpmn_graph.from_name[t_name], "resources": resources})

    return distributions


def validate_and_get_resource(event, bpmn_graph: BPMNGraph):
    """
    Validate that task from the log file exists in the BPMN diagram
    Validate that task has assigned resource
    Return the name of the resource
    """
    task_name = event["concept:name"]
    el_id = bpmn_graph.from_name.get(task_name)
    if el_id is None:
        raise InvalidLogFileException(f"Activity '{task_name}' could not be found in the BPMN diagram")

    element = bpmn_graph.element_info.get(el_id)
    if element is None:
        raise InvalidLogFileException(f"Cannot load details about activity '{task_name}' (element_id: {el_id})")

    if element.is_event() and is_event_resource_empty(event):
        # handling BIMP version of log file (with fake activities for start and end events)
        # and also case for intermediate events when we do not need assigned resource
        return task_name
    else:
        if element.type == BPMN.TASK and is_event_resource_empty(event):
            raise InvalidLogFileException(f"Activity '{task_name}' (element_id: {el_id}) should have assigned resource")
        else:
            return event["org:resource"]


def is_event_resource_empty(event):
    return "org:resource" not in event or event["org:resource"] == ""


def is_trace_event_start_or_end(event, bpmn_graph: BPMNGraph):
    """Check whether the trace event is start or end event"""

    element_id = get_element_id_from_event_info(event, bpmn_graph)

    if element_id == "":
        print("WARNING: Trace event could not be mapped to the BPMN element.")
        return False
    elif element_id in [bpmn_graph.starting_event, bpmn_graph.end_event]:
        return True

    return False


def is_event_in_bpmn_model(event, bpmn_graph: BPMNGraph):
    """Check whether the task name in the event matches a task in the BPMN process model"""

    return True if event["concept:name"] in bpmn_graph.from_name else False


def get_element_id_from_event_info(event, bpmn_graph: BPMNGraph):
    original_element_id = event.get("elementId", "")
    task_name = event.get("concept:name", "")

    if original_element_id != "" and original_element_id != task_name:
        # when log file is in CSV format, then task_name == original_element_id
        # and they both equals to task name
        return original_element_id

    # TODO: check whether 'from_name' handles duplicated names of elements in the BPMN model
    element_id = bpmn_graph.from_name.get(task_name, "")
    return element_id


def save_prosimos_json(to_save, file_path):
    resource_calendars = []
    for r_id in to_save["resource_calendars"]:
        resource_calendars.append(
            {"id": r_id + "timetable", "name": r_id + "timetable", "time_periods": to_save["resource_calendars"][r_id]}
        )

    assigned_tasks = dict()
    task_resource_distribution = []
    for t_id in to_save["task_resource_distribution"]:
        resources = []
        for r_id in to_save["task_resource_distribution"][t_id]:
            dist_info = to_save["task_resource_distribution"][t_id][r_id]
            if r_id not in assigned_tasks:
                assigned_tasks[r_id] = []
            if t_id not in assigned_tasks[r_id]:
                assigned_tasks[r_id].append(t_id)
            distribution_params = []
            for d_param in dist_info["distribution_params"]:
                distribution_params.append({"value": d_param})
            resources.append(
                {
                    "resource_id": r_id,
                    "distribution_name": dist_info["distribution_name"],
                    "distribution_params": distribution_params,
                }
            )
        task_resource_distribution.append({"task_id": t_id, "resources": resources})

    resource_profiles = []
    for rp_id in to_save["resource_profiles"]:
        rp_info = to_save["resource_profiles"][rp_id]
        resource_list = []
        for resource in rp_info["resource_list"]:
            resource_list.append(
                {
                    "id": resource["id"],
                    "name": resource["name"],
                    "cost_per_hour": resource["cost_per_hour"],
                    "amount": resource["amount"],
                    "calendar": resource["id"] + "timetable",
                    "assigned_tasks": assigned_tasks[resource["id"]],
                }
            )
        resource_profiles.append({"id": rp_id, "name": rp_info["name"], "resource_list": resource_list})

    gateway_branching = []
    for g_id in to_save["gateway_branching_probabilities"]:
        probabilities = []
        g_info = to_save["gateway_branching_probabilities"][g_id]
        for flow_arc in g_info:
            probabilities.append({"path_id": flow_arc, "value": g_info[flow_arc]})
        gateway_branching.append({"gateway_id": g_id, "probabilities": probabilities})

    arrival_dist_params = []
    if "distribution_params" in to_save["arrival_time_distribution"]:
        for d_param in to_save["arrival_time_distribution"]["distribution_params"]:
            arrival_dist_params.append({"value": d_param})
    histogram_data = (
        to_save["arrival_time_distribution"]["histogram_data"]
        if "histogram_data" in to_save["arrival_time_distribution"]
        else {}
    )
    arrival_time_distribution = {
        "distribution_name": to_save["arrival_time_distribution"]["distribution_name"],
        "distribution_params": arrival_dist_params,
        "histogram_data": histogram_data,
    }

    with open(file_path, "w") as file_writter:
        json.dump(
            {
                "resource_profiles": resource_profiles,
                "arrival_time_distribution": arrival_time_distribution,
                "arrival_time_calendar": to_save["arrival_time_calendar"],
                "gateway_branching_probabilities": gateway_branching_to_json,
                "task_resource_distribution": task_resource_distribution,
                "resource_calendars": resource_calendars,
            },
            file_writter,
        )


def sort_by_completion_times(trace_info: Trace):
    trace_info.sort_by_completion_date(False)
    task_sequence = list()
    for e_info in trace_info.event_list:
        task_sequence.append(e_info.task_id)
    return task_sequence


def map_task_id_from_names(task_resource_dist, from_name):
    id_task_resource_dist = dict()
    for t_name in task_resource_dist:
        id_task_resource_dist[from_name[t_name]] = task_resource_dist[t_name]
    return id_task_resource_dist


def build_resource_profiles(task_resource_dist, from_name):
    resource_profiles = []
    for t_name in task_resource_dist:
        t_id = from_name[t_name]
        resource_list = []
        for r_id in task_resource_dist[t_name]:
            assigned_tasks = []
            for t_cand in task_resource_dist:
                if r_id in task_resource_dist[t_cand]:
                    assigned_tasks.append(from_name[t_cand])

            resource_list.append(
                {
                    "id": r_id,
                    "name": r_id,
                    "cost_per_hour": 1,
                    "amount": 1,
                    "calendar": "%s_timetable" % r_id,
                    "assigned_tasks": assigned_tasks,
                }
            )
        resource_profiles.append({"id": t_id, "name": t_name, "resource_list": resource_list})
    return resource_profiles


def gateway_branching_to_json(gateways_branching):
    gateways_json = []
    for g_id in gateways_branching:
        probabilities = []
        g_prob = gateways_branching[g_id]
        for flow_arc in g_prob:
            probabilities.append({"path_id": flow_arc, "value": g_prob[flow_arc]})

        gateways_json.append({"gateway_id": g_id, "probabilities": probabilities})
    return gateways_json


def fix_enablement_from_incorrect_models(from_i: int, task_enablement: list, trace: list):
    started_at = trace[from_i].started_at
    enabled_at = task_enablement[from_i]
    i = from_i
    while i > 0:
        i -= 1
        if enabled_at == trace[i].completed_at:
            task_enablement[from_i] = started_at
            return True
    return False


def discover_resource_calendars(calendar_factory, task_resource_events, min_confidence, min_support, min_participation):
    # print("Discovering Resource Calendars ...")
    calendar_candidates = calendar_factory.build_weekly_calendars(min_confidence, min_support, min_participation)

    joint_event_candidates = dict()
    joint_task_resources = dict()
    pools_json = dict()

    task_event_freq = dict()
    task_event_covered_freq = dict()
    joint_resource_freq = dict()
    coverage_map = dict()

    for task_name in task_resource_events:
        unfit_resource_events = list()
        joint_task_resources[task_name] = list()

        task_event_freq[task_name] = 0
        task_event_covered_freq[task_name] = 0

        for r_name in task_resource_events[task_name]:
            joint_task_resources[task_name].append(r_name)
            if (
                r_name not in calendar_candidates
                or calendar_candidates[r_name] is None
                or calendar_candidates[r_name].total_weekly_work == 0
            ):
                unfit_resource_events += task_resource_events[task_name][r_name]
            else:
                task_event_covered_freq[task_name] += 2 * len(task_resource_events[task_name][r_name])
            task_event_freq[task_name] += 2 * len(task_resource_events[task_name][r_name])

        if len(unfit_resource_events) > 0:
            joint_events = _max_disjoint_intervals(unfit_resource_events)
            for i in range(0, len(joint_events)):
                j_name = f"Joint_{task_name}_{i}"
                joint_resource_freq[j_name] = 2 * len(joint_events[i])
                joint_event_candidates[j_name] = joint_events[i]
                joint_task_resources[task_name].append(j_name)
                for ev_info in joint_events[i]:
                    calendar_factory.check_date_time(j_name, task_name, ev_info.started_at, True)
                    calendar_factory.check_date_time(j_name, task_name, ev_info.completed_at, True)

    calendar_candidates = calendar_factory.build_weekly_calendars(min_confidence, min_support, min_participation)

    resource_calendars = dict()
    task_resources = dict()
    joint_resource_events = dict()

    discarded_joint = dict()
    for task_name in joint_task_resources:
        discarded_joint[task_name] = list()
        pools_json[task_name] = {"name": task_name, "resource_list": list()}
        resource_list = list()
        task_resources[task_name] = list()
        for r_name in joint_task_resources[task_name]:
            if (
                r_name in calendar_candidates
                and calendar_candidates[r_name] is not None
                and calendar_candidates[r_name].total_weekly_work > 0
            ):
                resource_list.append(_create_resource_profile_entry(r_name, r_name))
                resource_calendars[r_name] = calendar_candidates[r_name]
                task_resources[task_name].append(r_name)
                if r_name in joint_event_candidates:
                    task_event_covered_freq[task_name] += joint_resource_freq[r_name]
                    joint_resource_events[r_name] = joint_event_candidates[r_name]
            elif r_name in joint_event_candidates:
                discarded_joint[task_name].append([r_name, joint_resource_freq[r_name]])

        if calendar_factory.task_coverage(task_name) < min_support:
            discarded_joint[task_name].sort(key=lambda x: x[1], reverse=True)
            for d_info in discarded_joint[task_name]:
                resource_calendars[d_info[0]] = calendar_factory.build_unrestricted_resource_calendar(
                    d_info[0], task_name
                )
                task_event_covered_freq[task_name] += joint_resource_freq[d_info[0]]
                resource_list.append(_create_resource_profile_entry(d_info[0], d_info[0]))
                task_resources[task_name].append(d_info[0])
                joint_resource_events[d_info[0]] = joint_event_candidates[d_info[0]]
                if calendar_factory.task_coverage(task_name) >= min_support:
                    break

        if task_event_covered_freq[task_name] != 0 and task_event_freq[task_name] != 0:
            coverage_map[task_name] = task_event_covered_freq[task_name] / task_event_freq[task_name]
        else:
            coverage_map[task_name] = 0
        pools_json[task_name]["resource_list"] = resource_list

    return resource_calendars, task_resources, joint_resource_events, pools_json, coverage_map


def _max_disjoint_intervals(interval_list):
    if len(interval_list) == 1:
        return [interval_list]
    interval_list.sort(key=lambda ev_info: ev_info.completed_at)
    disjoint_intervals = list()
    while True:
        max_set = list()
        discarded_list = list()
        max_set.append(interval_list[0])
        current_last = interval_list[0].completed_at
        for i in range(1, len(interval_list)):
            if interval_list[i].started_at >= current_last:
                max_set.append(interval_list[i])
                current_last = interval_list[i].completed_at
            else:
                discarded_list.append(interval_list[i])
        if len(max_set) > 1:
            disjoint_intervals.append(max_set)
        if len(max_set) == 1 or len(discarded_list) == 0:
            break
        interval_list = discarded_list
    return disjoint_intervals


def _create_resource_profile_entry(r_id, r_name, amount=1, cost_per_hour=1):
    return {"id": r_id, "name": r_name, "cost_per_hour": cost_per_hour, "amount": amount}


def build_default_calendar(r_name):
    r_calendar = RCalendar("%s_Default" % r_name)
    r_calendar.add_calendar_item("MONDAY", "SUNDAY", "00:00:00.000" "", "23:59:59.999")
    return r_calendar


def discover_arrival_calendar(initial_events, minutes_x_granule, min_confidence, min_support):
    # print("Discovering Arrival Calendar ...")
    arrival_calendar_factory = CalendarFactory(minutes_x_granule)
    for case_id in initial_events:
        arrival_calendar_factory.check_date_time("arrival", "arrival", initial_events[case_id])
    arrival_calendar = arrival_calendar_factory.build_weekly_calendars(min_confidence, min_support, 0.9)
    # Printing Calendar Info (Testing) -----------------------------------
    kpi_calendar = arrival_calendar_factory.kpi_calendar
    t_name = "arrival"
    # print("Coverage: %.2f" % (kpi_calendar.task_coverage('arrival')))
    # print("In Timetable: %d events, Discarded: %d events"
    #       % (kpi_calendar.task_events_in_calendar[t_name],
    #          kpi_calendar.task_events_count[t_name] - kpi_calendar.task_events_in_calendar[t_name]))
    confidence, support = kpi_calendar.compute_confidence_support(t_name)
    participation_ratio = kpi_calendar.resource_participation_ratio(t_name)
    task_participation = kpi_calendar.resource_task_participation_ratio(t_name, t_name)
    # print("    %s -> (%d events), Confidence: %.2f, Support: %.2f, "
    #       "Task Part.: %.2f, Proc. Part.: %.2f"
    #       % (t_name, kpi_calendar.resource_task_freq[t_name][t_name],
    #          confidence, support, task_participation, participation_ratio))
    if print_info:
        for c_id in arrival_calendar:
            arrival_calendar[c_id].print_calendar_info()
    # End Testing --------------------------------------------------------------------
    return arrival_calendar["arrival"]


def discover_arrival_time_distribution(initial_events, arrival_calendar, use_observed_arrival_times=False):
    # print("Discovering Arrival-Time Distribution ...")
    arrival = list()
    j = 0
    for case_id in initial_events:
        is_working, interval_info = arrival_calendar.is_working_datetime(initial_events[case_id])
        if is_working:
            arrival.append(interval_info)
    arrival.sort(key=lambda x: x.date_time)
    durations = list()
    for i in range(1, len(arrival)):
        durations.append((arrival[i].date_time - arrival[i - 1].date_time).total_seconds())

    if print_info:
        print("In Calendar Event Ratio: %.2f" % (len(arrival) / len(initial_events)))
        print("---------------------------------------------------")
    # If we want to use the observed arrival times instead of fitting them to a distribution
    if use_observed_arrival_times:
        # TODO: use pix_framework for this
        # The arrival distribution is "histogram_sampling" so we compute the CDF and BINs of the observations histogram
        num_bins = 20
        filtered_durations = _reject_outliers(durations)
        bins = np.linspace(min(filtered_durations), max(filtered_durations), num_bins + 1)
        hist, _ = np.histogram(filtered_durations, bins=bins)
        cdf = np.cumsum(hist)
        cdf = cdf / cdf[-1]
        bin_midpoints = (bins[:-1] + bins[1:]) / 2
        arrival_distribution = {
            "distribution_name": "histogram_sampling",
            "histogram_data": {
                "cdf": [float(num) for num in cdf],
                "bin_midpoints": [float(num) for num in bin_midpoints],
            },
        }
    else:
        # Otherwise, find the best fitting distribution
        arrival_distribution = get_best_fitting_distribution(durations)
    return arrival_distribution


def _reject_outliers(data, m=5.0):
    # https://stackoverflow.com/a/16562028
    data = np.asarray(data)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.0
    return data[s < m]


def discover_aggregated_task_distributions(task_events, fit_cal, res_calendar: RCalendar):
    durations = list()
    for ev_info in task_events:
        if ev_info.started_at is None or ev_info.completed_at is None:
            continue
        real_duration = (ev_info.completed_at - ev_info.started_at).total_seconds()
        if fit_cal and res_calendar is not None and res_calendar.total_weekly_work > 0:
            real_duration = res_calendar.find_working_time(ev_info.started_at, ev_info.completed_at)
        durations.append(real_duration)
    aggregated_task_distribution = get_best_fitting_distribution(durations)
    # if print_info:
    #     # print("Total Events: %d, Distribution: %s"
    #     #       % (len(durations), str(aggregated_task_distribution)))
    #     # print('------------------------------------')
    return aggregated_task_distribution


def discover_resource_task_duration_distribution(
    task_res_evts, res_calendars, task_res, joint_events, fit_c, min_evts=50
):
    task_resource_distribution = dict()
    for t_id in task_res:
        if print_info:
            print("Task ID: %s" % t_id)
        if t_id not in task_resource_distribution:
            task_resource_distribution[t_id] = dict()
        full_task_durations = list()
        pending_resources = list()
        for r_id in task_res[t_id]:
            event_list = list()
            if res_calendars[r_id].total_weekly_work > 0 and r_id in task_res_evts[t_id]:
                event_list = task_res_evts[t_id][r_id]
            elif r_id in joint_events:
                event_list = joint_events[r_id]
            durations = list()
            for ev_info in event_list:
                real_duration = (
                    (ev_info.completed_at - ev_info.started_at).total_seconds()
                    if not fit_c
                    else res_calendars[r_id].find_working_time(ev_info.started_at, ev_info.completed_at)
                )
                durations.append(real_duration)
            full_task_durations += durations
            if len(durations) < min_evts:
                pending_resources.append(r_id)
            else:
                task_resource_distribution[t_id][r_id] = get_best_fitting_distribution(durations)
                if print_info:
                    print(
                        "Resource: %s, Total Events: %d, Distribution: %s"
                        % (r_id, len(durations), str(task_resource_distribution[t_id][r_id]))
                    )

        agregated_distribution = get_best_fitting_distribution(full_task_durations)
        for r_id in pending_resources:
            task_resource_distribution[t_id][r_id] = agregated_distribution
            if print_info:
                print(
                    "Resource: %s, Total Events: %d, Aggregated Distribution: %s"
                    % (r_id, len(full_task_durations), str(task_resource_distribution[t_id][r_id]))
                )
        if print_info:
            print("---------------------------------------------------")
    return task_resource_distribution


def print_initial_resource_calendar_info(resource_calendars, resource_freq, max_resource_freq):
    removed_resources = set()
    print("Resources to Remove ...")
    for r_name in resource_calendars:
        if resource_calendars[r_name].total_weekly_work == 0:
            removed_resources.add(r_name)
            print("%s: %.3f (%d)" % (r_name, resource_freq[r_name] / max_resource_freq, resource_freq[r_name]))
    print("-------------------------------------------------------")
    return removed_resources


def print_joint_resource_calendar_info(
    res_calendars, kpi_calendar, task_resources, task_resource_events, joint_resource_events, coverage_map
):
    for t_name in task_resources:
        print("Task Name: %s, Coverage: %.2f" % (t_name, kpi_calendar.task_coverage(t_name)))
        print(
            "In Timetable: %d events, Discarded: %d events"
            % (
                kpi_calendar.task_events_in_calendar[t_name],
                kpi_calendar.task_events_count[t_name] - kpi_calendar.task_events_in_calendar[t_name],
            )
        )
        removed_resources = list()

        for r_name in task_resources[t_name]:
            if r_name in res_calendars and res_calendars[r_name].total_weekly_work > 0:
                confidence, support = kpi_calendar.compute_confidence_support(r_name)
                participation_ratio = kpi_calendar.resource_participation_ratio(r_name)
                task_participation = kpi_calendar.resource_task_participation_ratio(r_name, t_name)
                print(
                    "    %s -> (%d events), Confidence: %.2f, Support: %.2f, "
                    "Task Part.: %.2f, Proc. Part.: %.2f"
                    % (
                        r_name,
                        kpi_calendar.resource_task_freq[r_name][t_name],
                        confidence,
                        support,
                        task_participation,
                        participation_ratio,
                    )
                )
            else:
                removed_resources.append(r_name)
        print("----------------------------------------------------------------")

        # for r_name in task_resources[t_name]:
        #     if r_name in task_resource_freq[t_name][1]:
        #         print("%s: %.3f (%d)" % (r_name,
        #                                  task_resource_freq[t_name][1][r_name] / task_resource_freq[t_name][0],
        #                                  task_resource_freq[t_name][1][r_name]))
        #     else:
        #         print("%s: JOINT EXTERNAL RESOURCE" % r_name)
        # for r_name in task_resource_freq[t_name][1]:
        #     if r_name not in task_resources[t_name]:
        #         print("(%s) %s: %.3f (%d)" % ('-', r_name,
        #                                       task_resource_freq[t_name][1][r_name] / task_resource_freq[t_name][0],
        #                                       task_resource_freq[t_name][1][r_name]))


def _cases_to_del(resource_calendars, resource_freq, max_resource_freq, resource_cases, cases_to_remove, total_traces):
    print("Resources to Remove ...")
    for r_name in resource_calendars:
        if resource_calendars[r_name].total_weekly_work == 0:
            print("%s: %.3f (%d)" % (r_name, resource_freq[r_name] / max_resource_freq, resource_freq[r_name]))
            for case_id in resource_cases[r_name]:
                cases_to_remove.add(case_id)
    print("Original Total Cases:      %d" % total_traces)
    print("Postprocessed Total Cases: %d" % (total_traces - len(cases_to_remove)))
    print("Cases to remove: %d" % len(cases_to_remove))
    print("-------------------------------------------------------")


def _update_first_last(start_date, end_date, current_date):
    if start_date is None:
        start_date = current_date
        end_date = current_date
    start_date = min(start_date, current_date)
    end_date = max(end_date, current_date)
    return start_date, end_date
