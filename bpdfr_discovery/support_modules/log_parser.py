import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
import pytz
from pm4py.objects.log.importer.xes import importer as xes_importer

from bpdfr_simulation_engine.execution_info import ProcessInfo, Trace
from bpdfr_simulation_engine.probability_distributions import best_fit_distribution

import ntpath

from bpdfr_simulation_engine.resource_calendar import RCalendar, update_calendar_from_log

date_format = "%Y-%m-%dT%H:%M:%S.%f%z"


# def parse_xes_log_1(log_path, bpmn_graph):
#
#     print(1)
#     tree = ET.parse(log_path)
#     root = tree.getroot()
#     ns = {'xes': root.tag.split('}')[0].strip('{')}
#     tags = dict(trace='xes:trace',
#                 string='xes:string',
#                 event='xes:event',
#                 date='xes:date')
#     traces = root.findall(tags['trace'], ns)
#
#     return _extract_log_info(bpmn_graph, traces, tags, ns)


def parse_xes_log(log_path, bpmn_graph, output_path):
    f_name = ntpath.basename(log_path).split('.')[0]
    print('Parsing Event Log %s ...' % f_name)
    process_info = ProcessInfo()
    i = 0
    total_traces = 0
    resource_list = set()

    task_resource = dict()
    task_distribution = dict()
    flow_arcs_frequency = dict()
    correct_traces = 0
    correct_activities = 0
    total_activities = 0
    task_fired_ratio = dict()
    task_missed_tokens = 0
    missed_tokens = dict()

    log_traces = xes_importer.apply(log_path)

    arrival_times = list()

    start_date = end_date = None
    resource_calendars = dict()
    arrival_dates = list()

    for trace in log_traces:
        arrival_dates.append(trace[0]['time:timestamp'])
        # if previous_arrival_date is not None:
        #     arrival_times.append((trace[0]['time:timestamp'] - previous_arrival_date).total_seconds())
        # previous_arrival_date = trace[0]['time:timestamp']

        caseid = trace.attributes['concept:name']
        total_traces += 1
        started_events = dict()
        trace_info = Trace(caseid)
        task_sequence = list()
        previous_date = None
        for event in trace:
            task_name = event['concept:name']
            task_id = bpmn_graph.from_name[task_name]
            resource = event['org:resource']
            state = event['lifecycle:transition'].lower()
            timestamp = event['time:timestamp']
            if previous_date is not None and previous_date > timestamp:
                print("Unsorted event %s" % task_name)
            previous_date = timestamp
            start_date, end_date = _update_first_last(start_date, end_date, timestamp)
            if resource not in resource_list:
                resource_list.add(resource)
                resource_calendars[resource] = RCalendar("%s_Schedule" % resource)
            update_calendar_from_log(resource_calendars[resource], timestamp, state in ["start", "assign"])
            if state in ["start", "assign"]:
                started_events[task_id] = trace_info.start_event(task_id, task_name, timestamp, resource, timestamp,
                                                                 None)
                task_sequence.append(task_id)
            elif state == "complete":
                if task_id in started_events:
                    event_info = trace_info.complete_event(started_events.pop(task_id), timestamp)
                    if task_id not in task_resource:
                        task_resource[task_id] = dict()
                        task_distribution[task_id] = dict()
                    if resource not in task_resource[task_id]:
                        task_resource[task_id][resource] = list()
                    task_resource[task_id][resource].append(event_info)
        is_correct, fired_tasks, pending_tokens = bpmn_graph.reply_trace(task_sequence, flow_arcs_frequency)
        if len(pending_tokens) > 0:
            task_missed_tokens += 1
            for flow_id in pending_tokens:
                if flow_id not in missed_tokens:
                    missed_tokens[flow_id] = 0
                missed_tokens[flow_id] += 1
        # if not is_correct:
        #     print(caseid)
        #     print('------------------------------------------------------')
        if is_correct:
            correct_traces += 1
        for i in range(0, len(task_sequence)):
            if task_sequence[i] not in task_fired_ratio:
                task_fired_ratio[task_sequence[i]] = [0, 0]
            if fired_tasks[i]:
                correct_activities += 1
                task_fired_ratio[task_sequence[i]][0] += 1
            task_fired_ratio[task_sequence[i]][1] += 1
        total_activities += len(fired_tasks)
        process_info.traces[caseid] = trace_info
        i += 1

    t_r = 100 * correct_traces / total_traces
    a_r = 100 * correct_activities / total_activities
    print("Correct Traces Ratio %.2f (Pass: %d, Fail: %d, Total: %d)" % (
        t_r, correct_traces, total_traces - correct_traces, total_traces))
    print("Correct Tasks  Ratio %.2f (Fire: %d, Fail: %d, Total: %d)" % (
        a_r, correct_activities, total_activities - correct_activities, total_activities))
    print("Missed Tokens Ratio  %.2f" % (100 * task_missed_tokens / total_traces))
    print('----------------------------------------------')
    # for task_id in task_fired_ratio:
    #     print("%s: %.2f (Fail: %d / %d)" % (
    #         task_id, 100 * task_fired_ratio[task_id][0] / task_fired_ratio[task_id][1],
    #         task_fired_ratio[task_id][1] - task_fired_ratio[task_id][0],
    #         task_fired_ratio[task_id][1]))
    # print('-----------------------------------------------')
    # for task_name in missed_tokens:
    #     print("%s: %d" % (task_name, missed_tokens[task_name]))
    min_dur = sys.float_info.max
    max_dur = 0
    for r_id in resource_calendars:
        min_dur = min(min_dur, resource_calendars[r_id].total_weekly_work)
        max_dur = max(max_dur, resource_calendars[r_id].total_weekly_work)
        # resource_calendars[r_id].print_calendar_info()
    print('Min Resource Weekly Work: %.2f ' % (min_dur / 3600))
    print('Max Resource Weekly Work: %.2f ' % (max_dur / 3600))
    return # Renmove this
    print('Saving Resource Calendars ...')
    json_map = dict()
    for r_id in resource_calendars:
        json_map[r_id] = resource_calendars[r_id].to_json()
    with open('./input_files/resource_calendars/%s_calendars.json' % f_name, 'w') as file_writter:
        json.dump(json_map, file_writter)

    # create_calendar_from_rule_associations(resource_assoc_times, resource_days_freq, start_date, end_date)

    print('Computing Branching Probability ...')
    gateways_branching = bpmn_graph.compute_branching_probability(flow_arcs_frequency)
    with open('./input_files/probability_distributions/%s_gateways_branching.json' % f_name, 'w') as file_writter:
        json.dump(gateways_branching, file_writter)

    print('Computing Arrival Times Distribution ...')
    arrival_dates.sort()
    for i in range(1, len(arrival_dates)):
        arrival_times.append((arrival_dates[i] - arrival_dates[i - 1]).total_seconds())
    with open('./input_files/probability_distributions/%s_arrival_times_distribution.json' % f_name,
              'w') as file_writter:
        json.dump(best_fit_distribution(arrival_times), file_writter)

    print('Computing Task-Resource Distributions ...')
    for task_id in task_resource:
        for resorce_id in task_resource[task_id]:
            real_durations = list()
            for e_info in task_resource[task_id][resorce_id]:
                real_durations.append(resource_calendars[resorce_id].find_working_time(e_info.started_at,
                                                                                       e_info.completed_at))

                if real_durations[len(real_durations) - 1] <= 0 and e_info.started_at != e_info.completed_at:
                    x = resource_calendars[resorce_id].find_working_time(e_info.started_at, e_info.completed_at)
                    print(real_durations[len(real_durations) - 1])
            task_distribution[task_id][resorce_id] = best_fit_distribution(real_durations)
    with open('./input_files/probability_distributions/%s_task_distribution.json' % f_name, 'w') as file_writter:
        json.dump(task_distribution, file_writter)
    print('----------------------------------------------------------------------------------')
    return process_info


def _update_first_last(start_date, end_date, current_date):
    if start_date is None:
        start_date = current_date
        end_date = current_date
    start_date = min(start_date, current_date)
    end_date = max(end_date, current_date)
    return start_date, end_date