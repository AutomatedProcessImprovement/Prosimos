from bpdfr_discovery.log_comparison_metrics import compute_enabling_processing_times
from bpdfr_discovery.log_parser import event_list_from_csv

from prosimos.simulation_properties_parser import parse_simulation_model
from prosimos.simulation_stats import format_duration


def get_starting_datetimes(csv_log_path):
    traces = event_list_from_csv(csv_log_path)
    arrival_dates = []
    for trace in traces:
        trace.event_list.sort(key=lambda i_info: i_info.started_at)
        arrival_dates.append(trace.event_list[0].started_at)
    arrival_dates.sort()
    current_date = arrival_dates[0]
    current_second = 0
    arrival_times = [0]
    for i in range(1, len(arrival_dates)):
        current_second += (arrival_dates[i] - current_date).total_seconds()
        current_date = arrival_dates[i]
        arrival_times.append(current_second)
    return arrival_times, arrival_dates[0]


def compute_log_statistics(csv_log_path, bpmn_path):
    traces = event_list_from_csv(csv_log_path)
    bpmn_graph = parse_simulation_model(bpmn_path)
    max_waiting, max_processing = dict(), dict()

    task_mean_waiting_time = dict()
    task_mean_processing_time = dict()
    tasks_frequency = dict()
    max_t = 0
    total_events = 0
    for trace in traces:
        compute_enabling_processing_times(trace, bpmn_graph, max_waiting, max_processing)
        for ev in trace.event_list:
            total_events += 1
            if ev.task_id not in task_mean_waiting_time:
                max_t = max(max_t, len(ev.task_id))
                task_mean_waiting_time[ev.task_id] = 0.0
                tasks_frequency[ev.task_id] = 0.0
                task_mean_processing_time[ev.task_id] = 0
            task_mean_waiting_time[ev.task_id] += (ev.started_at - ev.enabled_at).total_seconds()
            tasks_frequency[ev.task_id] += 1
            task_mean_processing_time[ev.task_id] += (ev.completed_at - ev.started_at).total_seconds()
    for t_id in task_mean_waiting_time:
        task_mean_waiting_time[t_id] = task_mean_waiting_time[t_id] / tasks_frequency[t_id]
        task_mean_processing_time[t_id] = task_mean_processing_time[t_id] / tasks_frequency[t_id]

    print("Total Events: %d" % total_events)
    print('| %s | %s | %s | %s | ' % ('Task Name'.ljust(max_t),
                                      'Count'.ljust(25),
                                      'Waiting Time'.ljust(25),
                                      'Idle Processing Time'.ljust(25),))

    for t_id in task_mean_waiting_time:
        print('| %s | %s | %s | %s |' % (t_id.ljust(max_t),
                                         str(tasks_frequency[t_id]).ljust(25),
                                         format_duration(task_mean_waiting_time[t_id], 25),
                                         format_duration(task_mean_processing_time[t_id], 25)))
