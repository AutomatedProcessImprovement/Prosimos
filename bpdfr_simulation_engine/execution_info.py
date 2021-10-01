import datetime
import pytz


class TaskEvent:
    def __init__(self, p_case, task_id, task_name, enabled_at, enabled_by):
        self.p_case = p_case
        self.task_id = task_id
        self.task_name = task_name
        self.enabled_at = enabled_at
        self.enabled_by = enabled_by
        self.started_at = None
        self.completed_at = None
        self.idle_time = None
        self.resource_id = None

    def print_event_info(self):
        print("Task: %s(%s)" % (self.task_name, str(self.p_case)))

    def start_event(self, started_at, resource_id):
        self.started_at = started_at
        self.resource_id = resource_id

    def complete_event(self, ended_at, idle_time):
        self.completed_at = ended_at
        self.idle_time = idle_time

    def waiting_time(self):
        return (self.started_at - self.enabled_at).total_seconds()

    def idle_processing_time(self):
        return (self.completed_at - self.started_at).total_seconds()

    def processing_time(self):
        return self.idle_processing_time() - self.idle_time

    def idle_cycle_time(self):
        return (self.completed_at - self.enabled_at).total_seconds()

    def cycle_time(self):
        return self.idle_cycle_time() - self.idle_time


class Trace:
    def __init__(self, p_case, started_at=datetime.datetime(9999, 12, 31, 23, 59, 59, 999999, pytz.utc)):
        self.p_case = p_case
        self.started_at = started_at
        self.completed_at = started_at
        self.event_list = list()
        self.next_parallel_tasks = list()

        self.cycle_time = None
        self.idle_cycle_time = None
        self.processing_time = None
        self.idle_processing_time = None
        self.waiting_time = None
        self.idle_time = None

    def start_event(self, task_id, task_name, started_at, started_by, enabled_at, enabled_by):
        event_info = TaskEvent(self.p_case, task_id, task_name, enabled_at, enabled_by)
        event_index = len(self.event_list)
        self.event_list.append(event_info)
        self.started_at = min(self.started_at, enabled_at)
        self.next_parallel_tasks.append(list())
        if enabled_by is not None:
            self.next_parallel_tasks[enabled_by].append(event_index)
        self.event_list[event_index].start_event(started_at, started_by)
        return event_index

    def complete_event(self, event_index, completed_at, idle_time=0):
        self.event_list[event_index].complete_event(completed_at, idle_time)
        self.completed_at = max(self.completed_at, self.event_list[event_index].completed_at)
        return self.event_list[event_index]


class ProcessInfo:
    def __init__(self):
        self.traces = dict()
        self.resource_profiles = dict()




