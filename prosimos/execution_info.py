import datetime
import pytz

from prosimos.control_flow_manager import BPMN, BatchInfoForExecution


class EnabledEvent:
    def __init__(self, p_case, p_state, task_id, enabled_at, enabled_datetime, 
        batch_info_exec: BatchInfoForExecution = None, duration_sec = None, is_inter_event = False):
        self.p_case = p_case
        self.p_state = p_state
        self.task_id = task_id
        self.enabled_datetime = enabled_datetime
        self.enabled_at = enabled_at
        self.batch_info_exec = batch_info_exec
        self.duration_sec = duration_sec        # filled only in case of event-based gateway
        self.is_inter_event = is_inter_event    # whether the enabled event is the intermediate event


class ProcessInfo:
    def __init__(self):
        self.traces = dict()
        self.resource_profiles = dict()


class TaskEvent:
    def __init__(self, p_case, task_id, resource_id, resource_available_at=None, 
        enabled_at=None, enabled_datetime=None, bpm_env=None, num_tasks_in_batch=0):
        self.p_case = p_case  # ID of the current trace, i.e., index of the trace in log_info list
        self.task_id = task_id  # Name of the task related to the current event
        self.type = BPMN.TASK # showing whether it's task or event
        self.resource_id = resource_id  # ID of the resource performing to the event
        self.waiting_time = None
        self.processing_time = None
        self.normalized_waiting = None
        self.normalized_processing = None
        self.worked_intervals = []

        if resource_available_at is not None:
            # Time moment in seconds from beginning, i.e., first event has time = 0
            self.enabled_at = enabled_at
            # Datetime of the time-moment calculated from the starting simulation datetime
            self.enabled_datetime = enabled_datetime

            # Time moment in seconds from beginning, i.e., first event has time = 0
            self.started_at = max(resource_available_at, enabled_at)
            # Datetime of the time-moment calculated from the starting simulation datetime
            self.started_datetime = bpm_env.simulation_datetime_from(self.started_at)

            # Ideal duration from the distribution-function if allocate resource doesn't rest
            self.ideal_duration = bpm_env.sim_setup.ideal_task_duration(task_id, resource_id, num_tasks_in_batch)
            # Actual duration adding the resource resting-time according to their calendar
            self.real_duration = bpm_env.sim_setup.real_task_duration(self.ideal_duration, self.resource_id,
                                                                      self.started_datetime, self.worked_intervals)

            # Time moment in seconds from beginning, i.e., first event has time = 0
            self.completed_at = self.started_at + self.real_duration
            # Datetime of the time-moment calculated from the starting simulation datetime
            self.completed_datetime = bpm_env.simulation_datetime_from(self.completed_at)

            # Time of a resource was resting while performing a task (in seconds)
            self.idle_time = self.real_duration - self.ideal_duration
            # Time from an event is enabled until it is started by any resource
            self.waiting_time = self.started_at - self.enabled_at
            self.idle_cycle_time = self.completed_at - self.enabled_at
            self.idle_processing_time = self.completed_at - self.started_at
            self.cycle_time = self.idle_cycle_time - self.idle_time
            self.processing_time = self.idle_processing_time - self.idle_time
        else:
            self.task_name = None
            self.enabled_at = enabled_at
            self.enabled_by = None
            self.started_at = None
            self.completed_at = None
            self.idle_time = None

    @classmethod
    def create_event_entity(cls, c_event: EnabledEvent, ended_at, ended_datetime):
        cls.p_case = c_event.p_case  # ID of the current trace, i.e., index of the trace in log_info list
        cls.task_id = c_event.task_id  # Name of the task related to the current event
        cls.type = BPMN.INTERMEDIATE_EVENT
        cls.enabled_at = c_event.enabled_at
        cls.enabled_datetime = c_event.enabled_datetime
        cls.started_at = c_event.enabled_at
        cls.started_datetime = c_event.enabled_datetime
        cls.completed_at = ended_at
        cls.completed_datetime = ended_datetime
        cls.idle_time = 0.0
        cls.waiting_time = 0.0
        cls.idle_cycle_time = 0.0
        cls.idle_processing_time = 0.0
        cls.cycle_time = 0.0
        cls.processing_time = 0.0

        return cls

    def update_enabling_times(self, enabled_at):
        # what's the use case ?
        if self.started_at is None or enabled_at > self.started_at:
            # print(self.task_id)
            # print(str(enabled_at))
            # print(str(self.started_at))
            # print("--------------------------------------------")
            enabled_at = self.started_at
            # raise Exception("Task ENABLED after STARTED")
        self.enabled_at = enabled_at
        self.waiting_time = (self.started_at - self.enabled_at).total_seconds()
        self.processing_time = (self.completed_at - self.started_at).total_seconds()


class LogEvent:
    def __int__(self, task_id, started_datetime, resource_id):
        self.task_id = task_id
        self.started_datetime = started_datetime
        self.resource_id = resource_id
        self.completed_datetime = None


class Trace:
    def __init__(self, p_case, started_at=datetime.datetime(9999, 12, 31, 23, 59, 59, 999999, pytz.utc)):
        self.p_case = p_case
        self.started_at = started_at
        self.completed_at = started_at
        self.event_list = list()

        self.cycle_time = None
        self.idle_cycle_time = None
        self.processing_time = None
        self.idle_processing_time = None
        self.waiting_time = None
        self.idle_time = None

    def start_event(self, task_id, task_name, started_at, resource_name):
        event_info = TaskEvent(self.p_case, task_id, resource_name)
        event_info.task_name = task_name
        event_info.started_at = started_at
        event_index = len(self.event_list)
        self.event_list.append(event_info)
        self.started_at = min(self.started_at, started_at)
        return event_index

    def complete_event(self, event_index, completed_at, idle_time=0):
        self.event_list[event_index].completed_at = completed_at
        self.event_list[event_index].idle_time = idle_time
        self.completed_at = max(self.completed_at, self.event_list[event_index].completed_at)
        return self.event_list[event_index]

    def sort_by_completion_date(self, completed_at=False):
        if completed_at:
            self.event_list.sort(key=lambda e_info: e_info.completed_at)
        else:
            self.event_list.sort(key=lambda e_info: e_info.started_at)
        self.started_at = self.event_list[0].started_at
        self.completed_at = self.event_list[len(self.event_list) - 1].completed_at

    def filter_incomplete_events(self):
        filtered_list = list()
        filtered_events = 0
        for ev_info in self.event_list:
            if ev_info.started_at is not None and ev_info.completed_at is not None:
                filtered_list.append(ev_info)
            else:
                filtered_events += 2
        self.event_list = filtered_list
        return filtered_events
