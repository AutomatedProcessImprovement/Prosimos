import datetime
import pytz


class TaskEvent:
    def __init__(self, p_case, task_id, resource_id, resource_available_at, enabled_at, enabled_datetime, bpm_env):
        self.p_case = p_case  # ID of the current trace, i.e., index of the trace in log_info list
        self.task_id = task_id  # Name of the task related to the current event
        self.resource_id = resource_id  # ID of the resource performing to the event

        # Time moment in seconds from beginning, i.e., first event has time = 0
        self.enabled_at = enabled_at
        # Datetime of the time-moment calculated from the starting simulation datetime
        self.enabled_datetime = enabled_datetime

        # Time moment in seconds from beginning, i.e., first event has time = 0
        self.started_at = max(resource_available_at, enabled_at)
        # Datetime of the time-moment calculated from the starting simulation datetime
        self.started_datetime = bpm_env.simulation_datetime_from(self.started_at)

        # Ideal duration from the distribution-function if allocate resource doesn't rest
        self.ideal_duration = bpm_env.sim_setup.ideal_task_duration(task_id, resource_id)
        # Actual duration adding the resource resting-time according to their calendar
        self.real_duration = bpm_env.sim_setup.real_task_duration(self.ideal_duration, self.resource_id,
                                                                  self.started_datetime)

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
        self.next_parallel_tasks = list()

        self.cycle_time = None
        self.idle_cycle_time = None
        self.processing_time = None
        self.idle_processing_time = None
        self.waiting_time = None
        self.idle_time = None


class EnabledEvent:
    def __init__(self, p_case, p_state, task_id, enabled_at, enabled_datetime):
        self.p_case = p_case
        self.p_state = p_state
        self.task_id = task_id
        self.enabled_datetime = enabled_datetime
        self.enabled_at = enabled_at
