import random
from datetime import datetime

from pix_framework.discovery.probabilistic_multitasking.discovery import MultiType
from prosimos.simulation_queues_ds import PriorityQueue


class MultiTaskDS:
    def __init__(self, mt_type: str, g_size: int = 0):
        self.mt_type = MultiTaskDS._extract_type(mt_type)
        self.g_size = g_size
        self.total_granules = 1440 // g_size if g_size > 0 else 1
        self.res_multitask_info = dict()
        self.allocated_tasks = dict()
        self.active_datetimes = dict()
        self.relative_workload = dict()
        self.expected_workload = dict()
        self.executed_tasks = dict()
        self.total_tasks = dict()

    def init_relative_workload(self, task_res_distr: dict):
        for t_id in task_res_distr:
            total_workload = 0.0
            for r_id in task_res_distr[t_id]:
                if r_id not in self.relative_workload:
                    self.relative_workload[r_id] = dict()
                    self.executed_tasks[r_id] = dict()
                total_workload += self.expected_workload[r_id]
            for r_id in task_res_distr[t_id]:
                if total_workload > 0:
                    self.relative_workload[r_id][t_id] = self.expected_workload[r_id] / total_workload
                else:
                    self.relative_workload[r_id][t_id] = 0.0
                self.executed_tasks[r_id][t_id] = 0
            self.total_tasks[t_id] = 0

    def update_expected_workload(self, r_id, workload):
        self.expected_workload[r_id] = workload

    def resource_workload(self, r_id, t_id):
        if r_id in self.executed_tasks and self.total_tasks[t_id] > 0:
            return self.executed_tasks[r_id][t_id] / self.total_tasks[t_id]
        return 0.0

    def workload_diff(self, r_id, t_id):
        if r_id in self.executed_tasks:
            return abs(self.resource_workload(r_id, t_id) - self.relative_workload[r_id][t_id])
        return 0.0

    def allocate_task_to(self, r_id: str, t_id: str, completed_at: datetime):
        if r_id in self.allocated_tasks:
            self.allocated_tasks[r_id] += 1
            if self.active_datetimes[r_id] is None:
                self.active_datetimes[r_id] = completed_at
            else:
                self.active_datetimes[r_id] = max(self.active_datetimes[r_id], completed_at)
            # self.executed_tasks[r_id][t_id] += 1
            # self.total_tasks[t_id] += 1

    def release_tasks_from(self, r_id: str, completed_at: datetime):
        if r_id in self.allocated_tasks and self.active_datetimes[r_id] is not None:
            completed_at = max(self.active_datetimes[r_id], completed_at)
            self.active_datetimes[r_id] = None
            self.allocated_tasks[r_id] = 0
        return completed_at

    def can_get_new_tasks(self, r_id: str, at_datetime: datetime):
        if r_id in self.allocated_tasks and self.allocated_tasks[r_id] > 0:
            if self.mt_type is MultiType.GLOBAL:
                if self.allocated_tasks[r_id] >= len(self.res_multitask_info[r_id]):
                    return False
                return random.random() <= self.res_multitask_info[r_id][self.allocated_tasks[r_id]]
            else:
                wd = at_datetime.weekday()
                gr = self.interval_index(at_datetime)
                if self.allocated_tasks[r_id] >= len(self.res_multitask_info[r_id][wd][gr]):
                    return False
                return random.random() <= self.res_multitask_info[r_id][wd][gr][self.allocated_tasks[r_id]]
        return True

    def register_resource(self, r_id: str):
        if r_id not in self.res_multitask_info:
            self.allocated_tasks[r_id] = 0
            self.active_datetimes[r_id] = None
            if self.mt_type is MultiType.GLOBAL:
                self.res_multitask_info[r_id] = [0.0]
            else:
                total_granules = self.total_granules
                self.res_multitask_info[r_id] = [[[0.0] for _ in range(total_granules)] for _ in range(7)]

    def register_multitasks(self, r_id: str, task_freq: int, prob: float, week_day: int = None, granule: int = None):
        if not 0 <= prob <= 1.0:
            raise ValueError("Probability 'prob' must be between 0 and 1.0 inclusive")
        if self.mt_type is MultiType.GLOBAL:
            self.res_multitask_info[r_id].extend([prob] * (task_freq - len(self.res_multitask_info[r_id])))
            self.res_multitask_info[r_id].append(prob)
        else:
            self.res_multitask_info[r_id][week_day][granule].extend(
                [prob] * (task_freq - len(self.res_multitask_info[r_id][week_day][granule]))
            )
            self.res_multitask_info[r_id][week_day][granule].append(prob)

    def register_local_multitasks(self, r_id: str, week_day: int, from_dt: datetime, to_dt: datetime, task_freq: int,
                                  prob: float):
        if week_day is None or not 0 <= week_day <= 6:
            raise ValueError("Weekdays must be between 0 and 6 inclusive - from 0: MONDAY to 6: SUNDAY")
        from_i = self.interval_index(from_dt)
        to_i = self.interval_index(to_dt) - 1
        if to_i < 0:
            to_i = self.total_granules - 1

        while from_i <= to_i:
            self.register_multitasks(r_id, task_freq, prob, week_day, from_i)
            from_i += 1

    def interval_index(self, current_date: datetime):
        return (current_date.hour * 60 + current_date.minute) // self.g_size

    @staticmethod
    def _extract_type(type_str: str):
        upp = type_str.upper()
        if upp == 'GLOBAL':
            return MultiType.GLOBAL
        elif upp == 'LOCAL':
            return MultiType.LOCAL
        raise ValueError("Multitasking Type (mt_type) must be GLOBAL or LOCAL")
