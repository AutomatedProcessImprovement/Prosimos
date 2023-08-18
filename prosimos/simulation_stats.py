import csv

import pytz

from prosimos.resource_profile import PoolInfo
from prosimos.simulation_stats_calculator import KPIMap, KPIInfo

import datetime
import re


class AggregatedPoolInfo:
    def __init__(self, pool_id, pool_name):
        self.pool_info = PoolInfo(pool_id, pool_name)
        self.kpi_allocated_tasks = KPIInfo()
        self.kpi_worked_time = KPIInfo()
        self.kpi_available_time = KPIInfo()
        self.kpi_utilization = KPIInfo()


class SimulationResult:
    def __init__(self, started_at, ended_at):
        self.started_at = started_at
        self.ended_at = ended_at

        self.process_kpi_map = KPIMap()
        self.tasks_kpi_map = dict()

        self.resource_utilization = dict()
        self.resource_info = dict()
        self.aggregated_pool_info = dict()
        self._max_res_s = 26

    def print_simulation_results(self):
        print("First process instance started at:  %s" % str(self.started_at))
        print("Last process instance completed at: %s" % str(self.ended_at))
        print("Total Duration: %s" % str(datetime.timedelta(seconds=(self.simulation_duration()))))
        print("Total instances simulated: %d" % self.process_kpi_map.cycle_time.count)
        print('------------------------------------------------------------')
        print('AVERAGE KPI VALUES -- FULL PROCESS')
        print('Waiting Time .......... %s' % format_duration(self.process_kpi_map.waiting_time.avg, 25))
        print('Idle Cycle Time ....... %s' % format_duration(self.process_kpi_map.idle_cycle_time.avg, 25))
        print('Cycle Time ............ %s' % format_duration(self.process_kpi_map.cycle_time.avg, 25))
        print('Idle Processing Time .. %s' % format_duration(self.process_kpi_map.idle_processing_time.avg, 25))
        print('Processing Time ....... %s' % format_duration(self.process_kpi_map.processing_time.avg, 25))
        print('Idle Time  ............ %s' % format_duration(self.process_kpi_map.idle_time.avg, 25))
        print('------------------------------------------------------------')
        # print('MIN KPI VALUES -- FULL PROCESS')
        # print('Waiting Time .......... %s' % format_duration(self.process_kpi_map.waiting_time.min, 25))
        # print('Idle Cycle Time ....... %s' % format_duration(self.process_kpi_map.idle_cycle_time.min, 25))
        # print('Cycle Time ............ %s' % format_duration(self.process_kpi_map.cycle_time.min, 25))
        # print('Idle Processing Time .. %s' % format_duration(self.process_kpi_map.idle_processing_time.min, 25))
        # print('Processing Time ....... %s' % format_duration(self.process_kpi_map.processing_time.min, 25))
        # print('Idle Time  ............ %s' % format_duration(self.process_kpi_map.idle_time.min, 25))
        # print('------------------------------------------------------------')
        # print('AGGREGATED RESOURCE POOL UTILIZATION')
        r_s = max(self._max_res_s, 13)
        # print('| %s | %s | %s | %s | %s | %s |' % ('Pool Name'.ljust(r_s),
        #                                            'Avg Resource Utilization'.ljust(25),
        #                                            'Avg Tasks Allocated'.ljust(20),
        #                                            'Total Tasks Allocated'.ljust(21),
        #                                            'Avg Time Worked'.ljust(25),
        #                                            'Avg Time Available'.ljust(25)))
        # for pool_id in self.aggregated_pool_info:
        #     pool_kpi: AggregatedPoolInfo = self.aggregated_pool_info[pool_id]
        #     print('| %s | %s | %s | %s | %s | %s |' % (pool_kpi.pool_info.pool_name.ljust(r_s),
        #                                                str(round(pool_kpi.kpi_utilization.avg, 3)).ljust(25),
        #                                                str(round(pool_kpi.kpi_allocated_tasks.avg)).ljust(20),
        #                                                str(pool_kpi.kpi_allocated_tasks.total).ljust(21),
        #                                                format_duration(pool_kpi.kpi_worked_time.avg, 25),
        #                                                format_duration(pool_kpi.kpi_available_time.avg, 25)))
        # print(len(self.resource_utilization))
        # print('------------------------------------------------------------')
        print('INDIVIDUAL RESOURCE UTILIZATION')
        print('| %s | %s | %s | %s | %s | %s |' % ('Resource Name'.ljust(r_s),
                                                   'Utilization'.ljust(11),
                                                   'Tasks Allocated'.ljust(16),
                                                   'Time Worked'.ljust(25),
                                                   'Time Available'.ljust(25),
                                                   'Pool Name'.ljust(15)))
        for res_name in self.resource_utilization:
            print('| %s | %s | %s | %s | %s | %s |' % (res_name.ljust(r_s),
                                                       str(round(self.resource_utilization[res_name], 3)).ljust(11),
                                                       str(self.resource_info[res_name][0]).ljust(16),
                                                       format_duration(self.resource_info[res_name][1], 25),
                                                       format_duration(self.resource_info[res_name][2], 25),
                                                       self.resource_info[res_name][3].ljust(15)))
        print('------------------------------------------------------------')
        print('INDIVIDUAL TASK KPI')
        max_t = 0
        d_s = 23
        c_s = 5
        for task_name in self.tasks_kpi_map:
            max_t = max(max_t, len(task_name))
            c_s = max(c_s, len(str(self.tasks_kpi_map[task_name].cycle_time.count)))
        print('| %s | %s | %s | %s | %s | %s | %s | %s |' % ('Task Name'.ljust(max_t),
                                                             'Count'.ljust(c_s),
                                                             'Waiting Time'.ljust(d_s),
                                                             'Idle Cycle Time'.ljust(d_s),
                                                             'Cycle Time'.ljust(d_s),
                                                             'Idle Processing Time'.ljust(d_s),
                                                             'Processing Time'.ljust(d_s),
                                                             'Idle Time'.ljust(d_s),))

        for task_name in self.tasks_kpi_map:
            kpi_info: KPIMap = self.tasks_kpi_map[task_name]
            print('| %s | %s | %s | %s | %s | %s | %s | %s |' % (task_name.ljust(max_t),
                                                                 str(kpi_info.cycle_time.count).ljust(c_s),
                                                                 format_duration(kpi_info.waiting_time.avg, d_s),
                                                                 format_duration(kpi_info.idle_cycle_time.avg, d_s),
                                                                 format_duration(kpi_info.cycle_time.avg, d_s),
                                                                 format_duration(kpi_info.idle_processing_time.avg,
                                                                                 d_s),
                                                                 format_duration(kpi_info.processing_time.avg, d_s),
                                                                 format_duration(kpi_info.idle_time.avg, d_s)))

        print('------------------------------------------------------------')

    def get_kpi_ref_list(self):
        return [self.process_kpi_map.cycle_time, self.process_kpi_map.processing_time,
                self.process_kpi_map.idle_cycle_time, self.process_kpi_map.idle_processing_time,
                self.process_kpi_map.waiting_time, self.process_kpi_map.idle_time]

    def update_resource_utilization(self, r_name, r_utilization, alloc_tasks=None, work_time=None, available_time=None,
                                    p_id=None, p_name=None):
        self._max_res_s = max(self._max_res_s, len(r_name))
        self.resource_utilization[r_name] = r_utilization
        if alloc_tasks is not None:
            self.resource_info[r_name] = [alloc_tasks, work_time, available_time, p_name]
            if p_id not in self.aggregated_pool_info:
                self.aggregated_pool_info[p_id] = AggregatedPoolInfo(p_id, p_name)
            self.aggregated_pool_info[p_id].kpi_allocated_tasks.add_value(alloc_tasks)
            self.aggregated_pool_info[p_id].kpi_worked_time.add_value(work_time)
            self.aggregated_pool_info[p_id].kpi_available_time.add_value(available_time)
            self.aggregated_pool_info[p_id].kpi_utilization.add_value(r_utilization)

    def simulation_duration(self):
        return (self.ended_at - self.started_at).total_seconds()

    def resource_utilization_for(self, pool_name):
        return self.resource_utilization[pool_name] if pool_name in self.resource_utilization[pool_name] else 0


def load_bimp_simulation_results(results_file_path, simulation_log):
    started_at, ended_at = extract_simulation_dates_from_simulation_log(simulation_log)
    bimp_sim_info = SimulationResult(started_at, ended_at)
    with open(results_file_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        output_section = 0
        for row in csv_reader:
            if len(row) > 1:
                if row[0] == "Resource":
                    output_section = 1
                    continue
                elif row[0] == "Name":
                    output_section = 2
                    continue
                elif row[0] == "KPI":
                    output_section = 3
                    continue
                if output_section == 1:
                    bimp_sim_info.update_resource_utilization(row[0], float(row[1]) / 100)
                elif output_section == 2:
                    task_name = row[0]
                    task_kpi = KPIMap()
                    occur = int(row[25])

                    task_kpi.cycle_time.set_values(float(row[2]), float(row[3]), float(row[1]), float(row[4]), occur)
                    task_kpi.waiting_time.set_values(float(row[6]), float(row[7]), float(row[5]), float(row[8]), occur)
                    task_kpi.idle_time.set_values(float(row[10]), float(row[11]), float(row[9]), float(row[12]), occur)

                    bimp_sim_info.tasks_kpi_map[task_name] = task_kpi

                elif output_section == 3:
                    if row[0] == "Process Cycle Time (s)":
                        bimp_sim_info.process_kpi_map.cycle_time.set_values(float(row[1]), float(row[3]), float(row[2]))
                    if row[0] == "Process Cycle Time excluding out of timetable hours (s)":
                        bimp_sim_info.process_kpi_map.idle_cycle_time.set_values(float(row[1]), float(row[3]),
                                                                                 float(row[2]))
                    if row[0] == "Process Waiting Time (s)":
                        bimp_sim_info.process_kpi_map.waiting_time.set_values(float(row[1]), float(row[3]),
                                                                              float(row[2]))
    return bimp_sim_info


def load_diff_simulation_results(csv_stats_path):
    sim_info = None
    started_at = None
    with open(csv_stats_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        output_section = 0
        kpi_index = 0
        kpi_array = None
        for row in csv_reader:
            if len(row) > 1:
                if row[0] == "started_at":
                    started_at = parse_date(row[1])
                    continue
                if row[0] == "completed_at":
                    sim_info = SimulationResult(started_at, parse_date(row[1]))
                    kpi_array = sim_info.get_kpi_ref_list()
                    continue
                if row[0] == "Resource ID":
                    output_section = 1
                    continue
                elif row[0] == "Name":
                    output_section = 2
                    continue
                elif row[0] == "KPI":
                    output_section = 3
                    continue
                if output_section == 1:
                    sim_info.update_resource_utilization(row[1], float(row[2]), int(row[3]), float(row[4]),
                                                         float(row[5]), row[6], row[7])
                elif output_section == 2:
                    task_name = row[0]
                    task_kpi = KPIMap()
                    occur = int(row[1])

                    task_kpi.duration.set_values(float(row[2]), float(row[3]), float(row[4]), float(row[5]), occur)
                    task_kpi.waiting_time.set_values(float(row[6]), float(row[7]), float(row[8]), float(row[9]), occur)
                    task_kpi.processing_time.set_values(float(row[10]), float(row[11]), float(row[12]), float(row[13]),
                                                        occur)
                    task_kpi.cycle_time.set_values(float(row[14]), float(row[15]), float(row[16]), float(row[17]),
                                                   occur)
                    task_kpi.idle_time.set_values(float(row[18]), float(row[19]), float(row[20]), float(row[21]), occur)
                    task_kpi.idle_cycle_time.set_values(float(row[22]), float(row[23]), float(row[24]), float(row[25]),
                                                        occur)
                    task_kpi.idle_processing_time.set_values(float(row[26]), float(row[27]), float(row[28]),
                                                             float(row[29]), occur)
                    task_kpi.cost.set_values(float(row[30]), float(row[31]), float(row[32]), float(row[33]), occur)

                    sim_info.tasks_kpi_map[task_name] = task_kpi
                elif output_section == 3:
                    kpi_array[kpi_index].set_values(float(row[1]), float(row[2]), float(row[3]),
                                                    float(row[4]), float(row[5]))
                    kpi_index += 1

    return sim_info


def _verify_file_section(row):
    if row[0] == "Resource":
        return True, 1
    elif row[0] == "Name":
        return True, 2
    elif row[0] == "KPI":
        return True, 3
    return False, 0


def extract_simulation_dates_from_simulation_log(file_path):
    simulation_start_date = None
    simulation_end_date = None
    with open(file_path) as file_reader:
        for line in file_reader:
            if 'Simulation started at' in line:
                simulation_start_date = parse_date(line)
                if simulation_end_date is not None:
                    break
            elif 'Simulation ended at' in line:
                simulation_end_date = parse_date(line)
                if simulation_start_date is not None:
                    break
    return simulation_start_date, simulation_end_date


def parse_date(date_str):
    return pytz.utc.localize(datetime.datetime.strptime(
        re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', date_str).group(), '%Y-%m-%d %H:%M:%S'))


def format_duration(time_seconds, space_size):
    return str(datetime.timedelta(seconds=time_seconds)).ljust(space_size)
