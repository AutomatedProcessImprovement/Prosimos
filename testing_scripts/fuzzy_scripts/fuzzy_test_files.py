from enum import Enum

test_processes_fuzzy = {
    0: 'BPI_2012',
    1: 'BPI_2017',
    2: 'consulta_data_mining',
    3: 'loan_application',
    4: 'production',
    5: 'insurance',
    6: 'call_centre',
}

is_syntetic = {
    'BPI_2012': False,
    'BPI_2017': False,
    'consulta_data_mining': False,
    'production': True,
    'insurance': True,
    'loan_application': True,
    'call_centre': False,
    'GOV': False,
    'POC': False,
    'AC-CRD': False,
    'WK-ORD': False
}

has_gateway_info = {
    'BPI_20120': False,
    'BPI_2017': False,
    'consulta_data_mining': False,
    'production': True,
    'insurance': True,
    'loan_application': False,
    'call_centre': True,
}


class FileType(Enum):
    BPMN = 0
    ORIGINAL_CSV_LOG = 1
    TRAINING_CSV_LOG = 2
    TESTING_CSV_LOG = 3
    SIMULATION_JSON = 4
    SIMULATED_LOG = 5
    SIMULATION_STATS = 6
    CRISP_JSON = 7
    CRISP_LOG = 8
    CRISP_STATS = 9
    JOINT_STATS = 10
    GENERATOR_LOG = 11
    SYNTHETIC = 12
    GATEWAYS_INFO = 13
    TASK_DISTRIBUTIONS = 14


calendar_type_folder = {
    1: '1_c_24_7',
    2: '2_c_8_5',
    3: '3_c_8_5_m_a',
    4: '4_c_8_5_m_a_1_2',
    5: 'vacation'
}

root_folder_path = './../assets/fuzzy_calendars/'

original_data = "./../assets/fuzzy_calendars/original_data"
bpmn_models_path = f'{original_data}/bpmn_models'
original_csv_path = f'{original_data}/csv_logs'

discovery_path = "./../assets/fuzzy_calendars/discovery"
simulation_path = "./../assets/fuzzy_calendars/simulation"

root_path_in = "./../assets/fuzzy_calendars/in"
root_path_out = "./../assets/fuzzy_calendars/out"
train_logs_path = f'{root_path_in}/train'
test_logs_path = f'{root_path_in}/test'


def get_file_path_fuzzy(proc_name: str, file_type: FileType, granule=15, angle=0, file_index=0, calendar_type=1, even=True):
    if is_syntetic[proc_name]:
        return get_synthetic_file_path(proc_name, file_type, granule, angle, file_index, calendar_type, even)
    else:
        return get_real_file_path(proc_name, file_type, granule, angle, file_index)


def get_synthetic_file_path(proc_name, file_type, granule=60, angle=0, file_index=1, calendar_type=1, even=True):
    root_folder = f'./../assets/fuzzy_calendars/{proc_name}'
    inner_folder = f'{root_folder}/{calendar_type_folder[calendar_type]}'

    if file_type is FileType.BPMN:
        return f'{root_folder}/bpmn_model.bpmn'
    elif file_type is FileType.ORIGINAL_CSV_LOG:
        return f'{inner_folder}/original_csv_log_t.csv' if even else f'{inner_folder}/original_csv_log_f.csv'
    elif file_type is FileType.TRAINING_CSV_LOG:
        return f'{inner_folder}/training_csv_log_t.csv' if even else f'{inner_folder}/training_csv_log_f.csv'
    elif file_type is FileType.TESTING_CSV_LOG:
        return f'{inner_folder}/testing_csv_log_t.csv' if even else f'{inner_folder}/testing_csv_log_f.csv'
    elif file_type is FileType.SIMULATION_JSON:
        return f'{inner_folder}/jsons_params/gr{str(granule)}_an{str(angle)}_t.json' if even \
            else f'{inner_folder}/jsons_params/gr{str(granule)}_an{str(angle)}_f.json'
    elif file_type is FileType.CRISP_JSON:
        return f'{inner_folder}/jsons_params/crisp_calendar_t.json' if even \
            else f'{inner_folder}/jsons_params/crisp_calendar_f.json'
    elif file_type is FileType.SIMULATED_LOG:
        return f'{inner_folder}/simulated_csv_logs/gr{str(granule)}_an{str(angle)}_l{str(file_index)}_t.csv' if even \
            else f'{inner_folder}/simulated_csv_logs/gr{str(granule)}_an{str(angle)}_l{str(file_index)}_f.csv'
    elif file_type is FileType.CRISP_LOG:
        return f'{inner_folder}/simulated_csv_logs/crisp_simulated_log_l{str(file_index)}_t.csv' if even \
            else f'{inner_folder}/simulated_csv_logs/crisp_simulated_log_l{str(file_index)}_f.csv'
    elif file_type is FileType.SIMULATION_STATS:
        return f'{inner_folder}/simulated_csv_stats/gr{str(granule)}_an{str(angle)}_l{str(file_index)}_t.csv' if even \
            else f'{inner_folder}/simulated_csv_stats/gr{str(granule)}_an{str(angle)}_l{str(file_index)}_f.csv'
    elif file_type is FileType.CRISP_STATS:
        return f'{inner_folder}/simulated_csv_stats/crisp_simulation_stats_l{str(file_index)}_t.csv' if even \
            else f'{inner_folder}/simulated_csv_stats/crisp_simulation_stats_l{str(file_index)}_f.csv'
    elif file_type is FileType.JOINT_STATS:
        return f'{inner_folder}/simulated_csv_stats/joint_log_distance_metrics_t.csv' if even else \
            f'{inner_folder}/simulated_csv_stats/joint_log_distance_metrics_f.csv'
    elif file_type is FileType.GENERATOR_LOG:
        return f'{root_folder}/generator_log_1.csv' if even else f'{root_folder}/generator_log_2.csv'
    elif file_type is FileType.GATEWAYS_INFO:
        return f'{root_folder}/gateways_prob.json'
    elif file_type is FileType.TASK_DISTRIBUTIONS:
        return f'{root_folder}/res_task_distr.json'

    return None


def get_real_file_path(proc_name: str, file_type: FileType, granule=15, angle=0, file_index=0):
    root_folder = f'./../assets/fuzzy_calendars/{proc_name}'
    if file_type is FileType.BPMN:
        return f'{root_folder}/bpmn_model.bpmn'
    elif file_type is FileType.ORIGINAL_CSV_LOG:
        return f'{root_folder}/original_csv_log.csv'
    elif file_type is FileType.TRAINING_CSV_LOG:
        return f'{root_folder}/training_csv_log.csv'
    elif file_type is FileType.TESTING_CSV_LOG:
        return f'{root_folder}/testing_csv_log.csv'
    elif file_type is FileType.SIMULATION_JSON:
        return f'{root_folder}/jsons_params/gr{str(granule)}_an{str(angle)}.json'
    elif file_type is FileType.CRISP_JSON:
        return f'{root_folder}/jsons_params/crisp_calendar.json'
    elif file_type is FileType.SIMULATED_LOG:
        return f'{root_folder}/simulated_csv_logs/gr{str(granule)}_an{str(angle)}_l{str(file_index)}.csv'
    elif file_type is FileType.CRISP_LOG:
        return f'{root_folder}/simulated_csv_logs/crisp_simulated_log_l{str(file_index)}.csv'
    elif file_type is FileType.SIMULATION_STATS:
        return f'{root_folder}/simulated_csv_stats/gr{str(granule)}_an{str(angle)}_l{str(file_index)}.csv'
    elif file_type is FileType.CRISP_STATS:
        return f'{root_folder}/simulated_csv_stats/crisp_simulation_stats_l{str(file_index)}.csv'
    elif file_type is FileType.JOINT_STATS:
        return f'{root_folder}/simulated_csv_stats/joint_log_distance_metrics.csv'
    elif file_type is FileType.GENERATOR_LOG:
        return f'{root_folder}/generator_log.csv'
    return None


crisp_discovery_params = {
    'BPI_2012': [60, 0.5, 0.5, 0.1, True],
    'BPI_2017': [60, 0.3, 1.0, 0.2, True],
    'consulta_data_mining': [60, 0.1, 0.8, 0.1, True],
    'loan_application': [60, 0.1, 0.6, 0.1, True],
    'production': [60, 0.3, 1.0, 0.5, False],
    'insurance': [60, 0.3, 1.0, 0.5, False],
    'call_centre': [60, 0.3, 1.0, 0.5, False]
}

process_files = {
    'loan_SC_LU': {
        'bpmn_path': f'{root_path_in}/LoanOriginationModel.bpmn',
        'fuzzy_json': f'{root_path_in}/loan_SC_LU_fuzzy.json',
        'crisp_json': f'{root_path_in}/loan_SC_LU_full.json',
        'real_csv_log': f'{root_path_in}/loan_SC_LU.csv',
        'sim_csv_log': f'{root_path_out}/loan_SC_LU_log.csv',
        'sim_statistics': f'{root_path_out}/loan_SC_LU_stat.csv',
        'disc_params': [60, 0.1, 1.0, 0.2, True],
        'start_datetime': '2015-03-06 15:47:26+00:00',
        'total_cases': 1000,
    }

}
