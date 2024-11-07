from testing_scripts.fuzzy_scripts.fuzzy_test_files import FileType, calendar_type_folder

root_folder_path = './../assets/multitasking/'

original_data = "./../assets/multitasking/original_data"
bpmn_models_path = f'{original_data}/bpmn_models'
original_csv_path = f'{original_data}/csv_logs'

discovery_path = "./../assets/multitasking/discovery"
simulation_path = "./../assets/multitasking/simulation"

root_path_in = "./../assets/multitasking/in"
root_path_out = "./../assets/multitasking/out"
train_logs_path = f'{root_path_in}/train'
test_logs_path = f'{root_path_in}/test'

test_processes_multi = {
    0: 'loan_application',
    1: 'GOV',
    2: 'POC',
    3: 'AC-CRD',
    4: 'WK-ORD'
}

is_syntetic = {
    'loan_application': True,
    'GOV': False,
    'POC': False,
    'AC-CRD': False,
    'WK-ORD': False
}


def get_file_path_multi(proc_name: str, file_type: FileType, granule=15, angle=0, file_index=0, calendar_type=1,
                        even=True):
    if is_syntetic[proc_name]:
        return get_synthetic_file_path(proc_name, file_type, granule, angle, file_index, calendar_type, even)
    else:
        return get_real_file_path(proc_name, file_type, granule, angle, file_index)


def get_synthetic_file_path(proc_name, file_type, granule=60, angle=0, file_index=1, calendar_type=1, even=True):
    root_folder = f'./../assets/multitasking/{proc_name}'
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
    root_folder = f'./../assets/multitasking/{proc_name}'
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
