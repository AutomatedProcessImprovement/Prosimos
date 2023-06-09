import os
import datetime

from bpdfr_discovery.log_parser import preprocess_xes_log
from bpdfr_simulation_engine.resource_calendar import parse_datetime
from fuzzy_engine.event_log_analyser import compute_log_statistics, get_starting_datetimes
from fuzzy_engine.fuzzy_parser import parse_json_sim_parameters
from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation
from testing_scripts.fuzzy_test_files import process_files


from log_distance_measures.config import EventLogIDs


def main():
    for proc_id in process_files:
        # discover_prosimos_original_calendars(proc_id)
        # return

        model_info = process_files[proc_id]

        # compute_log_statistics(model_info['csv_log'], model_info["bpmn"])
        # return

        print('--------------------------------------------------------------------------')
        print("Starting Simulation of process %s (%d instances)" % (proc_id, model_info['total_cases']))
        print('--------------------------------------------------------------------------')
        start = datetime.datetime.now()

        fixed_arrival_times = get_starting_datetimes(model_info['csv_log'])

        # diff_sim_result = run_crisp_prosimos(model_info, fixed_arrival_times)
        diff_sim_result = run_fuzzy_prosimos(model_info, fixed_arrival_times)
        print(
            "Simulation Time: %s" % str(datetime.timedelta(seconds=(datetime.datetime.now() - start).total_seconds())))
        diff_sim_result.print_simulation_results()

        # compute_log_statistics(model_info['csv_log'], model_info["bpmn"])

    os._exit(0)









def run_crisp_prosimos(model_info, fixed_arrival_times):
    _, diff_sim_result = run_diff_res_simulation(parse_datetime(model_info['start_datetime'], True),
                                                 model_info['total_cases'],
                                                 model_info["bpmn"],
                                                 model_info["full_json"],
                                                 False,
                                                 model_info['sim_stats'],
                                                 None, fixed_arrival_times)
    return diff_sim_result


def run_fuzzy_prosimos(model_info, fixed_arrival_times):
    _, diff_sim_result = run_diff_res_simulation(parse_datetime(model_info['start_datetime'], True),
                                                 model_info['total_cases'],
                                                 model_info["bpmn"],
                                                 model_info["fuzzy_json"],
                                                 True,
                                                 model_info['sim_stats'],
                                                 None, fixed_arrival_times)
    return diff_sim_result


def run_metrics():
    event_log_ids = EventLogIDs(  # These values are stored in DEFAULT_CSV_IDS
        case="case_id",
        activity="Activity",
        start_time="start_time",
        end_time="end_time"
    )




def discover_prosimos_original_calendars(model_name):
    model_info = process_files[model_name]
    print('--------------------------------------------------------------------------')
    print("Starting Discovery of demo example ...")
    print('--------------------------------------------------------------------------')
    [granule, conf, supp, part, adj_c] = model_info['disc_params']

    preprocess_xes_log(model_info['csv_log'], model_info['bpmn'], model_info['full_json'], granule, conf, supp, part,
                       adj_c, True)


if __name__ == "__main__":
    main()
