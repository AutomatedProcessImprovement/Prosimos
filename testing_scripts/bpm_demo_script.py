import os
import datetime

from bpdfr_discovery.log_parser import preprocess_xes_log
from bpdfr_simulation_engine.resource_calendar import parse_datetime
from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation
from testing_scripts.bpm_2022_testing_files import process_files


def main():
    model_info = process_files['demo_example']
    print('--------------------------------------------------------------------------')
    print("Starting Discovery of demo example ...")
    print('--------------------------------------------------------------------------')
    [granule, conf, supp, part, adj_c] = model_info['disc_params']

    start = datetime.datetime.now()
    preprocess_xes_log(model_info['xes_log'], model_info['bpmn'], model_info['json'], granule, conf, supp, part, adj_c)
    # preprocess_xes_log(model_info['csv_log'], model_info['bpmn'], model_info['json'], granule, conf, supp, part, adj_c, True)
    print("Discovery Time: %s" % str(datetime.timedelta(seconds=(datetime.datetime.now() - start).total_seconds())))

    print('--------------------------------------------------------------------------')
    print("Starting Simulation of demo example (%d instances)" % (model_info['total_cases']))
    print('--------------------------------------------------------------------------')
    start = datetime.datetime.now()
    _, diff_sim_result = run_diff_res_simulation(parse_datetime(model_info['start_datetime'], True),
                                                 model_info['total_cases'],
                                                 model_info["bpmn"],
                                                 model_info["json"],
                                                 model_info['demo_stats'],
                                                 None)
    print("Simulation Time: %s" % str(datetime.timedelta(seconds=(datetime.datetime.now() - start).total_seconds())))
    diff_sim_result.print_simulation_results()
    os._exit(0)


if __name__ == "__main__":
    main()
