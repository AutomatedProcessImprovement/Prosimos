import os
import datetime

from bpdfr_discovery.log_parser import preprocess_xes_log
from bpdfr_simulation_engine.simulation_engine import run_simulation
from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation
from testing_scripts.bpm_2022_testing_files import process_files


def main():
    model_info = process_files['and_example']
    # print('--------------------------------------------------------------------------')
    # print("Starting Discovery of demo example ...")
    # print('--------------------------------------------------------------------------')
    # [granule, conf, supp, part, adj_c] = model_info['disc_params']
    #
    # start = datetime.datetime.now()
    # # preprocess_xes_log(model_info['xes_log'], model_info['bpmn'], model_info['json'], granule, conf, supp, part, adj_c)
    # preprocess_xes_log(model_info['csv_log'], model_info['bpmn'], model_info['json'], granule, conf, supp, part, adj_c,
    #                    True)
    # print("Discovery Time: %s" % str(datetime.timedelta(seconds=(datetime.datetime.now() - start).total_seconds())))

    print('--------------------------------------------------------------------------')
    print("Starting Simulation of demo example (%d instances)" % (model_info['total_cases']))
    print('--------------------------------------------------------------------------')
    start = datetime.datetime.now()

    # run_diff_res_simulation(start_date, total_cases, bpmn_model, json_sim_params, out_stats_csv_path, out_log_csv_path)
    # run_simulation(bpmn_model, json_sim_params, total_cases, out_stats_csv_path, out_log_csv_path, start_date)

    # sim_kpi = run_simulation(model_info["bpmn"], model_info["json"], model_info['total_cases'], None, None,
    #                          model_info['start_datetime'])

    _, diff_sim_result = run_diff_res_simulation(model_info['start_datetime'],
                                                 model_info['total_cases'],
                                                 model_info["bpmn"],
                                                 model_info["json"],
                                                 model_info['demo_stats'],
                                                 model_info['sim_log'])
    print("Simulation Time: %s" % str(datetime.timedelta(seconds=(datetime.datetime.now() - start).total_seconds())))
    diff_sim_result.print_simulation_results()
    os._exit(0)


if __name__ == "__main__":
    main()
