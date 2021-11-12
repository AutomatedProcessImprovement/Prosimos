import datetime
import os

from bpdfr_simulation_engine.resource_calendar import parse_datetime
from bpdfr_simulation_engine.simulation_engine import run_simulation
from bpdfr_simulation_engine.simulation_stats import load_bimp_simulation_results, load_diff_simulation_results

experiment_models = {'bimp_example': {'bpmn': './../bimp_test_examples/bimp_example.bpmn',
                                      'json': './../bimp_test_examples/bimp_example.json',
                                      'total_cases': 1000,
                                      'start_datetime': '2016-12-04T16:40:51.000Z'},
                     'insurance_claim': {
                         'bpmn': './../bimp_test_examples/ch7_InsuranceClaimsSimulatio-StormScenario.bpmn',
                         'json': './../bimp_test_examples/ch7_InsuranceClaimsSimulatio-StormScenario.json',
                         'total_cases': 1000,
                         'start_datetime': '2020-03-24T07:00:00.000Z'},
                     'csv_pharmacy': {'bpmn': './../bimp_test_examples/CVS-Pharmacy.bpmn',
                                      'json': './../bimp_test_examples/CVS-Pharmacy.json',
                                      'total_cases': 1000,
                                      'start_datetime': '2019-03-25T06:00:00.000Z'},
                     'production': {'bpmn': './../bimp_test_examples/ihar/production.bpmn',
                                    'json': './../bimp_test_examples/ihar/production.json',
                                    'total_cases': 1000,
                                    'start_datetime': '2012-03-12T23:59:59.999999+00:00'},

                     'purchasing_example': {'bpmn': './../bimp_test_examples/ihar/purchasing_example.bpmn',
                                            'json': './../bimp_test_examples/ihar/purchasing_example.json',
                                            'total_cases': 1000,
                                            'start_datetime': '2011-06-20T18:43:59.999999+00:00'},
                     }

output_dir_path = './../bimp_test_examples/sim_output/'


def run_bimp_simulation(model_file_path, results_file_path, simulation_log,
                        bimp_engine_path="./../bimp_simulation_engine/qbp-simulator-engine.jar"):
    s_t = datetime.datetime.now()
    if os.system(
            "java -jar {bimp_engine_path} {model_file_path} -csv {results_file_path} > {simulation_log}".format(
                bimp_engine_path=bimp_engine_path,
                model_file_path=model_file_path,
                results_file_path=simulation_log,
                simulation_log=results_file_path)):
        raise RuntimeError('program {} failed!')
    print("BimpSim Execution Times: %s" %
          str(datetime.timedelta(seconds=(datetime.datetime.now() - s_t).total_seconds())))
    # return load_bimp_simulation_results(results_file_path, simulation_log)


def run_diff_res_simulation(start_date, total_cases, bpmn_model, json_sim_params, out_stats_csv_path, out_log_csv_path):
    s_t = datetime.datetime.now()
    run_simulation(bpmn_model, json_sim_params, total_cases, out_stats_csv_path, out_log_csv_path, start_date)
    print("DiffSim Execution Times: %s" %
          str(datetime.timedelta(seconds=(datetime.datetime.now() - s_t).total_seconds())))
    return load_diff_simulation_results(out_stats_csv_path)


def main():
    for model_name in experiment_models:
        # parse_qbp_simulation_process(experiment_models[model_name]['bpmn'], experiment_models[model_name]['json'])

        p_cases = 1000
        if "total_cases" in experiment_models[model_name]:
            p_cases = experiment_models[model_name]["total_cases"]

        print('--------------------------------------------------------------------------')
        print("Starting Simulation of process %s (%d instances)" % (model_name, p_cases))
        print('--------------------------------------------------------------------------')
        run_bimp_simulation(experiment_models[model_name]["bpmn"],
                            '%sbimp_%s_%d_stats.csv' % (output_dir_path, model_name, p_cases),
                            '%sbimp_%s_%d_log.csv' % (output_dir_path, model_name, p_cases))

        diff_sim_result = run_diff_res_simulation(parse_datetime(experiment_models[model_name]['start_datetime'], True),
                                                  p_cases,
                                                  experiment_models[model_name]["bpmn"],
                                                  experiment_models[model_name]["json"],
                                                  '%sdiff_%s_%d_stats.csv' % (output_dir_path, model_name, p_cases),
                                                  '%sdiff_%s_%d_log.csv' % (output_dir_path, model_name, p_cases))
        # diff_sim_result.print_simulation_results()
        # break

    os._exit(0)


def code_for_generating_simulation_parameters():
    for model_name in experiment_models:
        # # this funtion is for generating the JSON if they don't exist
        # # this function takes as input the BPMN model extended with simulation parameters, i.e., BIMP model
        # parse_qbp_simulation_process(experiment_models[model_name]['bpmn'], experiment_models[model_name]['json'])

        # # Extracting the simulation parameters from event-log (it saves them results to JSON files)
        # # This discovers differentated resources model from XES log, i.e., required by the new simulation engine
        # xes_path = xes_simodbpmn_file_paths[log_name][0]
        # bpmn_graph = parse_simulation_model(log_name)
        # parse_xes_log(xes_path, bpmn_graph)

        break


if __name__ == "__main__":
    main()
