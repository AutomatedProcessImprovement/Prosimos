import os

from bpdfr_simulation_engine.simulation_stats import load_bimp_simulation_results, load_diff_simulation_results
from diff_res_bpsim import load_simulation_info, start_simulation

experiment_models = {'bimp_example': {'bpmn': './../bimp_test_examples/BIMP_example.bpmn',
                                      'json': './../bimp_test_examples/bimp_example.json'},
                     'insurance_claim': {
                         'bpmn': './../bimp_test_examples/ch7_InsuranceClaimsSimulatio-StormScenario.bpmn',
                         'json': './../bimp_test_examples/ch7_InsuranceClaimsSimulatio-StormScenario.json'},
                     'csv_pharmacy': {'bpmn': './../bimp_test_examples/CVS-Pharmacy.bpmn',
                                      'json': './../bimp_test_examples/CVS-Pharmacy.json'}}

output_dir_path = './../bimp_test_examples/sim_output/'


def run_bimp_simulation(model_file_path, results_file_path, simulation_log,
                        bimp_engine_path="./../bimp_simulation_engine/qbp-simulator-engine.jar"):
    if os.system(
            "java -jar {bimp_engine_path} {model_file_path} -csv {results_file_path} > {simulation_log}".format(
                bimp_engine_path=bimp_engine_path,
                model_file_path=model_file_path,
                results_file_path=results_file_path,
                simulation_log=simulation_log)):
        raise RuntimeError('program {} failed!')
    return load_bimp_simulation_results(results_file_path, simulation_log)


def run_diff_res_simulation(start_date, total_cases, bpmn_model, json_sim_params, out_stats_csv_path, out_log_csv_path):
    load_simulation_info(bpmn_model, json_sim_params)
    start_simulation(total_cases, out_stats_csv_path, out_log_csv_path, start_date)
    return load_diff_simulation_results(out_stats_csv_path)


def main():
    p_cases = 500
    for model_name in experiment_models:
        bimp_result = run_bimp_simulation(experiment_models[model_name]["bpmn"],
                                          '%sbimp_%s_%d_stats.csv' % (output_dir_path, model_name, p_cases),
                                          '%sbimp_%s_%d_log.csv' % (output_dir_path, model_name, p_cases))

        diff_sim_result = run_diff_res_simulation(bimp_result.started_at, p_cases,
                                                  experiment_models[model_name]["bpmn"],
                                                  experiment_models[model_name]["json"],
                                                  '%sdiff_%s_%d_stats.csv' % (output_dir_path, model_name, p_cases),
                                                  '%sdiff_%s_%d_log.csv' % (output_dir_path, model_name, p_cases))
        diff_sim_result.print_simulation_results()
        break

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
