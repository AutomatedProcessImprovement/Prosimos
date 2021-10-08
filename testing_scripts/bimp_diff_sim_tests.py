import os

from bpdfr_simulation_engine.simulation_properties_parser import parse_qbp_simulation_process
from diff_res_bpsim import load_simulation_info, start_simulation

experiment_models = {'bimp_example': {'bpmn': './../bimp_test_examples/BIMP_example.bpmn',
                                      'json': './../bimp_test_examples/bimp_example.json'},
                     'insurance_claim': {
                         'bpmn': './../bimp_test_examples/ch7_InsuranceClaimsSimulatio-StormScenario.bpmn',
                         'json': './../bimp_test_examples/ch7_InsuranceClaimsSimulatio-StormScenario.json'},
                     'csv_pharmacy': {'bpmn': './../bimp_test_examples/CVS-Pharmacy.bpmn',
                                      'json': './../bimp_test_examples/CVS-Pharmacy.json'}}


def main():
    for model_name in experiment_models:
        # this funtion is for generating the JSON if they don't exist
        # parse_qbp_simulation_process(experiment_models[model_name]['bpmn'], experiment_models[model_name]['json'])

        load_simulation_info(experiment_models[model_name]["bpmn"],
                             experiment_models[model_name]["json"])

        # # Extracting the simulation parameters from event-log (it saves them results to JSON files)
        # # xes_path = xes_simodbpmn_file_paths[log_name][0]
        # # bpmn_graph = parse_simulation_model(log_name)
        # # parse_xes_log(xes_path, bpmn_graph)
        #
        # # Loading the simulation parameters from file, the following method loads all the parameters from files,
        # # see the constructor of the class 'SimDiffSetup'

        total_cases = 5

        start_simulation(total_cases,
                         './../output_files/diffsim_logs/%s_%d_stats.csv' % (model_name, total_cases),
                         './../output_files/diffsim_logs/%s_%d_log.csv' % (model_name, total_cases))
        break

    os._exit(0)


if __name__ == "__main__":
    main()
