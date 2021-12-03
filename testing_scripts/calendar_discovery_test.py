
import os
from pathlib import Path

from bpdfr_discovery.support_modules.log_parser import parse_xes_log, preprocess_xes_log
from bpdfr_simulation_engine.simulation_properties_parser import parse_simulation_model

experiment_logs = {0: 'production',
                   1: 'purchasing_example',
                   2: 'consulta_data_mining',
                   3: 'insurance',
                   4: 'call_centre',
                   5: 'bpi_challenge_2012',
                   6: 'bpi_challenge_2017_filtered',
                   7: 'bpi_challenge_2017'}

xes_simodbpmn_file_paths = {
    'purchasing_example': ['./../input_files/xes_files/PurchasingExample.xes',
                           './../input_files/bpmn_simod_models/purchasing_example.bpmn'],
    'production': ['./../input_files/xes_files/production.xes',
                   './../input_files/bpmn_simod_models/Production.bpmn'],
    'insurance': ['./../input_files/xes_files/insurance.xes',
                  './../input_files/bpmn_simod_models/insurance.bpmn'],
    'call_centre': ['./../input_files/xes_files/callcentre.xes',
                    './../input_files/bpmn_simod_models/callcentre.bpmn'],
    'bpi_challenge_2012': ['./../input_files/xes_files/BPI_Challenge_2012_W_Two_TS.xes',
                           './../input_files/bpmn_simod_models/BPI_Challenge_2012_W_Two_TS.bpmn'],
    'bpi_challenge_2017_filtered': ['./../input_files/xes_files/BPI_Challenge_2017_W_Two_TS_filtered.xes',
                                    './../input_files/bpmn_simod_models/BPI_Challenge_2017_W_Two_TS_filtered.bpmn'],
    'bpi_challenge_2017': ['./../input_files/xes_files/BPI_Challenge_2017_W_Two_TS.xes',
                           './../input_files/bpmn_simod_models/BPI_Challenge_2017_W_Two_TS.bpmn'],
    'consulta_data_mining': ['./../input_files/xes_files/ConsultaDataMining201618.xes',
                             './../input_files/bpmn_simod_models/consulta_data_mining.bpmn']
}

output_dir_path = './../output_files/discovery/'

def main():
    for i in range(2, 8):
        log_name = experiment_logs[i]

        # Extracting the simulation parameters from event-log (it saves them results to JSON files)
        xes_path = xes_simodbpmn_file_paths[log_name][0]
        bpmn_path = xes_simodbpmn_file_paths[log_name][1]
        preprocess_xes_log(xes_path, 15, 0.5, 0.5)

        # bpmn_graph = parse_simulation_model(bpmn_path)
        # parse_xes_log(xes_path, bpmn_graph, output_dir_path)

        # Loading the simulation parameters from file, the following method loads all the parameters from files,
        # see the constructor of the class 'SimDiffSetup'

        total_cases = 100

        break

    os._exit(0)


if __name__ == "__main__":
    main()