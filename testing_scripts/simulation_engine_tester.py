
import os


experiment_logs = {0: 'production',
                   1: 'purchasing_example',
                   2: 'consulta_data_mining',
                   3: 'insurance',
                   4: 'call_centre',
                   5: 'bpi_challenge_2012',
                   6: 'bpi_challenge_2017_filtered',
                   7: 'bpi_challenge_2017'}

xes_simodbpmn_file_paths = {
    'purchasing_example': ['./input_files/xes_files/PurchasingExample.xes',
                           './input_files/bpmn_simod_models/PurchasingExample.bpmn'],
    'production': ['./input_files/xes_files/production.xes',
                   './input_files/bpmn_simod_models/Production.bpmn'],
    'insurance': ['./input_files/xes_files/insurance.xes',
                  './input_files/bpmn_simod_models/insurance.bpmn'],
    'call_centre': ['./input_files/xes_files/callcentre.xes',
                    './input_files/bpmn_simod_models/callcentre.bpmn'],
    'bpi_challenge_2012': ['./input_files/xes_files/BPI_Challenge_2012_W_Two_TS.xes',
                           './input_files/bpmn_simod_models/BPI_Challenge_2012_W_Two_TS.bpmn'],
    'bpi_challenge_2017_filtered': ['./input_files/xes_files/BPI_Challenge_2017_W_Two_TS_filtered.xes',
                                    './input_files/bpmn_simod_models/BPI_Challenge_2017_W_Two_TS_filtered.bpmn'],
    'bpi_challenge_2017': ['./input_files/xes_files/BPI_Challenge_2017_W_Two_TS.xes',
                           './input_files/bpmn_simod_models/BPI_Challenge_2017_W_Two_TS.bpmn'],
    'consulta_data_mining': ['./input_files/xes_files/ConsultaDataMining201618.xes',
                             './input_files/bpmn_simod_models/ConsultaDataMining201618.bpmn']
}

def main():
    for i in range(0, 8):
        log_name = experiment_logs[i]

        # Extracting the simulation parameters from event-log (it saves them results to JSON files)
        # xes_path = xes_simodbpmn_file_paths[log_name][0]
        # bpmn_graph = parse_simulation_model(log_name)
        # parse_xes_log(xes_path, bpmn_graph)

        # Loading the simulation parameters from file, the following method loads all the parameters from files,
        # see the constructor of the class 'SimDiffSetup'

        total_cases = 100

        break

    os._exit(0)


if __name__ == "__main__":
    main()