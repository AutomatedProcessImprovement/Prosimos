experiment_logs = {0: 'production',
                   1: 'purchasing_example',
                   2: 'consulta_data_mining',
                   3: 'bpi_2012',
                   4: 'bpi_2017',
                   5: 'loan_SC_LU',
                   6: 'loan_SC_HU',
                   7: 'loan_MC_LU',
                   8: 'loan_MC_HU',
                   9: 'insurance',
                   10: 'government'
                   }

process_files = {
    'purchasing_example':
        {
            'xes_log': './../input_output_files/discovery_input_files/xes_logs/PurchasingExample.xes',
            'csv_log': './../input_output_files/discovery_input_files/csv_logs/purchasing_example.csv',
            'real_csv_log': './../input_output_files/discovery_output_files/real_csv_logs/purchasing_example.csv',
            'bpmn': './../input_output_files/discovery_input_files/bpmn_models/purchasing_example.bpmn',
            'json': './../input_output_files/discovery_output_files/json/purchasing_example.json',
            'sim_log': './../input_output_files/discovery_output_files/prosimos_logs/purchasing_example.csv',
            'start_datetime': '2011-01-01T00:00:00.000000-05:00',
            'total_cases': 608,
            'disc_params': [60, 0.2, 0.8, 0.4, False]
        },
    'production':
        {
            'xes_log': './../input_output_files/discovery_input_files/xes_logs/production.xes',
            'csv_log': './../input_output_files/discovery_input_files/csv_logs/production.csv',
            'real_csv_log': './../input_output_files/discovery_output_files/real_csv_logs/production.csv',
            'bpmn': './../input_output_files/discovery_input_files/bpmn_models/Production.bpmn',
            'json': './../input_output_files/discovery_output_files/json/production.json',
            'sim_log': './../input_output_files/discovery_output_files/prosimos_logs/production.csv',
            'start_datetime': '2012-01-02T07:00:00.000000+02:00',
            'total_cases': 225,
            'disc_params': [60, 0.3, 1.0, 0.5, False]
        },
    'consulta_data_mining':
        {
            'xes_log': './../input_output_files/discovery_input_files/xes_logs/ConsultaDataMining201618.xes',
            'csv_log': './../input_output_files/discovery_input_files/csv_logs/consulta_data_mining.csv',
            'real_csv_log': './../input_output_files/discovery_output_files/real_csv_logs/consulta_data_mining.csv',
            'bpmn': './../input_output_files/discovery_input_files/bpmn_models/consulta_data_mining.bpmn',
            'json': './../input_output_files/discovery_output_files/json/consulta_data_mining.json',
            'sim_log': './../input_output_files/discovery_output_files/prosimos_logs/consulta_data_mining.csv',
            'start_datetime': '2016-02-01 13:23:52+02:00',
            'total_cases': 954,
            'disc_params': [60, 0.1, 0.8, 0.1, True]
        },
    'bpi_2012':
        {
            'xes_log': './../input_output_files/discovery_input_files/xes_logs/BPI_Challenge_2012_W_Two_TS.xes',
            'csv_log': './../input_output_files/discovery_input_files/csv_logs/bpi_2012.csv',
            'real_csv_log': './../input_output_files/discovery_output_files/real_csv_logs/BPI_Challenge_2012.csv',
            'bpmn': './../input_output_files/discovery_input_files/bpmn_models/BPI_Challenge_2012_W_Two_TS.bpmn',
            'json': './../input_output_files/discovery_output_files/json/bpi_2012.json',
            'sim_log': './../input_output_files/discovery_output_files/prosimos_logs/bpi_2012.csv',
            'start_datetime': '2011-10-01 11:08:36.700000+03:00',
            'total_cases': 8616,
            'disc_params': [60, 0.5, 0.5, 0.1, True]
        },
    'bpi_2017':
        {
            'xes_log': './../input_output_files/discovery_input_files/xes_logs/BPI_Challenge_2017_W_Two_TS.xes',
            'csv_log': './../input_output_files/discovery_input_files/csv_logs/bpi_2017.csv',
            'real_csv_log': './../input_output_files/discovery_output_files/real_csv_logs/BPI_Challenge_2017.csv',
            'bpmn': './../input_output_files/discovery_input_files/bpmn_models/BPI_Challenge_2017_W_Two_TS.bpmn',
            'json': './../input_output_files/discovery_output_files/json/bpi_2017.json',
            'sim_log': './../input_output_files/discovery_output_files/prosimos_logs/bpi_2017.csv',
            'start_datetime': '2016-01-02 09:05:02+02:00',
            'total_cases': 30276,
            'disc_params': [60, 0.3, 1.0, 0.2, True]
        },
    'loan_MC_HU':
        {
            'xes_log': './../input_output_files/discovery_input_files/xes_logs/LoanOrigination-MC-HU.xes',
            'csv_log': './../input_output_files/discovery_input_files/csv_logs/loan_MC_HU.csv',
            'real_csv_log': './../input_output_files/discovery_output_files/real_csv_logs/LoanOrigination-MC-HU.csv',
            'bpmn': './../input_output_files/discovery_input_files/bpmn_models/LoanOriginationModel.bpmn',
            'json': './../input_output_files/discovery_output_files/json/loan_MC_HU.json',
            'sim_log': './../input_output_files/discovery_output_files/prosimos_logs/loan_MC_HU.csv',
            'start_datetime': '2015-03-09 09:00:26+00:00',
            'total_cases': 1000,
            'disc_params': [60, 0.1, 0.6, 0.2, True]
        },
    'loan_MC_LU':
        {
            'xes_log': './../input_output_files/discovery_input_files/xes_logs/LoanOrigination-MC-LU.xes',
            'csv_log': './../input_output_files/discovery_input_files/csv_logs/loan_MC_LU.csv',
            'real_csv_log': './../input_output_files/discovery_output_files/real_csv_logs/LoanOrigination-MC-LU.csv',
            'bpmn': './../input_output_files/discovery_input_files/bpmn_models/LoanOriginationModel.bpmn',
            'json': './../input_output_files/discovery_output_files/json/loan_MC_LU.json',
            'sim_log': './../input_output_files/discovery_output_files/prosimos_logs/loan_MC_LU.csv',
            'start_datetime': '2015-03-09 09:00:26+00:00',
            'total_cases': 1000,
            'disc_params': [60, 0.1, 0.6, 0.2, True]
        },
    'loan_SC_HU':
        {
            'xes_log': './../input_output_files/discovery_input_files/xes_logs/LoanOrigination-SC-HU.xes',
            'csv_log': './../input_output_files/discovery_input_files/csv_logs/loan_SC_HU.csv',
            'real_csv_log': './../input_output_files/discovery_output_files/real_csv_logs/LoanOrigination-SC-HU.csv',
            'bpmn': './../input_output_files/discovery_input_files/bpmn_models/LoanOriginationModel.bpmn',
            'json': './../input_output_files/discovery_output_files/json/loan_SC_HU.json',
            'sim_log': './../input_output_files/discovery_output_files/prosimos_logs/loan_SC_HU.csv',
            'start_datetime': '2015-03-06 15:47:26+00:00',
            'total_cases': 1000,
            'disc_params': [60, 0.1, 0.6, 0.2, True]
        },
    'loan_SC_LU':
        {
            'xes_log': './../input_output_files/discovery_input_files/xes_logs/LoanOrigination-SC-LU.xes',
            'csv_log': './../input_output_files/discovery_input_files/csv_logs/loan_SC_LU.csv',
            'real_csv_log': './../input_output_files/discovery_output_files/real_csv_logs/LoanOrigination-SC-LU.csv',
            'bpmn': './../input_output_files/discovery_input_files/bpmn_models/LoanOriginationModel.bpmn',
            'json': './../input_output_files/discovery_output_files/json/loan_SC_LU.json',
            'sim_log': './../input_output_files/discovery_output_files/prosimos_logs/loan_SC_LU.csv',
            'start_datetime': '2015-03-06 15:47:26+00:00',
            'total_cases': 1000,
            'disc_params': [60, 0.1, 0.6, 0.2, True]
        },
    'insurance':
        {
            'xes_log': './../input_output_files/discovery_input_files/xes_logs/insurance.xes',
            'csv_log': './../input_output_files/discovery_input_files/csv_logs/insurance.csv',
            'real_csv_log': './../input_output_files/discovery_output_files/real_csv_logs/insurance.csv',
            'bpmn': './../input_output_files/discovery_input_files/bpmn_models/insurance.bpmn',
            'json': './../input_output_files/discovery_output_files/json/insurance.json',
            'sim_log': './../input_output_files/discovery_output_files/prosimos_logs/insurance.csv',
            'start_datetime': '1971-01-01 00:00:00+00:00',
            'total_cases': 1182,
            'disc_params': [60, 0.3, 1.0, 0.5, False]
        },
    'government':
        {
            'xes_log': './../input_output_files/discovery_input_files/xes_logs/AA_Government.xes',
            'csv_log': './../input_output_files/discovery_input_files/csv_logs/AA_Government.csv',
            'real_csv_log': './../input_output_files/discovery_output_files/real_csv_logs/AA_Government.csv',
            'bpmn': './../input_output_files/discovery_input_files/bpmn_models/AA_Government.bpmn',
            'json': './../input_output_files/discovery_output_files/json/AA_Government.json',
            'sim_log': './../input_output_files/discovery_output_files/prosimos_logs/AA_Government.csv',
            'start_datetime': '2016-03-16 20:48:00+00:00',
            'total_cases': 46330,
            'disc_params': [60, 0.3, 1.0, 0.5, True]
        }
}

canonical_json = {
    'purchasing_example': './../input_output_files/exp_extra/simod_json/PurchasingExample_canon.json',
    'production': './../input_output_files/exp_extra/simod_json/Production_canon.json',
    'consulta_data_mining': './../input_output_files/exp_extra/simod_json/ConsultaDataMining201618_canon.json',
    'insurance': './../input_output_files/exp_extra/simod_json/insurance.xes_canon.json',
    'bpi_2012': './../input_output_files/exp_extra/simod_json/BPI_Challenge_2012_W_Two_TS_canon.json',
    'bpi_2017': './../input_output_files/exp_extra/simod_json/BPI_Challenge_2017_W_Two_TS_canon.json',
    'loan_MC_HU': './../input_output_files/exp_extra/simod_json/loan_origination_MC_HU_canon.json',
    'loan_MC_LU': './../input_output_files/exp_extra/simod_json/loan_origination_MC_LU_canon.json',
    'loan_SC_HU': './../input_output_files/exp_extra/simod_json/loan_origination_SC_HU_canon.json',
    'loan_SC_LU': './../input_output_files/exp_extra/simod_json/loan_origination_SC_LU_canon.json'
}

out_folder = './../input_output_files/exp_extra/cases_json/'
