test_processes = {0: 'loan_SC_LU'
                  }

process_files = {
    'loan_SC_LU': {
        'bpmn': './assets/fuzzy_calendars/in/LoanOriginationModel.bpmn',
        'fuzzy_json': './assets/fuzzy_calendars/in/loan_SC_LU_fuzzy.json',
        'full_json': './assets/fuzzy_calendars/in/loan_SC_LU_full.json',
        'csv_log': './assets/fuzzy_calendars/in/loan_SC_LU.csv',
        'sim_log': './assets/fuzzy_calendars/out/loan_SC_LU_log.csv',
        'sim_stats': './assets/fuzzy_calendars/out/loan_SC_LU_stat.csv',
        'disc_params': [60, 0.1, 1.0, 0.2, True],
        'start_datetime': '2015-03-06 15:47:26+00:00',
        'total_cases': 1000,
    }

}
