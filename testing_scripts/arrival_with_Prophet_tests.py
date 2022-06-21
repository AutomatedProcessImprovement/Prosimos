import os

from bpdfr_discovery.inter_arrival_cases_discovery import discover_inter_arrival, GranuleSize
from bpdfr_discovery.log_parser import event_list_from_csv
from testing_scripts.bpm_2022_testing_files import experiment_logs, process_files


def main():
    for i in range(4, 11):
        model_name = experiment_logs[i]
        log_traces = event_list_from_csv(process_files[model_name]['csv_log'])
        discover_inter_arrival(log_traces, GranuleSize.Day)
        break


    os._exit(0)


if __name__ == "__main__":
    main()
