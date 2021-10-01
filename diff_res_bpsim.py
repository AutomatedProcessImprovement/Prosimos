import csv
import os
from pathlib import Path

import click

from bpdfr_simulation_engine.simulation_engine import run_simulation
from bpdfr_simulation_engine.simulation_setup import SimDiffSetup

diffsim_info: SimDiffSetup


@click.group()
def cli():
    pass


@cli.command()
@click.option('--bpmn_path', required=True,
              help='Path to the BPMN file with the process model')
@click.option('--arrival_dist', required=True,
              help='Path to the JSON file with the arrival time distributions, i.e., new case creation')
@click.option('--gateway_prob', required=True,
              help='Path to the JSON file with the branching probabilities in decision gateways')
@click.option('--task_res_dist', required=True,
              help='Path to the JSON file with the processing time distributions per pair task-resource')
@click.option('--res_calendar', required=True,
              help='Path to the JSON file with the calendars of each resource')
@click.pass_context
def load_simulation_info_cmd(bpmn_path, arrival_dist, gateway_prob, task_res_dist, res_calendar):
    load_simulation_info(bpmn_path, arrival_dist, gateway_prob, task_res_dist, res_calendar)


def load_simulation_info(bpmn_path, arrival_dist, gateway_prob, task_res_dist, res_calendar):
    global diffsim_info
    diffsim_info = SimDiffSetup(bpmn_path, arrival_dist, gateway_prob, task_res_dist, res_calendar)


@cli.command()
@click.option('--total_cases', required=True,
              help='Number of process instances to simulate')
@click.option('--stat_out_path', required=False,
              help='Path to the CSV file to produce with the statistics/metrics after running the simulations.'
                   'If this file path is not provided, one is created by default in the current directory.')
@click.option('--log_out_path', required=False,
              help='Path to the CSV file to produce with the event-log of the simulation. This parameter is optional.'
                   'If the parameter is NONE, no event-log is generated, which leads to lower execution times.')
@click.option('--starting_at', required=False,
              help='Date-time of the first process case in the simulation.'
                   'If this parameter is not provided, the current date-time is assigned.')
def start_simulation_cmd(total_cases, stat_out_path=None, log_out_path=None, starting_at=None):
    start_simulation(total_cases, stat_out_path, log_out_path, starting_at)


def start_simulation(total_cases, stat_out_path=None, log_out_path=None, starting_at=None):
    if diffsim_info:
        if starting_at:
            diffsim_info.set_starting_satetime(starting_at)
        if not stat_out_path:
            stat_out_path = os.path.join(os.path.dirname(__file__), Path("%s.csv" % diffsim_info.process_name))
        with open(stat_out_path, mode='w', newline='') as stat_csv_file:
            if log_out_path:
                with open(log_out_path, mode='w', newline='') as log_csv_file:
                    run_simulation(diffsim_info, total_cases,
                                   csv.writer(stat_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL),
                                   csv.writer(log_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL))
            else:
                run_simulation(diffsim_info, total_cases,
                               csv.writer(stat_csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL))
    else:
        print('Simulation model NOT found. Run function load_simulation_info to fix it.')


cli.add_command(load_simulation_info_cmd)
cli.add_command(start_simulation_cmd)


if __name__ == "__main__":
    cli()
