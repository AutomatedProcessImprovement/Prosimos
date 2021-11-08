import csv
import os
from pathlib import Path

import click

from bpdfr_simulation_engine.simulation_engine import run_simulation
from bpdfr_simulation_engine.simulation_setup import SimDiffSetup


@click.group()
def cli():
    pass


@cli.command()
@click.option('--bpmn_path', required=True,
              help='Path to the BPMN file with the process model')
@click.option('--json_path', required=True,
              help='Path to the JSON file with the differentiated simulation parameters')
@click.option('--total_cases', required=True, type=click.INT,
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
@click.option('--with_enabled_state', required=False,
              help='Boolean variable indicating if the state "enabled" should be added to the resulting event log.'
                   'For example, if the variable is set to True, the resulting event log will store, for every task,'
                   'the events with the states, "enabled", "started", "completed". If not only the states "started" '
                   'and "completed" will be registered in the resulting event log.')
@click.option('--with_csv_state_column', required=False,
              help='Boolean variable indicating if the resulting event log should contain the event states.'
                   'If the variable is set to True, each row of the final CSV will have the structure:'
                   '[case_id, task, resource, state, timestamp] or [case_id, task, resource, timestamp]'
                   'If the variable is set to True, each row of the final CSV will have the structure:'
                   '[case_id, task, enabled_timestamp, start_timestamp, end_timestamp, resource] or '
                   '[case_id, task, start_timestamp, end_timestamp, resource]')
@click.pass_context
def start_simulation(ctx, bpmn_path, json_path, total_cases, stat_out_path=None, log_out_path=None, starting_at=None,
                     with_enabled_state=False, with_csv_state_column=False):
    if not run_simulation(bpmn_path, json_path, total_cases, stat_out_path, log_out_path, starting_at,
                          with_enabled_state, with_csv_state_column):
        print('Simulation model NOT found.')


if __name__ == "__main__":
    cli()
