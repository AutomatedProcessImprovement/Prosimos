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
@click.pass_context
def start_simulation(ctx, bpmn_path, json_path, total_cases, stat_out_path=None, log_out_path=None, starting_at=None):
    run_simulation(bpmn_path, json_path, total_cases, stat_out_path, log_out_path, starting_at)


if __name__ == "__main__":
    cli()
