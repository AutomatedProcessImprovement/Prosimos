import csv
import os
from pathlib import Path

import click

from bpdfr_discovery.log_parser import preprocess_xes_log
from prosimos.simulation_engine import run_simulation
from prosimos.simulation_setup import SimDiffSetup


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
@click.option('--is_event_added_to_log', required=False,
              help='Boolean showing whether event should be added to the resulted simulation log.'
                   'If this parameter is not provided, False is considered as the parameter value.')
@click.pass_context
def start_simulation(ctx, bpmn_path, json_path, total_cases, stat_out_path=None, log_out_path=None, starting_at=None, is_event_added_to_log=False):
    run_simulation(bpmn_path, json_path, total_cases, stat_out_path, log_out_path, starting_at, is_event_added_to_log)


@cli.command()
@click.option('--bpmn_path', required=True,
              help='Path to the BPMN file with the process model')
@click.option('--log_path', required=True,
              help='Path to the event log in XES format')
@click.option('--out_json', required=True,
              help='Path to the JSON file to produce as output with the differentiated simulation parameters')
@click.option('--granule_size', required=False,
              help='Length of the working intervals in the calendars in minutes, 60 minutes by default')
@click.option('--conf', required=False,
              help='Confidence ratio in [0, 1] to consider a timestamp in the event-log as outlier, 0.1 by default.'
                   'Timestamps under the confidence threshold are not included in the calendar.')
@click.option('--supp', required=False,
              help='Support ratio in [0, 1] describing the coverage of timestamps by the calendar, 0.7 by default.'
                   'The calendar will include a ratio of timestamps in the log over the support threshold.')
@click.option('--part', required=False,
              help='Resource participation ratio in [0, 1] to mark a resource as external, 0.3 by default.'
                   'Resources with participation ratio under the threshold in the event-log are discarded.')
@click.pass_context
def discover_simulation_parameters(ctx, bpmn_path, log_path, out_json, granule_size=60, conf=0.1, supp=0.7, part=0.3):
    preprocess_xes_log(log_path, bpmn_path, out_json, granule_size, conf, supp, part, True)


if __name__ == "__main__":
    cli()
