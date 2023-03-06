import datetime
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from bpdfr_simulation_engine.simulation_properties_parser import (
    EVENT_DISTRIBUTION_SECTION,
)
from performance_exp.events.testing_files import process_files
from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation


def main():
    total_number_of_events_to_add = 9
    number_of_events_to_add_list = range(0, 1 + total_number_of_events_to_add)
    print(number_of_events_to_add_list)
    sim_time_list = []
    median_results_str = ""
    for index in number_of_events_to_add_list:
        print(
            "--------------------------------------------------------------------------"
        )
        print(f"Starting Simulation with {index} inserted events")
        print(
            "--------------------------------------------------------------------------"
        )

        same_index_sim_time_list = []
        for iter_num in range(0, 5):
            sim_time = run_one_iteration(index)
            print(f"iter {iter_num}: {sim_time}")
            same_index_sim_time_list.append(sim_time)

        median_sim_time = np.mean(same_index_sim_time_list)
        sim_time_list.append(median_sim_time)

        # collect results for writing them as txt later
        median_results_str += f"{index},{median_sim_time}\n"

    # save received results (number_inserted_events, simulation_time) as a separate file
    model_info = process_files["events_exp"]
    demo_stats = os.path.join(
        os.path.dirname(__file__),
        model_info["results_folder"],
        f"all_simulation_times.csv",
    )
    with open(demo_stats, "w+") as logs_file:
        logs_file.write(median_results_str)

    print(sim_time_list)

    # show plot of the results
    xpoints = np.array(number_of_events_to_add_list)
    ypoints = np.array(sim_time_list)

    plt.plot(xpoints, ypoints)
    plt.show()


def run_one_iteration(num_inserted_events: int):
    model_info = process_files["events_exp"]
    initial_json_path = os.path.join(os.path.dirname(__file__), model_info["json"])
    bpmn_path = os.path.join(os.path.dirname(__file__), model_info["bpmn"])
    demo_stats = os.path.join(
        os.path.dirname(__file__),
        model_info["results_folder"],
        f"{num_inserted_events}_stats.csv",
    )
    sim_log = os.path.join(
        os.path.dirname(__file__),
        model_info["results_folder"],
        f"{num_inserted_events}_logs.csv",
    )
    new_json_path = _setup_event_distribution(initial_json_path, num_inserted_events)

    start = datetime.datetime.now()

    _, _ = run_diff_res_simulation(
        model_info["start_datetime"],
        model_info["total_cases"],
        bpmn_path,
        new_json_path,
        demo_stats,
        sim_log,
        True,
        num_inserted_events,
    )
    simulation_time = (datetime.datetime.now() - start).total_seconds()

    return simulation_time
    # diff_sim_result.print_simulation_results()


def _setup_event_distribution(initial_json_path, num_events: int):
    """
    Create event distribution for all events that will be later added to the model
    Save the newly created json in new location to keep track of the setup for simulations
    """

    event_distr_list = [
        {
            "event_id": f"event_{index}",
            "distribution_name": "fix",
            "distribution_params": [{"value": 900.0}],
        }
        for index in range(num_events)
    ]

    with open(initial_json_path, "r") as f:
        json_dict = json.load(f)

    json_dict[EVENT_DISTRIBUTION_SECTION] = event_distr_list

    # save modified json as a new file specifying the number of experiment
    # in order to keep track of run experiments
    folder_loc = os.path.dirname(initial_json_path)
    new_filename = f"{num_events}_events_exp.json"
    new_json_path = os.path.join(folder_loc, new_filename)

    with open(new_json_path, "w+") as json_file:
        json.dump(json_dict, json_file)

    return new_json_path


if __name__ == "__main__":
    main()
