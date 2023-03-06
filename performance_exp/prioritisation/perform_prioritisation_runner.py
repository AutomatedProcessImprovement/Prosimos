import json
import os

import matplotlib.pyplot as plt
import numpy as np

from bpdfr_simulation_engine.simulation_properties_parser import (
    PRIORITISATION_RULES_SECTION,
)
from performance_exp.prioritisation.testing_files import process_files_setup
from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation


def main():
    model_name = "simple_example"
    model_info = process_files_setup[model_name]
    max_number_prioritisation_rules = 6

    prioritisation_rules_to_add_list = range(0, 1 + max_number_prioritisation_rules)
    # run_one_iteration(1, model_info)

    sim_time_list = []
    median_results_str = ""
    for index in prioritisation_rules_to_add_list:
        print("-------------------------------------------")
        print(
            f"Starting Simulation with {index} priority levels in the simulation scenario"
        )
        print("-------------------------------------------")

        same_index_sim_time_list = []
        for iter_num in range(0, 5):
            sim_time = run_one_iteration(index, model_info)
            print(f"iter {iter_num}: {sim_time}")
            same_index_sim_time_list.append(sim_time)

        # TODO: should we use mean or median?
        median_sim_time = np.mean(same_index_sim_time_list)
        print(f"median: {median_sim_time}")

        sim_time_list.append(median_sim_time)

        # collect results for writing them as txt later
        median_results_str += f"{index},{median_sim_time}\n"

    # save received results (number_priority_levels, simulation_time) as a separate file
    demo_stats = _get_abs_path(
        model_info["results_folder"],
        f"all_simulation_times.csv",
    )
    with open(demo_stats, "w+") as logs_file:
        logs_file.write(median_results_str)

    print(sim_time_list)

    # show plot of the relationship: number of priority levels - simulation time
    _show_plot(
        np.array(prioritisation_rules_to_add_list),
        np.array(sim_time_list),
        model_name,
        model_info["total_cases"],
    )


def run_one_iteration(num_prioritisation_rules: int, model_info):
    initial_json_path = _get_abs_path(model_info["json"])
    bpmn_path = _get_abs_path(model_info["bpmn"])
    demo_stats = _get_abs_path(
        model_info["results_folder"], f"{num_prioritisation_rules}_stats.csv"
    )
    sim_log = _get_abs_path(
        model_info["results_folder"],
        f"{num_prioritisation_rules}_logs.csv",
    )
    new_json_path = _setup_sim_scenario(initial_json_path, num_prioritisation_rules)

    simulation_time, _ = run_diff_res_simulation(
        model_info["start_datetime"],
        model_info["total_cases"],
        bpmn_path,
        new_json_path,
        demo_stats,
        sim_log,
        False,  # no events in the log
        None,  # no added events
    )

    return simulation_time
    # diff_sim_result.print_simulation_results()


def _setup_sim_scenario(initial_json_path, num_prioritisation_rules: int):
    """
    Create case-based prioritisation rules based on the required number (num_prioritisation_rules)
    Save the newly created json in new location to keep track of the setup for simulations
    """

    prioritisation_rules = [
        {
            "priority_level": 1,
            "rules": [
                [
                    {
                        "attribute": "loan_amount",
                        "comparison": "in",
                        "value": [2000, "inf"],
                    }
                ],
            ],
        },
        {
            "priority_level": 2,
            "rules": [
                [
                    {
                        "attribute": "loan_amount",
                        "comparison": "in",
                        "value": [1500, 2000],
                    }
                ],
            ],
        },
        {
            "priority_level": 3,
            "rules": [
                [
                    {
                        "attribute": "loan_amount",
                        "comparison": "in",
                        "value": [1000, 1500],
                    }
                ],
            ],
        },
        {
            "priority_level": 4,
            "rules": [
                [
                    {
                        "attribute": "loan_amount",
                        "comparison": "in",
                        "value": [800, 1000],
                    }
                ],
            ],
        },
        {
            "priority_level": 5,
            "rules": [
                [{"attribute": "loan_amount", "comparison": "in", "value": [500, 800]}],
            ],
        },
        {
            "priority_level": 6,
            "rules": [
                [{"attribute": "loan_amount", "comparison": "in", "value": [0, 500]}],
            ],
        },
    ]

    with open(initial_json_path, "r") as f:
        json_dict = json.load(f)

    json_dict[PRIORITISATION_RULES_SECTION] = prioritisation_rules[
        :num_prioritisation_rules
    ]

    # save modified json as a new file specifying the number of experiment
    # in order to keep track of run experiments
    folder_loc = os.path.dirname(initial_json_path)
    new_filename = f"{num_prioritisation_rules}_prioritisation_rules_exp.json"
    new_json_path = os.path.join(folder_loc, new_filename)

    with open(new_json_path, "w+") as json_file:
        json.dump(json_dict, json_file)

    return new_json_path


def _show_plot(xpoints, ypoints, model_name, num_of_instances):
    # give a general title
    plt.title(f"Model: {model_name}, instances: {num_of_instances}")

    # name axis
    plt.xlabel("Number of priority levels")
    plt.ylabel("Simulation time, sec")

    # provide data points
    plt.plot(xpoints, ypoints)

    # visualize
    plt.show()


def _get_abs_path(*args):
    return os.path.join(os.path.dirname(__file__), *args)


if __name__ == "__main__":
    main()
