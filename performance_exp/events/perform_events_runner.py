import json
import os

import matplotlib.pyplot as plt
from bpdfr_simulation_engine.simulation_properties_parser import (
    EVENT_DISTRIBUTION_SECTION,
)
from performance_exp.events.testing_files import process_files_setup
from performance_exp.shared_func import (
    run_whole_experiment,
)
from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation


def main():
    all_models = [
        "events_exp",
    ]
    for model_name in all_models:
        run_whole_experiment(
            model_name,
            process_files_setup[model_name],
            "number_of_added_events",
            _get_abs_path,
            run_one_iteration,
            _save_plot,
        )


def run_one_iteration(num_inserted_events: int, model_info):
    initial_json_path = _get_abs_path(model_info["json"])
    bpmn_path = _get_abs_path(model_info["bpmn"])
    demo_stats = _get_abs_path(
        model_info["results_folder"], f"{num_inserted_events}_stats.csv"
    )
    sim_log = _get_abs_path(
        model_info["results_folder"],
        f"{num_inserted_events}_logs.csv",
    )
    new_json_path = _setup_event_distribution(initial_json_path, num_inserted_events)

    simulation_time, _ = run_diff_res_simulation(
        model_info["start_datetime"],
        model_info["total_cases"],
        bpmn_path,
        new_json_path,
        demo_stats,
        sim_log,
        True,
        num_inserted_events,
    )

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


def _save_plot(xpoints, ypoints, model_name, num_of_instances, plt_path):
    # give a general title
    plt.title(f"Model: {model_name}, instances: {num_of_instances}")

    # name axis
    plt.xlabel("Number of added events")
    plt.ylabel("Simulation time, sec")

    # provide data points
    plt.plot(xpoints, ypoints)

    # save as a file
    plt.savefig(plt_path, bbox_inches="tight")


def _get_abs_path(*args):
    return os.path.join(os.path.dirname(__file__), *args)


if __name__ == "__main__":
    main()
