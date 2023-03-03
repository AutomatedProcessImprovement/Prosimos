import datetime
import json
import os

from bpdfr_simulation_engine.simulation_properties_parser import (
    EVENT_DISTRIBUTION_SECTION,
)
from performance_exp.events.testing_files import process_files
from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation


def main():
    run_one_iteration()

    os._exit(0)


def run_one_iteration():
    model_info = process_files["events_exp"]
    initial_json_path = os.path.join(os.path.dirname(__file__), model_info["json"])
    bpmn_path = os.path.join(os.path.dirname(__file__), model_info["bpmn"])
    demo_stats = os.path.join(os.path.dirname(__file__), model_info["demo_stats"])
    sim_log = os.path.join(os.path.dirname(__file__), model_info["sim_log"])
    new_json_path = _setup_event_distribution(initial_json_path, 5)

    print("--------------------------------------------------------------------------")
    print(
        "Starting Simulation of demo example (%d instances)"
        % (model_info["total_cases"])
    )
    print("--------------------------------------------------------------------------")
    start = datetime.datetime.now()

    _, diff_sim_result = run_diff_res_simulation(
        model_info["start_datetime"],
        model_info["total_cases"],
        bpmn_path,
        new_json_path,
        demo_stats,
        sim_log,
        True,
    )
    print(
        "Simulation Time: %s"
        % str(
            datetime.timedelta(
                seconds=(datetime.datetime.now() - start).total_seconds()
            )
        )
    )
    diff_sim_result.print_simulation_results()


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
    new_filename = f"events_exp_{num_events}.json"
    new_json_path = os.path.join(folder_loc, new_filename)

    with open(new_json_path, "w+") as json_file:
        json.dump(json_dict, json_file)

    return new_json_path


if __name__ == "__main__":
    main()
