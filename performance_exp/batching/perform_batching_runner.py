import json
import os
import uuid

import matplotlib.pyplot as plt
import numpy as np
from bpdfr_simulation_engine.simulation_properties_parser import (
    BATCH_PROCESSING_SECTION,
    TASK_RESOURCE_DISTR_SECTON,
)
from mpl_toolkits import mplot3d
from performance_exp.batching.testing_files import process_files_setup
from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation


def main():
    model_name = "bpi2012"
    # model_name = "simple_example"
    model_info = process_files_setup[model_name]
    max_num_tasks_with_batching = model_info["max_num_tasks_with_batching"]
    max_iter_num = model_info["max_iter_num"]
    max_complexity_level = model_info["max_complexity_level"]

    num_batched_tasks_range = range(0, 1 + max_num_tasks_with_batching)
    rules_level_range = range(1, 1 + max_complexity_level)

    # unique identifier of the experiment run
    unique_run_id = uuid.uuid4()

    # file for saving received results (number_inserted_events, simulation_time)
    final_plot_results = _get_abs_path(
        model_info["results_folder"],
        f"{unique_run_id}_plot_data.csv",
    )

    # add name of the model used during this experiment
    with open(final_plot_results, "a") as plot_file:
        plot_file.write(f"{model_name}\n\n")

    sim_time_list = []
    all_rule_complex_level = []
    all_batched_task = []

    for index in num_batched_tasks_range:
        print("-------------------------------------------")
        print(
            f"Starting Simulation with {index} priority levels in the simulation scenario"
        )
        print("-------------------------------------------")

        for rule_complexity_level in rules_level_range:
            median_sim_time = get_avg_after_all_iters(
                max_iter_num, index, model_info, rule_complexity_level
            )

            all_batched_task.append(index)
            all_rule_complex_level.append(rule_complexity_level)
            sim_time_list.append(median_sim_time)

            # collect data points used for plotting
            with open(final_plot_results, "a") as plot_file:
                plot_file.write(f"{index},{rule_complexity_level},{median_sim_time}\n")

    print(sim_time_list)

    # save plot of the relationship: number of batched tasks - simulation time
    plt_path = _get_abs_path(
        model_info["results_folder"],
        f"{unique_run_id}_plot.png",
    )

    # show plot of the relationship: number of priority levels - simulation time
    print(all_batched_task)
    print(sim_time_list)
    print(f"np {np.array(sim_time_list)}")
    print(all_rule_complex_level)
    print(f"np {np.array(all_rule_complex_level)}")
    _save_plot_2(
        # all_batched_task,
        num_batched_tasks_range,
        sim_time_list,
        all_rule_complex_level,
        max_complexity_level,
        model_name,
        model_info["total_cases"],
        plt_path,
    )


def get_avg_after_all_iters(
    max_iter_num: int, current_run_index: int, model_info, rule_complexity: int
):
    same_index_sim_time_list = []
    for iter_num in range(0, max_iter_num):
        sim_time = run_one_iteration(current_run_index, model_info, rule_complexity)
        print(f"iter {iter_num}: {sim_time}")
        same_index_sim_time_list.append(sim_time)

    median_sim_time = np.median(same_index_sim_time_list)
    print(f"median: {median_sim_time}")

    return median_sim_time


def run_one_iteration(num_tasks_with_batching: int, model_info, rule_complexity: int):
    results_folder = model_info["results_folder"]
    initial_json_path = _get_abs_path(model_info["json"])
    bpmn_path = _get_abs_path(model_info["bpmn"])
    demo_stats = _get_abs_path(results_folder, f"{num_tasks_with_batching}_stats.csv")
    sim_log = _get_abs_path(results_folder, f"{num_tasks_with_batching}_logs.csv")

    new_json_path = _setup_sim_scenario(
        initial_json_path, num_tasks_with_batching, rule_complexity
    )

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


def _setup_sim_scenario(
    initial_json_path, num_tasks_with_batching: int, rule_complexity: int
):
    """
    Create case-based prioritisation rules based on the required number (num_prioritisation_rules)
    Save the newly created json in new location to keep track of the setup for simulations
    """

    one_batching_rule = _get_rule_by_complexity_level(rule_complexity)

    with open(initial_json_path, "r") as f:
        json_dict = json.load(f)

    # collect all ids of activities in the BPMN model
    all_tasks_distr = json_dict[TASK_RESOURCE_DISTR_SECTON]
    all_tasks_id = map(lambda item: item["task_id"], all_tasks_distr)

    # select only number of activities that should have an assigned batching rule
    selected_tasks_id = list(all_tasks_id)[:num_tasks_with_batching]

    # create batching setup
    new_batching_rules_section = [
        {
            "task_id": task_id,
            "type": "Sequential",
            "batch_frequency": 1.0,
            "size_distrib": [
                {"key": "2", "value": 1},
            ],
            "duration_distrib": [{"key": "3", "value": 0.8}],
            "firing_rules": one_batching_rule,
        }
        for task_id in selected_tasks_id
    ]

    json_dict[BATCH_PROCESSING_SECTION] = new_batching_rules_section

    # save modified json as a new file specifying the number of experiment
    # in order to keep track of run experiments
    folder_loc = os.path.dirname(initial_json_path)
    new_filename = f"{num_tasks_with_batching}_batching_exp.json"
    new_json_path = os.path.join(folder_loc, new_filename)

    with open(new_json_path, "w+") as json_file:
        json.dump(json_dict, json_file)

    return new_json_path


def _get_rule_by_complexity_level(rule_complexity: int):
    all_rules = {
        "1": [
            [
                {"attribute": "size", "comparison": ">=", "value": 4},
            ]
        ],
        "2": [
            [
                {"attribute": "daily_hour", "comparison": "<", "value": "12"},
                {"attribute": "week_day", "comparison": "=", "value": "Friday"},
            ]
        ],
        "3": [
            [
                {"attribute": "daily_hour", "comparison": "<", "value": "12"},
                {"attribute": "week_day", "comparison": "=", "value": "Friday"},
            ],
            [
                {"attribute": "size", "comparison": ">=", "value": 4},
            ],
        ],
        "4": [
            [
                {"attribute": "daily_hour", "comparison": "<", "value": "12"},
                {"attribute": "week_day", "comparison": "=", "value": "Friday"},
            ],
            [
                {"attribute": "size", "comparison": ">=", "value": 4},
                {"attribute": "large_wt", "comparison": "<", "value": 3600},
            ],
        ],
    }

    return all_rules[str(rule_complexity)]


def _save_plot(xpoints, ypoints, zpoints, model_name, num_of_instances, plt_path):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter3D(xpoints, ypoints, zpoints, c=zpoints, cmap="Greens")

    # give a general title
    ax.set_title(f"Model: {model_name}, instances: {num_of_instances}")

    # name axis
    ax.set_xlabel("Number of priority levels")
    ax.set_ylabel("Simulation time, sec")

    plt.show()
    # save as a file
    # plt.savefig(plt_path, bbox_inches="tight")


def _save_plot_2(
    num_batched_task_arr,
    sim_time_arr,
    rule_complexity_level_arr,
    max_complexity_level,
    model_name,
    num_of_instances,
    plt_path,
):
    fig = plt.figure()

    # give a general title
    # plt.title(f"Model: {model_name}, instances: {num_of_instances}")

    # # provide data points
    # plt.plot(xpoints, ypoints)
    ax = fig.add_subplot(projection="3d")
    colors = ["b", "g", "r", "c", "m", "y"]

    for i, (current_num_batched_task, color) in enumerate(
        zip(num_batched_task_arr, colors)
    ):
        gap = max_complexity_level
        start = i * gap  # 1 - current_num_batched_task
        end = start + gap
        xs = sim_time_arr[start:end]
        ys = rule_complexity_level_arr[start:end]
        print(f"zs = {current_num_batched_task}")
        print(start, end, xs, ys)
        ax.bar(ys, xs, zs=current_num_batched_task, zdir="y", color=color, alpha=0.8)

    # name axis
    ax.set_xlabel("Complexity of the batching rule")
    ax.set_ylabel("Number of batched tasks")
    ax.set_zlabel("Simulation time, sec")
    ax.set_title(f"Model: {model_name}, instances: {num_of_instances}")

    # plt.colorbar()  # show color scale
    plt.show()
    # save as a file
    # plt.savefig(plt_path, bbox_inches="tight")


def _get_abs_path(*args):
    return os.path.join(os.path.dirname(__file__), *args)


if __name__ == "__main__":
    main()
