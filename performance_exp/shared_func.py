import os
import uuid

import numpy as np
import pandas as pd


def get_central_tendency_over_all_iters(
    max_iter_num,
    run_one_iteration,
    current_index,
    model_info,
    measure_central_tendency,
    _get_abs_path,
    is_normalised=False,
):
    """
    Run one iteration n number of times and calculate the central tendency of simulation times

    :param int max_iter_num: Number of iteration to run
    :param func run_one_iteration: Function that needs to be run per one iteration
    :param int current_index:
    :param obj model_info: Description of the input provided by the user
    :param func measure_central_tendency: numpy function used for calculating the central tendency (e.g., np.mean, np.median)
    :param bool is_normalised: whether the simulation time is divided by the total number of activities during simulation
    """
    same_index_sim_time_list = []

    results_folder_path = _get_abs_path(model_info["results_folder"])
    current_log_path = get_log_filepath(results_folder_path, current_index)
    normalised_data_path = os.path.join(results_folder_path, "normalised_table.csv")

    for iter_num in range(0, max_iter_num):
        sim_time = run_one_iteration(current_index, model_info)

        if is_normalised:
            # in case of normalising, we divide the simulation time
            # by the number of activities run during the simulation
            sim_time, num_total_activities, norm_sim_time = get_normalised_sim_time(
                current_log_path,
                sim_time,
                normalised_data_path,
                current_index,
                iter_num,
            )

        print(f"iter {iter_num}: {sim_time}")
        same_index_sim_time_list.append(sim_time)

    median_sim_time = measure_central_tendency(same_index_sim_time_list)
    print(f"central_tendency: {median_sim_time}")

    if is_normalised:
        # append row with statistics for the median value
        with open(normalised_data_path, "a") as plot_file:
            plot_file.write(
                f"{current_index},median,{sim_time},{num_total_activities},{norm_sim_time}\n"
            )

    return median_sim_time


def run_whole_experiment(
    model_name,
    model_info,
    metric_under_performance_range_str,
    _get_abs_path,
    run_one_iteration,
    _save_plot,
    is_sim_time_normalised,
):
    total_number_of_x_values = model_info[metric_under_performance_range_str]
    measure_central_tendency = (
        model_info["measure_central_tendency"]
        if "measure_central_tendency" in model_info
        else np.median
    )
    max_iter_num = model_info["max_iter_num"]

    print(f"Selected log: {model_name}")
    print(f"Selected function for central tendency: {measure_central_tendency}")

    number_of_events_to_add_list = range(0, 1 + total_number_of_x_values)

    # array to save ordinate (y coordinate) of data points
    sim_time_list = []

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

    # clear file with normalised data
    results_folder_path = _get_abs_path(model_info["results_folder"])
    normalised_data = os.path.join(results_folder_path, "normalised_table.csv")

    with open(normalised_data, "a") as plot_file:
        plot_file.write(
            f"num_priority_levels,iter,sim_time_sec,num_total_activities,norm_sim_time_ms\n"
        )

    for index in number_of_events_to_add_list:
        print("-------------------------------------------")
        print(f"Starting Simulation with {index} inserted events")
        print("-------------------------------------------")

        median_sim_time = get_central_tendency_over_all_iters(
            max_iter_num,
            run_one_iteration,
            index,
            model_info,
            measure_central_tendency,
            _get_abs_path,
            is_sim_time_normalised,
        )
        sim_time_list.append(median_sim_time)

        with open(final_plot_results, "a") as plot_file:
            plot_file.write(f"{index},{median_sim_time}\n")

    print(sim_time_list)

    # save plot of the relationship: number of added events - simulation time
    plt_path = _get_abs_path(
        model_info["results_folder"],
        f"{unique_run_id}_plot.png",
    )
    _save_plot(
        np.array(number_of_events_to_add_list),
        sim_time_list,
        model_name,
        model_info["total_cases"],
        plt_path,
        is_sim_time_normalised,  # use ms if True
    )


def get_log_filepath(results_folder_path, index):
    return os.path.join(results_folder_path, f"{index}_logs.csv")


def get_total_num_activities(log_path):
    df = pd.read_csv(log_path)
    return df.shape[0]  # number of rows


def get_normalised_sim_time(
    current_log_path, sim_time, normalised_data_path, current_index, iter_num
):
    num_total_activities = get_total_num_activities(current_log_path)
    norm_sim_time = sim_time / num_total_activities  # seconds
    norm_sim_time = norm_sim_time * 1000  # miliseconds

    # append row with statistics for this current iteration
    with open(normalised_data_path, "a") as plot_file:
        plot_file.write(
            f"{current_index},{iter_num},{sim_time},{num_total_activities},{norm_sim_time}\n"
        )

    # normalised sim_time used as a final one
    return norm_sim_time, num_total_activities, norm_sim_time
