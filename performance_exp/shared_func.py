def get_central_tendency_over_all_iters(
    max_iter_num, run_one_iteration, current_index, model_info, measure_central_tendency
):
    """
    Run one iteration n number of times and calculate the central tendency of simulation times

    :param int max_iter_num: Number of iteration to run
    :param func run_one_iteration: Function that needs to be run per one iteration
    :param int current_index:
    :param obj model_info: Description of the input provided by the user
    :param func measure_central_tendency: numpy function used for calculating the central tendency (e.g., np.mean, np.median)
    """
    same_index_sim_time_list = []
    for iter_num in range(0, max_iter_num):
        sim_time = run_one_iteration(current_index, model_info)
        print(f"iter {iter_num}: {sim_time}")
        same_index_sim_time_list.append(sim_time)

    median_sim_time = measure_central_tendency(same_index_sim_time_list)
    print(f"central_tendency: {median_sim_time}")

    return median_sim_time
