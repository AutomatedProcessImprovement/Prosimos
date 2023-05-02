import numpy as np


process_files_setup = {
    "simple_example": {
        "bpmn": "simple_example/batch-example-end-task.bpmn",
        "json": "simple_example/batch-example-with-batch.json",
        "results_folder": "simple_example/results",
        "start_datetime": "2022-06-21 13:22:30.035185+03:00",
        "total_cases": 1000,
        "disc_params": [60, 0.1, 0.9, 0.6, True],
        "max_num_tasks_with_batching": 5,  # should be the max number of tasks in the model
        "measure_central_tendency": np.median,
        "max_iter_num": 2,
        "max_complexity_level": 4,
    },
    "bpi2012": {
        "bpmn": "bpi2012/input/BPI_Challenge_2012_W_Two_TS.bpmn",
        "json": "bpi2012/input/bpi_2012.json",
        "results_folder": "bpi2012/results",
        "start_datetime": "2011-10-01 11:08:36.700000+03:00",  # used to be as close to the real log as possible
        "total_cases": 8616,
        "disc_params": [
            60,
            0.5,
            0.5,
            0.1,
            True,
        ],  # to know the input required for discovering the json from the log
        "max_num_tasks_with_batching": 6,  # should be the max number of tasks in the model
        "measure_central_tendency": np.median,  # median since this metric is not impacted by outliers, compared to the mean one
        "max_iter_num": 5,
        "max_complexity_level": 4,
    },
}
