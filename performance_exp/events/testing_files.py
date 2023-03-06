process_files = {
    "events_exp": {
        "bpmn": "input/batch-example-end-task.bpmn",
        "json": "input/batch-example-with-batch.json",
        "results_folder": "results",
        "start_datetime": "2022-06-21 13:22:30.035185+03:00",
        "total_cases": 1000,
        "disc_params": [60, 0.1, 0.9, 0.6, True],
        "number_of_added_events": 9,  # should be equal to the number of sequence flows in the BPMN model
    }
}
