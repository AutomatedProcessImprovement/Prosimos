import pytest
import json
from prosimos.batch_processing import BATCH_TYPE
from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation
from testing_scripts.test_batching import JSON_FILENAME, JSON_ONE_RESOURCE_FILENAME, MODEL_FILENAME, SIM_LOGS_FILENAME, SIM_STATS_FILENAME, _setup_initial_scenario
from testing_scripts.test_batching import assets_path

data_resource_utilization = [
    # parallel execution results in duration of each item in the log
    # multiplied by number of items in the batch
    # thus we need to make sure that we take only one record per batch while calculating the worked hours
    (
        BATCH_TYPE.PARALLEL.value,
        96.0 * 3 * 2 + 120.0,           # two batches with three activities per each and one task alone in batch
        1368                            # 22.8 minutes
    ),
    # sequential execution results in every item in the log file
    # having its proper duration, we just sum up them to have the total
    (
        BATCH_TYPE.SEQUENTIAL.value,
        96.0 * 3 * 2 + 120.0,           # two batches with three activities per each and one task alone in batch
        1368                            # 22.8 minutes
    ),
    # when no batch should be created,
    # we validate that all tasks have its initial performance
    (
        None,       # no batches, execute right away
        120.0 * 7,    # 7 tasks, initial duration of the task is 120 sec
        1200        # 18 minutes
    )
]

#TODO: write separate test for validating the end time in the log file

@pytest.mark.parametrize(
    "batch_type, expected_worked_time, expected_available_time", 
    data_resource_utilization
)
def test_resource_utilization_correct(batch_type, expected_worked_time, expected_available_time, assets_path):
    """
    Resource_1 is being assigned only to the batched task ('D').
    New case arrive every two minutes. In total, we simulate 7 cases.
    
    Validate that the worked hours and total hours of the Resource_1 are correct
    """
    # ====== ARRANGE ======
    model_path = assets_path / MODEL_FILENAME
    json_path = assets_path / JSON_ONE_RESOURCE_FILENAME
    sim_stats = assets_path / SIM_STATS_FILENAME
    sim_logs = assets_path / SIM_LOGS_FILENAME
    _setup_file_with_one_resource_for_batched_task(assets_path, json_path)

    start_string = "2022-06-21 13:22:30.035185+03:00"

    firing_rules = [[{"attribute": "size", "comparison": "=", "value": 3}]]
    
    # either execute batched tasks in batches
    # or execute all batched tasks alone
    size_distr = { } if batch_type != None \
        else \
        [
            {"key": "1", "value": 1},
            {"key": "2", "value": 0 }
        ]

    _setup_initial_scenario(json_path, firing_rules, batch_type, size_distr)

    # ====== ACT ======
    _, diff_sim_result = run_diff_res_simulation(
        start_string, 7, model_path, json_path, sim_stats, sim_logs, True
    )

    # ====== ASSERT ======
    actual_worked_time = diff_sim_result.resource_info["Resource_1"][1] 
    assert actual_worked_time == expected_worked_time

    actual_total_available_hours = diff_sim_result.resource_info["Resource_1"][2]
    assert actual_total_available_hours == expected_available_time
    
    actual_resource_utili = diff_sim_result.resource_utilization["Resource_1"]
    expected_resource_utili = expected_worked_time / expected_available_time
    assert actual_resource_utili == expected_resource_utili


def _setup_file_with_one_resource_for_batched_task(assets_path, new_json_path):
    json_path = assets_path / JSON_FILENAME

    with open(json_path, "r") as f:
        json_dict = json.load(f)

    for task_dist in json_dict["task_resource_distribution"]:
        if task_dist["task_id"] != "sid-503A048D-6344-446A-8D67-172B164CF8FA":
            # leave everything except Resource_1
            task_dist["resources"] = [resource for idx, resource in enumerate(task_dist["resources"]) if idx != 0]
        else:
            # remove everything but Resource_1
            task_dist["resources"] = [resource for idx, resource in enumerate(task_dist["resources"]) if idx == 0]

    with open(new_json_path, "w+") as json_file:
        json.dump(json_dict, json_file)