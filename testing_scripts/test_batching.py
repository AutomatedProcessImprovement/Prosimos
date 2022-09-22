import pandas as pd
import numpy as np
import datetime
import os
from pathlib import Path
import pytest
import json
from pandas import testing as tm

from bpdfr_simulation_engine.resource_calendar import parse_datetime
from bpdfr_simulation_engine.simulation_properties_parser import parse_json_sim_parameters
from testing_scripts.bimp_diff_sim_tests import run_diff_res_simulation
from testing_scripts.test_simulation import _verify_activity_count_and_duration

distribution = {
    "1": 0.8,
    "3": 0.75,
    "5": 0.6
}

data_nearest_neighbors = [
    (
        "assets_path", 
        distribution,
        4,
        120*0.75
    ),
    (
        "assets_path", 
        distribution,
        7,
        120*0.6
    ),
    (
        "assets_path", 
        distribution,
        10,
        120*0.6
    )
]

@pytest.fixture
def assets_path(request) -> Path:
    entry_path: Path
    if os.path.basename(os.getcwd()) == 'testing_scripts':
        entry_path = Path('assets')
    else:
        entry_path = Path('testing_scripts/assets')

    def teardown():
        output_paths = [
            entry_path / 'batch_stats.csv',
            entry_path / 'batch_logs.csv',
            entry_path / 'batch-example-nearest-coef.json'
        ]
        for output_path in output_paths:
            if output_path.exists():
                os.remove(output_path)
    request.addfinalizer(teardown)

    return entry_path

@pytest.mark.skip(reason="used only for testing while developing")
def test(assets_path):
    """
    """

    # ====== ARRANGE ======
    json_path = assets_path / 'batch_processing_basic.json'
    
    _, _, _, _, _, _, batch_processing \
        = parse_json_sim_parameters(json_path)

    # ====== ACT ======
    print(batch_processing)

    # ====== ASSERT ======


def test_seq_batch_count_firing_rule_correct_duration(assets_path):
    # ====== ARRANGE ======
    model_path = assets_path / 'batch-example-end-task.bpmn'
    json_path = assets_path / 'batch-example-with-batch.json'
    sim_stats = assets_path / 'batch_stats.csv'
    sim_logs = assets_path / 'batch_logs.csv'

    start_string = '2022-06-21 13:22:30.035185+03:00'
    start_date = parse_datetime(start_string, True)

    # ====== ACT ======
    _, diff_sim_result = run_diff_res_simulation(start_date,
                                    6,
                                    model_path,
                                    json_path,
                                    sim_stats,
                                    sim_logs,
                                    True)

    # ====== ASSERT ======
    # verify stats
    task_d_sim_result = diff_sim_result.tasks_kpi_map['D']
    expected_average_processing_time = 96.0 # 80% from the initially defined task performance
    assert task_d_sim_result.processing_time.avg == expected_average_processing_time
    assert task_d_sim_result.processing_time.count == 6

    # make sure the next tasks after batching were executed
    task_d_sim_result = diff_sim_result.tasks_kpi_map['E']
    assert task_d_sim_result.processing_time.count == 6

    # verify logs
    df = pd.read_csv(sim_logs)
    
    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')

    # verify that there are 6 instances of D activity executed
    # and that theirs duration is exactly 96 seconds
    logs_d_task = df[df['activity'] == 'D']
    expected_activity_timedelta = datetime.timedelta(seconds=96)
    _verify_activity_count_and_duration(logs_d_task, 6, expected_activity_timedelta)


@pytest.mark.parametrize("assets_path_fixture,duration_distrib,firing_count,expected_duration_sec", data_nearest_neighbors)
def test_batch_count_firing_rule_nearest_neighbor_correct(
    assets_path_fixture, duration_distrib, firing_count, expected_duration_sec, request):
    # ====== ARRANGE ======
    assets_path = request.getfixturevalue(assets_path_fixture)
    model_path = assets_path / 'batch-example-end-task.bpmn'
    basic_json_path = assets_path / 'batch-example-with-batch.json'
    json_path = assets_path / 'batch-example-nearest-coef.json'
    sim_stats = assets_path / 'batch_stats.csv'
    sim_logs = assets_path / 'batch_logs.csv'

    start_string = '2022-06-21 13:22:30.035185+03:00'
    start_date = parse_datetime(start_string, True)

    with open(basic_json_path, 'r') as f:
        json_dict = json.load(f)

    _setup_sim_scenario_file(json_dict, duration_distrib, firing_count, None, None)

    with open(json_path, 'w+') as json_file:
        json.dump(json_dict, json_file)

    # ====== ACT ======
    _, diff_sim_result = run_diff_res_simulation(start_date,
                                    firing_count,
                                    model_path,
                                    json_path,
                                    sim_stats,
                                    sim_logs)

    # ====== ASSERT ======
    # verify duration in stats
    task_d_sim_result = diff_sim_result.tasks_kpi_map['D']
    assert task_d_sim_result.processing_time.avg == expected_duration_sec

    # verify duration in logs
    df = pd.read_csv(sim_logs)
    
    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')

    logs_d_task = df[df['activity'] == 'D']
    expected_activity_timedelta = datetime.timedelta(seconds=expected_duration_sec)
    _verify_activity_count_and_duration(logs_d_task, firing_count, expected_activity_timedelta)


def test_seq_batch_waiting_time_correct(assets_path):
    # ====== ARRANGE ======
    model_path = assets_path / 'batch-example-end-task.bpmn'
    json_path = assets_path / 'batch-example-with-batch.json'
    sim_stats = assets_path / 'batch_stats.csv'
    sim_logs = assets_path / 'batch_logs.csv'

    start_string = '2022-06-21 13:22:30.035185+03:00'
    start_date = parse_datetime(start_string, True)

    # ====== ACT ======
    _, diff_sim_result = run_diff_res_simulation(start_date,
                                    6,
                                    model_path,
                                    json_path,
                                    sim_stats,
                                    sim_logs,
                                    True)

    # ====== ASSERT ======
    diff_sim_result.print_simulation_results()

    task_d_sim_result = diff_sim_result.tasks_kpi_map['D']
    full_act_dur = 120
    expected_activity_in_batch_duration_sec = full_act_dur * 0.8
    assert task_d_sim_result.processing_time.avg == expected_activity_in_batch_duration_sec

    # the first task in the batch will wait for the longest
    # this happens because dur_task_prior_to_batch > dur_task_in_batch (120 > 96)
    # calculation: duration of the task prior to the batch *   of tasks
    # since batch is of size 3, one is current, two more is needed to enable the batch execution
    assert task_d_sim_result.waiting_time.max == full_act_dur * 2

    # the last task in the batch (sorted by case_id) will wait the lowest
    # this happens because dur_task_prior_to_batch > dur_task_in_batch (120 > 96)
    assert task_d_sim_result.waiting_time.min == expected_activity_in_batch_duration_sec * 2

    # verify duration in logs
    df = pd.read_csv(sim_logs)
    
    df['enable_time'] = pd.to_datetime(df['enable_time'], errors='coerce')
    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')
    
    logs_d_task = df[df['activity'] == 'D']
    
    start_enable_start_diff_for_task = logs_d_task['start_time'] - logs_d_task['enable_time']
    expected_waiting_times = pd.Series([
        datetime.timedelta(seconds=full_act_dur * 2),
        datetime.timedelta(seconds=full_act_dur     + expected_activity_in_batch_duration_sec),
        datetime.timedelta(seconds=0                + expected_activity_in_batch_duration_sec * 2),
        datetime.timedelta(seconds=full_act_dur * 2),
        datetime.timedelta(seconds=full_act_dur     + expected_activity_in_batch_duration_sec),
        datetime.timedelta(seconds=0                + expected_activity_in_batch_duration_sec * 2)
    ])
    
    tm.assert_series_equal(start_enable_start_diff_for_task, expected_waiting_times, check_index=False)

def test_parallel_batch_enable_start_waiting_correct(assets_path):
    # ====== ARRANGE ======
    model_path = assets_path / 'batch-example-end-task.bpmn'
    basic_json_path = assets_path / 'batch-example-with-batch.json'
    json_path = assets_path / 'batch-example-nearest-coef.json'
    sim_stats = assets_path / 'batch_stats.csv'
    sim_logs = assets_path / 'batch_logs.csv'

    start_string = '2022-06-21 13:22:30.035185+03:00'
    start_date = parse_datetime(start_string, True)

    with open(basic_json_path, 'r') as f:
        json_dict = json.load(f)

    _setup_sim_scenario_file(json_dict, None, None, "Parallel", None)

    with open(json_path, 'w+') as json_file:
        json.dump(json_dict, json_file)

    # ====== ACT ======
    _, diff_sim_result = run_diff_res_simulation(start_date,
                                    3, # one batch 
                                    model_path,
                                    json_path,
                                    sim_stats,
                                    sim_logs)

    # ====== ASSERT ======
    # verify duration in stats
    full_act_dur = 120
    expected_activity_in_batch_duration_sec = full_act_dur * 0.8

    task_d_sim_result = diff_sim_result.tasks_kpi_map['D']
    assert task_d_sim_result.processing_time.avg == expected_activity_in_batch_duration_sec

    # verify min and max waiting times in logs
    
    # the first task in the batch will wait for the longest
    # calculation: duration of the task prior to the batch * num of tasks
    # since batch is of size 3, one is current, two more is needed to enable the batch execution
    assert task_d_sim_result.waiting_time.max == full_act_dur * 2

    # the last task in the batch (sorted by case_id) will wait the lowest
    # once we finish the prior task to batching, we enable and eventually start the batching
    # and if resource is available we start straight away
    assert task_d_sim_result.waiting_time.min == 0

    # verify waiting time in logs
    df = pd.read_csv(sim_logs)
    
    df['enable_time'] = pd.to_datetime(df['enable_time'], errors='coerce')
    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')
    
    logs_d_task = df[df['activity'] == 'D']
    
    start_enable_start_diff_for_task = logs_d_task['start_time'] - logs_d_task['enable_time']
    expected_waiting_times = pd.Series([
        datetime.timedelta(seconds=full_act_dur * 2),
        datetime.timedelta(seconds=full_act_dur),
        datetime.timedelta(seconds=0)
    ])
    
    tm.assert_series_equal(start_enable_start_diff_for_task, expected_waiting_times, check_index=False)


def test_two_batches_duration_correct(assets_path):
    """
    Input: two tasks are set up in the batch configuration: D and E.
    Expected: all tasks inside the batches are correctly executed based on the provided configuration.
    Verified that the start_time of all tasks E inside batch has the correct start_time.
    """

    # ====== ARRANGE ======
    model_path = assets_path / 'batch-example-end-task.bpmn'
    basic_json_path = assets_path / 'batch-example-with-batch.json'
    json_path = assets_path / 'batch-example-nearest-coef.json'
    sim_stats = assets_path / 'batch_stats.csv'
    sim_logs = assets_path / 'batch_logs.csv'

    start_string = '2022-06-21 13:22:30.035185+03:00'
    start_date = parse_datetime(start_string, True)

    with open(basic_json_path, 'r') as f:
        json_dict = json.load(f)

    # _setup_sim_scenario_file(json_dict, None, None, "Parallel")
    _add_batch_task(json_dict)

    with open(json_path, 'w+') as json_file:
        json.dump(json_dict, json_file)

    # ====== ACT ======
    _, diff_sim_result = run_diff_res_simulation(start_date,
                                    3, # one batch 
                                    model_path,
                                    json_path,
                                    sim_stats,
                                    sim_logs)

    # ====== ASSERT ======

    # verify the second batch (the one consisting of activity E) has correct start_time
    df = pd.read_csv(sim_logs)
    df['enable_time'] = pd.to_datetime(df['enable_time'], errors='coerce')
    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')

    grouped_by_case_id = df.groupby(by="case_id")
    last_d_activity = grouped_by_case_id.get_group(2)
    last_d_activity_end_time = \
        last_d_activity[last_d_activity['activity'] == 'D']['end_time'].values[0]
    for case_id, group in grouped_by_case_id:
        for row_index, row in group.iterrows():
            if row['activity'] != 'E':
                continue
            
            curr_start_time = np.datetime64(row['start_time'])
            expected_e_activity_duration = 120 * 0.5
            expected_e_activity_delta = \
                np.timedelta64(int(row['case_id'] * expected_e_activity_duration), 's')
            expected_start_time = last_d_activity_end_time + expected_e_activity_delta
            assert curr_start_time == expected_start_time, \
                f"The row {row_index} for case {case_id} contains incorrect start_time. \
                    Expected: {expected_start_time}, but was {curr_start_time}"


def test_waiting_time_rule_correct_firing(assets_path):
    """
    Input: firing rule of waiting time > 200 seconds. 
    Expected: batch of tasks will be executed once the oldest task in the batch pull will be > 200 seconds.
    This happens during the 3rd case, so that's when the batch execution is enabled. 
    Verified the appropriate start_time and end_time (tasks are executed in parallel).
    """

    # ====== ARRANGE ======
    model_path = assets_path / 'batch-example-end-task.bpmn'
    basic_json_path = assets_path / 'batch-example-with-batch.json'
    json_path = assets_path / 'batch-example-nearest-coef.json'
    sim_stats = assets_path / 'batch_stats.csv'
    sim_logs = assets_path / 'batch_logs.csv'

    start_string = '2022-06-21 13:22:30.035185+03:00'
    start_date = parse_datetime(start_string, True)

    with open(basic_json_path, 'r') as f:
        json_dict = json.load(f)

    firing_rules = [
        [
            {
                "attribute": "waiting_time",
                "comparison": ">",
                "value": 210 # 3.5 minutes
            }
        ]
    ]
    _setup_sim_scenario_file(json_dict, None, None, "Parallel", firing_rules)

    with open(json_path, 'w+') as json_file:
        json.dump(json_dict, json_file)

    # ====== ACT ======
    _, diff_sim_result = run_diff_res_simulation(start_date,
                                    3, # one batch 
                                    model_path,
                                    json_path,
                                    sim_stats,
                                    sim_logs)


    # ====== ASSERT ======

    # verify that batch was execute right after 8 min of simulation
    # verify that batch was of size 3 and all tasks have the same start and end date (parallel execution)

    expected_batch_start_time = start_date + datetime.timedelta(minutes=8)
    expected_batch_end_time = expected_batch_start_time + datetime.timedelta(seconds=96.0) # 80% of the full performance

    df = pd.read_csv(sim_logs)
    
    df['enable_time'] = pd.to_datetime(df['enable_time'], errors='coerce')
    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')
    
    logs_d_tasks = df[df['activity'] == 'D']

    expected_start_times = pd.Series([expected_batch_start_time] * 3)
    expected_end_times = pd.Series([expected_batch_end_time] * 3)
    
    tm.assert_series_equal(logs_d_tasks['start_time'], expected_start_times, check_index=False, check_dtype=False, check_names=False)
    tm.assert_series_equal(logs_d_tasks['end_time'], expected_end_times, check_index=False, check_dtype=False, check_names=False)


def test_waiting_time_rule_correct_firing(assets_path):
    """
    Input: firing rule of waiting time > 200 seconds. 
    Expected: batch of tasks will be executed once the oldest task in the batch pull will be > 200 seconds.
    This happens during the 3rd case, so that's when the batch execution is enabled. 
    Verified the appropriate start_time and end_time (tasks are executed in parallel).
    """

    # ====== ARRANGE ======
    model_path = assets_path / 'batch-example-end-task.bpmn'
    basic_json_path = assets_path / 'batch-example-with-batch.json'
    json_path = assets_path / 'batch-example-nearest-coef.json'
    sim_stats = assets_path / 'batch_stats.csv'
    sim_logs = assets_path / 'batch_logs.csv'

    start_string = '2022-06-21 13:22:30.035185+03:00'
    start_date = parse_datetime(start_string, True)

    with open(basic_json_path, 'r') as f:
        json_dict = json.load(f)

    firing_rules = [
        [
            {
                "attribute": "waiting_time",
                "comparison": ">",
                "value": 210 # 3.5 minutes
            }
        ]
    ]
    _setup_sim_scenario_file(json_dict, None, None, "Parallel", firing_rules)

    with open(json_path, 'w+') as json_file:
        json.dump(json_dict, json_file)

    # ====== ACT ======
    _, diff_sim_result = run_diff_res_simulation(start_date,
                                    3, # one batch 
                                    model_path,
                                    json_path,
                                    sim_stats,
                                    sim_logs)


    # ====== ASSERT ======

    # verify that batch was execute right after 8 min of simulation
    # verify that batch was of size 3 and all tasks have the same start and end date (parallel execution)

    expected_batch_start_time = start_date + datetime.timedelta(minutes=8)
    expected_batch_end_time = expected_batch_start_time + datetime.timedelta(seconds=96.0) # 80% of the full performance

    df = pd.read_csv(sim_logs)
    
    df['enable_time'] = pd.to_datetime(df['enable_time'], errors='coerce')
    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')
    
    logs_d_tasks = df[df['activity'] == 'D']

    expected_start_times = pd.Series([expected_batch_start_time] * 3)
    expected_end_times = pd.Series([expected_batch_end_time] * 3)
    
    tm.assert_series_equal(logs_d_tasks['start_time'], expected_start_times, check_index=False, check_dtype=False, check_names=False)
    tm.assert_series_equal(logs_d_tasks['end_time'], expected_end_times, check_index=False, check_dtype=False, check_names=False)


def _add_batch_task(json_dict):
    batch_processing = json_dict['batch_processing']
    batch_processing.append(
      {
         "task_id": "Activity_0ngxjs9",
         "type": "Sequential",
         "duration_distrib": {
            "3": 0.5
         },
         "firing_rules": [
            [
               {
                  "attribute": "size",
                  "comparison": "=",
                  "value": 3
               }
            ]
         ]
      }
    )


def _setup_sim_scenario_file(json_dict, duration_distrib, firing_count, batch_type, firing_rules):        
    batch_processing = json_dict['batch_processing'][0]
    if batch_type != None:
        batch_processing['type'] = batch_type

    if duration_distrib != None:
        batch_processing['duration_distrib'] = duration_distrib

    if firing_count != None:
        batch_processing['firing_rules'] = [
            [
                {
                    "attribute": "size",
                    "comparison": "=",
                    "value": firing_count
                }
            ]
        ]

    if firing_rules != None:
        batch_processing['firing_rules'] = firing_rules
