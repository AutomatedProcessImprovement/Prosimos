# Prosimos

![build](https://github.com/AutomatedProcessImprovement/prosimos/actions/workflows/python.yml/badge.svg)
![release](https://github.com/AutomatedProcessImprovement/prosimos/actions/workflows/release-pypi.yml/badge.svg)
![version](https://img.shields.io/github/v/tag/AutomatedProcessImprovement/prosimos)

Prosimos is a Business Process Simulation Engine that supports differentiated resources. 
Prosimos considers resource pools formed by a set of resources (process participants) with different profiles. 
For example, each resource has individual calendars, costs, and a differentiated performance to execute the process activities. 
Besides, pools can share resources, i.e., a resource may play different roles in the organization, thus performing several tasks.

## Requirements

- Python 3.9+
- Poetry 1.4.2
- For dependencies, please, check `pyproject.toml`

## Getting Started

    git clone https://github.com/AutomatedProcessImprovement/Prosimos.git

Set up Python environment using Poetry tool:

    poetry install

Once all the dependencies all installed, open a terminal and from the root folder run the following comand (i.e., all in one line):

    poetry run prosimos start-simulation --bpmn_path <Path to the BPMN file with the process model> 
                                         --json_path <Path to the JSON file with the differentiated simulation parameters>
                                         --total_cases <Number of process instances to simulate>
                                         --log_out_path <(Optional) Path to the CSV file to save the statistics/metrics after running the simulations>
                                         --stat_out_path <(Optional) Path to the CSV file to save the event-log of the simulation>
                                         --starting_at <(Optional) Date-time of the first process case in the simulation as a string. For example, 2022-06-21T13:22:30.035185+03:00>

The last three parameters are optional. 
If none of the output file paths **_stat_out_path_** and **_log_out_path_** are provided, then **_stat_out_path_** is used by default, and the statistics file generated in the current directory. 
If parameter **_starting_at_** is not provided, the current date-time is assigned as starting point for the simulation.


## Simulation Input File Formats 

The first input parameter described by **_--bpmn_path_** is a process model written in the Business Process Model and Notation 
([BPMN Standard](https://www.bpmn.org/#:~:text=BPMN%20is%20a%20standard%20set,the%20communication%20between%20independent%20processes)).
Note that **Prosimos** only extracts the information related to the control-flow from the BPMN model, 
i.e., the tasks, gateways, and their relations from the flow-arcs. 
The information associated with the remaining simulation parameters, 
e.g., arrival time distributions, resource tasks associations, calendars, branching probabilities and task duration distributions, 
are specified separately in a JSON file.  

The JSON file with the simulation parameters is split into 6 sections described below. 
Note that the order of the sections is not relevant, i.e., they can appear in any order in the JSON file.

* "resource_profiles": Contains the information of the resources, grouped into pools. 
   Specifically, it includes a set of resource pools. Each resource pool is represented by its ID, 
   containing a name and a "resource_list". Besides, each resource in "resource_list" contains id, name, 
   cost per hour, and the amount.
* "arrival_time_calendar": List of time intervals in which new process cases can be started on a weekly calendar basis. 
   Each calendar interval is described starting from weekday (Monday, ..., Sunday) at some beginTime, 
   until another (not necessarily different) weekday to some endTime.
* "arrival_time_distribution": Probability distribution function that describes how a new process case is started 
   across the arrival calendar. **Prosimos** allows any of the functions supported by the Python library 
   [Scipy Stats](https://docs.scipy.org/doc/scipy/reference/stats.html#module-scipy.stats). Specifically, 
   the distribution function is represented by its name and a list of parameters 
   (i.e., typically ranging from 1 to 3 floating numbers depending on the function). 
   For the complete list of available distributions, please check the 
   [Scipy Stats documentation](https://docs.scipy.org/doc/scipy/reference/stats.html#module-scipy.stats). The unit of measurement of values provided as `distribution_params` is seconds.
* "gateway_branching_probabilities": Represents the probability for the process execution to move towards any outgoing 
   flow of each split (inclusive or exclusive) gateway in the process model. So, for each gateway, the set of pairs 
   outgoing_flow -> probability (between 0 and 1) is represented. Note that the sum of all the probabilities for a 
   given gateway must always be 1. 
* "task_resource_distribution": Maps each task in the process model and the list of resources that can perform it. 
   For each task, represented by its ID, the list of allowed resources and a probability distribution function 
   (per resource) that describes its duration are represented. Note that the distribution function of a task may vary 
   per resource. As for the arrival time distributions, **Prosimos** allows any of the functions supported by the Python 
   library [Scipy Stats](https://docs.scipy.org/doc/scipy/reference/stats.html#module-scipy.stats). The unit of measurement of values provided as `distribution_params` is seconds.
* "resource_calendars": List of time intervals in which a resource is available to perform a task on a weekly calendar basis. 
   Each calendar interval is described starting from weekday (Monday, ..., Sunday) at some beginTime, 
   until another (not necessarily different) weekday to some endTime.
* "batch_processing": List of tasks that are batched (= could be executed together based on the specified rules).
  * `type`. There might be batched tasks of two types: `Sequential` or `Parallel`. 
  * `size_distrib` defines the probability of tasks to be executed in batches of the specified size. The total sum of values should be 1 (which equals 100% of tasks). For example, if we want to imply that all tasks are being batched, we say:
  ```
    "size_distrib": [
        {
            "key": "1",
            "value": 0
        },
        {
            "key": "2",
            "value": 1
        }
    ]
  ```
  * `duration_distrib` defines the scaling factor for the activity's duration based on the number of tasks in the batch. Let's say one defines the array in the following way:
  ```
    "duration_distrib": [
        {
            "key": "3",
            "value": 0.8
        }
    ]
  ```
  This means that batches with 1 or 2 tasks inside will be executed with the defined (original) duration. Starting from batches with 3 tasks inside, the duration of the individual task inside the batch will be scaled by `0.8`. The array should contain objects with unique keys. The tool will sort values internally, so the user does not need to provide them in ascending order.
  * `firing_rules`. 
* "case_attributes": Description on which case attributes should be generated during the simulation and what value it should have. There is a possibility to introduce two types of case attributes: `discrete` and `continuous`. The `values` property of the `continuous` case attribute is defined as a distribution function selected from the Python library [Scipy Stats](https://docs.scipy.org/doc/scipy/reference/stats.html#module-scipy.stats). The unit of measurement of values provided as `distribution_params` is seconds. The `values` property of the `discrete` case attribute is defined as an array of possible choices with theirs probability accordingly. The sum of probabilities should be equal to 1. 

An example of the general structure of the input JSON parameters with the simulation parameters could be found [here](./simulation_scenario_example.json).

## Running Experiments BPM-2022

      git clone https://github.com/AutomatedProcessImprovement/Prosimos.git

* Set up Python environment using the built-in venv module from requirements.txt. 
* Unzip the file input_output_files.zip, to add the folder **input_output_files** containing all the input files used in 
the experimentation (excluding the log insurance which is private) in the root folder.
* Once all the dependencies all installed, run the script **bpm22_experiments_script.py**, in the folder **testing_scripts**.
Then check the information printed in the terminal. 

## Running tests and receive a coverage report 

```
pytest --cov-config=.coveragerc --cov --cov-report=html --cov-branch
```
* If one wants to skip running the tests and just overview a coverage report, one needs to unzip **htmlcov.zip** archive. Unpacked folder **htmlcov** contains HTML coverage report, in general and per each file.

## Notes on releasing a new version

Once you are done with the development and all changes are in `master` branch, we can move forward with releasing PyPI package. The workflow grabs the package version from `pyproject.toml`. So before proceeding further, make sure that you updated a version of the package to-be-released in `pyproject.toml` file. Next, take the following steps to release the new version:

0. `Build & Test` job succeeded on the `master` branch. This means all tests are passing. (The job is triggered automatically when you push new changes to `master`)
1. Go to `Actions` tab.
2. Find `Release PyPI package` from the list of all existing actions on the left-side menu.
3. Start the workflow by clicking on `Run workflow` button.
4. After the task success, double verify that the new version was published on PyPI (https://pypi.org/project/prosimos/).


<details><summary>Development notes</summary>

#### Use local version of `pix-framework`

In case you want to introduce and test changes both to `pix-framework` and `Prosimos`, we need to install local version of `pix-framework` instead of the PyPI released version. For this, the following command needs to be run:

```
poetry add --editable <relative-path> 
```
, where `<relative-path>` should be changed with the path to the root folder of `pix-framework` (e.g., ...`/pix-framework`).
</details>
