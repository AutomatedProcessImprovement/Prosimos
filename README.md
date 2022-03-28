# Prosimos

Prosimos is a Business Process Simulation Engine that supports differentiated resources. 
Prosimos considers resource pools formed by a set of resources (process participants) with different profiles. 
For example, each resource has individual calendars, costs, and a differentiated performance to execute the process activities. 
Besides, pools can share resources, i.e., a resource may play different roles in the organization, thus performing several tasks.

## Requirements

- Python 3.8+
- PIP 21.2.3+ (upgrade with python -m pip install --upgrade pip)
- For dependencies, please, check requirements.txt

## Getting Started

    git clone https://github.com/AutomatedProcessImprovement/Prosimos.git

Set up Python environment using the built-in venv module from requirements.txt. 
Once all the dependencies all installed, open a terminal and from the root folder run the following comand (i.e., all in one line):

    .\diff_res_bpsim.py start-simulation --bpmn_path <Path to the BPMN file with the process model> 
                                         --json_path <Path to the JSON file with the differentiated simulation parameters>
                                         --total_cases <Number of process instances to simulate>
                                         --log_out_path <(Optional) Path to the CSV file to save the statistics/metrics after running the simulations>
                                         --stat_out_path <(Optional) Path to the CSV file to save the event-log of the simulation>
                                         --starting_at <(Optional) Date-time of the first process case in the simulation>

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
   [Scipy Stats documentation](https://docs.scipy.org/doc/scipy/reference/stats.html#module-scipy.stats).
* "gateway_branching_probabilities": Represents the probability for the process execution to move towards any outgoing 
   flow of each split (inclusive or exclusive) gateway in the process model. So, for each gateway, the set of pairs 
   outgoing_flow -> probability (between 0 and 1) is represented. Note that the sum of all the probabilities for a 
   given gateway must always be 1. 
* "task_resource_distribution": Maps each task in the process model and the list of resources that can perform it. 
   For each task, represented by its ID, the list of allowed resources and a probability distribution function 
   (per resource) that describes its duration are represented. Note that the distribution function of a task may vary 
   per resource. As for the arrival time distributions, **Prosimos** allows any of the functions supported by the Python 
   library [Scipy Stats](https://docs.scipy.org/doc/scipy/reference/stats.html#module-scipy.stats).
* "resource_calendars": List of time intervals in which a resource is available to perform a task on a weekly calendar basis. 
   Each calendar interval is described starting from weekday (Monday, ..., Sunday) at some beginTime, 
   until another (not necessarily different) weekday to some endTime.

The following snippet outlines an example of the general structure of the input JSON parameters with the simulation parameters.

      {
          "resource_profiles": [
              {
                  "id": "Profile ID_1",
                  "name": "Credit Officer",
                  "resource_list": [
                      {
                          "id": "resource_id_1",
                          "name": "Credit Officer_1",
                          "cost_per_hour": "35",
                          "amount": 1,
                          "calendar": "sid-222A1118-4766-43B2-A004-7DADE521982D",
                          "assignedTasks": ["sid-622A1118-4766-43B2-A004-7DADE521982D"]
                      },
                      {
                          "id": "resource_id_2",
                          "name": "Credit Officer_2",
                          "cost_per_hour": "35",
                          "amount": 1,
                          "calendar": "sid-222A1118-4766-43B2-A004-7DADE521982D",
                          "assignedTasks": ["sid-622A1118-4766-43B2-A004-7DADE521982D"]
                      }
                  ]
              }
          ],
          "arrival_time_distribution": {
              "distribution_name": "expon",
              "distribution_params": [
                  { "value": 0 },
                  { "value": 1800.0 },
                  { "value": 90.0 }
              ]
          },
          "arrival_time_calendar": [{
              "from": "MONDAY",
              "to": "FRIDAY",
              "beginTime": "09:00:00.000",
              "endTime": "17:00:00.000"
          }],
          "gateway_branching_probabilities": [
              {
                  "gateway_id": "sid-64FC5B46-47E5-4940-A0AF-ECE87483967D",
                  "probabilities": [
                      {
                          "path_id": "sid-8AE82A7B-75EE-401B-8ABE-279FB05A3946",
                          "value": "0.7"
                      },
                      {
                          "path_id": "sid-789335C6-205C-4A03-9AD6-9655893C1FFB",
                          "value": "0.3"
                      }
                  ]
              },
              {
                  "gateway_id": "sid-FACFF0AE-6A1B-47AC-B289-F5E60CB12B2A",
                  "probabilities": [
                      {
                          "path_id": "sid-AFEC7074-8C12-43E2-A1FE-87D5CEF395C8",
                          "value": "0.3"
                      },
                      {
                          "path_id": "sid-AE313010-5715-438C-AD61-1C02F03DCF77",
                          "value": "0.7"
                      }
                  ]
              }
          ],
          "task_resource_distribution": [
              {
                  "task_id": "sid-622A1118-4766-43B2-A004-7DADE521982D",
                  "resources": [
                      {
                          "resource_id": "resource_id_1",
                          "distribution_name": "norm",
                          "distribution_params": [
                              { "value": 600.0 },
                              { "value": 120.0 }
                          ]
                      },
                      {
                          "resource_id": "resource_id_2",
                          "distribution_name": "norm",
                          "distribution_params": [
                              { "value": 60.0 },
                              { "value": 12.0 }
                          ]             
                      }
                  ]
              }
          ],
          "resource_calendars": [
              {
                  "id": "sid-222A1118-4766-43B2-A004-7DADE521982D",
                  "name": "calendar1",
                  "time_periods": [
                      {
                          "from": "MONDAY",
                          "to": "FRIDAY",
                          "beginTime": "09:00:00.000",
                          "endTime": "17:00:00.000"
                      },
                      {
                          "from": "SATURDAY",
                          "to": "SATURDAY",
                          "beginTime": "09:00:00.000",
                          "endTime": "13:00:00.000"
                      }
                  ]
              }
          ]
      }

## Running Experiments BPM-2022

      git clone https://github.com/AutomatedProcessImprovement/Prosimos.git

* Set up Python environment using the built-in venv module from requirements.txt. 
* Unzip the file input_output_files.zip, to add the folder **input_output_files** containing all the input files used in 
the experimentation (excluding the log insurance which is private) in the root folder.
* Once all the dependencies all installed, run the script **bpm22_experiments_script.py**, in the folder **testing_scripts**.
Then check the information printed in the terminal. 


