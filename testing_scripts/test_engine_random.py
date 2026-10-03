"""
One random generator per engine (docs/engine-internals.md, "Random numbers"): each ProsimosEngine
draws from its own Python and NumPy generator states, seeded from the simulation seed and its
process name, so one process's draws don't depend on which other processes run beside it.
"""
import csv
import json
import random
from datetime import datetime

import numpy as np
import pytz

from prosimos.orchestrator import ProcessSpec, SimulationConfig, run_orchestrator
from prosimos.simulation_engine import run_simulation

EXAMPLE = "testing_scripts/assets/running_example"
START = pytz.utc.localize(datetime(2024, 1, 1, 9))
SEED = 1


def _sales(name="Sales"):
    return ProcessSpec(name, f"{EXAMPLE}/sales.bpmn", f"{EXAMPLE}/sales.json", 20)


BILLING = ProcessSpec("Billing", f"{EXAMPLE}/billing.bpmn", f"{EXAMPLE}/billing.json")  # one random-length invoice per order


def _rows(tmp_path, processes, seed=SEED, name="log"):
    """The merged log of a run, without the process column, keyed by process."""
    path = tmp_path / f"{name}.csv"
    run_orchestrator(SimulationConfig(processes, START, seed), path)
    with open(path, encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    by_process = {}
    for row in rows:
        by_process.setdefault(row.pop("process"), []).append(row)
    return by_process


def _sales_columns(rows):
    return [{key: row[key] for key in ("case_id", "activity", "enable_time", "start_time", "end_time", "resource",
                                       "city")} for row in rows]


def test_sales_log_is_the_same_whether_or_not_billing_runs_beside_it(tmp_path):
    alone = _rows(tmp_path, [_sales()], name="alone")
    with_billing = _rows(tmp_path, [_sales(), BILLING], name="with_billing")

    assert len(with_billing["Billing"]) == 20  # Billing really ran, drawing an invoice time per order
    assert _sales_columns(alone["Sales"]) == _sales_columns(with_billing["Sales"])


def test_two_engines_of_the_same_model_under_different_names_draw_different_values(tmp_path):
    rows = _rows(tmp_path, [_sales("SalesA"), _sales("SalesB")])

    assert _sales_columns(rows["SalesA"]) != _sales_columns(rows["SalesB"])


def test_the_same_seed_gives_the_same_run_and_another_seed_a_different_one(tmp_path):
    first = _rows(tmp_path, [_sales(), BILLING], name="first")
    second = _rows(tmp_path, [_sales(), BILLING], name="second")
    other_seed = _rows(tmp_path, [_sales(), BILLING], seed=SEED + 1, name="other")

    assert first == second
    assert first != other_seed


def test_a_run_leaves_the_callers_random_generators_alone(tmp_path):
    random.seed(123)
    np.random.seed(123)
    before = random.getstate(), np.random.get_state()

    _rows(tmp_path, [_sales(), BILLING])

    python_state, numpy_state = random.getstate(), np.random.get_state()
    assert python_state == before[0]
    assert all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b for a, b in zip(numpy_state, before[1]))


def test_ties_at_an_event_based_gateway_are_repeatable_with_a_seed(tmp_path):
    # after Task A, an event-based gateway races two timers of the same length, so every case is a tie
    bpmn = tmp_path / "tie.bpmn"
    bpmn.write_text("""<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL"><bpmn:process id="P">
      <bpmn:startEvent id="Start"/><bpmn:task id="Task_A" name="Task A"/><bpmn:eventBasedGateway id="Race"/>
      <bpmn:intermediateCatchEvent id="Timer_X"><bpmn:timerEventDefinition/></bpmn:intermediateCatchEvent>
      <bpmn:intermediateCatchEvent id="Timer_Y"><bpmn:timerEventDefinition/></bpmn:intermediateCatchEvent>
      <bpmn:task id="Task_X" name="Task X"/><bpmn:task id="Task_Y" name="Task Y"/>
      <bpmn:exclusiveGateway id="Merge"/><bpmn:endEvent id="End"/>
      <bpmn:sequenceFlow id="F1" sourceRef="Start" targetRef="Task_A"/>
      <bpmn:sequenceFlow id="F2" sourceRef="Task_A" targetRef="Race"/>
      <bpmn:sequenceFlow id="F3" sourceRef="Race" targetRef="Timer_X"/>
      <bpmn:sequenceFlow id="F4" sourceRef="Race" targetRef="Timer_Y"/>
      <bpmn:sequenceFlow id="F5" sourceRef="Timer_X" targetRef="Task_X"/>
      <bpmn:sequenceFlow id="F6" sourceRef="Timer_Y" targetRef="Task_Y"/>
      <bpmn:sequenceFlow id="F7" sourceRef="Task_X" targetRef="Merge"/>
      <bpmn:sequenceFlow id="F8" sourceRef="Task_Y" targetRef="Merge"/>
      <bpmn:sequenceFlow id="F9" sourceRef="Merge" targetRef="End"/>
    </bpmn:process></bpmn:definitions>""")
    tasks = ["Task_A", "Task_X", "Task_Y"]
    settings = {
        "resource_profiles": [{"id": "Team", "name": "Team", "resource_list": [
            {"id": "Worker", "name": "Worker", "cost_per_hour": 1, "amount": 3, "calendar": "Always",
             "assigned_tasks": tasks}]}],
        "arrival_time_distribution": {"distribution_name": "fix", "distribution_params": [{"value": 600}]},
        "arrival_time_calendar": [{"from": "MONDAY", "to": "SUNDAY", "beginTime": "00:00:00", "endTime": "23:59:59"}],
        "gateway_branching_probabilities": [],
        "task_resource_distribution": [{"task_id": task, "resources": [
            {"resource_id": "Worker", "distribution_name": "fix", "distribution_params": [{"value": 60}]}]}
            for task in tasks],
        "resource_calendars": [{"id": "Always", "name": "Always", "time_periods": [
            {"from": "MONDAY", "to": "SUNDAY", "beginTime": "00:00:00", "endTime": "23:59:59"}]}],
        "event_distribution": [{"event_id": timer, "distribution_name": "fix", "distribution_params": [{"value": 300}]}
                               for timer in ("Timer_X", "Timer_Y")],
    }
    json_path = tmp_path / "tie.json"
    json_path.write_text(json.dumps(settings))

    def branches(seed):
        random.seed(seed)
        np.random.seed(seed)
        run_simulation(str(bpmn), str(json_path), 20, None, tmp_path / "tie.csv", "2024-01-01T09:00:00+00:00")
        with open(tmp_path / "tie.csv") as file:
            return [row["activity"] for row in csv.DictReader(file) if row["activity"] != "Task A"]

    first = branches(SEED)
    assert set(first) == {"Task X", "Task Y"}  # ties really go both ways
    assert all(branches(SEED) == first for _ in range(5))
