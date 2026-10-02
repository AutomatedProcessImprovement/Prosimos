"""
finish(): after the loop, every engine reports its stalled cases and the warnings it raised, and the
orchestrator merges them into the run report under the engine's process name.
"""
import json
import random
from datetime import datetime

import numpy as np
import pytest
import pytz

from prosimos.orchestrator import ProcessSpec, ProsimosEngine, StalledCase, run_engines
from prosimos.warning_logger import warning_logger
from testing_scripts.scripted_engine import ScriptedEngine

ASSETS = "testing_scripts/assets/messaging"
START = pytz.utc.localize(datetime(2024, 1, 1, 9))
DURATION_IGNORED = "duration of Catch_Shipment is ignored: it waits for a message"
WEIGHT_MISSING = "Attribute weight has no value when case 0 passes Throw_First; its First message carries None"


class _Log:
    def __init__(self):
        self.rows = []

    def writerow(self, header):
        pass

    def writerows(self, rows):
        self.rows.extend(rows)


def _time(logged):
    return logged if isinstance(logged, datetime) else datetime.fromisoformat(logged)


def _sales(cases, json_path=f"{ASSETS}/sales.json"):
    log = _Log()
    return ProsimosEngine(ProcessSpec("Sales", f"{ASSETS}/sales.bpmn", json_path, cases), START, log), log


def _sales_with_a_duration(tmp_path):
    """Sales settings that give the waiting catch event a duration, so loading raises a warning."""
    with open(f"{ASSETS}/sales.json") as file:
        settings = json.load(file)
    settings["event_distribution"] = [
        {"event_id": "Catch_Shipment", "distribution_name": "fix", "distribution_params": [{"value": 3600}]}]
    path = tmp_path / "sales.json"
    path.write_text(json.dumps(settings))
    return str(path)


@pytest.fixture(autouse=True)
def seeded():
    random.seed(1)
    np.random.seed(1)


def test_a_run_where_one_order_never_gets_its_shipment_lists_exactly_that_case_as_stalled():
    sales, log = _sales(3)
    warehouse = ScriptedEngine("Warehouse")
    for order_id in ("Sales-0", "Sales-2"):  # Sales-1 is never shipped
        warehouse.publish_at(START.replace(hour=12), "Shipment", order_id=order_id)

    report = run_engines({"Sales": sales, "Warehouse": warehouse}, None, 1)

    reached = next(_time(row[4]) for row in log.rows if row[0] == 1 and row[1] == "Take order")
    assert report.stalled == [("Sales", StalledCase("Sales-1", "Catch_Shipment", ["Shipment"], reached))]


def test_a_case_at_an_event_accepting_several_types_lists_them_all(tmp_path):
    # Catch_Shipment is listed twice under consume: a Shipment or a Delay for the order resumes it
    with open(f"{ASSETS}/sales.json") as file:
        settings = json.load(file)
    settings["messages"]["consume"].append(dict(settings["messages"]["consume"][0], type="Delay"))
    path = tmp_path / "sales.json"
    path.write_text(json.dumps(settings))
    sales, _ = _sales(1, json_path=str(path))

    report = run_engines({"Sales": sales}, None, 1)

    assert [(process, case.case_id, case.message_types) for process, case in report.stalled] == [
        ("Sales", "Sales-0", ["Delay", "Shipment"])]


def test_a_warning_raised_inside_an_engine_appears_in_the_run_report_under_its_name(tmp_path):
    sales, _ = _sales(1, json_path=_sales_with_a_duration(tmp_path))  # warns while the model is loaded

    report = run_engines({"Sales": sales}, None, 1)

    assert report.engine_warnings == [("Sales", DURATION_IGNORED)]


def test_the_warnings_of_two_engines_never_mix(tmp_path):
    # Sales warns while it is loaded, Bursts while it runs (a published attribute without a value)
    sales, _ = _sales(1, json_path=_sales_with_a_duration(tmp_path))
    bursts = ProsimosEngine(ProcessSpec("Bursts", f"{ASSETS}/bursts.bpmn", f"{ASSETS}/bursts.json", 3), START)
    warning_logger.warnings_queue[:] = ["raised outside any engine"]

    report = run_engines({"Sales": sales, "Bursts": bursts}, None, 1)

    assert report.engine_warnings == [("Bursts", WEIGHT_MISSING), ("Sales", DURATION_IGNORED)]
    assert sales.finish().warnings == [DURATION_IGNORED] and bursts.finish().warnings == [WEIGHT_MISSING]
    # Prosimos's shared list only ever holds what was raised outside the engines
    assert warning_logger.get_all_warnings() == ["raised outside any engine"]
    warning_logger.clear_warnings()


def test_the_orchestrator_merges_what_every_engine_reports_in_process_name_order():
    # given in reverse name order, so the report's order comes from the names, not the dict
    b, a, c = ScriptedEngine("B"), ScriptedEngine("A"), ScriptedEngine("C")
    a.report.warnings.append("A noticed something")
    a.report.stalled.append(StalledCase("A-0", "Catch_Y", ["Y"], START))
    b.report.warnings.extend(["B noticed something", "B noticed something else"])
    b.report.stalled.append(StalledCase("B-4", "Catch_X", ["X"], START))
    # C reports nothing

    report = run_engines({"B": b, "A": a, "C": c}, None, 1)

    assert report.stalled == [("A", StalledCase("A-0", "Catch_Y", ["Y"], START)),
                              ("B", StalledCase("B-4", "Catch_X", ["X"], START))]
    assert report.engine_warnings == [("A", "A noticed something"), ("B", "B noticed something"),
                                      ("B", "B noticed something else")]


def test_an_engine_whose_finish_returns_something_else_is_rejected():
    engine = ScriptedEngine("Old")
    engine.finish = lambda: []

    with pytest.raises(TypeError, match="must return an EngineReport"):
        run_engines({"Old": engine}, None, 1)
