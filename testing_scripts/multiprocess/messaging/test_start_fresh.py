"""
Starting fresh on a loop back (docs/messaging.md, "append" and "aggregate"): a case that reaches a catch event it
waited at before starts that event's appended lists and aggregated values over, as at its first wait, and the 'with'
values over those lists follow; the OCEL file writes each as cleared at that moment.
"""
import json
from datetime import datetime

import pytz

from prosimos.orchestration.catalogue.model import CatalogueObject, Computed, Reference
from prosimos.orchestration.catalogue.picks import drawn_catalogues, picks_of
from prosimos.orchestration.catalogue.timetable import Timetable
from prosimos.orchestration.config import ProcessSpec, SimulationConfig
from prosimos.orchestration.orchestrator import orchestrate
from prosimos.orchestration.prosimos_engine import ProsimosEngine
from prosimos.orchestration.protocol import Message
from prosimos.orchestration.simulation import run_multiprocess_simulation
from testing_scripts.multiprocess.orchestrator_settings import one_organization
from testing_scripts.multiprocess.scripted_engine import ScriptedEngine

START = pytz.utc.localize(datetime(2024, 1, 1, 9))
ASSETS = "testing_scripts/assets/messaging"
# truck_rounds: a truck ready at 09:00 waits at Packages loaded, which collects every Package offered then, appends
# its object_id to package_ids and sums its weight into load; it delivers them (1 hour) and, after a load of 20 kg
# or more, comes back for more
WEIGHTS = {"P1": 10, "P2": 10, "P3": 10, "P4": 6, "P5": 6}
ROUND_1, ROUND_2 = ("P1", "P2", "P3"), ("P4", "P5")


def _run(tmp_path, rounds, settings=None, references=(), listed=("package_ids", "load")):
    """The truck's object in the OCEL file after a run in which Packing sends each round's packages, at once, at
    09:00 and at 11:00, and the report. settings changes the truck's JSON settings."""
    json_path = f"{ASSETS}/truck_rounds.json"
    if settings is not None:
        with open(json_path) as file:
            content = json.load(file)
        settings(content)
        json_path = tmp_path / "truck_rounds.json"
        json_path.write_text(json.dumps(content))
    packing = ScriptedEngine("Packing")
    for hour, packages in zip((9, 11), rounds):
        packing.at(START.replace(hour=hour), lambda now, packages=packages: [
            Message("Package", {"object_id": package, "weight": WEIGHTS[package]}) for package in packages])
    spec = ProcessSpec("Trucks", f"{ASSETS}/truck_rounds.bpmn", str(json_path), 1, "truck",
                       references=tuple(references), ocel_attributes=listed)
    catalogues = {"package": [CatalogueObject(package, (("weight", weight),)) for package, weight in WEIGHTS.items()]}
    config = SimulationConfig(one_organization([spec], catalogues), START, 1, extra_processes=["Packing"])
    report = run_multiprocess_simulation(config, None, {"Packing": packing}, ocel_out_path=tmp_path / "log.json")
    with open(tmp_path / "log.json") as file:
        [truck] = [obj for obj in json.load(file)["objects"] if obj["type"] == "truck"]
    return [(value["name"], value["value"], value["time"][11:16]) for value in truck["attributes"]], report


def test_a_truck_back_at_its_catch_event_collects_from_zero_with_a_clearing_at_its_arrival(tmp_path):
    # round 1 loads 30 kg, so the truck comes back at 10:00; round 2 starts from nothing: 12 kg, and the truck is
    # done (with round 1's 30 kg still counted it would come back again and wait forever)
    values, report = _run(tmp_path, [ROUND_1, ROUND_2])

    assert values == [("package_ids", '["P1", "P2", "P3"]', "09:00"), ("load", 30, "09:00"),
                      ("load", None, "10:00"), ("package_ids", None, "10:00"),
                      ("package_ids", '["P4", "P5"]', "11:00"), ("load", 12, "11:00")]
    assert report.stalled == []


def test_an_aggregate_with_initial_starts_from_its_initial_again(tmp_path):
    # the truck weighs 5 kg empty: 35 kg in round 1, 17 in round 2
    def with_initial(settings):
        settings["messages"]["consume"][0]["aggregate"]["load"]["initial"] = 5

    values, report = _run(tmp_path, [ROUND_1, ROUND_2], with_initial, listed=("load",))

    assert values == [("load", 35, "09:00"), ("load", None, "10:00"), ("load", 17, "11:00")]
    assert report.stalled == []


def test_a_with_over_an_appended_list_follows_the_list(tmp_path):
    # the packages' total weight, from the catalogue, over package_ids
    reference = Reference("package_ids", "package", link_events=(),
                          with_attributes=(("total", Computed("sum", "weight")),))

    values, _ = _run(tmp_path, [ROUND_1, ROUND_2], references=[reference], listed=("total",))

    assert values == [("total", 30, "09:00"), ("total", None, "10:00"), ("total", 12, "11:00")]


def test_while_the_case_waits_again_a_with_over_the_list_is_its_value_over_no_objects():
    # only round 1 comes: the truck comes back at 10:00 and waits to the end of the run, its list empty and its
    # total 0, the sum over no packages, as when it was created
    reference = Reference("package_ids", "package", link_events=(),
                          with_attributes=(("total", Computed("sum", "weight")),))
    spec = ProcessSpec("Trucks", f"{ASSETS}/truck_rounds.bpmn", f"{ASSETS}/truck_rounds.json", 1, "truck",
                       references=(reference,))
    catalogues = {"package": [CatalogueObject(package, (("weight", weight),)) for package, weight in WEIGHTS.items()]}
    [organization] = SimulationConfig(one_organization([spec], catalogues), START, 1).organizations
    picks = picks_of(organization.processes[0], 1, drawn_catalogues(organization, 1), Timetable((), START))
    trucks = ProsimosEngine(organization.processes[0], START, None, 1, tuple(picks))
    packing = ScriptedEngine("Packing")
    packing.at(START, lambda now: [Message("Package", {"object_id": package, "weight": WEIGHTS[package]})
                                   for package in ROUND_1])

    report = orchestrate({"Trucks": trucks, "Packing": packing}, None, 1)

    assert [case.object_id for _, case in report.stalled] == ["truck-0"]
    values = trucks._env.sim_setup.bpmn_graph.all_attributes[0]
    assert (values["package_ids"], values["load"], values["total"]) == ([], 0, 0)


def test_a_case_that_reaches_its_catch_event_once_is_unchanged(tmp_path):
    # round 1 loads 12 kg, and the truck is done after it
    values, report = _run(tmp_path, [ROUND_2])

    assert values == [("package_ids", '["P4", "P5"]', "09:00"), ("load", 12, "09:00")]
    assert report.stalled == []
