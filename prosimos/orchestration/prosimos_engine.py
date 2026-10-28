"""
Prosimos as an engine (docs/orchestrator.md, "Engine interface"): one implementation of the engine interface, which
makes a Prosimos simulation answer the orchestrator.
"""
import json
import tempfile
from pathlib import Path

from prosimos.messaging_parser import unknown_keys
from prosimos.simulation_setup import SimDiffSetup


# the top-level sections of a process's JSON settings: Prosimos's own, with object_attributes in place of its
# case_attributes, and messages
PROCESS_SETTINGS_KEYS = {"resource_profiles", "resource_calendars", "task_resource_distribution",
                         "arrival_time_distribution", "arrival_time_calendar", "gateway_branching_probabilities",
                         "branch_rules", "event_distribution", "batch_processing", "prioritisation_rules",
                         "object_attributes", "event_attributes", "global_attributes", "gateway_execution_limit",
                         "multitask", "model_type", "granule_size", "messages"}


def load_process(bpmn_path, json_path, arrivals, start_datetime, picks=()) -> SimDiffSetup:
    """Prosimos's setup of one process, from its JSON settings in the orchestrator's names: Prosimos reads
    object_attributes as its case_attributes and arrivals as its total_cases, so the settings are handed to it
    translated, in a copy that is deleted once it has read them. A key that isn't one of the sections is
    rejected, as a misspelled one would be ignored. picks are attributes given from outside the settings (see
    _Pick)."""
    with open(json_path) as file:
        settings = json.load(file)
    problem = unknown_keys("top level", settings, PROCESS_SETTINGS_KEYS)
    if problem:
        raise ValueError(f"{json_path}: {problem}")
    if "object_attributes" in settings:
        settings["case_attributes"] = settings.pop("object_attributes")
    with tempfile.TemporaryDirectory() as folder:
        translated = Path(folder) / Path(json_path).name
        translated.write_text(json.dumps(settings))
        return SimDiffSetup(bpmn_path, str(translated), False, arrivals, start_datetime, picks, "arrivals")
