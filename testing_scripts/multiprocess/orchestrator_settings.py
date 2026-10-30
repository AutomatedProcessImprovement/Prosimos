"""
Models moved between single-process Prosimos and the orchestrator in tests: Prosimos's JSON settings name the drawn
attributes case_attributes and may carry keys Prosimos ignores, while the orchestrator reads object_attributes and
rejects keys it doesn't know. And configurations built in Python, whose processes all go in one organization.
"""
import json
import tempfile
from pathlib import Path

from prosimos.orchestration.config import Organization
from prosimos.orchestration.prosimos_engine import PROCESS_SETTINGS_KEYS

_FOLDER = Path(tempfile.mkdtemp(prefix="orchestrator_settings_"))


def orchestrator_settings(json_path, name=None):
    """The path of a copy of a single-process model's JSON settings in the orchestrator's names: case_attributes
    becomes object_attributes, and keys Prosimos ignores are left out."""
    with open(json_path) as file:
        settings = json.load(file)
    if "case_attributes" in settings:
        settings["object_attributes"] = settings.pop("case_attributes")
    path = _FOLDER / (name or Path(json_path).name)
    path.write_text(json.dumps({key: value for key, value in settings.items() if key in PROCESS_SETTINGS_KEYS}))
    return str(path)


def prosimos_settings(json_path, name=None):
    """The path of a copy of a process's JSON settings in the orchestrator's names, in single-process Prosimos's:
    object_attributes becomes case_attributes."""
    with open(json_path) as file:
        settings = json.load(file)
    if "object_attributes" in settings:
        settings["case_attributes"] = settings.pop("object_attributes")
    path = _FOLDER / (name or f"prosimos_{Path(json_path).name}")
    path.write_text(json.dumps(settings))
    return str(path)


def with_qualifiers(json_path, name=None):
    """The path of a copy of a process's JSON settings whose message entries without a qualifier get their message
    type as one, so that both ends of every message write their OCEL links."""
    with open(json_path) as file:
        settings = json.load(file)
    for entry in settings.get("messages", {}).get("publish", []) + settings.get("messages", {}).get("consume", []):
        entry.setdefault("qualifier", entry["type"])
    path = _FOLDER / (name or f"qualified_{Path(json_path).name}")
    path.write_text(json.dumps(settings))
    return str(path)


def one_organization(processes, catalogues=None):
    """The organizations of a configuration built in Python: one, holding all its processes and the catalogues."""
    return [Organization("Organization", list(processes), dict(catalogues or {}))]
