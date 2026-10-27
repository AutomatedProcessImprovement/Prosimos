"""
The contract between the engines and the orchestrator (docs/orchestrator.md, "Engine interface"): what a message
is, what an engine answers when it is offered one, what it records for the output, and the interface an engine
implements. It doesn't depend on Prosimos: another engine can implement it.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class Message:
    """One object passed between processes. The publishing process sets type and attributes;
    the orchestrator sets id, source and time when it stamps the message."""

    type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    source: Optional[str] = None
    time: Optional[datetime] = None
