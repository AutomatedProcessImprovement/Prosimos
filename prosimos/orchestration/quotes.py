"""Quotes: the orchestrator asks the members of the group that supplies a quote's lines (docs/orchestrator.md)."""
from datetime import datetime
from typing import Callable, Dict, List

from prosimos.orchestration.protocol import Estimate, Message, SimulationEngine


def ask_members(message: Message, members: List[str], engine: Callable[[str], SimulationEngine],
                now: datetime) -> Dict[str, Estimate]:
    """Each member's estimate for the stamped quote message, asked at now; engine(name) gives a member's engine."""
    return {member: engine(member).estimate(message, now) for member in members}
