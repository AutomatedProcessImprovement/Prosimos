"""Quotes: the orchestrator asks the members of the group that supplies a quote's lines (docs/orchestrator.md)."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from prosimos.orchestration.protocol import Estimate, Line, Message, SimulationEngine


@dataclass
class Plan:
    """Where and when each line of a quote would be ready: line id -> (member, time), and the latest time."""

    lines: Dict[str, Tuple[str, datetime]] = field(default_factory=dict)
    ready_by: Optional[datetime] = None


def ask_members(message: Message, members: List[str], engine: Callable[[str], SimulationEngine],
                now: datetime) -> Dict[str, Estimate]:
    """Each member's estimate for the stamped quote message, asked at now; engine(name) gives a member's engine."""
    return {member: engine(member).estimate(message, now) for member in members}


def plan_lines(lines: List[Line], estimates: Dict[str, Estimate]) -> Optional[Plan]:
    """The slot rule: each line takes the earliest unused slot across the members, whose estimates are in the order
    the members are asked; identical lines share one slot list, so they take its 1st, 2nd, ... slot. None if a line
    has no slot."""
    plan = Plan()
    for same in _identical(lines):
        # a member answers the same list for identical lines: its 1st, 2nd, ... matching item
        slots = sorted((time, order, index, member)
                       for order, (member, estimate) in enumerate(estimates.items())
                       for index, time in enumerate(estimate.slots.get(same[0].id, [])))
        if len(slots) < len(same):
            return None
        for line, (time, _, _, member) in zip(same, slots):
            plan.lines[line.id] = (member, time)
    plan.ready_by = max((time for _, time in plan.lines.values()), default=None)
    return plan


def _identical(lines):
    """The lines grouped by their attributes, each group and the groups in the order of the lines."""
    groups = []
    for line in lines:
        group = next((group for group in groups if group[0].attributes == line.attributes), None)
        if group is None:
            groups.append([line])
        else:
            group.append(line)
    return groups
