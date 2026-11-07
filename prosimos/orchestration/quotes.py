"""Quotes: the orchestrator asks the members of the group that supplies a quote's lines (docs/orchestrator.md)."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from prosimos.orchestration.protocol import Line, Message, SimulationEngine, Slots


@dataclass
class Plan:
    """Where and when each line of a quote would be ready: line id -> (member, time), and the latest time."""

    lines: Dict[str, Tuple[str, datetime]] = field(default_factory=dict)
    ready_by: Optional[datetime] = None


def ask_members(message: Message, members: List[str], engine: Callable[[str], SimulationEngine],
                now: datetime) -> Dict[str, Slots]:
    """Each member's slots for the stamped quote message, asked at now; engine(name) gives a member's engine."""
    return {member: engine(member).estimate(message, now) for member in members}


class QuoteState(Enum):
    """Where a tracked quote stands."""

    OPEN = "open"
    DECLINED = "declined"
    LOCKED = "locked"  # some member's share has moved on: it can no longer be modified or canceled


@dataclass
class TrackedQuote:
    """A quote the orchestrator tracks: its state, its limit date (None: no limit), and per line id the line, the
    member holding it, and the message that reserved it there."""

    state: QuoteState
    limit: Optional[datetime] = None
    lines: Dict[str, Tuple[Line, str, Message]] = field(default_factory=dict)


class QuoteBook:
    """The quotes the orchestrator tracks, by requester and quote type: the requesting process, its object's id,
    since that id is unique only within a process, and the quote type."""

    def __init__(self):
        self.quotes: Dict[Tuple[str, str, str], TrackedQuote] = {}

    def answer(self, message: Message, members: List[str], engine: Callable[[str], SimulationEngine],
               now: datetime, warn: Callable[[str], None]) -> Message:
        """Asks the members, and if their plan is ready by the limit, the quote's time plus within (or there is no
        limit), reserves its lines at each member and returns the accepted reply; else the declined reply. A quote
        without lines is accepted at once, and a second one while the requester's quote of its type is open is
        declined at once, each with a warning; once some of the first has moved on, the second replaces it."""
        quote = message.quote
        if quote.accepted is None or quote.declined is None:
            raise ValueError(f"{message.type} {message.id} from {message.source} asks a quote without the types of "
                             f"its replies: a quote needs accepted and declined")
        key = (message.source, quote.request, quote.quote_type)
        if key in self.quotes and self._still_open(self.quotes[key], engine, now):
            warn(f"{quote.request} already has an open {quote.quote_type}; use modify")
            return Message(quote.declined, {"request": quote.request, "earliest": None})
        limit = None if quote.limit is None else message.time + quote.limit
        if not quote.lines:
            warn(f"{message.type} {message.id} from {message.source} has no lines")
            self.quotes[key] = TrackedQuote(QuoteState.OPEN, limit)
            return Message(quote.accepted, {"request": quote.request, "ready_by": message.time, "deliveries": 0})
        slots_of = ask_members(message, members, engine, now)
        plan = plan_lines(quote.lines, slots_of)
        if plan is None or limit is not None and plan.ready_by > limit:
            self.quotes[key] = TrackedQuote(QuoteState.DECLINED, limit)
            return Message(quote.declined, {"request": quote.request, "earliest": plan.ready_by if plan else None})
        tracked = self.quotes[key] = TrackedQuote(QuoteState.OPEN, limit)
        line_of = {line.id: line for line in quote.lines}
        deliveries = 0  # the members holding lines: each delivers its share at once
        for member in slots_of:  # in the order the members were asked
            line_ids = [line_id for line_id, (holder, _) in plan.lines.items() if holder == member]
            if line_ids:
                engine(member).reserve(message, line_ids, now)
                tracked.lines.update((line_id, (line_of[line_id], member, message)) for line_id in line_ids)
                deliveries += 1
        return Message(quote.accepted, {"request": quote.request, "ready_by": plan.ready_by,
                                        "deliveries": deliveries})


    @staticmethod
    def _still_open(tracked, engine, now):
        """Whether the tracked quote is open and nothing of it has moved on: its members are asked with the
        messages that reserved their lines. One without lines has nothing to deliver."""
        if tracked.state is not QuoteState.OPEN or not tracked.lines:
            return False
        holders = {(member, id(reserving)): (member, reserving) for _, member, reserving in tracked.lines.values()}
        return not any(engine(member).moved(reserving, now) for member, reserving in holders.values())


def plan_lines(lines: List[Line], slots_of: Dict[str, Slots]) -> Optional[Plan]:
    """The slot rule: each line takes the earliest unused slot across the members, whose slots are in the order
    the members are asked; identical lines share one slot list, so they take its 1st, 2nd, ... slot. None if a line
    has no slot."""
    plan = Plan()
    for same in _identical(lines):
        # a member answers the same list for identical lines: its 1st, 2nd, ... matching item
        slots = sorted((time, order, index, member)
                       for order, (member, member_slots) in enumerate(slots_of.items())
                       for index, time in enumerate(member_slots.get(same[0].id, [])))
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
