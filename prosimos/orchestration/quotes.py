"""Quotes: the orchestrator asks the members of the group that supplies a quote's lines (docs/orchestrator.md)."""
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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


# what became of the quote, modify, and cancel messages of a quote type, in the report's quotes section
OUTCOMES = ("accepted", "declined", "canceled", "cancel_too_late", "modified_accepted", "modified_declined",
            "declined_as_second")


class QuoteState(Enum):
    """Where a tracked quote stands."""

    OPEN = "open"
    DECLINED = "declined"
    LOCKED = "locked"  # some member's share has moved on: it can no longer be modified or canceled
    COMPLETE = "complete"  # every member's share has moved on: the quote was delivered in full


@dataclass
class HeldLine:
    """A line of a tracked quote: the member holding it, the message that reserved it there, and when it would be
    ready."""

    line: Line
    member: str
    reserving: Message
    ready: datetime


@dataclass
class TrackedQuote:
    """A quote the orchestrator tracks: its state, its limit date (None: no limit), its held lines by line id, and
    the ready_by of its last accepted reply, its promise."""

    state: QuoteState
    limit: Optional[datetime] = None
    lines: Dict[str, HeldLine] = field(default_factory=dict)
    ready_by: Optional[datetime] = None


class QuoteBook:
    """The quotes the orchestrator tracks, by requester and quote type: the requesting process, its object's id,
    since that id is unique only within a process, and the quote type."""

    def __init__(self):
        self.quotes: Dict[Tuple[str, str, str], TrackedQuote] = {}
        # the members a cancel or modify released lines at, since the orchestrator last took them: its pooled copies
        # are offered again, as a freed item may take one
        self.released_at: List[str] = []
        self.outcomes: Dict[str, Counter] = defaultdict(Counter)  # quote type -> how often each of OUTCOMES
        # (book key, quote) of each quote a fresh quote of the same requester and type replaced: its promise stands
        self.replaced: List[Tuple[Tuple[str, str, str], TrackedQuote]] = []

    def answer(self, message: Message, members: List[str], engine: Callable[[str], SimulationEngine],
               now: datetime, warn: Callable[[str], None]) -> Message:
        """Asks the members, and if their plan is ready by the limit, the quote's time plus within (or there is no
        limit), reserves its lines at each member and returns the accepted reply; else the declined reply. A quote
        without lines is accepted at once, and a second one while the requester's quote of its type is open is
        declined at once, each with a warning; once some of the first has moved on, the second replaces it."""
        quote, key = _replied(message), _key(message)
        if key in self.quotes and self._still_open(self.quotes[key], engine, now):
            warn(f"{quote.request} already has an open {quote.quote_type}; use modify")
            self.outcomes[quote.quote_type]["declined_as_second"] += 1
            return _declined(quote, None)
        limit = None if quote.limit is None else message.time + quote.limit
        reply = self._quote(message, quote.lines, members, engine, now, limit, warn)
        self.outcomes[quote.quote_type]["accepted" if reply.type == quote.accepted else "declined"] += 1
        return reply

    def modify(self, message: Message, members: List[str], engine: Callable[[str], SimulationEngine],
               now: datetime, warn: Callable[[str], None]) -> Message:
        """Changes the requester's open quote to the modify's lines (see _modify), and counts the reply."""
        reply = self._modify(message, members, engine, now, warn)
        quote = message.quote
        self.outcomes[quote.quote_type]["modified_" + ("accepted" if reply.type == quote.accepted else "declined")] += 1
        return reply

    def _modify(self, message, members, engine, now, warn):
        """Changes the requester's open quote to the modify's lines: lines in both lists, matched by their
        attributes, stay where they are, and the added ones are quoted; within the original limit date, the dropped
        lines are released, the added ones reserved, and the reply is accepted, else nothing changes and it is
        declined. Too late once some line has moved on; after a decline, a fresh quote of the new list; without a
        quote, declined with a warning."""
        quote, key = _replied(message), _key(message)
        tracked = self.quotes.get(key)
        if tracked is None:
            warn(f"{quote.request} has no {quote.quote_type} to modify")
            return _declined(quote, None)
        if tracked.state is QuoteState.DECLINED:
            return self._quote(message, quote.lines, members, engine, now, tracked.limit, warn)
        if tracked.state is QuoteState.OPEN:
            tracked.state = _moved_state(tracked, engine, now)
        if tracked.state is not QuoteState.OPEN:
            return _declined(quote, None)
        kept, added = _matched(tracked, quote.lines)
        plan = plan_lines(added, ask_members(message, members, engine, now)) if added else Plan()
        if plan is None:
            return _declined(quote, None)
        ready_by = max([tracked.lines[line_id].ready for line_id in kept] + [time for _, time in plan.lines.values()],
                       default=message.time)
        if tracked.limit is not None and ready_by > tracked.limit:
            return _declined(quote, ready_by)
        dropped = [line_id for line_id in tracked.lines if line_id not in kept]
        for member, reserving, line_ids in _held(tracked, dropped):
            engine(member).release(reserving, line_ids, now)
            self.released_at.append(member)
        for line_id in dropped:
            del tracked.lines[line_id]
        _reserve(tracked, message, added, plan, members, engine, now)
        return _accepted(quote, tracked, ready_by)

    def cancel(self, message: Message, engine: Callable[[str], SimulationEngine], now: datetime):
        """Releases every line of the requester's open quote at its member, unless some share has moved on: then
        nothing is released, and the quote is complete if every member's share has, else locked, too late. One
        without lines simply leaves the book. A locked quote becomes complete once every share has moved on; a
        declined, complete, or unknown one is left as it is. A cancel gets no reply."""
        key, outcomes = _key(message), self.outcomes[message.quote.quote_type]
        tracked = self.quotes.get(key)
        if tracked is not None and tracked.state in (QuoteState.OPEN, QuoteState.LOCKED):
            tracked.state = _moved_state(tracked, engine, now)  # a share that has moved on stays moved on
        if tracked is None or tracked.state is not QuoteState.OPEN:
            # after a full delivery the cancel isn't too late: there is nothing left to cancel
            if tracked is not None and tracked.state is QuoteState.LOCKED:
                outcomes["cancel_too_late"] += 1
            return
        for member, reserving, line_ids in _held(tracked, tracked.lines):
            engine(member).release(reserving, line_ids, now)
            self.released_at.append(member)
        del self.quotes[key]
        outcomes["canceled"] += 1

    def _quote(self, message, lines, members, engine, now, limit, warn):
        """A fresh quote of lines against limit: tracked open with its reserved lines and accepted, or declined.
        Without lines, it is accepted at once, with a warning."""
        quote = message.quote
        if _key(message) in self.quotes:
            self.replaced.append((_key(message), self.quotes[_key(message)]))
        if not lines:
            warn(f"{message.type} {message.id} from {message.source} has no lines")
            self.quotes[_key(message)] = TrackedQuote(QuoteState.OPEN, limit)
            return _accepted(quote, self.quotes[_key(message)], message.time)
        plan = plan_lines(lines, ask_members(message, members, engine, now))
        if plan is None or limit is not None and plan.ready_by > limit:
            self.quotes[_key(message)] = TrackedQuote(QuoteState.DECLINED, limit)
            return _declined(quote, plan.ready_by if plan else None)
        tracked = self.quotes[_key(message)] = TrackedQuote(QuoteState.OPEN, limit)
        _reserve(tracked, message, lines, plan, members, engine, now)
        return _accepted(quote, tracked, plan.ready_by)

    @staticmethod
    def _still_open(tracked, engine, now):
        """Whether the tracked quote is open and nothing of it has moved on: its members are asked with the
        messages that reserved their lines. One without lines has nothing to deliver."""
        return tracked.state is QuoteState.OPEN and bool(tracked.lines) and \
            _moved_state(tracked, engine, now) is QuoteState.OPEN


def _moved_state(tracked, engine, now):
    """What the open quote is now, by whether its members' shares have moved on, asking every member holding lines
    with the messages that reserved them: complete if all have, locked if some have, else still open."""
    moved = [engine(member).moved(reserving, now) for member, reserving, _ in _held(tracked, tracked.lines)]
    if moved and all(moved):
        return QuoteState.COMPLETE
    return QuoteState.LOCKED if any(moved) else QuoteState.OPEN


def _key(message):
    """A quote's place in the book: the requesting process, its object's id, and the quote type."""
    return message.source, message.quote.request, message.quote.quote_type


def _replied(message):
    """The quote of a quote or modify message, which the orchestrator replies to with its accepted or declined
    type."""
    if message.quote.accepted is None or message.quote.declined is None:
        raise ValueError(f"{message.type} {message.id} from {message.source} asks a quote without the types of its "
                         f"replies: a quote needs accepted and declined")
    return message.quote


def _matched(tracked, lines):
    """(the held line ids the new lines keep, the new lines added): a new line keeps the first held line with the
    same attributes not kept yet."""
    unmatched, kept, added = list(tracked.lines.items()), [], []
    for line in lines:
        found = next((index for index, (_, held) in enumerate(unmatched) if held.line.attributes == line.attributes),
                     None)
        if found is None:
            added.append(line)
        else:
            kept.append(unmatched.pop(found)[0])
    return kept, added


def _reserve(tracked, message, lines, plan, members, engine, now):
    """Reserves the plan's lines at each member with message, in the order the members are asked, and tracks them."""
    line_of = {line.id: line for line in lines}
    for member in members:
        line_ids = [line_id for line_id, (holder, _) in plan.lines.items() if holder == member]
        if line_ids:
            engine(member).reserve(message, line_ids, now)
            tracked.lines.update((line_id, HeldLine(line_of[line_id], member, message, plan.lines[line_id][1]))
                                 for line_id in line_ids)


def _accepted(quote, tracked, ready_by):
    tracked.ready_by = ready_by
    # deliveries: the members holding lines, each delivering its share at once
    return Message(quote.accepted, {"request": quote.request, "ready_by": ready_by,
                                    "deliveries": len({held.member for held in tracked.lines.values()})})


def _declined(quote, earliest):
    return Message(quote.declined, {"request": quote.request, "earliest": earliest})


def _held(tracked, line_ids):
    """(member, the message that reserved them there, the line ids) of each member and message holding some of
    line_ids, in the order of the lines."""
    held = {}
    for line_id in line_ids:
        line = tracked.lines[line_id]
        held.setdefault((line.member, id(line.reserving)), (line.member, line.reserving, []))[2].append(line_id)
    return list(held.values())


def broken_promises(book: QuoteBook, moved: Dict[Tuple[str, str], datetime], end: Optional[datetime]):
    """(quote type -> how late each member's share of its quotes, also those later ones replaced, moved on after the
    quote's ready_by, quote type -> how long past it each share not moved on by end, the end of the run, is). moved:
    (member, the id of a message that reserved lines there) -> when the share holding them moved on."""
    late, overdue = defaultdict(list), defaultdict(list)
    for (_, _, quote_type), tracked in [*book.replaced, *book.quotes.items()]:
        if tracked.ready_by is None:
            continue
        shares = {}  # member -> when its share moved on: its lines from the quote and from modifies move together
        held = _held(tracked, tracked.lines)
        for member, reserving, _ in held:
            if (member, reserving.id) in moved:
                shares[member] = moved[(member, reserving.id)]
        late[quote_type].extend(at - tracked.ready_by for at in shares.values() if at > tracked.ready_by)
        if end is not None and tracked.ready_by < end:  # one promised after the end may still be kept
            overdue[quote_type].extend(end - tracked.ready_by for member in {member for member, _, _ in held}
                                       if member not in shares)
    return dict(late), dict(overdue)


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
