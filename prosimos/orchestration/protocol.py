"""
The contract between the engines and the orchestrator (docs/orchestrator.md, "Engine interface"). It depends on no
particular simulator.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


ORCHESTRATOR = "orchestrator"  # the source of the replies to quotes, so no process may have this name


@dataclass
class Line:
    """One line of a quote, matched by a member's consume condition for the quote type like a message of that type.
    The requester sets attributes; the orchestrator sets id when it stamps the quote."""

    attributes: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None


class QuoteKind(Enum):
    """What a message with a Quote asks for."""

    QUOTE = "quote"
    MODIFY = "modify"
    CANCEL = "cancel"


@dataclass
class Quote:
    """What a quote, modify, or cancel message asks of the members that supply its lines. request: the requester's
    object_id; limit: the quote's within, None for no limit; accepted and declined: the types of the replies the
    orchestrator publishes to a quote or modify."""

    kind: QuoteKind
    quote_type: str
    request: str
    lines: List[Line] = field(default_factory=list)
    limit: Optional[timedelta] = None
    accepted: Optional[str] = None
    declined: Optional[str] = None


@dataclass
class Message:
    """One object passed between processes. The publishing process sets type and attributes, and a quote's
    requester its quote; the orchestrator sets id, source and time when it stamps the message."""

    type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    source: Optional[str] = None
    time: Optional[datetime] = None
    quote: Optional[Quote] = None


# a member's answer to a quote: line id -> the times its matching items not yet promised would be ready, earliest first
Slots = Dict[str, List[datetime]]


class Verdict(Enum):
    """An engine's answer to one offered message."""

    CLAIMED = "claimed"
    DISCARDED = "discarded"
    PENDING = "pending"


@dataclass
class StalledCase:
    """A case still waiting for a message when the run ended."""

    object_id: str
    event_id: str
    message_types: List[str]  # the types it waits for: any one of them resumes it (several in a race)
    waiting_since: datetime
    collected: int = 0  # messages it had claimed at that event...
    needed: int = 1  # ...of the ones it needed (more than 1 with collect)


@dataclass(frozen=True)
class PublishRecord:
    """An engine's record that one of its cases published a message at an element. task_rows: the rows of its log
    the message's OCEL links attach to, the last task before the element (through a parallel or inclusive join,
    one per branch)."""

    message_id: str
    object_id: str
    element_id: str
    qualifier: str = ""
    task_rows: Tuple[int, ...] = ()
    o2o: Optional[str] = None  # qualifier of the OCEL link from its case's object to the other end's, if any
    # the later events of its case that also link the other end's object: True for all, or the activities named
    carry: Union[bool, Tuple[str, ...]] = False


@dataclass(frozen=True)
class ClaimRecord:
    """An engine's record that one of its cases took a message at an element. task_rows: the rows of its log the
    message's OCEL links attach to, the next task after it (after a parallel or inclusive split, the first run of
    every task on the branches taken)."""

    message_id: str
    object_id: str
    element_id: str
    qualifier: str = ""
    task_rows: Tuple[int, ...] = ()
    o2o: Optional[str] = None  # qualifier of the OCEL link from its case's object to the other end's, if any
    # the later events of its case that also link the other end's object: True for all, or the activities named
    carry: Union[bool, Tuple[str, ...]] = False


@dataclass
class CaseObject:
    """One case of an engine as an OCEL object, with its attributes over time, (attribute, value, time), from its
    creation on; the OCEL output leaves some out (see ocel_writer._written)."""

    object_id: str
    created: datetime
    attributes: List[Tuple[str, Any, datetime]] = field(default_factory=list)
    # the attributes whose value at creation is only their start, which no message wrote yet
    only_started: Tuple[str, ...] = ()


@dataclass
class EngineReport:
    """What one engine has to report when the run is over."""

    stalled: List[StalledCase] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)  # raised inside this engine during the run
    # which of its cases and elements published and took each message
    published: List[PublishRecord] = field(default_factory=list)
    claimed: List[ClaimRecord] = field(default_factory=list)
    objects: List[CaseObject] = field(default_factory=list)  # every case it created
    logged_elements: List[str] = field(default_factory=list)  # the element id of each row of its log, in order
    # the id of each quote or modify message that reserved lines here -> when the share holding them moved on
    moved: Dict[str, datetime] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseElement:
    """One element of one case of one process; for a message, also how its OCEL links attach, as in its record."""

    process: str
    object_id: str
    element_id: str
    qualifier: str = ""
    task_rows: Tuple[int, ...] = ()
    o2o: Optional[str] = None  # qualifier of the OCEL link from its case's object to the other end's, if any
    # the later events of its case that also link the other end's object: True for all, or the activities named
    carry: Union[bool, Tuple[str, ...]] = False


@dataclass
class MessageRecord:
    """Who published a message and who took it, joined after the run: several publishers with fan-in, several
    claimers with capacity, none if no case took it, and no publishers if its engine keeps no records."""

    message_id: str
    type: str
    publishers: List[CaseElement] = field(default_factory=list)
    claimers: List[CaseElement] = field(default_factory=list)


class SimulationEngine(ABC):
    """The only methods the orchestrator may call on an engine; engines never call the orchestrator."""

    @abstractmethod
    def subscriptions(self) -> List[str]:
        """The message types this process consumes, from its model configuration. Called once, at setup."""

    @abstractmethod
    def next_event_time(self) -> Optional[datetime]:
        """When your next event is due; None when nothing is (cases waiting for a message don't count)."""

    @abstractmethod
    def step(self) -> List[Message]:
        """Perform exactly one event and return the messages it published; the only way to publish."""

    @abstractmethod
    def deliver(self, message: Message, now: datetime) -> Verdict:
        """Offers one pending message at now. CLAIMED binds it to one case at now; this copy is never offered again.
        DISCARDED: the engine will never want it, whatever happens. PENDING: not now; an engine that isn't sure
        must answer PENDING."""

    @abstractmethod
    def estimate(self, message: Message, now: datetime) -> Slots:
        """When this member could have each line of the stamped quote message (its quote in message.quote) ready at
        now; changes nothing."""

    @abstractmethod
    def moved(self, message: Message, now: datetime) -> bool:
        """Whether some line the stamped quote message reserved here has already moved on at now; changes
        nothing."""

    @abstractmethod
    def reserve(self, message: Message, line_ids: List[str], now: datetime) -> None:
        """Books the lines line_ids of the stamped quote message at now, for its requester."""

    @abstractmethod
    def release(self, message: Message, line_ids: List[str], now: datetime) -> None:
        """Frees the lines line_ids of the stamped quote message at now, which this member booked."""

    @abstractmethod
    def finish(self) -> EngineReport:
        """Called once, after the loop; returns the engine's EngineReport. Nothing is published or delivered."""
