"""
The contract between the engines and the orchestrator (docs/orchestrator.md, "Engine interface"): what a message
is, what an engine answers when it is offered one, what it records for the output, and the interface an engine
implements. It doesn't depend on Prosimos: another engine can implement it.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Message:
    """One object passed between processes. The publishing process sets type and attributes;
    the orchestrator sets id, source and time when it stamps the message."""

    type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    source: Optional[str] = None
    time: Optional[datetime] = None


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
    """An engine's record that one of its cases published a message, at one of its elements. task_rows are the
    rows of its log (0 for the first) that the message's OCEL links attach to: the last task before the
    element, and through a parallel or inclusive join the last task of every branch that came in; empty if
    there is none. qualifier qualifies those links."""

    message_id: str
    object_id: str
    element_id: str
    qualifier: str = ""
    task_rows: Tuple[int, ...] = ()
    o2o: Optional[str] = None  # qualifier of the OCEL link from its case's object to the other end's, if any


@dataclass(frozen=True)
class ClaimRecord:
    """An engine's record that one of its cases took a message, at one of its elements. task_rows are the rows
    of its log that the message's OCEL links attach to: the next task after the element, and after a
    parallel or inclusive split the first run of every task on the branches taken; empty if there is none. qualifier
    qualifies those links."""

    message_id: str
    object_id: str
    element_id: str
    qualifier: str = ""
    task_rows: Tuple[int, ...] = ()
    o2o: Optional[str] = None  # qualifier of the OCEL link from its case's object to the other end's, if any


@dataclass
class CaseObject:
    """One case of an engine, as an object of the OCEL output: its id in the engine, when it was created, and its
    case attributes over time, as (attribute, value, time): the values it had when it was created, then every
    later change. All of them are kept here; the OCEL output leaves some out (see ocel_writer._written)."""

    object_id: str
    created: datetime
    attributes: List[Tuple[str, Any, datetime]] = field(default_factory=list)
    # its sums and counts whose value at creation is only their starting 0, no message having written them yet
    only_started: Tuple[str, ...] = ()


@dataclass
class EngineReport:
    """What one engine has to report when the run is over."""

    stalled: List[StalledCase] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)  # raised inside this engine during the run
    # which of its cases and elements published and took each message, kept by the engine during the run
    published: List[PublishRecord] = field(default_factory=list)
    claimed: List[ClaimRecord] = field(default_factory=list)
    objects: List[CaseObject] = field(default_factory=list)  # every case it created
    logged_elements: List[str] = field(default_factory=list)  # the element id of each row of its log, in order


@dataclass(frozen=True)
class CaseElement:
    """One element of one case of one process; for a message, also the qualifier of its OCEL links and the
    rows of the process's log they attach to (none if there is no such task)."""

    process: str
    object_id: str
    element_id: str
    qualifier: str = ""
    task_rows: Tuple[int, ...] = ()
    o2o: Optional[str] = None  # qualifier of the OCEL link from its case's object to the other end's, if any


@dataclass
class MessageRecord:
    """Who published a message and who took it, joined from the engines' records after the run. No
    claimers for a message that was discarded or never claimed; several with capacity; no publisher
    if the publishing engine keeps no records."""

    message_id: str
    type: str
    publisher: Optional[CaseElement]
    claimers: List[CaseElement] = field(default_factory=list)
