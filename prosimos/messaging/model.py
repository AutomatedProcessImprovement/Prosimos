"""
What a process's 'messages' section says once read (docs/messaging.md): where it publishes and what, where it waits
and for which messages, and what it does with those it takes. The parser builds these from the JSON settings; the
engine acts on them.
"""
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

COMPARISONS = ("=", "!=", "<", "<=", ">", ">=", "in")  # the ones branch rules understand
AGGREGATE_FUNCTIONS = {"sum", "count", "min", "max"}


@dataclass(frozen=True)
class ConditionTerm:
    """Compares a message attribute (or 'source') with a fixed value or with a case attribute."""

    attribute: str
    comparison: str
    value: Any = None
    object_attribute: Optional[str] = None


@dataclass(frozen=True)
class PublishPoint:
    event_id: str
    type: str
    attributes: Tuple[str, ...]  # copied from the case's current values; 'object_id' is the object's id,
    # 'index' the message's number among the ones count publishes
    count: Optional[int] = None  # how many messages are published at once, unless read from the case
    count_attribute: Optional[str] = None  # the case attribute holding that number, if any
    count_list: Optional[str] = None  # the case attribute holding a list: one message per entry, if any
    count_as: Optional[str] = None  # the attribute under which each of those messages carries its entry
    qualifier: Optional[str] = None  # of the OCEL links its messages give; None: no links
    o2o: Optional[str] = None  # OCEL object link from the publishing case's object to each claimer's; None: none

    @property
    def counted(self):
        """Whether the entry has a count (else it publishes one message, without an index)."""
        return self.count is not None or self.count_attribute is not None or self.count_list is not None


@dataclass(frozen=True)
class ConsumePoint:
    event_id: str
    type: str
    condition: Optional[Tuple[Tuple[ConditionTerm, ...], ...]]  # any alternative whose terms all hold; None accepts every message
    copy: Tuple[Tuple[str, str], ...] = ()  # (case attribute, message attribute): copied into the case
    starts_case: bool = False  # at the start event: an accepted message starts a new case
    # max: how many waiting cases one message resumes at most (math.inf: no limit), unless read from the message
    capacity: float = 1
    capacity_attribute: Optional[str] = None  # the message attribute holding that max, if any
    collect: int = 1  # min: how many messages a case must claim here before it continues, unless read from the case
    collect_attribute: Optional[str] = None  # the case attribute holding that number, if any
    qualifier: Optional[str] = None  # of the OCEL links the messages it takes give; None: no links
    o2o: Optional[str] = None  # OCEL object link from the claiming case's object to the publisher's; None: none
    # (case attribute, message attribute): the message attribute's value is added to a list in the case, which
    # starts empty
    append: Tuple[Tuple[str, str], ...] = ()
    # (case attribute, function, message attribute, initial): the function over the message attribute's values of
    # every message claimed, at this entry and any other aggregating into the same case attribute, and over initial,
    # a number already among them, if not None
    aggregate: Tuple[Tuple[str, str, str, Optional[float]], ...] = ()
    # max: how many messages a case may claim here in all; once it has min, it also takes every other matching
    # message offered at that instant, up to max. None means the same as min, math.inf no limit; unless read from
    # the case attribute collect_max_attribute
    collect_max: Optional[float] = None
    collect_max_attribute: Optional[str] = None
    # message attributes the first message a case claims here sets; every later one must have the same values
    same: Tuple[str, ...] = ()
    # min: how many cases must be waiting for one message to resume them; with fewer, it waits. Unless read from the
    # message attribute capacity_min_attribute
    capacity_min: int = 1
    capacity_min_attribute: Optional[str] = None


@dataclass(frozen=True)
class MessagingModel:
    publish: Tuple[PublishPoint, ...] = ()
    consume: Tuple[ConsumePoint, ...] = ()

    def subscriptions(self) -> List[str]:
        return sorted({point.type for point in self.consume})

    @property
    def started_by_messages(self) -> bool:
        """Whether the process's cases are started by messages instead of an arrival schedule."""
        return any(point.starts_case for point in self.consume)
