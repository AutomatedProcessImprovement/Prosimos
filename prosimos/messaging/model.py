"""A process's 'messages' section once read (docs/messaging.md): the parser builds it, the engine acts on it."""
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

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
    """A 'publish' entry: the messages a case publishes when it passes event_id."""

    event_id: str
    type: str
    attributes: Tuple[str, ...]  # from the case's values; the engine fills in 'object_id' and 'index'
    fan_out: Optional[int] = None
    fan_out_attribute: Optional[str] = None  # the case attribute holding the number
    fan_out_list: Optional[str] = None  # a case attribute holding a list: one message per entry, carried as fan_out_as
    fan_out_as: Optional[str] = None
    qualifier: Optional[str] = None  # of its messages' OCEL links; None: no links
    o2o: Optional[str] = None  # OCEL link from the publishing case's object to each claimer's
    # the later events of the case that also link the claimers' objects: True for all, or the activities named
    carry: Union[bool, Tuple[str, ...]] = False

    @property
    def fans_out(self):
        """Whether the entry has a fan_out (else it publishes one message, without an index)."""
        return self.fan_out is not None or self.fan_out_attribute is not None or self.fan_out_list is not None


@dataclass(frozen=True)
class ConsumePoint:
    """A 'consume' entry: the messages a case waiting at event_id takes, and what it does with them."""

    event_id: str
    type: str
    condition: Optional[Tuple[Tuple[ConditionTerm, ...], ...]]  # alternatives of terms; None takes every message
    copy: Tuple[Tuple[str, str], ...] = ()  # (case attribute, message attribute)
    starts_case: bool = False
    capacity: float = 1  # max cases one message resumes (math.inf: no limit)
    capacity_attribute: Optional[str] = None  # the message attribute holding that max
    collect: int = 1  # min messages a case claims here before it goes on
    collect_attribute: Optional[str] = None  # the case attribute holding that min
    qualifier: Optional[str] = None  # of its messages' OCEL links; None: no links
    o2o: Optional[str] = None  # OCEL link from the claiming case's object to the publisher's
    append: Tuple[Tuple[str, str], ...] = ()  # (case attribute, message attribute): added to a list
    # (case attribute, function, message attribute, initial): over every message claimed into that case attribute
    aggregate: Tuple[Tuple[str, str, str, Optional[float]], ...] = ()
    collect_max: Optional[float] = None  # max messages a case claims here; None: the same as min (math.inf: no limit)
    collect_max_attribute: Optional[str] = None
    same: Tuple[str, ...] = ()  # message attributes every message a case claims here has the same values of
    capacity_min: int = 1  # min cases waiting before one message resumes them
    capacity_min_attribute: Optional[str] = None
    # the later events of the case that also link the publisher's object: True for all, or the activities named
    carry: Union[bool, Tuple[str, ...]] = False


@dataclass(frozen=True)
class MessagingModel:
    """A process's 'publish' and 'consume' entries."""

    publish: Tuple[PublishPoint, ...] = ()
    consume: Tuple[ConsumePoint, ...] = ()

    def subscriptions(self) -> List[str]:
        return sorted({point.type for point in self.consume})

    @property
    def started_by_messages(self) -> bool:
        """Whether the process's cases are started by messages instead of an arrival schedule."""
        return any(point.starts_case for point in self.consume)
