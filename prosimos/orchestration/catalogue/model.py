"""The catalogue's configuration types: catalogues, references and catalogue changes (docs/catalogue.md)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass(frozen=True)
class CatalogueObject:
    """An object with no events of its own (a customer, a product), in an organization's catalogue."""

    id: str
    attributes: Tuple[Tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class CataloguePool:
    """A generated catalogue: count objects, <name>-1 to <name>-<count>, their attributes drawn at the start."""

    count: int
    attributes: Tuple[Dict[str, Any], ...] = ()


@dataclass(frozen=True)
class GrowingCatalogue:
    """A catalogue that starts empty; a "new" or new_share pick creates its objects, <name>-1, <name>-2, ..."""

    attributes: Tuple[Dict[str, Any], ...] = ()


@dataclass(frozen=True)
class NewShare:
    """A pick on a growing catalogue: share of the cases create an object, the rest reuse one created so far."""

    share: float


@dataclass(frozen=True)
class Computed:
    """A 'with' value: function over the attribute source of the objects, and initial, as aggregate does."""

    function: str
    source: str
    initial: Optional[float] = None


@dataclass(frozen=True)
class QuantityFrom:
    """A pick's quantity read from a case attribute when the case is created."""

    object_attribute: str


@dataclass(frozen=True)
class CatalogueChange:
    """A change of a catalogue attribute on the clock, from catalogue_changes (docs/catalogue.md, "Changes");
    objects None changes every object of the catalogue."""

    catalogue: str
    attribute: str
    operation: str
    value: Any
    every: Optional[timedelta] = None
    at: Optional[datetime] = None
    round: Optional[int] = None
    objects: Optional[Tuple[str, ...]] = None

    def applied(self, value):
        """The attribute's value after this change, value before it."""
        if self.operation == "multiply":
            value = value * self.value
        elif self.operation == "add":
            value = value + self.value
        else:
            value = self.value
        return value if self.round is None else round(value, self.round)


@dataclass(frozen=True)
class Reference:
    """Says that a case attribute holds ids of a catalogue's objects, which OCEL links to write for them, and with
    pick, how to choose them (docs/catalogue.md, "References"). link_events None means every event, () none."""

    object_attribute: str
    catalogue: str
    o2o: Optional[str] = None
    reverse: bool = False
    link_events: Optional[Tuple[str, ...]] = None
    pick: Union[None, str, Tuple[Tuple[str, float], ...], NewShare] = None
    with_attributes: Tuple[Tuple[str, Union[str, Computed]], ...] = ()  # JSON "with"
    tag_along: bool = False
    quantity: Union[None, int, QuantityFrom] = None
    process: str = ""  # set by its ProcessSpec

    @property
    def creates(self) -> bool:
        return self.pick == "new" or isinstance(self.pick, NewShare)


def catalogue_ids(name, entry) -> List[str]:
    """The ids known when the run starts: none yet for a growing catalogue."""
    if isinstance(entry, GrowingCatalogue):
        return []
    if isinstance(entry, CataloguePool):
        return [f"{name}-{number}" for number in range(1, entry.count + 1)] if isinstance(entry.count, int) else []
    return [obj.id for obj in entry]
