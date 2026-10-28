"""
Catalogues and picks (docs/catalogue.md): objects with no events of their own, a process's references to them, and
the picks that assign them to its cases.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass(frozen=True)
class CatalogueObject:
    """An object with no events of its own (a customer, a product), declared in a catalogue of a process."""

    id: str
    attributes: Tuple[Tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class CataloguePool:
    """A generated catalogue, for sets too large to list: count objects with ids <catalogue name>-1 to
    <catalogue name>-<count>, each with attributes drawn once at the start of the run, given in the format
    of object_attributes (discrete or continuous)."""

    count: int
    attributes: Tuple[Dict[str, Any], ...] = ()


@dataclass(frozen=True)
class GrowingCatalogue:
    """A catalogue with no list and no count, for populations that aren't fixed: it starts empty, and a reference
    that picks "new" or new_share creates its objects as cases are created, with ids <catalogue name>-1,
    <catalogue name>-2, ... in creation order, each with attributes in the format of object_attributes."""

    attributes: Tuple[Dict[str, Any], ...] = ()


@dataclass(frozen=True)
class NewShare:
    """A pick on a growing catalogue: share of the cases create a new object, the rest reuse one created so far,
    uniformly; the first case always creates one."""

    share: float


@dataclass(frozen=True)
class Computed:
    """A with value computed over the objects a reference picked, as aggregate computes over messages: function
    ("sum", "count", "min" or "max") of their attribute source, and of initial, if not None, a number already
    among them."""

    function: str
    source: str
    initial: Optional[float] = None


@dataclass(frozen=True)
class CountFrom:
    """A pick's count read from a case attribute when the case is created."""

    object_attribute: str


@dataclass(frozen=True)
class Reference:
    """Says that a case attribute's value of its process is the id of an object of one of its catalogues, or a list
    of such ids (each one linked as a single value is), and which OCEL links to write for it: from the case's object
    to it with o2o as qualifier (reverse turns it around), and from the case's events named in link_events (task
    names or ids; None for every event, () for none), and, with tag_along, from every event of another case that
    links the case's object.
    With pick, the reference also assigns the value, when the case is created: "uniform" (every object
    equally likely) or, for a listed catalogue, weights as ((id, weight), ...) (ids left out are never picked);
    for a growing catalogue, "new" (every case creates its own object) or NewShare. Without it, the value comes
    from object_attributes or a message. with_attributes (JSON "with") gives the case starting values from the
    picked objects, as ((attribute, source), ...): source is an attribute of the picked object, copied, or Computed
    over the picked objects (a single pick is a list of one). count, with "uniform" or weights, makes the pick
    choose that many objects, each independently, into a list: a whole number, or CountFrom, an attribute the case
    draws in its object_attributes. process is the process whose entry lists it, set by its ProcessSpec."""

    object_attribute: str
    catalogue: str
    o2o: Optional[str] = None
    reverse: bool = False
    link_events: Optional[Tuple[str, ...]] = None
    pick: Union[None, str, Tuple[Tuple[str, float], ...], NewShare] = None
    with_attributes: Tuple[Tuple[str, Union[str, Computed]], ...] = ()
    tag_along: bool = False
    count: Union[None, int, CountFrom] = None
    process: str = ""

    @property
    def creates(self) -> bool:
        return self.pick == "new" or isinstance(self.pick, NewShare)


def _catalogue_ids(name, entry) -> List[str]:
    """The ids known when the run starts: none yet for a growing catalogue."""
    if isinstance(entry, GrowingCatalogue):
        return []
    if isinstance(entry, CataloguePool):
        return [f"{name}-{number}" for number in range(1, entry.count + 1)] if isinstance(entry.count, int) else []
    return [obj.id for obj in entry]
