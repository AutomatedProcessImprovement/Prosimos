"""Picks, the values 'with' brings, and the catalogues' objects as the run starts and ends (docs/catalogue.md)."""
from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

from prosimos.case_attributes import CaseAttribute
from prosimos.messaging.engine import aggregate_of
from prosimos.messaging.parser import is_whole_number
from prosimos.orchestration.catalogue.model import CatalogueObject, Computed, CountFrom, NewShare, catalogue_ids
from prosimos.orchestration.catalogue.timetable import CatalogueValues, Timetable
from prosimos.orchestration.random_generators import engine_random_states
from prosimos.warning_logger import warning_logger

if TYPE_CHECKING:
    from prosimos.orchestration.config import Organization, ProcessSpec


class Pick:
    """A case attribute a reference picks from its catalogue's ids, drawn like Prosimos's CaseAttribute. Its
    generator is its own (seed, process, attribute), so it doesn't change the engine's other draws."""

    def __init__(self, name: str, ids: List[str], weights: Optional[List[float]], seed: Optional[int], process: str,
                 count: Union[None, int, CountFrom] = None):
        self.name, self._ids, self._weights, self.count = name, ids, weights, count
        self.last = None  # the id picked for the latest case
        self._counted = None  # the value of count's attribute drawn for the latest case
        self._warned = False
        self._generator = random.Random()
        self._generator.setstate(engine_random_states(seed, f"pick/{process}/{name}")[0])

    def read_count_from(self, attribute):
        """Makes the count's attribute, drawn before the picks, tell this pick its value for each case."""
        draw = attribute.get_next_value

        def drawn():
            self._counted = draw()
            return self._counted

        attribute.get_next_value = drawn

    def get_next_value(self):
        if self.count is None:
            self.last = self._generator.choices(self._ids, self._weights)[0]
        else:
            self.last = self._generator.choices(self._ids, self._weights, k=self._how_many())
        return self.last

    def _how_many(self):
        """count, or the value its attribute drew for this case; an invalid one picks one, with one warning."""
        if not isinstance(self.count, CountFrom):
            return int(self.count)
        if is_whole_number(self._counted, minimum=0):
            return int(self._counted)
        if not self._warned:
            self._warned = True
            warning_logger.add_warning(f"{self.count.object_attribute} is {self._counted!r}, no whole number of at "
                                       f"least 0; the pick of {self.name} picks one")
        return 1


class Brought:
    """A value a Pick's objects give the case: an attribute of the picked object, or a function over the picked
    objects. It draws nothing: it reads its pick's last value, from the values before any change, since a planned
    case's attributes are drawn before its arrival time is known; the engine works it out again at creation."""

    def __init__(self, name: str, pick: Pick, values: Dict[str, Any], function: Optional[str] = None,
                 initial: Optional[float] = None, catalogue: str = ""):
        self.name, self._pick, self.values, self.function, self.initial = name, pick, values, function, initial
        self.catalogue = catalogue

    @property
    def source(self) -> str:
        """The attribute whose objects it reads: its pick's."""
        return self._pick.name

    def get_next_value(self):
        if self.function is None:
            return self.values[self._pick.last]
        picked = self._pick.last if isinstance(self._pick.last, list) else [self._pick.last]
        if not picked and self.function in ("min", "max") and self.initial is None:
            return None
        return aggregate_of(self.function, [self.values[object_id] for object_id in picked], self.initial)


class Received:
    """A value 'with' gives the case from the object an attribute names without a pick. It draws nothing: the
    engine fills it in (see EngineMessaging._apply_received)."""

    def __init__(self, name: str, source: str, values: Dict[str, Any], function: Optional[str] = None,
                 initial: Optional[float] = None, catalogue: str = ""):
        self.name, self.source, self.values, self.function, self.initial = name, source, values, function, initial
        self.catalogue = catalogue

    def get_next_value(self):
        return None


class NewPick:
    """A case attribute picked from a growing catalogue: a new id for every case, or for share of them, the rest
    reusing one created so far; like Pick, from a generator of its own."""

    def __init__(self, name: str, catalogue: str, share: float, seed: Optional[int], process: str):
        self.name, self.catalogue, self._share = name, catalogue, share
        self.created = 0
        self._generator = random.Random()
        self._generator.setstate(engine_random_states(seed, f"pick/{process}/{name}")[0])

    def get_next_value(self):
        if self.created == 0 or self._share == 1 or self._generator.random() < self._share:
            self.created += 1
            return f"{self.catalogue}-{self.created}"
        return f"{self.catalogue}-{self._generator.randint(1, self.created)}"


def picks_of(spec: ProcessSpec, seed: Optional[int], catalogues: Dict[str, List[CatalogueObject]],
             timetable: Timetable) -> List[Pick]:
    """The attributes a process's references pick, and those their 'with' brings, as timetable changes them."""
    def values(reference, attribute):
        return CatalogueValues({obj.id: dict(obj.attributes)[attribute] for obj in catalogues[reference.catalogue]},
                                reference.catalogue, attribute, timetable)

    picks = []
    for reference in spec.references:
        if reference.creates:
            share = reference.pick.share if isinstance(reference.pick, NewShare) else 1
            picks.append(NewPick(reference.object_attribute, reference.catalogue, share, seed, spec.name))
        elif reference.pick is None:
            for name, source in reference.with_attributes:
                function, attribute, initial = ((source.function, source.source, source.initial)
                                                if isinstance(source, Computed) else (None, source, None))
                picks.append(Received(name, reference.object_attribute, values(reference, attribute), function,
                                      initial, reference.catalogue))
        else:
            ids = [obj.id for obj in catalogues[reference.catalogue]]
            weights = None if reference.pick == "uniform" else [dict(reference.pick).get(object_id, 0) for object_id in ids]
            pick = Pick(reference.object_attribute, ids, weights, seed, spec.name, reference.count)
            picks.append(pick)
            for name, source in reference.with_attributes:
                function, attribute, initial = ((source.function, source.source, source.initial)
                                                if isinstance(source, Computed) else (None, source, None))
                picks.append(Brought(name, pick, values(reference, attribute), function, initial,
                                     reference.catalogue))
    return picks


def drawn_catalogues(organization: Organization, seed: Optional[int]) -> Dict[str, List[CatalogueObject]]:
    """An organization's catalogues, name -> their objects as the run starts; none yet for a growing one."""
    return {name: list(entry) if isinstance(entry, list) else _drawn_objects(seed, name, entry.attributes,
                                                                            catalogue_ids(name, entry))
            for name, entry in organization.catalogues.items()}


def grown_catalogues(organization: Organization, seed: Optional[int], catalogues: Dict[str, List[CatalogueObject]],
                     picks: List[Pick]) -> Dict[str, List[CatalogueObject]]:
    """An organization's catalogues after the run, with the objects each growing one's pick created. Drawn now,
    they get the values they'd have got when created, since nothing else draws from their generators."""
    grown = dict(catalogues)
    for pick in picks:
        if isinstance(pick, NewPick):
            attribute_specs = organization.catalogues[pick.catalogue].attributes
            grown[pick.catalogue] = _drawn_objects(seed, pick.catalogue, attribute_specs,
                                                   [f"{pick.catalogue}-{number}" for number in range(1, pick.created + 1)])
    return grown


def _drawn_objects(seed, name, attribute_specs, ids) -> List[CatalogueObject]:
    """The objects with these ids, their attributes drawn from generators seeded from the seed and the catalogue."""
    attributes = [CaseAttribute(attribute["name"], attribute["type"], attribute["values"]) for attribute in attribute_specs]
    python_state, numpy_state = engine_random_states(seed, f"catalogue/{name}")
    shared_python, shared_numpy = random.getstate(), np.random.get_state()
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    try:
        return [CatalogueObject(object_id, tuple((attribute.name, _plain_value(attribute.get_next_value()))
                                                 for attribute in attributes))
                for object_id in ids]
    finally:
        random.setstate(shared_python)
        np.random.set_state(shared_numpy)


def _plain_value(value):
    return value.item() if hasattr(value, "item") else value  # a NumPy number, from a continuous attribute
