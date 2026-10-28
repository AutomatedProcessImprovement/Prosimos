"""
Catalogues and picks (docs/catalogue.md): objects with no events of their own, a process's references to them, and
the picks that assign them to its cases.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from prosimos.case_attributes import CaseAttribute
from prosimos.control_flow_manager import BPMN
from prosimos.messaging_parser import AGGREGATE_FUNCTIONS, is_number, is_whole_number
from prosimos.orchestration.random_states import _engine_random_states
from prosimos.simulation_engine import _aggregated, _is_number
from prosimos.simulation_properties_parser import parse_simulation_model
from prosimos.warning_logger import warning_logger

if TYPE_CHECKING:
    from prosimos.orchestrator import ProcessSpec


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


def _checked_catalogue(processes):
    """Catalogue names unique across the processes and unlike their object types, since both are OCEL object types;
    ids unique across all catalogues and unlike the processes' object ids (<object type>-<number>), since both are
    OCEL object ids; pools with a count of at least 1 and attributes in the format of object_attributes; and
    references only naming catalogues of their own process and, in link_events, tasks of it (by name or id). A
    pick is "uniform" or weights on known ids of a listed catalogue, "uniform" on a pool, or "new" or new_share on
    a growing catalogue, for an attribute that the process's object_attributes don't have and its consume entries
    don't copy, append or aggregate into, and at most one per process and attribute. A growing catalogue's objects are created by exactly one reference, and no other id looks like
    them (<catalogue name>-<number>). with gives attributes that the process doesn't otherwise pick, draw, copy,
    append or aggregate into, and not object_id, from attributes that every object of the picked listed catalogue
    or pool has, as numbers for a sum, min or max; a pick with count only computes them."""
    problems = []
    owners = defaultdict(list)  # catalogue name -> the processes declaring it
    for spec in processes:
        for name in spec.catalogues:
            owners[name].append(spec.name)
    for name, declaring in owners.items():
        if len(declaring) > 1:
            problems.append(f"catalogue {name!r} is declared by more than one process: {', '.join(declaring)}")
    catalogue = {name: entry for spec in processes for name, entry in spec.catalogues.items()}
    # (where it is in the configuration, reference)
    references = [(f"processes[{index}].references[{number}]", reference)
                  for index, spec in enumerate(processes) for number, reference in enumerate(spec.references)]
    for name, entry in catalogue.items():
        if not isinstance(entry, (CataloguePool, GrowingCatalogue)):
            continue
        if isinstance(entry, CataloguePool) and (not isinstance(entry.count, int) or isinstance(entry.count, bool)
                                                 or entry.count < 1):
            problems.append(f"catalogue {name!r}: count must be a whole number of at least 1, got {entry.count!r}")
        for number, attribute in enumerate(entry.attributes):
            try:
                CaseAttribute(attribute["name"], attribute["type"], attribute["values"])
            except Exception as error:  # Prosimos's own messages, as for object_attributes
                problems.append(f"catalogue {name!r}: attributes[{number}]: {type(error).__name__}: {error}")
    ids = [object_id for name, entry in catalogue.items() for object_id in _catalogue_ids(name, entry)]
    repeated = sorted(object_id for object_id, times in Counter(ids).items() if times > 1)
    if repeated:
        problems.append(f"catalogue ids are repeated: {repeated}")
    object_types = {spec.object_type: spec.name for spec in processes if spec.object_type is not None}
    object_id = re.compile("|".join(rf"{re.escape(object_type)}-\d+" for object_type in object_types))
    like_object_ids = sorted({obj for obj in ids if object_types and object_id.fullmatch(obj)})
    if like_object_ids:
        problems.append(f"catalogue ids look like object ids (<object type>-<number>): {like_object_ids}")
    growing = [name for name, entry in catalogue.items() if isinstance(entry, GrowingCatalogue)]
    for name in growing:
        created_id = re.compile(rf"{re.escape(name)}-\d+")
        like_created = sorted({object_id for object_id in ids if created_id.fullmatch(object_id)})
        if like_created:
            problems.append(f"catalogue ids look like the ids growing catalogue {name!r} creates "
                            f"({name}-<number>): {like_created}")
        creators = [reference for _, reference in references if reference.catalogue == name and reference.creates]
        if len(creators) != 1:
            problems.append(f"growing catalogue {name!r} needs exactly one reference that creates its objects "
                            f'("new" or new_share), got {len(creators)}')
    for name in catalogue:
        if name in object_types:
            problems.append(f"catalogue {name!r} is the object type of process {object_types[name]}")
    specs = {spec.name: spec for spec in processes}
    tasks_of = {}
    for where, reference in references:
        if reference.catalogue not in specs[reference.process].catalogues:
            # what another process knows reaches this one only in messages
            problems.append(f"{where} names catalogue {reference.catalogue!r} of process "
                            f"{', '.join(owners[reference.catalogue])}; a reference names a catalogue of its own process"
                            if reference.catalogue in owners else f"{where} names unknown catalogue {reference.catalogue!r}")
            continue
        if reference.link_events:
            if reference.process not in tasks_of:
                graph = parse_simulation_model(specs[reference.process].bpmn_path)
                tasks_of[reference.process] = {key for info in graph.element_info.values() if info.type == BPMN.TASK
                                               for key in (info.id, info.name)}
            unknown = [task for task in reference.link_events if task not in tasks_of[reference.process]]
            if unknown:
                problems.append(f"{where} link_events names {unknown}, which are no tasks of process {reference.process}")
        if reference.pick is not None:
            problems.extend(f"{where} {problem}"
                            for problem in _pick_problems(reference, catalogue[reference.catalogue], specs))
        if reference.count is not None:
            problems.extend(f"{where} {problem}" for problem in _count_problems(reference, specs))
    picked = [(reference.process, reference.object_attribute) for _, reference in references if reference.pick is not None]
    for process, attribute in sorted({pair for pair in picked if picked.count(pair) > 1}):
        problems.append(f"more than one reference picks {attribute} of process {process}")
    brought = [(reference.process, name) for _, reference in references for name in _brought_names(reference)]
    for where, reference in references:
        if reference.with_attributes and reference.catalogue in specs[reference.process].catalogues:
            problems.extend(f"{where} {problem}" for problem in _with_problems(
                reference, catalogue[reference.catalogue], specs, set(picked), brought))
    if problems:
        raise ValueError("Invalid catalogue: " + "; ".join(problems))


def _pick_problems(reference, entry, specs):
    if isinstance(entry, GrowingCatalogue) and not reference.creates:
        return [f'picks from growing catalogue {reference.catalogue!r}, which only takes "new" or {{"new_share": ...}}']
    if reference.creates:
        if entry is not None and not isinstance(entry, GrowingCatalogue):
            return [f'picks "new" or new_share from catalogue {reference.catalogue!r}, which is no growing catalogue']
        share = reference.pick.share if isinstance(reference.pick, NewShare) else 1
        if not isinstance(share, (int, float)) or isinstance(share, bool) or not 0 < share <= 1:
            return [f"new_share must be a number above 0 and at most 1, got {share!r}"]
    elif reference.pick != "uniform":
        if not isinstance(reference.pick, tuple):
            return ['pick must be "uniform", {"weights": {id: weight, ...}}, "new" or {"new_share": share}']
        if isinstance(entry, CataloguePool):
            return [f"has weights on pool {reference.catalogue!r}; a pool can only be picked from uniformly"]
        if entry is not None:
            unknown = [object_id for object_id, _ in reference.pick if object_id not in {obj.id for obj in entry}]
            if unknown:
                return [f"weights name {unknown}, which are no ids of catalogue {reference.catalogue!r}"]
        weights = [weight for _, weight in reference.pick]
        if not all(isinstance(weight, (int, float)) and not isinstance(weight, bool) and weight >= 0
                   for weight in weights) or sum(weights) <= 0:
            return [f"weights must be numbers of at least 0, not all 0, got {dict(reference.pick)}"]
    return _settings_problems("picks", reference.object_attribute, reference.process, specs)


def _count_problems(reference, specs):
    count = reference.count
    if reference.pick is None:
        return ["has count but no pick; count says how many objects the pick chooses"]
    if reference.creates:
        return ['has count with "new" or new_share; count works with "uniform" or weights, which choose among the '
                'objects there are']
    if isinstance(count, CountFrom):
        with open(specs[reference.process].json_path) as file:
            declared = {entry.get("name") for entry in json.load(file).get("object_attributes", [])}
        if count.object_attribute not in declared:
            return [f"count reads {count.object_attribute!r}, which process {reference.process} doesn't draw in its "
                    f"object_attributes; the count is read when the case is created"]
        return []
    if not is_whole_number(count, minimum=0):
        return [f'count must be a whole number of at least 0 or {{"object_attribute": <object attribute>}}, '
                f"got {count!r}"]
    return []


def _with_json(pairs):
    """with as the configuration wrote it, for messages."""
    if not isinstance(pairs, tuple):
        return pairs
    def written(source):
        if not isinstance(source, Computed):
            return source
        return {source.function: source.source, **({} if source.initial is None else {"initial": source.initial})}

    return {pair[0]: written(pair[1]) for pair in pairs if isinstance(pair, tuple) and len(pair) == 2}


def _drawn_as_numbers(attribute):
    """Whether a pool attribute, in the format of object_attributes, always draws a number: a continuous one, or a
    discrete one whose keys are all numbers."""
    if attribute.get("type") == "continuous":
        return True
    values = attribute.get("values")
    return isinstance(values, list) and all(isinstance(value, dict) and _is_number(value.get("key")) for value in values)


def _settings_problems(verb, attribute, process, specs):
    """An attribute that the orchestrator gives a process's cases when they are created, which its JSON settings
    also set: drawn in object_attributes, or set from a message."""
    with open(specs[process].json_path) as file:
        settings = json.load(file)
    declared = {entry.get("name") for entry in settings.get("object_attributes", [])}
    if attribute in declared:
        return [f"{verb} {attribute}, which process {process} also draws in its object_attributes"]
    # a copied value would replace the given one (with "new", the replaced object is never written), an
    # appended attribute is a list, and an aggregated one is computed over the messages claimed
    messages = settings.get("messages")
    consume = messages.get("consume", []) if isinstance(messages, dict) else []
    for key, also in (("copy", "copies"), ("append", "appends to"), ("aggregate", "aggregates into")):
        found_at = [entry.get("event_id") for entry in consume if isinstance(entry, dict)
                    and isinstance(entry.get(key), dict) and attribute in entry[key]]
        if found_at:
            return [f"{verb} {attribute}, which process {process} also {also} from a message (consume entries at "
                    f"{found_at})"]
    return []


def _brought_names(reference):
    """The attributes a reference's with gives its cases (none if with is malformed, which _with_problems reports)."""
    pairs = reference.with_attributes
    return [pair[0] for pair in pairs if isinstance(pair, tuple) and len(pair) == 2] if isinstance(pairs, tuple) else []


def _with_problems(reference, entry, specs, picked, brought):
    """picked: (process, attribute) of every pick; brought: (process, attribute) of every attribute a with gives."""
    pairs = reference.with_attributes
    if isinstance(pairs, tuple):
        for name, source in (pair for pair in pairs if isinstance(pair, tuple) and len(pair) == 2):
            if isinstance(source, dict) and set(source) == {"initial"}:
                return [f"with {name!r} has initial but no function; initial is the first value a sum, count, min "
                        f"or max works over, and a plain copy has none"]
            initial = source.get("initial") if isinstance(source, dict) else getattr(source, "initial", None)
            if (isinstance(source, Computed) and initial is not None or isinstance(source, dict) and "initial" in source) \
                    and not is_number(initial):
                return [f"with {name!r} initial must be a number, got {initial!r}"]
    if not (isinstance(pairs, tuple) and all(
            isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[0], str) and pair[0]
            and (isinstance(pair[1], str) and pair[1]
                 or isinstance(pair[1], Computed) and pair[1].function in AGGREGATE_FUNCTIONS
                 and isinstance(pair[1].source, str) and pair[1].source) for pair in pairs)):
        return ['with must map attribute names to an attribute of the picked object or to one function of one, '
                f'{{"sum" | "count" | "min" | "max": <attribute>}}, got {_with_json(pairs)!r}']
    if reference.pick is None:
        return ["has with but no pick; a catalogue object brings its attributes only when the reference picks it"]
    if isinstance(entry, GrowingCatalogue):
        return [f"has with on growing catalogue {reference.catalogue!r}, whose objects don't exist yet when a "
                f"case is created"]
    copied = [name for name, source in pairs if isinstance(source, str)]
    if reference.count is not None and copied:
        return [f"copies {copied} on a pick with count; a list pick needs a function over the picked objects, "
                f'e.g. {{"sum": "price"}}']
    sources = list(dict.fromkeys(source.source if isinstance(source, Computed) else source for _, source in pairs))
    numeric = list(dict.fromkeys(source.source for _, source in pairs
                                 if isinstance(source, Computed) and source.function != "count"))
    if entry is not None:
        if isinstance(entry, CataloguePool):
            specs_of = {attribute.get("name"): attribute for attribute in entry.attributes}
            missing = [name for name in sources if name not in specs_of]
            not_numbers = [name for name in numeric if name in specs_of and not _drawn_as_numbers(specs_of[name])]
        else:
            missing = [name for name in sources if not all(name in dict(obj.attributes) for obj in entry)]
            not_numbers = [name for name in numeric if name not in missing
                           and not all(_is_number(dict(obj.attributes)[name]) for obj in entry)]
        if missing:
            return [f"with names {missing}, which not every object of catalogue {reference.catalogue!r} has"]
        if not_numbers:
            return [f"with sums, or takes the min or max of, {not_numbers}, which not every object of catalogue "
                    f"{reference.catalogue!r} has as a number"]
    problems = []
    for name, _ in pairs:
        if name == "object_id":
            problems.append("brings object_id, which is reserved for the object's own id")
        elif (reference.process, name) in picked:
            problems.append(f"brings {name}, which process {reference.process} also picks")
        elif brought.count((reference.process, name)) > 1:
            problems.append(f"brings {name}, which another with also brings to process {reference.process}")
        else:
            problems.extend(_settings_problems("brings", name, reference.process, specs))
    return problems


class _Pick:
    """A case attribute that a reference picks (pick) from its catalogue's ids. It has what the engine uses of
    Prosimos's CaseAttribute, name and get_next_value(), so it is drawn with the case's other attributes, when
    the case is created; its generator is its own, seeded from the seed, the process and the attribute, so it
    doesn't change the engine's other draws. With count, it picks that many ids, each independently, into a
    list."""

    def __init__(self, name: str, ids: List[str], weights: Optional[List[float]], seed: Optional[int], process: str,
                 count: Union[None, int, CountFrom] = None):
        self.name, self._ids, self._weights, self.count = name, ids, weights, count
        self.last = None  # the id picked for the latest case
        self._counted = None  # the value of count's attribute drawn for the latest case
        self._warned = False
        self._generator = random.Random()
        self._generator.setstate(_engine_random_states(seed, f"pick/{process}/{name}")[0])

    def read_count_from(self, attribute):
        """Has the case attribute that holds the count, one the engine draws before its picks, tell this pick the
        value it draws for each case. The attribute keeps its own draws."""
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
        """count, or the value its attribute drew for this case; one that isn't a whole number of at least 0 picks
        one, with one warning."""
        if not isinstance(self.count, CountFrom):
            return int(self.count)
        if is_whole_number(self._counted, minimum=0):
            return int(self._counted)
        if not self._warned:
            self._warned = True
            warning_logger.add_warning(f"{self.count.object_attribute} is {self._counted!r}, no whole number of at "
                                       f"least 0; the pick of {self.name} picks one")
        return 1


class _Brought:
    """A value a _Pick's objects give the case it has just picked them for: an attribute of the picked object,
    copied, or, with function, the function over that attribute of every object it picked, each as often as it
    was picked (a single pick is a list of one), and over initial, if given, as aggregate computes over messages.
    It comes right after its pick in the case's attributes, which are drawn in order, so it reads the pick's last
    value; it draws nothing. Over no objects, a sum and a count are 0 and a min or max has no value (None), unless
    initial gives them one."""

    def __init__(self, name: str, pick: _Pick, values: Dict[str, Any], function: Optional[str] = None,
                 initial: Optional[float] = None):
        self.name, self._pick, self._values, self._function, self._initial = name, pick, values, function, initial

    def get_next_value(self):
        if self._function is None:
            return self._values[self._pick.last]
        picked = self._pick.last if isinstance(self._pick.last, list) else [self._pick.last]
        if not picked and self._function in ("min", "max") and self._initial is None:
            return None
        return _aggregated(self._function, [self._values[object_id] for object_id in picked], self._initial)


class _NewPick:
    """A case attribute that a reference picks from a growing catalogue: a new id, <catalogue name>-<number> in
    creation order, for every case ("new") or for share of them, the rest reusing one created so far, uniformly.
    Like _Pick, it is drawn with the case's other attributes, from a generator of its own."""

    def __init__(self, name: str, catalogue: str, share: float, seed: Optional[int], process: str):
        self.name, self.catalogue, self._share = name, catalogue, share
        self.created = 0
        self._generator = random.Random()
        self._generator.setstate(_engine_random_states(seed, f"pick/{process}/{name}")[0])

    def get_next_value(self):
        if self.created == 0 or self._share == 1 or self._generator.random() < self._share:
            self.created += 1
            return f"{self.catalogue}-{self.created}"
        return f"{self.catalogue}-{self._generator.randint(1, self.created)}"


def _picks(spec: ProcessSpec, seed: Optional[int], catalogues: Dict[str, List[CatalogueObject]]) -> List[_Pick]:
    """The attributes a process's references pick, from its own catalogues as the run starts (see
    _drawn_catalogues)."""
    picks = []
    for reference in spec.references:
        if reference.creates:
            share = reference.pick.share if isinstance(reference.pick, NewShare) else 1
            picks.append(_NewPick(reference.object_attribute, reference.catalogue, share, seed, spec.name))
        elif reference.pick is not None:
            ids = [obj.id for obj in catalogues[reference.catalogue]]
            weights = None if reference.pick == "uniform" else [dict(reference.pick).get(object_id, 0) for object_id in ids]
            pick = _Pick(reference.object_attribute, ids, weights, seed, spec.name, reference.count)
            picks.append(pick)
            for name, source in reference.with_attributes:
                function, attribute, initial = ((source.function, source.source, source.initial)
                                                if isinstance(source, Computed) else (None, source, None))
                values = {obj.id: dict(obj.attributes)[attribute] for obj in catalogues[reference.catalogue]}
                picks.append(_Brought(name, pick, values, function, initial))
    return picks


def _drawn_catalogues(spec: ProcessSpec, seed: Optional[int]) -> Dict[str, List[CatalogueObject]]:
    """A process's catalogues, name -> their objects when the run starts: a listed catalogue's as given, a pool's
    with their attributes drawn (see _drawn_objects), none yet for a growing catalogue."""
    return {name: list(entry) if isinstance(entry, list) else _drawn_objects(seed, name, entry.attributes,
                                                                            _catalogue_ids(name, entry))
            for name, entry in spec.catalogues.items()}


def _grown_catalogues(spec: ProcessSpec, seed: Optional[int], catalogues: Dict[str, List[CatalogueObject]],
                      picks: List[_Pick]) -> Dict[str, List[CatalogueObject]]:
    """A process's catalogues after the run: each growing catalogue with the objects its pick created, <name>-1 to
    <name>-<created>, their attributes drawn in creation order (see _drawn_objects). Drawing them now rather
    than when each was created gives the same values, since nothing else draws from their generators."""
    grown = dict(catalogues)
    for pick in picks:
        if isinstance(pick, _NewPick):
            grown[pick.catalogue] = _drawn_objects(seed, pick.catalogue, spec.catalogues[pick.catalogue].attributes,
                                                   [f"{pick.catalogue}-{number}" for number in range(1, pick.created + 1)])
    return grown


def _drawn_objects(seed, name, attribute_specs, ids) -> List[CatalogueObject]:
    """The objects with these ids, their attributes drawn object after object by Prosimos's case attribute code,
    from generators seeded from the seed and the catalogue's name."""
    attributes = [CaseAttribute(attribute["name"], attribute["type"], attribute["values"]) for attribute in attribute_specs]
    python_state, numpy_state = _engine_random_states(seed, f"catalogue/{name}")
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
