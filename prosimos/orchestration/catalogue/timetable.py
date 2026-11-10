"""The timetable of catalogue changes (docs/catalogue.md, "Changes")."""
import hashlib
import random
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from prosimos.orchestration.catalogue.model import Between, CatalogueChange


class Timetable:
    """An organization's catalogue values over the run, worked out from its catalogue_changes: a change is valid
    from its moment on, and changes at the same moment apply in the listed order. An amount drawn from a range is
    fixed by the seed, the change, the object, and the moment, so the timetable is the same whenever it is read, and
    it draws from no other generator; without a seed, it differs from run to run."""

    def __init__(self, changes: Tuple[CatalogueChange, ...], start: datetime, seed: Optional[int] = None):
        self._start = start
        if seed is None and any(isinstance(change.value, Between) for change in changes):
            seed = random.getrandbits(64)
        self._seed = seed
        # (catalogue, attribute) -> [(its place in the list, change)]
        self._changes = defaultdict(list)
        for index, change in enumerate(changes):
            self._changes[(change.catalogue, change.attribute)].append((index, change))
        # (catalogue, attribute, object id) -> [[moment of each change so far], [the value from then on], read up to]
        self._series = {}

    def value(self, catalogue: str, attribute: str, object_id: str, value, when: datetime):
        """What the attribute of one catalogue object is at the moment 'when', given value, what it was before any
        change: e.g. a Kindle's price of 129, read 61 days into the run with a 3% rise every 60 days, is 132.87."""
        changes = self._changes_of(catalogue, attribute, object_id)
        if not changes:
            return value
        series = self._series.setdefault((catalogue, attribute, object_id), [[], [], None])
        moments, values, read_up_to = series
        if read_up_to is None or when > read_up_to:
            current = values[-1] if values else value
            for moment, index, change in sorted(self._moments(changes, read_up_to, when), key=lambda found: found[:2]):
                current = change.applied(current, self._amount(index, change, object_id, moment))
                moments.append(moment)
                values.append(current)
            series[2] = when
        number = bisect_right(moments, when)
        return values[number - 1] if number else value

    def changed_at(self, catalogue: str, attribute: str, object_id: str, after: datetime, until: datetime):
        """The moments in ('after', 'until'] at which the attribute of one catalogue object changes, each once."""
        changes = self._changes_of(catalogue, attribute, object_id)
        return sorted({moment for moment, _, _ in self._moments(changes, after, until)})

    def _amount(self, index, change, object_id, moment):
        """The change's amount for one object at one moment: its own, or drawn from its range."""
        if not isinstance(change.value, Between):
            return change.value
        key = f"{self._seed}/catalogue_changes[{index}]/{object_id}/{moment.isoformat()}"
        draw = random.Random(int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big"))
        return draw.uniform(change.value.low, change.value.high)

    def _changes_of(self, catalogue, attribute, object_id):
        """(place in the list, change) of every change of the attribute of one catalogue object."""
        return [(index, change) for index, change in self._changes.get((catalogue, attribute), ())
                if change.objects is None or object_id in change.objects]

    def _moments(self, changes, after, until):
        """(moment, place in the list, change) of the changes in ('after', 'until'], all up to 'until' if 'after' is
        None."""
        for index, change in changes:
            if change.at is not None:
                if (after is None or change.at > after) and change.at <= until:
                    yield change.at, index, change
                continue
            number = 1 if after is None or after < self._start else (after - self._start) // change.every + 1
            while self._start + number * change.every <= until:
                yield self._start + number * change.every, index, change
                number += 1


class CatalogueValues(dict):
    """One attribute of a catalogue's objects: object id -> its value before any change; at(id, when): its value
    then."""

    def __init__(self, values: Dict[str, Any], catalogue: str, attribute: str, timetable: Timetable):
        super().__init__(values)
        self.catalogue, self.attribute, self._timetable = catalogue, attribute, timetable

    def at(self, object_id: str, when: datetime):
        return self._timetable.value(self.catalogue, self.attribute, object_id, self[object_id], when)
