"""
The timetable of catalogue changes (docs/catalogue.md, "Changes"): the value of a catalogue attribute at any moment
of the run, worked out in advance from an organization's catalogue_changes.
"""
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Tuple

from prosimos.orchestration.catalogue.model import CatalogueChange


class Timetable:
    """The values of an organization's catalogue attributes over the run, worked out from its catalogue_changes
    instead of changed as the run goes: whoever reads a value reads the one valid at that moment. A change at a
    moment is valid from that moment on, inclusive, and changes at the same moment apply in the order they are
    listed. An object's values are worked out once, as far as the latest moment read so far."""

    def __init__(self, changes: Tuple[CatalogueChange, ...], start: datetime):
        self._start = start
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
            for moment, _, change in sorted(self._moments(changes, read_up_to, when), key=lambda found: found[:2]):
                current = change.applied(current)
                moments.append(moment)
                values.append(current)
            series[2] = when
        number = bisect_right(moments, when)
        return values[number - 1] if number else value

    def changed_at(self, catalogue: str, attribute: str, object_id: str, after: datetime, until: datetime):
        """The moments after 'after' and at most at 'until' at which a change changes the attribute of one catalogue
        object, in order, each once however many changes it has."""
        changes = self._changes_of(catalogue, attribute, object_id)
        return sorted({moment for moment, _, _ in self._moments(changes, after, until)})

    def _changes_of(self, catalogue, attribute, object_id):
        """(place in the list, change) of every change of the attribute of one catalogue object."""
        return [(index, change) for index, change in self._changes.get((catalogue, attribute), ())
                if change.objects is None or object_id in change.objects]

    def _moments(self, changes, after, until):
        """(moment, place in the list, change) of every change in changes after 'after' (None: from the beginning) and
        at most at until."""
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
    """One attribute of a catalogue's objects, as a 'with' reads it: a dict of object id -> the value before any
    change, and at(object_id, when), the value at the moment when, from its organization's timetable."""

    def __init__(self, values: Dict[str, Any], catalogue: str, attribute: str, timetable: Timetable):
        super().__init__(values)
        self.catalogue, self.attribute, self._timetable = catalogue, attribute, timetable

    def at(self, object_id: str, when: datetime):
        return self._timetable.value(self.catalogue, self.attribute, object_id, self[object_id], when)
