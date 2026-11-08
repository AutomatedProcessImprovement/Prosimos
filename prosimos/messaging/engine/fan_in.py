"""Fan-in: parts handed in, piled per group and sent as one message (docs/messaging.md, "Fan-in")."""
from datetime import timedelta
from heapq import heapify, heappush

from prosimos.messaging.engine.values import aggregate_of
from prosimos.messaging.parser.common import is_number, is_whole_number_of_at_least_one
from prosimos.warning_logger import warning_logger


class Pile:
    """The parts handed in so far for one fan-in message: the first part's size decides when it is sent."""

    def __init__(self, point, key, element_id, size, fewest, given_fewest, same_values):
        self.point, self.key, self.element_id = point, key, element_id
        # by its first part, which later parts are compared with
        self.given_size, self.given_fewest = size, given_fewest
        self.size = max(size, fewest)
        self.fewest = fewest  # the parts it needs at its deadline; without a deadline, its size
        self.same_values = same_values
        self.parts = []  # (case id, when it was handed in, {object attribute: its value then})
        self.deadline_held = None  # its item in the held heap, if its entry has a deadline
        self.overdue = False  # its deadline passed while it had fewer parts than it needs


class FanIn:
    """Hands in parts, keeps their piles and sends a pile's message."""

    def _hand_in(self, point, p_case, element_id, due):
        """Puts the part of p_case on its group's pile, at its own time; the part that completes the pile publishes
        the group's one message."""
        values = self.sim_setup.bpmn_graph.get_all_attributes(p_case)
        same = tuple(self._part_value(point, p_case, values, name) for name in point.same)
        key = (point, tuple(map(_hashable, same)))
        size, valid = self._fan_in_size(point, p_case, values)
        given_fewest, valid_fewest = self._given_min(point, p_case, values)
        pile = self._piles.get(key)
        if pile is None:
            fewest = self._fan_in_min(point, p_case, size, given_fewest)
            pile = self._piles[key] = Pile(point, key, element_id, size, fewest, given_fewest, same)
            if point.deadline is not None:
                # due a microsecond late, as a race timer, so every part of that instant comes first
                pile.deadline_held = (due + point.deadline + timedelta(microseconds=1), p_case,
                                      next(self._pass_order), pile)
                heappush(self._held, pile.deadline_held)
        elif valid and size != pile.given_size and (point.event_id, point.fan_in_attribute) not in \
                self._warned_fan_in_sizes:
            self._warned_fan_in_sizes.add((point.event_id, point.fan_in_attribute))
            warning_logger.add_warning(
                f"case {p_case} passes {point.event_id} with {point.fan_in_attribute} {size}, but its fan-in group "
                f"has {pile.given_size} from its first part; the group keeps {pile.size}")
        if pile.parts and valid_fewest and given_fewest != pile.given_fewest:
            self._warn_fan_in(point, point.fan_in_min_attribute,
                              f"case {p_case} passes {point.event_id} with {point.fan_in_min_attribute} "
                              f"{given_fewest}, but its fan-in group has {pile.given_fewest} from its first part; the "
                              f"group keeps min {pile.fewest}")
        sources = dict.fromkeys([source for _, source in point.gathered]
                                + [source for _, _, source, _ in point.computed])
        pile.parts.append((p_case, due, {source: self._part_value(point, p_case, values, source)
                                         for source in sources}))
        if len(pile.parts) == pile.size or pile.overdue and len(pile.parts) >= pile.fewest:
            self._send(pile)

    def _deadline_passed(self, pile):
        """Sends pile if it has the parts it needs; else it stays open until it has."""
        pile.deadline_held = None
        if len(pile.parts) >= pile.fewest:
            self._send(pile)
        else:
            pile.overdue = True

    def _send(self, pile):
        """Publishes the message of pile's parts, now: it is complete, or has its min after its deadline."""
        point = pile.point
        del self._piles[pile.key]
        if pile.deadline_held is not None and pile.deadline_held in self._held:
            # not due any more, so it doesn't keep the engine going
            self._held.remove(pile.deadline_held)
            heapify(self._held)
        attributes = {name: pile.same_values[point.same.index(name)] for name in point.attributes}
        attributes.update((name, [part[source] for _, _, part in pile.parts]) for name, source in point.gathered)
        attributes.update((name, self._over_parts(point, name, function, source, initial, pile.parts))
                          for name, function, source, initial in point.computed)
        self.outbox.append((point.type, attributes, tuple((member, time) for member, time, _ in pile.parts),
                            pile.element_id, point, None))

    def _given_min(self, point, p_case, values):
        """(the min this part gives, whether it is valid): None without one; an invalid one counts as 1, with one
        warning."""
        if point.fan_in_min_attribute is None:
            return point.fan_in_min, False
        fewest = values.get(point.fan_in_min_attribute)
        if is_whole_number_of_at_least_one(fewest):
            return int(fewest), True
        self._warn_fan_in(point, point.fan_in_min_attribute,
                          f"case {p_case} passes {point.event_id} with no valid {point.fan_in_min_attribute} (got "
                          f"{fewest!r}); its group needs 1 part at its deadline")
        return 1, False

    def _fan_in_min(self, point, p_case, size, fewest):
        """The parts a group needs at its deadline, from its first part's min: 1 with a deadline and no min, else
        size without one. A max below min counts as min, with one warning."""
        if fewest is None:
            return 1 if point.deadline is not None else size
        if fewest > size:
            names = (point.fan_in_attribute or "max", point.fan_in_min_attribute or "min")
            self._warn_fan_in(point, names, f"case {p_case} passes {point.event_id} with {names[0]} {size} below "
                                            f"{names[1]} {fewest}; its group's max counts as {fewest}")
        return fewest

    def _warn_fan_in(self, point, about, warning):
        if (point.event_id, about) not in self._warned_fan_in_sizes:
            self._warned_fan_in_sizes.add((point.event_id, about))
            warning_logger.add_warning(warning)

    def _fan_in_size(self, point, p_case, values):
        """(the group size this part gives, whether it is valid): an invalid one counts as 1, with one warning."""
        if point.fan_in_attribute is None:
            return point.fan_in, True
        size = values.get(point.fan_in_attribute)
        if is_whole_number_of_at_least_one(size):
            return int(size), True
        if (point.event_id, point.fan_in_attribute) not in self._warned_fan_in_sizes:
            self._warned_fan_in_sizes.add((point.event_id, point.fan_in_attribute))
            warning_logger.add_warning(
                f"case {p_case} passes {point.event_id} with no valid {point.fan_in_attribute} (got {size!r}); its "
                f"part gives a group size of 1")
        return 1, False

    def _part_value(self, point, p_case, values, name):
        if name == "object_id":
            return self.object_id(p_case)
        if name in values:
            return values[name]
        if (point.event_id, name) not in self._warned_missing_values:
            self._warned_missing_values.add((point.event_id, name))
            warning_logger.add_warning(f"Attribute {name} has no value when case {p_case} passes {point.event_id}; "
                                       f"its part of the {point.type} message carries None")
        return None

    def _over_parts(self, point, name, function, source, initial, parts):
        """function over the parts' values of source; 'count' counts those with a value, and a value that isn't a
        number is left out of the others, with one warning."""
        if function == "count":
            return aggregate_of(function, [part[source] for _, _, part in parts if part[source] is not None], initial)
        numbers_only = []
        for p_case, _, part in parts:
            if is_number(part[source]):
                numbers_only.append(part[source])
            elif (point.event_id, source) not in self._warned_fan_in_numbers:
                self._warned_fan_in_numbers.add((point.event_id, source))
                warning_logger.add_warning(
                    f"case {p_case} passes {point.event_id} with {source} {part[source]!r}, which isn't a number to "
                    f"{function} into {name}; its part is left out of {name}")
        if not numbers_only and initial is None and function in ("min", "max"):
            return None
        return aggregate_of(function, numbers_only, initial)

    def warn_unsent_fan_ins(self):
        """One warning per fan-in group still short of its size when the run ends."""
        for (point, _), pile in self._piles.items():
            group = " and ".join(f"{name} {value}" for name, value in zip(point.same, pile.same_values))
            needs = f" (min {pile.fewest})" if pile.fewest != pile.size else ""
            warning_logger.add_warning(f"fan-in {point.type}{' for ' + group if group else ''}: "
                                       f"{len(pile.parts)} of {pile.size} parts{needs}, never sent")


def _hashable(value):
    # a group's values key its pile; a list value becomes a tuple
    return tuple(map(_hashable, value)) if isinstance(value, list) else value
