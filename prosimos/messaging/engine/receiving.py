"""How a message offered to the engine is taken: deliver, claims, collect, capacity and same."""
import math

from prosimos.execution_info import EnabledEvent
from prosimos.messaging.parser.common import is_whole_number, is_whole_number_of_at_least_one
from prosimos.warning_logger import warning_logger


class Receiving:
    """Offering a message to the waiting cases: claims, collect, capacity and same."""

    def deliver(self, message_type, attributes, source, now):
        """Offers one message: 'claimed' (by the case waiting longest, or a new case), 'discarded' if it can
        never match, else 'pending'. After a claim, claimed_by lists each (case, event, entry) that took it."""
        self._ensure_arrivals_generated()
        self._close_collections(before=now)
        self.claimed_by = []
        values = dict(attributes, source=source)
        # the entry's capacity (min, max): how many cases waiting there claim it, oldest first; fewer than min, none
        claiming, takers, fewest, capacity = None, [], 1, 1
        for parked_event in self.parked_events():
            if (parked_event.p_case, parked_event.task_id) in self._set_aside:
                continue  # it holds a quote's line
            if claiming is None:
                claiming = next((point for point in self._consume_points[parked_event.task_id] if point.type == message_type
                                 and self._accepts(point, values, parked_event.p_case)), None)
                if claiming is not None:
                    if parked_event.race is not None:
                        parked_event, claiming = self._first_listed_branch(parked_event.race, message_type, values)
                    fewest, capacity = self._capacity(claiming, values)
                    takers.append(parked_event)
            elif parked_event.task_id == claiming.event_id and self._accepts(claiming, values, parked_event.p_case):
                takers.append(parked_event)
            if takers and len(takers) == capacity:
                break
        if takers and len(takers) < fewest:
            return "pending"  # offered again at this engine's next step, e.g. when another case starts waiting
        if takers:
            for parked_event in takers:
                self._claim(parked_event, claiming, values, now)
            self.claimed_by = [(parked_event.p_case, parked_event.task_id, claiming)
                               for parked_event in takers]
            return "claimed"
        points = [point for points in self._consume_points.values() for point in points if point.type == message_type]
        # a start event's condition looks only at the message, so it decides now: a new case, or never; of several
        # start events that accept it, the one whose entry comes first in the JSON starts the case, as in races
        starting = min((point for point in points
                        if point.starts_case and self._condition_holds(point.condition, values, None)),
                       key=lambda point: self._consume_order[id(point)], default=None)
        if starting is not None:
            def start_values(p_case, case_values):
                self._apply_copy(starting, values, case_values)
                self._apply_append(starting, values, case_values)
                self._apply_aggregate(starting, values, p_case, case_values)

            p_case = self.create_case(now, start_values, starting.event_id)
            self.claimed_by = [(p_case, starting.event_id, starting)]
            return "claimed"
        if any(self._could_match(point.condition, values) and self._has_same(point, values)
               for point in points if not point.starts_case):
            return "pending"
        return "discarded"

    def _claim(self, parked_event, point, values, now):
        """A waiting case claims one message, and is resumed once it has collected what it needs."""
        key = (parked_event.p_case, parked_event.task_id)
        case_values = self.sim_setup.bpmn_graph.all_attributes[parked_event.p_case]
        self._apply_copy(point, values, case_values)
        self._apply_append(point, values, case_values)
        aggregated = self._apply_aggregate(point, values, parked_event.p_case, case_values)
        self._copied_later.extend((parked_event.p_case, target, values[source], now)
                                  for target, source in point.copy if source in values)
        if self._received:  # the 'with' values follow what was copied or appended now
            changed = {target for target, source in (*point.copy, *point.append) if source in values}
            self._copied_later.extend((parked_event.p_case, name, case_values[name], now)
                                      for name in self._apply_received(case_values, now, changed))
        self._copied_later.extend((parked_event.p_case, target, case_values[target], now)
                                  for target, source in point.append if source in values)
        self._copied_later.extend((parked_event.p_case, target, case_values[target], now) for target in aggregated)
        self._collected[key][0] += 1
        if point.same and self._collected[key][3] is None:
            self._collected[key][3] = {name: values[name] for name in point.same}
        claimed, low, high, _ = self._collected[key]
        if key in self._closing:
            if claimed >= high:
                self._close(key)
        elif claimed >= low:
            self._resume(parked_event, now, keep_open=claimed < high)

    def _resume(self, parked_event, now, keep_open=False):
        """Resumes a waiting case at now; in a race, its branch wins. With keep_open the case has its min but not
        its max, and keeps taking the messages offered at now (see _close_collections)."""
        if parked_event.race is not None:
            self._win_race(parked_event.race, parked_event.task_id, parked_event.p_state)
        key = (parked_event.p_case, parked_event.task_id)
        if keep_open:
            self._closing[key] = now
        else:
            del self._parked_events[key]
            del self._collected[key]
        resumed = EnabledEvent(parked_event.p_case, parked_event.p_state, parked_event.task_id, self.simulation_at_from_datetime(now), now,
                               is_inter_event=True)
        resumed.parked_event = parked_event
        self.events_queue.append_event(resumed)

    def _close(self, key):
        del self._parked_events[key], self._collected[key], self._closing[key]

    def _close_collections(self, before=None):
        """Closes the collections kept open since their case got its min: all, or those opened before 'before'.
        The orchestrator offers all of a step's messages before it calls an engine again."""
        for key, opened in list(self._closing.items()):
            if before is None or opened < before:
                self._close(key)

    def _accepts(self, point, values, p_case):
        """Whether the case waiting at point's event takes the message: its condition and 'same' hold."""
        if not self._condition_holds(point.condition, values, p_case) or not self._has_same(point, values):
            return False
        collected = self._collected.get((p_case, point.event_id))
        first = collected[3] if collected is not None else None
        return first is None or all(values[name] == value for name, value in first.items())

    def _has_same(self, point, values):
        missing = [name for name in point.same if name not in values]
        for name in missing:
            if (point.event_id, name) not in self._warned_missing_same:
                self._warned_missing_same.add((point.event_id, name))
                warning_logger.add_warning(
                    f"{point.type} message offered at {point.event_id} has no {name}, which 'same' compares; it "
                    f"doesn't match there")
        return not missing

    def _capacity(self, point, values):
        """(min, max) waiting cases this message resumes at point; an invalid min counts as 1, a max as min."""
        fewest = point.capacity_min
        if point.capacity_min_attribute is not None:
            fewest = values.get(point.capacity_min_attribute)
            if not is_whole_number_of_at_least_one(fewest):
                self._warn_capacity(point, "min", point.capacity_min_attribute, f"(got {fewest!r}); its min counts as 1")
                fewest = 1
            fewest = int(fewest)
        most = point.capacity if point.capacity_attribute is None else values.get(point.capacity_attribute)
        if most == math.inf or is_whole_number_of_at_least_one(most) and most >= fewest:
            return fewest, most if most == math.inf else int(most)
        if fewest == 1:
            self._warn_capacity(point, "max", point.capacity_attribute, f"(got {most!r}); it resumes one waiting case")
        else:
            self._warn_capacity(point, "max", point.capacity_attribute or "max",
                                f"(got {most!r}, min {fewest}); it resumes no more than {fewest} waiting cases")
        return fewest, fewest

    def _warn_capacity(self, point, bound, attribute, consequence):
        if (point.event_id, bound, attribute) not in self._warned_capacities:
            self._warned_capacities.add((point.event_id, bound, attribute))
            warning_logger.add_warning(
                f"{point.type} message accepted at {point.event_id} has no valid {attribute} {consequence}")

    def _needed(self, p_case, event_id):
        """How many messages a case must claim at a catch event; an invalid value counts as 1, with one warning."""
        point = self._consume_points[event_id][0]  # an event with collect has exactly one entry
        if point.collect_attribute is None:
            return point.collect
        needed = self.sim_setup.bpmn_graph.get_all_attributes(p_case).get(point.collect_attribute)
        if is_whole_number(needed, minimum=0):
            return int(needed)
        if (point.event_id, point.collect_attribute) not in self._warned_collects:
            self._warned_collects.add((point.event_id, point.collect_attribute))
            warning_logger.add_warning(
                f"case {p_case} reaches {point.event_id} with no valid {point.collect_attribute} "
                f"(got {needed!r}); it collects one message")
        return 1

    def _most(self, p_case, event_id, needed):
        """How many messages a case may claim at a catch event in all, at least needed (math.inf: no limit)."""
        point = self._consume_points[event_id][0]
        if point.collect_max is None and point.collect_max_attribute is None:
            return needed
        if point.collect_max_attribute is None:
            most, name = point.collect_max, "max"
        else:
            most = self.sim_setup.bpmn_graph.get_all_attributes(p_case).get(point.collect_max_attribute)
            name = point.collect_max_attribute
        if isinstance(most, float) and most == math.inf:
            return most
        if is_whole_number(most, minimum=needed):
            return int(most)
        if (point.event_id, name) not in self._warned_collects:
            self._warned_collects.add((point.event_id, name))
            warning_logger.add_warning(
                f"case {p_case} reaches {point.event_id} with no valid {name} (got {most!r}, min {needed}); it "
                f"collects no more than {needed}")
        return needed

    def collected(self, parked_event):
        """(messages claimed so far, messages needed: min) by a case parked at its catch event."""
        return tuple(self._collected[(parked_event.p_case, parked_event.task_id)][:2])
