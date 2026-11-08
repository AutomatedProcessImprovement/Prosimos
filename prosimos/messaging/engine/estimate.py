"""Estimates: when this engine's items could be ready for a quote's lines."""
from datetime import timedelta

from pix_framework.statistics.distribution import DistributionType

from prosimos.control_flow_manager import BPMN


class Estimating:
    """Answers estimate(): per line of a quote, the times its matching items not yet promised would be ready."""

    def slots_for(self, quote_type, requester, lines, now):
        """line id -> the times, earliest first, the items matching each of lines (line id, attributes) from
        requester are expected at a catch event taking the quote type at now: slot k is when the expected count
        there reaches k. The open lines take the earliest expected items first; those are promised and skipped."""
        self._ensure_arrivals_generated()
        free = self._quote_candidates(quote_type, now)
        for line in self._open_lines:
            _take_one(item for item in free if self._line_matches(line["type"], line["values"], item))
        return {line_id: _expected_slots(item for item in free if self._line_matches(
                    quote_type, dict(attributes, source=requester), item))
                for line_id, attributes in lines}

    def _quote_candidates(self, quote_type, now):
        """[time, case id, event id, chance] of each item not holding a line that may reach a catch event taking
        quote_type: waiting there, queued to reach it, or forecast from a task it is queued at, in time order."""
        events = {point.event_id for points in self._consume_points.values() for point in points
                  if point.type == quote_type and not point.starts_case}
        waiting = [[parked_event.enabled_datetime, parked_event.p_case, parked_event.task_id, 1.0]
                   for parked_event in self.parked_events() if parked_event.task_id in events
                   and (parked_event.p_case, parked_event.task_id) not in self._set_aside]
        queued, waits = [], {}
        for event in self.events_queue.enabled_events.entry_finder:
            if event.task_id in events:
                if event.parked_event is None:
                    queued.append([event.enabled_datetime, event.p_case, event.task_id, 1.0])
                continue
            if self.sim_setup.bpmn_graph.element_info[event.task_id].type != BPMN.TASK:
                continue
            reached = self._forecast(event.p_case, event.task_id, event.enabled_datetime, events, now, waits)
            queued += [[ready, event.p_case, catch_event, chance]
                       for catch_event, (chance, ready) in (reached or {}).items()]
        return sorted(waiting + queued, key=lambda item: item[:3])

    def _forecast(self, p_case, element_id, ready, events, now, waits):
        """catch event in events -> (chance, average ready time) of the case p_case ready at element_id at ready;
        None if a way has anything but tasks, throw events, exclusive gateways, and end events. waits caches the
        average waits."""
        graph = self.sim_setup.bpmn_graph
        reached = {}
        ways = [(element_id, ready, 1.0, 0)]
        while ways:
            element_id, ready, chance, steps = ways.pop()
            # a loop back is followed until its chance is negligible; one the case's data decides never ends: it is cut
            if chance < _NEGLIGIBLE or steps > _MOST_STEPS:
                continue
            if element_id in events:
                reached.setdefault(element_id, []).append((chance, ready))
                continue
            info = graph.element_info[element_id]
            if info.type is BPMN.END_EVENT:
                continue
            if info.type is BPMN.EXCLUSIVE_GATEWAY and len(info.outgoing_flows) > 1:
                branches = self._branches(info, p_case)
            elif info.type in (BPMN.TASK, BPMN.INTERMEDIATE_THROW_EVENT, BPMN.EXCLUSIVE_GATEWAY) \
                    and len(info.outgoing_flows) == 1:
                branches = [(info.outgoing_flows[0], 1.0)]
            else:
                return None
            if info.type is BPMN.TASK:
                if element_id not in waits:
                    waits[element_id] = self._average_wait(element_id, now)
                ready = forecast_end(ready, self._staff_calendars(element_id),
                                     waits[element_id] + self._average_duration(element_id))
            ways += [(graph.flow_arcs[flow][1], ready, chance * share, steps + 1) for flow, share in branches]
        return {event: _average(ways) for event, ways in reached.items()}

    def _branches(self, info, p_case):
        """(flow, chance) of each branch of the exclusive split info for the case p_case, as Prosimos picks one:
        the branches the case's data passes, or the default branch, or the probabilities."""
        graph = self.sim_setup.bpmn_graph
        choice = graph.element_probability[info.id]
        shares = dict(zip(choice.candidates_list, choice.probability_list))
        conditions = graph.gateway_conditions.get(info.id)
        if conditions is not None and conditions.candidates_list:
            attributes = graph.get_all_attributes(p_case)
            passed = [flow for flow, rule in zip(conditions.candidates_list, conditions.rules_list)
                      if not rule or rule.is_true(attributes)]
            if len(passed) == 1:
                return [(passed[0], 1.0)]
            if passed:
                total = sum(shares[flow] for flow in passed)
                return [(flow, shares[flow] / total) for flow in passed]
            if conditions.get_default_path():
                return [(conditions.get_default_path(), 1.0)]
        return list(shares.items())

    def _average_wait(self, task_id, now):
        """The average working seconds from ready to started of task_id, on each performer's calendar, over the
        times it started by now; 0 while none has."""
        setup = self.sim_setup
        waits = [setup.calendars_map[setup.resources_map[event.resource_id].calendar_id].find_working_time(
                     event.enabled_datetime, event.started_datetime)
                 for trace in self.log_info.trace_list for event in trace.event_list
                 if event.type == BPMN.TASK and event.task_id == task_id and event.started_datetime <= now]
        return sum(waits) / len(waits) if waits else 0

    def _average_duration(self, task_id):
        """The mean of task_id's duration distributions over the staff who may do it, in working seconds."""
        distributions = self.sim_setup.task_resource[task_id].values()
        return sum(_mean(distribution) for distribution in distributions) / len(distributions)

    def _staff_calendars(self, task_id):
        setup = self.sim_setup
        return [setup.calendars_map[setup.resources_map[resource_id].calendar_id]
                for resource_id in setup.task_resource[task_id]]

    def _line_matches(self, quote_type, values, item):
        """Whether the item's catch event's consume entry for quote_type takes a line with values."""
        _, p_case, event_id, _ = item
        point = next((point for point in self._consume_points.get(event_id, []) if point.type == quote_type), None)
        return point is not None and self._accepts(point, values, p_case)


_NEGLIGIBLE = 1e-9
_MOST_STEPS = 1000
# an expected count summed from chances, less the negligible ones, may fall a little short of a whole number
_WHOLE = 1 - 1e-6


def _take_one(items):
    """Takes one expected item from items (lists whose last entry is a chance), earliest first."""
    needed = 1.0
    for item in items:
        taken = min(needed, item[-1])
        item[-1] -= taken
        needed -= taken
        if needed <= 1 - _WHOLE:
            return


def _expected_slots(items):
    """When the expected count of items ([time, ..., chance], in time order) reaches 1, 2, 3, and so on."""
    slots, count = [], 0.0
    for item in items:
        count += item[-1]
        while count >= len(slots) + _WHOLE:
            slots.append(item[0])
    return slots


def _average(ways):
    """(chance, ready) of all ways together: their summed chance and their ready time weighted by chance."""
    chance = sum(share for share, _ in ways)
    first = min(ready for _, ready in ways)
    return chance, first + sum(((ready - first) * share for share, ready in ways), timedelta()) / chance


def forecast_end(start, calendars, working_seconds):
    """When working_seconds of work from start end on the earliest of calendars, skipping their time off."""
    return min(start + timedelta(seconds=calendar.find_idle_time(start, working_seconds)) for calendar in calendars)


def _mean(distribution):
    # a uniform distribution has only its bounds
    if distribution.type == DistributionType.UNIFORM:
        return (distribution.min + distribution.max) / 2
    return distribution.mean
