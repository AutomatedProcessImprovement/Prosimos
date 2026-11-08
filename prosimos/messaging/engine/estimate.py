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
        quote_type: waiting there, queued to reach it, or forecast from where its branches are queued, in time
        order."""
        events = {point.event_id for points in self._consume_points.values() for point in points
                  if point.type == quote_type and not point.starts_case}
        waiting = [[parked_event.enabled_datetime, parked_event.p_case, parked_event.task_id, 1.0]
                   for parked_event in self.parked_events() if parked_event.task_id in events
                   and (parked_event.p_case, parked_event.task_id) not in self._set_aside]
        # a case waiting for a message elsewhere may be held up there in a branch: it isn't forecast
        held_up = {parked_event.p_case for parked_event in self.parked_events() if parked_event.task_id not in events}
        queued, ahead = [], {}
        for event in self.events_queue.enabled_events.entry_finder:
            if event.task_id not in events:
                ahead.setdefault(event.p_case, []).append(event)
            elif event.parked_event is None:
                queued.append([event.enabled_datetime, event.p_case, event.task_id, 1.0])
        waits = {}
        for p_case, case_events in ahead.items():
            if p_case not in held_up:
                reached = self._case_forecast(p_case, case_events, events, now, waits)
                queued += [[ready, p_case, catch_event, chance]
                           for catch_event, (chance, ready) in (reached or {}).items()]
        return sorted(waiting + queued, key=lambda item: item[:3])

    def _case_forecast(self, p_case, case_events, events, now, waits):
        """catch event in events -> (chance, average ready time) of the case p_case from its queued case_events;
        None if not forecast. The branches of a split go on together from their join when the latest gets there,
        the innermost join first, which then is one branch of the next."""
        forecasts = [self._forecast(p_case, event.task_id, event.enabled_datetime, events, now, waits)
                     for event in case_events]
        if None in forecasts:
            return None
        if all(target in events for forecast in forecasts for target in forecast):
            return {target: reached for forecast in forecasts for target, reached in forecast.items()}
        arriving = {}  # join -> when the branches on their way get there
        for forecast in forecasts:
            joined = _joined([forecast], events)
            if joined is None:
                return None
            arriving.setdefault(joined[0], []).append(joined[1])
        while True:
            onward = {join: self._forecast(p_case, join, self._join_ready(p_case, join, times), events, now, waits,
                                           onward=True) for join, times in arriving.items()}
            if None in onward.values():
                return None
            # no other join's way leads to this one: all its branches are in
            join = next((join for join in arriving
                         if not any(join in onward[other] for other in arriving if other != join)), None)
            if join is None:
                return None
            del arriving[join]
            if all(target in events for target in onward[join]):
                return None if arriving else onward[join]
            joined = _joined([onward[join]], events)
            if joined is None:
                return None
            arriving.setdefault(joined[0], []).append(joined[1])

    def _join_ready(self, p_case, join, times):
        """When the join goes on for the case p_case: at the latest of times, or of a branch already there, which
        got there at its task's end (Prosimos keeps the join's latest token time)."""
        arrived = self.sim_setup.bpmn_graph.last_datetime[join].get(p_case)
        return max(times + ([arrived.datetime] if arrived is not None else []))

    def _forecast(self, p_case, element_id, ready, events, now, waits, onward=False):
        """target -> (chance, average ready time) of the case p_case ready at element_id at ready; a target is a
        catch event in events or the first parallel join a way meets (onward: other than element_id). None if a way
        has anything but tasks, throw events, exclusive and parallel gateways, and end events. waits caches waits."""
        graph = self.sim_setup.bpmn_graph
        reached = {}
        ways = [(element_id, ready, 1.0, 0, onward)]  # passing: the way goes on from it, even from a join
        while ways:
            element_id, ready, chance, steps, passing = ways.pop()
            # a loop back is followed until its chance is negligible; one the case's data decides never ends: it is cut
            if chance < _NEGLIGIBLE or steps > _MOST_STEPS:
                continue
            info = graph.element_info[element_id]
            if element_id in events or (_is_join(info) and not passing):
                reached.setdefault(element_id, []).append((chance, ready))
                continue
            if info.type is BPMN.END_EVENT:
                continue
            if info.type not in (BPMN.TASK, BPMN.INTERMEDIATE_THROW_EVENT, BPMN.EXCLUSIVE_GATEWAY,
                                 BPMN.PARALLEL_GATEWAY):
                return None
            if info.type is BPMN.TASK:
                if element_id not in waits:
                    waits[element_id] = self._average_wait(element_id, now)
                ready = forecast_end(ready, self._staff_calendars(element_id),
                                     waits[element_id] + self._average_duration(element_id))
            if len(info.outgoing_flows) == 1:
                ways.append((graph.flow_arcs[info.outgoing_flows[0]][1], ready, chance, steps + 1, False))
            elif info.type is BPMN.EXCLUSIVE_GATEWAY:
                ways += [(graph.flow_arcs[flow][1], ready, chance * share, steps + 1, False)
                         for flow, share in self._branches(info, p_case)]
            elif info.type is BPMN.PARALLEL_GATEWAY:  # a split: on from its join when the longest branch gets there
                joined = _joined([self._forecast(p_case, graph.flow_arcs[flow][1], ready, events, now, waits)
                                  for flow in info.outgoing_flows], events)
                if joined is None:
                    return None
                join, ready = joined
                ways.append((join, ready, chance, steps + 1, True))
            else:
                return None
        return {target: _average(ways) for target, ways in reached.items()}

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


def _is_join(info):
    """Whether info is a parallel join. A task or throw event with several incoming flows never is one: Prosimos's
    loader puts a hidden exclusive join before it."""
    return info.type is BPMN.PARALLEL_GATEWAY and len(info.incoming_flows) > 1


def _joined(forecasts, events):
    """(join, ready) of branches whose forecasts all reach one join, not a catch event in events, for sure: ready
    when the latest gets there; None otherwise."""
    if None in forecasts:
        return None
    targets = {target for forecast in forecasts for target in forecast}
    if len(targets) != 1 or targets & events:
        return None
    [join] = targets
    if any(forecast.get(join, (0.0, None))[0] < _WHOLE for forecast in forecasts):
        return None
    return join, max(forecast[join][1] for forecast in forecasts)


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
