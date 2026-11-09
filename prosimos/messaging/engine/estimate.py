"""Estimates: when this engine's items could be ready for a quote's lines."""
import itertools
import math
from datetime import timedelta
from heapq import heappop, heappush

from pix_framework.statistics.distribution import DistributionType

from prosimos.control_flow_manager import BPMN


class Estimating:
    """Answers estimate(): per line of a quote, the times its matching items not yet promised would be ready."""

    def slots_for(self, quote_type, requester, lines, now):
        """line id -> the times, earliest first, the items matching each of lines (line id, attributes) from
        requester are expected at a catch event taking the quote type at now: slot k is when the expected count
        there reaches k. The open lines take the earliest expected items first; those are promised and skipped."""
        self._ensure_arrivals_generated()
        self._catch_up(now)
        free = self._quote_candidates(quote_type, now)
        for line in self._open_lines:
            _take_one(item for item in free if self._line_matches(line["type"], line["values"], item))
        return {line_id: _expected_slots(item for item in free if self._line_matches(
                    quote_type, dict(attributes, source=requester), item))
                for line_id, attributes in lines}

    def _quote_candidates(self, quote_type, now):
        """[time, case id, event id, chance] of each item not holding a line that may reach a catch event taking
        quote_type: waiting there, queued to reach it, or forecast from where its branches are queued or wait for a
        batch, in time order."""
        events = {point.event_id for points in self._consume_points.values() for point in points
                  if point.type == quote_type and not point.starts_case}
        waiting = [[parked_event.enabled_datetime, parked_event.p_case, parked_event.task_id, 1.0]
                   for parked_event in self.parked_events() if parked_event.task_id in events
                   and (parked_event.p_case, parked_event.task_id) not in self._set_aside]
        queued = []
        # case id -> (element id, ready time, how) of each of its branches: at the element, past it, or in its race
        ahead = {}
        for parked_event in self.parked_events():
            if parked_event.task_id not in events and parked_event.task_id not in self._race_gateway_of:
                # it goes on after its average wait there, or now if overdue
                ready = parked_event.enabled_datetime + timedelta(seconds=self._average_caught(parked_event.task_id))
                ahead.setdefault(parked_event.p_case, []).append((parked_event.task_id, max(ready, now), "past"))
        for event in self.events_queue.enabled_events.entry_finder:
            if event.task_id in events:
                if event.parked_event is None:
                    queued.append([event.enabled_datetime, event.p_case, event.task_id, 1.0])
            elif event.parked_event is not None:  # a case resumed by a message is past its catch event
                ahead.setdefault(event.p_case, []).append((event.task_id, event.enabled_datetime, "past"))
            elif event.task_id not in self._race_gateway_of:
                ahead.setdefault(event.p_case, []).append((event.task_id, event.enabled_datetime, "at"))
            elif event.armed_event is None:  # a branch of a race not started yet: the case is at its gateway
                start = (self._race_gateway_of[event.task_id], event.enabled_datetime, "at")
                if start not in ahead.setdefault(event.p_case, []):
                    ahead[event.p_case].append(start)
        for (p_case, gateway_id), race in self._races.items():
            ahead.setdefault(p_case, []).append((gateway_id, race.started, "race"))
        waits = {}
        for task_id, batch in self.sim_setup.bpmn_graph.batch_waiting_processes.items():
            for p_case, ready in batch.items():  # waiting for its batch: done after the average wait, or now
                calendars = self._staff_calendars(task_id)
                if task_id not in waits:
                    waits[task_id] = self._average_wait(task_id, now)
                started = max(forecast_end(ready.datetime, calendars, waits[task_id]), now)
                ahead.setdefault(p_case, []).append(
                    (task_id, forecast_end(started, calendars, self._average_duration(task_id)), "past"))
        for p_case, starts in ahead.items():
            reached = self._case_forecast(p_case, starts, events, now, waits)
            queued += [[ready, p_case, catch_event, chance] for catch_event, (chance, ready) in (reached or {}).items()]
        return sorted(waiting + queued, key=lambda item: item[:3])

    def _case_forecast(self, p_case, starts, events, now, waits):
        """catch event in events -> (chance, average ready time) of the case p_case from where its branches are,
        starts; None if not forecast. The branches of a split go on together from their join when the latest gets
        there, the innermost join first, which then is one branch of the next."""
        graph = self.sim_setup.bpmn_graph
        forecasts = []
        for element_id, ready, how in starts:
            outgoing = graph.element_info[element_id].outgoing_flows
            if how == "race":
                forecasts.append(self._race_forecast(p_case, element_id, ready, events, now, waits))
            elif how == "past" and len(outgoing) != 1:
                return None
            else:
                at = graph.flow_arcs[outgoing[0]][1] if how == "past" else element_id
                forecasts.append(self._forecast(p_case, at, ready, events, now, waits))
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
        catch event in events or the first parallel or inclusive join a way meets (onward: other than element_id).
        None if a way has anything but tasks, throw events, catch events outside races, gateways, and end events.
        'waits' caches waits."""
        graph = self.sim_setup.bpmn_graph
        reached = {}
        # path: the elements the way has passed; passing: the way goes on from element_id, even from a join
        ways = [(element_id, ready, 1.0, (), onward)]
        while ways:
            element_id, ready, chance, path, passing = ways.pop()
            if element_id in path:
                return None  # a loop back other than from an exclusive split along a plain way
            info = graph.element_info[element_id]
            if element_id in events or (_is_join(info) and not passing):
                reached.setdefault(element_id, []).append((chance, ready))
                continue
            if info.type is BPMN.END_EVENT:
                continue
            if info.type not in (BPMN.TASK, BPMN.INTERMEDIATE_THROW_EVENT, BPMN.INTERMEDIATE_EVENT,
                                 BPMN.EXCLUSIVE_GATEWAY, BPMN.PARALLEL_GATEWAY, BPMN.INCLUSIVE_GATEWAY,
                                 BPMN.EVENT_BASED_GATEWAY):
                return None
            ready = self._pass(element_id, ready, 1, now, waits)
            if ready is None:
                return None
            path += (element_id,)
            if len(info.outgoing_flows) == 1:
                ways.append((graph.flow_arcs[info.outgoing_flows[0]][1], ready, chance, path, False))
            elif info.type is BPMN.EVENT_BASED_GATEWAY:
                branches = self._race_ways(p_case, info.id, ready, events, now, waits)
                if branches is None:
                    return None
                ways += [(element, branch_ready, chance * share, path, False)
                         for element, branch_ready, share in branches]
            elif info.type is BPMN.EXCLUSIVE_GATEWAY:
                branches = self._exclusive_ways(p_case, info, ready, path, events, now, waits)
                if branches is None:
                    return None
                ways += [(element, branch_ready, chance * share, path, False)
                         for element, branch_ready, share in branches]
            elif info.type is BPMN.PARALLEL_GATEWAY:  # a split: on from its join when the longest branch gets there
                joined = _joined([self._forecast(p_case, graph.flow_arcs[flow][1], ready, events, now, waits)
                                  for flow in info.outgoing_flows], events)
                if joined is None:
                    return None
                join, ready = joined
                ways.append((join, ready, chance, path, True))
            else:  # an inclusive split: each combination of branches on from the join when its longest gets there
                combinations = self._combinations(info, p_case, path)
                forecasts = [self._forecast(p_case, graph.flow_arcs[flow][1], ready, events, now, waits)
                             for flow in info.outgoing_flows]
                branches = [_joined([forecast], events) for forecast in forecasts]
                if combinations is None or None in branches or len({join for join, _ in branches}) != 1:
                    return None
                join = branches[0][0]
                ways += [(join, max(branches[index][1] for index in chosen), chance * share, path, True)
                         for chosen, share in combinations]
        return {target: _average(ways) for target, ways in reached.items()}

    def _pass(self, element_id, ready, times, now, waits):
        """When the case ready at ready has passed element_id times over (a fraction for an expected count): a task
        takes its average wait and duration in working time, a catch event its average wait for a message, a timer
        or other timed event its drawn duration; None for an event without a duration."""
        graph = self.sim_setup.bpmn_graph
        info = graph.element_info[element_id]
        if info.type is BPMN.TASK:
            if element_id not in waits:
                waits[element_id] = self._average_wait(element_id, now)
            return forecast_end(ready, self._staff_calendars(element_id),
                                times * (waits[element_id] + self._average_duration(element_id)))
        if info.type is not BPMN.INTERMEDIATE_EVENT:
            return ready
        if element_id in self._consume_points:
            return ready + timedelta(seconds=times * self._average_caught(element_id))
        if element_id in graph.event_distribution:
            return ready + timedelta(seconds=times * _mean(graph.event_distribution[element_id]))
        return None

    def _exclusive_ways(self, p_case, info, ready, path, events, now, waits):
        """(element, ready time, chance) of each branch of the exclusive split info out of it, the case having passed
        path and got there at ready. A branch whose plain way leads back to an element on path loops: the way out
        is taken after the expected number of loop backs, each repeating the loop's elements, within the model's
        loop limit (Prosimos stops a case at a split it has passed that many times). None if it can't be followed."""
        graph = self.sim_setup.bpmn_graph
        ways_back = {}  # flow -> (the way back, its loop's elements) of each branch that loops
        for flow in info.outgoing_flows:
            element_id, way_back = graph.flow_arcs[flow][1], []
            # a plain way: elements with one way on, other than catch events in events and race branches
            while (element_id not in path and element_id not in events and element_id not in self._race_gateway_of
                   and len(graph.element_info[element_id].outgoing_flows) == 1 and element_id not in way_back):
                way_back.append(element_id)
                element_id = graph.flow_arcs[graph.element_info[element_id].outgoing_flows[0]][1]
            if element_id in path:
                ways_back[flow] = (way_back, path[path.index(element_id):])
        branches = self._branches(info, p_case, path)
        if branches is None:  # decided by data not known yet, with no history: the way out to events passes
            for flow in info.outgoing_flows:
                element_id = graph.flow_arcs[flow][1]
                if flow not in ways_back and (element_id in events or any(
                        target in events for target in self._forecast(p_case, element_id, ready, events, now, waits)
                        or {})):
                    return [(element_id, ready, 1.0)]
            return []
        out = [(graph.flow_arcs[flow][1], share) for flow, share in branches if flow not in ways_back]
        back = [(*ways_back[flow], share) for flow, share in branches if flow in ways_back]
        if not back:
            return [(element_id, ready, share) for element_id, share in out]
        looping = sum(share for _, _, share in back)
        passes = graph.gateway_execution_limit - graph.simulation_execution_stats.get_element_executions(
            p_case, info.id)  # including this one
        if looping >= _WHOLE or passes < 1:
            return []
        leaving = 1 - looping ** passes
        # loop backs before the way out within the passes left: a geometric count cut at passes - 1
        loops = looping * (1 - passes * looping ** (passes - 1) + (passes - 1) * looping ** passes) / (
            (1 - looping) * leaving)
        for way_back, loop, share in back:
            for element_id in way_back + list(loop):
                ready = self._pass(element_id, ready, loops * share / looping, now, waits)
                if ready is None:
                    return None
        return [(element_id, ready, share / (1 - looping) * leaving) for element_id, share in out]

    def _race_ways(self, p_case, gateway_id, ready, events, now, waits, race=None):
        """(element, ready time, chance) where each branch of the race at gateway_id goes on after it wins, the case
        having reached it at ready: from the history of its winners, each branch that has won with its share of the
        wins, after their average wait; with no history, the first branch on a way to a catch event in events wins
        at once. For race, the case's race still running at now: only past wins that took longer than it has waited
        count, a timer's at its known time; with none, a message branch wins now. None if it can't be followed."""
        graph = self.sim_setup.bpmn_graph
        if gateway_id not in self._race_branches:
            return None
        onward = {}  # branch -> where the case goes on after it: the branch itself if it is in events
        for branch in self._race_branches[gateway_id]:
            outgoing = graph.element_info[branch].outgoing_flows
            if branch not in events and len(outgoing) != 1:
                return None
            onward[branch] = branch if branch in events else graph.flow_arcs[outgoing[0]][1]
        due = {} if race is None else {  # the race's armed timers -> when they fire
            event.task_id: event.armed_event.enabled_datetime + timedelta(seconds=event.duration_sec)
            for event in self.events_queue.enabled_events.entry_finder if event.race is race and event.armed_event}
        waited = (now - ready).total_seconds()
        counted = []  # (element, ready time, wins counted)
        for branch, element_id in onward.items():
            won = [seconds for winner, seconds in self._race_wins.get(gateway_id, []) if winner == branch]
            if branch in due:
                if won:
                    counted.append((element_id, due[branch], len(won)))
                continue
            if race is not None:
                won = [seconds for seconds in won if seconds > waited]
            if won:
                counted.append((element_id, ready + timedelta(seconds=sum(won) / len(won)), len(won)))
        total = sum(wins for _, _, wins in counted)
        if total:
            return [(element_id, at, wins / total) for element_id, at, wins in counted]
        for branch, element_id in onward.items():
            if race is not None and branch not in self._consume_points:
                continue
            if branch in events or any(target in events for target in
                                       self._forecast(p_case, element_id, ready, events, now, waits) or {}):
                return [(element_id, ready if race is None else now, 1.0)]
        return []

    def _race_forecast(self, p_case, gateway_id, started, events, now, waits):
        """target -> (chance, average ready time) of the case p_case in its race at gateway_id since started, as
        _forecast gives; None if not forecast."""
        ways = self._race_ways(p_case, gateway_id, started, events, now, waits, self._races[(p_case, gateway_id)])
        if ways is None:
            return None
        reached = {}
        for element_id, ready, share in ways:
            forecast = self._forecast(p_case, element_id, ready, events, now, waits)
            if forecast is None:
                return None
            for target, (chance, at) in forecast.items():
                reached.setdefault(target, []).append((chance * share, at))
        return {target: _average(ways) for target, ways in reached.items()}

    def _combinations(self, info, p_case, path):
        """(indices of the outgoing flows taken, chance) of each combination the inclusive split info may take for
        the case p_case, having passed path, as Prosimos picks one: each flow with its own chance, a flow with a
        condition taken if the case's data passes it (by the flow's history if the data isn't known yet; taken with
        no history); if none is taken, the default flow, or one drawn by the chances. None if that draw is needed
        and the chances don't add up to 1 (Prosimos can't draw then)."""
        graph = self.sim_setup.bpmn_graph
        conditions = graph.gateway_conditions.get(info.id)
        choice = graph.element_probability[info.id]
        shares = dict(zip(choice.candidates_list, choice.probability_list))
        chances = [shares.get(flow, 0.0) for flow in info.outgoing_flows]
        if conditions is not None and conditions.candidates_list:
            attributes = graph.get_all_attributes(p_case)
            taken, passes = self._history(info.id)
            for flow, rule in zip(conditions.candidates_list, conditions.rules_list):
                if not rule:
                    continue
                if self._unknown(rule, p_case, path):
                    chance = taken.get(flow, 0) / passes if passes else 1.0
                else:
                    chance = 1.0 if rule.is_true(attributes) else 0.0
                chances[info.outgoing_flows.index(flow)] = chance
        none_taken = math.prod(1 - chance for chance in chances)
        default = conditions.get_default_path() if conditions is not None else None
        if none_taken > 1 - _WHOLE and default is None and abs(sum(shares.values()) - 1) > 1 - _WHOLE:
            return None
        drawn = [1.0 if flow == default else 0.0 for flow in info.outgoing_flows] if default else \
            [shares.get(flow, 0.0) for flow in info.outgoing_flows]
        combinations = []
        for chosen in itertools.product((False, True), repeat=len(chances)):
            taken = [index for index, take in enumerate(chosen) if take]
            share = math.prod(chance if take else 1 - chance for chance, take in zip(chances, chosen))
            share += none_taken * drawn[taken[0]] if len(taken) == 1 else 0
            if taken and share > 0:
                combinations.append((taken, share))
        return combinations

    def _history(self, gateway_id):
        """(flow -> how often the cases took it, how often they passed gateway_id) so far in this run, as of the
        last catch-up."""
        taken, passes = self._split_history.get(gateway_id, ({}, 0))
        return dict(taken), passes

    def _unknown(self, rule, p_case, path):
        """Whether the condition rule reads data the case p_case doesn't have yet: data it lacks, or data a task on
        path, still ahead of it, sets."""
        attributes = self.sim_setup.all_attributes
        set_ahead = {name for element_id in path
                     for source in (attributes.event_attributes, attributes.global_event_attributes)
                     for name in source.attributes.get(element_id, {})}
        known = self.sim_setup.bpmn_graph.get_all_attributes(p_case)
        return any(single.attribute not in known or single.attribute in set_ahead
                   for and_rule in rule.or_rule.or_rules for single in and_rule.and_rules)

    def _branches(self, info, p_case, path):
        """(flow, chance) of each branch of the exclusive split info for the case p_case, having passed path, as
        Prosimos picks one: the branches the case's data passes, or the default branch, or the probabilities. Data
        not known yet: the share of the cases that took each branch so far; None with no history yet."""
        graph = self.sim_setup.bpmn_graph
        choice = graph.element_probability[info.id]
        shares = dict(zip(choice.candidates_list, choice.probability_list))
        conditions = graph.gateway_conditions.get(info.id)
        if conditions is not None and conditions.candidates_list:
            if any(rule and self._unknown(rule, p_case, path) for rule in conditions.rules_list):
                taken, passes = self._history(info.id)
                return [(flow, taken[flow] / passes) for flow in info.outgoing_flows if taken.get(flow)] if passes \
                    else None
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

    def _task_started(self, event):
        """Keeps the task event until it starts, to count its wait then; only in an engine with a consume entry,
        the only kind that answers quotes."""
        if self._consume_points:
            heappush(self._not_yet, (event.started_datetime, next(self._not_yet_order), event.task_id,
                                     event))

    def _catch_up(self, now):
        """Counts the task starts and split passes due by now; Prosimos works them out ahead of the clock."""
        setup = self.sim_setup
        for _, split, at, flows in setup.bpmn_graph.split_passes:
            heappush(self._not_yet, (at, next(self._not_yet_order), split, flows))
        setup.bpmn_graph.split_passes.clear()
        while self._not_yet and self._not_yet[0][0] <= now:
            _, _, element_id, what = heappop(self._not_yet)
            if isinstance(what, list):  # the flows a split pass took
                history = self._split_history.setdefault(element_id, [{}, 0])
                for flow in what:
                    history[0][flow] = history[0].get(flow, 0) + 1
                history[1] += 1
            else:  # a task's wait to start, in working time on its performer's calendar
                calendar = setup.calendars_map[setup.resources_map[what.resource_id].calendar_id]
                waits = self._task_waits.setdefault(element_id, [0.0, 0])
                waits[0] += calendar.find_working_time(what.enabled_datetime, what.started_datetime)
                waits[1] += 1

    def _average_wait(self, task_id, now):
        """The average working seconds from ready to started of task_id, on each performer's calendar, over the
        times it started by now; 0 while none has."""
        self._catch_up(now)
        total, count = self._task_waits.get(task_id, (0, 0))
        return total / count if count else 0

    def _average_caught(self, event_id):
        """The average clock seconds items waited at the catch event event_id for a message so far; 0 while none
        has."""
        total, count = self._caught_waits.get(event_id, (0, 0))
        return total / count if count else 0

    def _average_duration(self, task_id):
        """The mean of task_id's duration distributions over the staff who may do it, in working seconds."""
        return self._model_average(task_id)[1]

    def _staff_calendars(self, task_id):
        return self._model_average(task_id)[0]

    def _model_average(self, task_id):
        """(the calendars of the staff who may do task_id, the mean of their duration distributions), worked out
        once."""
        if task_id not in self._model_averages:
            setup = self.sim_setup
            distributions = setup.task_resource[task_id]
            self._model_averages[task_id] = (
                [setup.calendars_map[setup.resources_map[resource_id].calendar_id] for resource_id in distributions],
                sum(_mean(distribution) for distribution in distributions.values()) / len(distributions))
        return self._model_averages[task_id]

    def _line_matches(self, quote_type, values, item):
        """Whether the item's catch event's consume entry for quote_type takes a line with values."""
        _, p_case, event_id, _ = item
        point = next((point for point in self._consume_points.get(event_id, []) if point.type == quote_type), None)
        return point is not None and self._accepts(point, values, p_case)


# an expected count summed from chances may fall a little short of a whole number
_WHOLE = 1 - 1e-6


def _is_join(info):
    """Whether info is a parallel or inclusive join. A task or throw event with several incoming flows never is one:
    Prosimos's loader puts a hidden exclusive join before it."""
    return info.type in (BPMN.PARALLEL_GATEWAY, BPMN.INCLUSIVE_GATEWAY) and len(info.incoming_flows) > 1


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
