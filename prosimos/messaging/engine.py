"""
The messaging part of the Prosimos engine (docs/engine-internals.md): the methods SimBPMEnv inherits to step one
event at a time, receive and publish messages, race the branches of event-based gateways, set case values from
messages, and tell the output what happened.
"""
import math
import numbers
from datetime import timedelta
from heapq import heappop, heappush

from prosimos.branch_condition_rules import BranchConditionRule
from prosimos.control_flow_manager import BPMN
from prosimos.execution_info import EnabledEvent
from prosimos.messaging.parser import is_whole_number, is_whole_number_of_at_least_one
from prosimos.warning_logger import warning_logger


class Race:
    """The branches of one event-based gateway, armed for one case: the first to happen wins."""

    def __init__(self, p_case, gateway_id):
        self.p_case = p_case
        self.gateway_id = gateway_id
        self.won = False
        self.wins_at_once = None  # the message branch with nothing to collect that wins at once, if any


class EngineMessaging:
    """The methods our messaging adds to SimBPMEnv, which inherits them as they are (same names, same self)."""

    def generate_all_arrival_events(self):
        sim_setup = self.sim_setup
        arrival_time = 0
        for p_case in range(0, sim_setup.total_num_cases or 0):
            enabled_datetime = self._update_initial_event_info(self.sim_setup, p_case, arrival_time)
            arrival_time += sim_setup.next_arrival_time(enabled_datetime)

    def generate_fixed_arrival_events(self, starting_times):
        p_case = 0
        for arrival_time in starting_times:
            self._update_initial_event_info(self.sim_setup, p_case, arrival_time)
            p_case += 1

    def _ensure_arrivals_generated(self, fixed_starting_times=None):
        """Generate the case arrivals exactly once, regardless of whether execute_full_process
        or next_event_time()/step() triggers it first."""
        if self._arrivals_generated:
            return
        if fixed_starting_times is None:
            self.generate_all_arrival_events()
        else:
            self.generate_fixed_arrival_events(fixed_starting_times)
        self._arrivals_generated = True

    def next_event_time(self):
        """Datetime of what this engine does next (the next event, or held effects that are due
        earlier), or None if it has nothing left to do."""
        self._close_collections()
        self._ensure_arrivals_generated()
        self._hold_passed_elements()
        next_event = self.events_queue.peek()
        times = [next_event.enabled_datetime] if next_event is not None else []
        if self._held:
            times.append(self._held[0][0])
        return min(times) if times else None

    def step(self):
        """Do what next_event_time() announced. If held effects are due no later than the next
        event, do all those due at that time (in order of time, case id, then order passed) and
        return None; they happened before anything that comes after them. Otherwise, execute exactly
        one event and return it. Returns None if there was nothing left to do."""
        self._close_collections()
        self._ensure_arrivals_generated()
        self._hold_passed_elements()
        next_event = self.events_queue.peek()
        if self._held and (next_event is None or self._held[0][0] <= next_event.enabled_datetime):
            self._release_held(self._held[0][0])
            return None
        if next_event is None:
            # the generator is only advanced while the queue has an event, so it never runs dry
            return None
        if self._event_generator is None:
            self._event_generator = self._full_process()
        executed = next(self._event_generator, None)
        self._hold_passed_elements()
        return executed

    def _hold_passed_elements(self):
        passed = self.sim_setup.bpmn_graph.passed_watched_elements
        for p_case, element_id, passed_at in passed:
            heappush(self._held, (passed_at.datetime, p_case, next(self._pass_order), element_id))
        passed.clear()

    def _release_held(self, due):
        while self._held and self._held[0][0] == due:
            _, p_case, _, element_id = heappop(self._held)
            for point in self._publish_points.get(element_id, []):
                for index in range(1, self._count(point, p_case) + 1):
                    self.outbox.append((point.type, self._message_attributes(point, p_case, index), p_case, element_id,
                                        point))

    def deliver(self, message_type, attributes, source, now):
        """Offers one message: 'claimed' if it resumed a waiting case (the one waiting longest, ties
        by case id) or, failing that, started a new case at the start event; 'discarded' if it can
        never match; 'pending' otherwise. After a claim, claimed_by lists who took it: the new case and
        the start event, or each waiting case it resumed and the catch event it waited at."""
        self._ensure_arrivals_generated()
        self._close_collections(before=now)
        self.claimed_by = []
        values = dict(attributes, source=source)
        # the oldest matching waiting case claims the message; its consume entry's capacity (min, max) says how
        # many waiting cases at the same entry claim it, oldest first: with fewer than min waiting, none does yet;
        # with min, up to max, without waiting to fill up
        claiming, takers, fewest, capacity = None, [], 1, 1
        for parked_event in self.parked_events():
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
        # a start event's condition looks only at the message, so it decides now: a new case, or never
        starting = next((point for point in points
                         if point.starts_case and self._condition_holds(point.condition, values, None)), None)
        if starting is not None:
            def start_values(p_case, case_values):
                self._apply_copy(starting, values, case_values)
                self._apply_append(starting, values, case_values)
                self._apply_aggregate(starting, values, p_case, case_values)

            p_case = self.create_case(now, start_values)
            self.claimed_by = [(p_case, starting.event_id, starting)]
            return "claimed"
        if any(self._could_match(point.condition, values) and self._has_same(point, values)
               for point in points if not point.starts_case):
            return "pending"
        return "discarded"

    def _claim(self, parked_event, point, values, now):
        """A waiting case claims one message: the values listed in point.copy are copied into it, and once it
        has collected all the messages it needs, it is resumed at now."""
        key = (parked_event.p_case, parked_event.task_id)
        case_values = self.sim_setup.bpmn_graph.all_attributes[parked_event.p_case]
        self._apply_copy(point, values, case_values)
        self._apply_append(point, values, case_values)
        aggregated = self._apply_aggregate(point, values, parked_event.p_case, case_values)
        self._copied_later.extend((parked_event.p_case, target, values[source], now)
                                  for target, source in point.copy if source in values)
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
        """Resumes a waiting case at now: its catch event is queued again, and completes at once. If the
        case was waiting in a race, its branch wins and the others are canceled. With keep_open, the case got its
        min but not its max: it keeps taking messages offered at now until the orchestrator's offers for that
        step are done (see _close_collections)."""
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
        """Closes the collections still open since their case got its min (all, or those opened before the time
        before). The orchestrator offers a step's messages one after another and then asks every engine what it
        does next, so the next call to next_event_time() or step() means the offers of that step are done; a
        message offered at a later time means so too."""
        for key, opened in list(self._closing.items()):
            if before is None or opened < before:
                self._close(key)

    def _accepts(self, point, values, p_case):
        """Whether the case p_case, waiting at point's event, takes the message: its condition holds and, with
        'same', the message has the values the first message the case claimed there set."""
        if not self._condition_holds(point.condition, values, p_case) or not self._has_same(point, values):
            return False
        collected = self._collected.get((p_case, point.event_id))
        first = collected[3] if collected is not None else None
        return first is None or all(values[name] == value for name, value in first.items())

    def _has_same(self, point, values):
        """Whether the message has every attribute point.same lists; a message without one never matches there,
        with one warning per event and attribute."""
        missing = [name for name in point.same if name not in values]
        for name in missing:
            if (point.event_id, name) not in self._warned_missing_same:
                self._warned_missing_same.add((point.event_id, name))
                warning_logger.add_warning(
                    f"{point.type} message offered at {point.event_id} has no {name}, which 'same' compares; it "
                    f"doesn't match there")
        return not missing

    def _capacity(self, point, values):
        """(min, max) waiting cases this message resumes at point, each fixed or read from the message. A min read
        from the message that is missing or not a whole number of at least 1 counts as 1; a max that is missing or
        not a whole number of at least min counts as min. Each comes with one warning per event and attribute."""
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
        """How many messages a case arriving at a catch event must claim there: the entries' fixed collect, or
        the case attribute holding it, read now; one that isn't a whole number of at least 0 counts as 1, with
        one warning per event and attribute."""
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
        """How many messages a case arriving at a catch event may claim there in all, at least needed (its min):
        the entry's fixed max, no limit (math.inf), or the case attribute holding it, read now; without a max, the
        min. One that isn't a whole number of at least needed counts as needed, with one warning per event and
        attribute."""
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

    def _first_listed_branch(self, race, message_type, values):
        """The parked branch of race, and its consume entry, that takes a message accepted by more than one
        of the race's branches: the one whose entry comes first in the JSON, with one warning per gateway."""
        accepting = []
        for branch in self._race_branches[race.gateway_id]:
            parked_event = self._parked_events.get((race.p_case, branch))
            if parked_event is None:
                continue  # a branch that doesn't wait for a message, e.g. a timer
            point = next((point for point in self._consume_points[branch] if point.type == message_type
                          and self._accepts(point, values, race.p_case)), None)
            if point is not None:
                accepting.append((parked_event, point))
        parked_event, point = min(accepting, key=lambda accepted: self._consume_order[id(accepted[1])])
        if len(accepting) > 1 and race.gateway_id not in self._warned_overlapping_races:
            self._warned_overlapping_races.add(race.gateway_id)
            branches = ", ".join(sorted(accepted[0].task_id for accepted in accepting))
            warning_logger.add_warning(
                f"a {message_type} message matches several branches of the race at {race.gateway_id} ({branches}); "
                f"it goes to {point.event_id}, whose 'consume' entry comes first")
        return parked_event, point

    def _race_step(self, c_event):
        """A branch of a race comes off the queue. Returns True if nothing more is to be done with it now."""
        if c_event.race is not None:  # a timer branch due to fire
            if c_event.race.won:
                return True  # canceled: another branch won
            incoming = self.sim_setup.bpmn_graph.element_info[c_event.task_id].incoming_flows
            if not any(c_event.p_state.has_token(flow) for flow in incoming):
                # the case left the race without it, e.g. ended by a terminate end event on another branch
                c_event.race.won = True
                self._races.pop((c_event.race.p_case, c_event.race.gateway_id), None)
                return True
            self._win_race(c_event.race, c_event.task_id, c_event.p_state)
            return False  # it completes now; execute_event logs it from when it was armed
        # armed when the case reached the gateway; the branches of one race share one Race
        incoming = self.sim_setup.bpmn_graph.element_info[c_event.task_id].incoming_flows
        if not any(c_event.p_state.has_token(flow) for flow in incoming):
            return True  # another branch already won at this instant, e.g. one with nothing to collect
        key = (c_event.p_case, self._race_gateway_of[c_event.task_id])
        if key not in self._races:  # the case's first branch to come off the queue starts the race
            self._races[key] = Race(*key)
            self._races[key].wins_at_once = self._branch_with_nothing_to_collect(*key)
        c_event.race = self._races[key]
        if c_event.race.wins_at_once not in (None, c_event.task_id):
            return True  # a branch with nothing to collect wins at once; this one is never armed
        if c_event.task_id in self._consume_points:
            return False  # the message branch parks, like any catch event waiting for a message
        # a message at exactly the timer's time wins: other engines may still publish one for that instant,
        # so the timer is due one microsecond later and every message of that instant is offered first
        [delay] = self.sim_setup.bpmn_graph.event_duration(c_event.task_id)
        due = EnabledEvent(c_event.p_case, c_event.p_state, c_event.task_id, c_event.enabled_at + delay + 1e-6,
                           c_event.enabled_datetime + timedelta(seconds=delay, microseconds=1), is_inter_event=True)
        due.race, due.armed_event, due.duration_sec = c_event.race, c_event, delay
        self.events_queue.append_event(due)
        return True

    def _branch_with_nothing_to_collect(self, p_case, gateway_id):
        """The message branch of the race that wins as soon as the case reaches the gateway: of the branches
        with nothing to collect, the one whose consume entry comes first in the JSON, with one warning per
        gateway if there are several; None if every message branch waits for something."""
        branches = [branch for branch in self._race_branches[gateway_id]
                    if branch in self._consume_points and self._needed(p_case, branch) == 0]
        if not branches:
            return None
        # a branch with collect has exactly one consume entry
        branches.sort(key=lambda branch: self._consume_order[id(self._consume_points[branch][0])])
        if len(branches) > 1 and gateway_id not in self._warned_races_won_at_once:
            self._warned_races_won_at_once.add(gateway_id)
            warning_logger.add_warning(
                f"several branches of the race at {gateway_id} have nothing to collect "
                f"({', '.join(sorted(branches))}); {branches[0]}, whose 'consume' entry comes first, wins")
        return branches[0]

    def _win_race(self, race, winner, p_state):
        """The branch winner wins race: every other branch is canceled. Its token is taken off the flow into
        it, a parked case's waiting record is dropped, and a queued timer is skipped when it comes off."""
        race.won = True
        del self._races[(race.p_case, race.gateway_id)]
        graph = self.sim_setup.bpmn_graph
        for branch in self._race_branches[race.gateway_id]:
            if branch == winner:
                continue
            for flow in graph.element_info[branch].incoming_flows:
                if p_state.tokens[flow] > 0:
                    p_state.tokens[flow] -= 1
                    if p_state.tokens[flow] == 0:
                        p_state.state_mask &= ~graph.arcs_bitset[flow]
            self._parked_events.pop((race.p_case, branch), None)
            self._collected.pop((race.p_case, branch), None)
        self._tokens_moved(race.p_case, p_state)

    def _tokens_moved(self, p_case, p_state):
        """Keeps _unfinished up to date after the tokens of the case p_case moved."""
        if any(p_state.tokens.values()):
            self._unfinished[p_case] = None
        else:
            self._unfinished.pop(p_case, None)

    def create_case(self, now, set_values=None):
        """Starts a new case at now, as a planned arrival would (case attributes drawn as usual), and
        returns its id: the next consecutive one, since the log's trace list is indexed by case id.
        set_values(p_case, case_values), if given, changes the drawn values before the case starts."""
        p_case = len(self.log_info.trace_list)
        case_values = self.sim_setup.case_attributes.get_values_calculated()
        case_values.update(self._starting_values())
        if set_values is not None:
            set_values(p_case, case_values)
        self.case_prioritisation.all_case_attributes[p_case] = case_values
        self.case_prioritisation.all_case_priorities[p_case] = \
            self.sim_setup.prioritisation_rules.get_priority(case_values)
        self.sim_setup.bpmn_graph.all_attributes[p_case] = case_values
        self._update_initial_event_info(self.sim_setup, p_case, self.simulation_at_from_datetime(now))
        return p_case

    def _starting_values(self):
        """What a case's appended and aggregated attributes start with: an empty list, and an aggregate's initial,
        else 0 for a sum or count (a min or max without initial has no value until the first message)."""
        values = {target: [] for target in self._appended}
        values.update((target, initial if initial is not None else 0) for target, (function, initial)
                      in self._aggregated.items() if initial is not None or function in ("sum", "count"))
        return values

    def _apply_aggregate(self, point, values, p_case, case_values):
        """Adds the message attributes listed in point.aggregate to the values aggregated so far for the case, and
        writes each function over them into the case's values; the case attributes it wrote. A message without one
        of them, or for sum, min and max with one that isn't a number, is skipped, with one warning per event and
        attribute."""
        written = []
        for target, function, source_name, initial in point.aggregate:
            if source_name not in values:
                if (point.event_id, source_name) not in self._warned_missing_aggregates:
                    self._warned_missing_aggregates.add((point.event_id, source_name))
                    warning_logger.add_warning(
                        f"{point.type} message accepted at {point.event_id} has no {source_name} to aggregate into "
                        f"{target}; {target} is left unchanged")
                continue
            value = values[source_name]
            if function != "count" and not _is_number(value):
                if (point.event_id, source_name) not in self._warned_not_numbers:
                    self._warned_not_numbers.add((point.event_id, source_name))
                    warning_logger.add_warning(
                        f"{point.type} message accepted at {point.event_id} has {source_name} {value!r}, which isn't "
                        f"a number to {function} into {target}; {target} is left unchanged")
                continue
            aggregated = self._aggregated_values.setdefault((p_case, target), [])
            aggregated.append(value)
            case_values[target] = _aggregated(function, aggregated, initial)
            written.append(target)
        return written

    def _apply_copy(self, point, values, case_values):
        """Writes the message attributes listed in point.copy into the case's values. A message without
        one of them leaves that case attribute unchanged, with one warning per event and attribute."""
        for target, source_name in point.copy:
            if source_name in values:
                case_values[target] = values[source_name]
            elif (point.event_id, source_name) not in self._warned_missing_copies:
                self._warned_missing_copies.add((point.event_id, source_name))
                warning_logger.add_warning(
                    f"{point.type} message accepted at {point.event_id} has no {source_name} to copy into "
                    f"{target}; {target} is left unchanged")

    def _apply_append(self, point, values, case_values):
        """Adds the message attributes listed in point.append to the lists in the case's values, as new lists, so
        the values recorded earlier keep their contents. A message without one of them adds nothing, with one
        warning per event and attribute."""
        for target, source_name in point.append:
            if source_name in values:
                case_values[target] = [*case_values.get(target, []), values[source_name]]
            elif (point.event_id, source_name) not in self._warned_missing_appends:
                self._warned_missing_appends.add((point.event_id, source_name))
                warning_logger.add_warning(
                    f"{point.type} message accepted at {point.event_id} has no {source_name} to append to "
                    f"{target}; nothing is added to {target}")

    def _condition_holds(self, condition, values, p_case):
        if condition is None:
            return True
        return any(all(self._term_holds(term, values, p_case) for term in alternative) for alternative in condition)

    def _term_holds(self, term, values, p_case):
        if term.attribute not in values:
            return False
        if term.object_attribute is None:
            compared_with = term.value
        elif term.object_attribute == "object_id":
            compared_with = self.object_id(p_case)
        else:
            case_values = self.sim_setup.bpmn_graph.get_all_attributes(p_case)
            if term.object_attribute not in case_values:
                return False
            compared_with = case_values[term.object_attribute]
        # the same comparison as branch rules
        return BranchConditionRule(term.attribute, term.comparison, compared_with).is_rule_true(
            {term.attribute: values[term.attribute]})

    def _could_match(self, condition, values):
        """Whether some case that hasn't finished, or (in a process started by messages) one that doesn't
        exist yet, could still accept the message. Only terms on fixed values and on object_id are decided
        now; other case attributes can still change."""
        unfinished = self._unfinished
        future_cases = self.sim_setup.messaging.started_by_messages
        if condition is None:
            return bool(unfinished) or future_cases
        for alternative in condition:
            decided = [term for term in alternative if term.object_attribute in (None, "object_id")]
            if any(all(self._term_holds(term, values, p_case) for term in decided) for p_case in unfinished):
                return True
            if future_cases and all(self._term_could_hold_for_a_future_case(term, values) for term in decided):
                return True
        return False

    def _term_could_hold_for_a_future_case(self, term, values):
        """For a case that a start message may still create. Its object_id isn't known yet, except that its
        number is at least the number of cases so far: an '=' term holds only for such an id; any other
        comparison might hold."""
        if term.object_attribute is None:
            return self._term_holds(term, values, None)
        if term.attribute not in values:
            return False
        if term.comparison != "=":
            return True
        prefix, _, number = str(values[term.attribute]).rpartition("-")
        return (prefix == self.id_prefix and number.isdigit() and str(int(number)) == number
                and int(number) >= len(self.log_info.trace_list))

    def _count(self, point, p_case):
        """How many messages point publishes for a case passing it now: 1 without count, else its fixed count
        or the case attribute holding it, read now; one that isn't a whole number of at least 0 counts as 1,
        with one warning per event and attribute. Over a list, one per entry; a missing or empty list publishes
        none, with one warning per event."""
        if not point.counted:
            return 1
        if point.count_list is not None:
            return len(self._count_list(point, p_case))
        if point.count_attribute is None:
            return point.count
        count = self.sim_setup.bpmn_graph.get_all_attributes(p_case).get(point.count_attribute)
        if is_whole_number(count, minimum=0):
            return int(count)
        if (point.event_id, point.count_attribute) not in self._warned_counts:
            self._warned_counts.add((point.event_id, point.count_attribute))
            warning_logger.add_warning(
                f"case {p_case} passes {point.event_id} with no valid {point.count_attribute} (got {count!r}); "
                f"it publishes one {point.type} message")
        return 1

    def _count_list(self, point, p_case):
        entries = self.sim_setup.bpmn_graph.get_all_attributes(p_case).get(point.count_list)
        if isinstance(entries, (list, tuple)) and entries:
            return entries
        if point.event_id not in self._warned_count_lists:
            self._warned_count_lists.add(point.event_id)
            kind = "an empty" if isinstance(entries, (list, tuple)) else "no"
            warning_logger.add_warning(
                f"case {p_case} passes {point.event_id} with {kind} list {point.count_list} (got {entries!r}); "
                f"it publishes no {point.type} message")
        return ()

    def object_id(self, p_case):
        """How a case's object is named in messages, both when publishing and in conditions: e.g. order-7."""
        return f"{self.id_prefix}-{p_case}"

    def _message_attributes(self, point, p_case, index):
        """The listed attributes, copied from the case's current values; index is the message's number among
        the ones point publishes for this pass."""
        values = self.sim_setup.bpmn_graph.get_all_attributes(p_case)
        attributes = dict()
        for name in point.attributes:
            if name == "object_id":
                attributes[name] = self.object_id(p_case)
            elif name == "index":  # only listed on an entry with count (checked at load time)
                attributes[name] = index
            elif name in values:
                attributes[name] = values[name]
            else:
                # declared (checked at load time) but not given a value on this case's path yet
                attributes[name] = None
                if (point.event_id, name) not in self._warned_missing_values:
                    self._warned_missing_values.add((point.event_id, name))
                    warning_logger.add_warning(
                        f"Attribute {name} has no value when case {p_case} passes {point.event_id}; "
                        f"its {point.type} message carries None")
        if point.count_list is not None:  # only called for an index within the list
            attributes[point.count_as] = values[point.count_list][index - 1]
        return attributes

    def tasks_next_to(self, element_id, forward):
        """The tasks the token reaches first after element_id (forward) or came from last before it: the flows
        are followed through gateways and events, and stop at tasks. Task id -> whether some path to it passes
        a parallel or inclusive gateway that joins (or splits, forward) several flows, i.e. whether it may run
        alongside the others."""
        key = (element_id, forward)
        if key not in self._tasks_next_to:
            graph = self.sim_setup.bpmn_graph
            tasks, seen, to_visit = {}, {(element_id, False)}, [(element_id, False)]
            while to_visit:
                current, concurrent = to_visit.pop()
                element = graph.element_info[current]
                if element.type in (BPMN.PARALLEL_GATEWAY, BPMN.INCLUSIVE_GATEWAY) and current != element_id:
                    concurrent = concurrent or (element.is_split() if forward else element.is_join())
                for flow in element.outgoing_flows if forward else element.incoming_flows:
                    neighbour = graph.flow_arcs[flow][1 if forward else 0]
                    if (neighbour, concurrent) in seen:
                        continue
                    seen.add((neighbour, concurrent))
                    if graph.element_info[neighbour].type is BPMN.TASK:
                        tasks[neighbour] = tasks.get(neighbour, False) or concurrent
                    else:
                        to_visit.append((neighbour, concurrent))
            self._tasks_next_to[key] = tasks
        return self._tasks_next_to[key]

    def case_attribute_history(self):
        """Case id -> its case attributes over time, as (attribute, value, time): every value the case had
        when it was created, then every value a message copied into it later, in the order they came."""
        history = {p_case: [(name, value, createdAt) for name, value in values.items()]
                   for p_case, (createdAt, values) in self._case_created.items()}
        for p_case, name, value, copiedAt in self._copied_later:
            history[p_case].append((name, value, copiedAt))
        return history

    def case_creation_times(self):
        """Case id -> when it was created."""
        return {p_case: createdAt for p_case, (createdAt, _) in self._case_created.items()}

    def only_started_values(self):
        """Case id -> the aggregated attributes whose value when it was created is only their start, 0 or their
        initial: no message had written them yet (a message that starts a case writes its own aggregates into the
        new case)."""
        return dict(self._only_started)

    def stalled_at(self, parked_event):
        """Where a parked case is reported stalled: the race gateway if it waits in a race, else its event."""
        return self._race_gateway_of.get(parked_event.task_id, parked_event.task_id)

    def stalled_waits(self):
        """The cases still waiting for a message, once per case and place where they are reported stalled:
        (the parked event, the place, the message types accepted there), the one waiting longest first.
        A case waiting in a race is reported once, at the gateway, with the types of all its message
        branches."""
        self._close_collections()
        waits = dict()  # (case id, place) -> (parked event, place, message types)
        for parked_event in self.parked_events():
            place = self.stalled_at(parked_event)
            _, _, types = waits.setdefault((parked_event.p_case, place), (parked_event, place, set()))
            types.update(self.waiting_for(parked_event.task_id))
        return [(parked_event, place, sorted(types)) for parked_event, place, types in waits.values()]

    def parked_events(self):
        """The queued catch events (EnabledEvents) at which cases are parked, waiting for a message: one
        per waiting case and event, the one waiting longest first (ties by case id)."""
        for key, parked_event in list(self._parked_events.items()):
            if not any(parked_event.p_state.has_token(flow) for flow in self.sim_setup.bpmn_graph.element_info[parked_event.task_id].incoming_flows):
                del self._parked_events[key]  # the case moved on without it (e.g. a terminate end event)
                del self._collected[key]
                self._closing.pop(key, None)
        return [self._parked_events[key] for key in sorted(self._parked_events, key=lambda key: (self._parked_events[key].enabled_datetime, key))]

    def waiting_for(self, event_id):
        """The message types a case waiting at event_id accepts."""
        return sorted({point.type for point in self._consume_points[event_id]})


def _is_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _aggregated(function, values, initial=None):
    """function over values: count, sum, min or max, and over initial, if given, a number already among them: a
    count adds the values to it, a sum, min or max takes it as one more value. A sum of whole numbers is a whole
    number; any other is rounded once, at the end (math.fsum), so it doesn't depend on the order of the messages:
    1.2, 0.8 and 2.1 add up to 4.1 in any order, where adding them one by one may give 4.1000000000000005."""
    if function == "count":
        return len(values) + (initial if initial is not None else 0)
    if initial is not None:
        values = [initial, *values]
    if function == "sum":
        return sum(values) if all(isinstance(value, numbers.Integral) for value in values) else math.fsum(values)
    return min(values) if function == "min" else max(values)
