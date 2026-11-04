"""The messaging methods SimBPMEnv inherits (docs/engine-internals.md, docs/messaging.md)."""
import math
import numbers
from datetime import timedelta
from heapq import heapify, heappop, heappush

from prosimos.branch_condition_rules import BranchConditionRule
from prosimos.control_flow_manager import BPMN
from prosimos.execution_info import EnabledEvent
from prosimos.messaging.parser import is_number, is_whole_number, is_whole_number_of_at_least_one
from prosimos.warning_logger import warning_logger


class Race:
    """The branches of one event-based gateway, armed for one case: the first to happen wins."""

    def __init__(self, p_case, gateway_id):
        self.p_case = p_case
        self.gateway_id = gateway_id
        self.won = False
        self.wins_at_once = None  # the message branch with nothing to collect that wins at once, if any


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
        """Generates the case arrivals once, whichever of execute_full_process and step() asks first."""
        if self._arrivals_generated:
            return
        if fixed_starting_times is None:
            self.generate_all_arrival_events()
        else:
            self.generate_fixed_arrival_events(fixed_starting_times)
        self._arrivals_generated = True

    def next_event_time(self):
        """When this engine acts next: its next event, or a held effect due earlier; None if nothing is left."""
        self._close_collections()
        self._ensure_arrivals_generated()
        self._hold_passed_elements()
        next_event = self.events_queue.peek()
        times = [next_event.enabled_datetime] if next_event is not None else []
        if self._held:
            times.append(self._held[0][0])
        return min(times) if times else None

    def step(self):
        """Does what next_event_time() announced: the held effects then due (returns None), or one event."""
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
        # a terminate end event ends its case at its time, which may still be ahead: held until then (element None)
        for p_case, when in self.sim_setup.bpmn_graph.terminated.items():
            if p_case not in self._terminations_held:
                self._terminations_held.add(p_case)
                heappush(self._held, (when, p_case, next(self._pass_order), None))

    def _release_held(self, due):
        while self._held and self._held[0][0] == due:
            _, p_case, _, element_id = heappop(self._held)
            if isinstance(element_id, Pile):  # its deadline
                self._deadline_passed(element_id)
                continue
            if element_id is None:
                self._end_terminated(p_case)
            if self.sim_setup.bpmn_graph.terminated_by(p_case, due):
                continue
            for point in self._publish_points.get(element_id, []):
                if point.fans_in:
                    self._hand_in(point, p_case, element_id, due)
                    continue
                for index in range(1, self._fan_out(point, p_case) + 1):
                    self.outbox.append((point.type, self._message_attributes(point, p_case, index), ((p_case, due),),
                                        element_id, point))

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

    def _first_listed_branch(self, race, message_type, values):
        """The parked branch and entry of a race that take a message several branches accept: the first listed."""
        accepting = []
        for branch in self._race_branches[race.gateway_id]:
            parked_event = self._parked_events.get((race.p_case, branch))
            if parked_event is None:
                continue
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
                return True
            incoming = self.sim_setup.bpmn_graph.element_info[c_event.task_id].incoming_flows
            if not any(c_event.p_state.has_token(flow) for flow in incoming):
                # the case left the race without it, e.g. ended by a terminate end event on another branch
                c_event.race.won = True
                self._races.pop((c_event.race.p_case, c_event.race.gateway_id), None)
                return True
            self._win_race(c_event.race, c_event.task_id, c_event.p_state)
            return False  # it completes now; execute_event logs it from when it was armed
        incoming = self.sim_setup.bpmn_graph.element_info[c_event.task_id].incoming_flows
        if not any(c_event.p_state.has_token(flow) for flow in incoming):
            return True  # another branch already won at this instant, e.g. one with nothing to collect
        key = (c_event.p_case, self._race_gateway_of[c_event.task_id])
        if key not in self._races:
            self._races[key] = Race(*key)
            self._races[key].wins_at_once = self._branch_with_nothing_to_collect(*key)
        c_event.race = self._races[key]
        if c_event.race.wins_at_once not in (None, c_event.task_id):
            return True  # a branch with nothing to collect wins at once; this one is never armed
        if c_event.task_id in self._consume_points:
            return False
        # due a microsecond late, so a message other engines publish for the same instant wins
        [delay] = self.sim_setup.bpmn_graph.event_duration(c_event.task_id)
        due = EnabledEvent(c_event.p_case, c_event.p_state, c_event.task_id, c_event.enabled_at + delay + 1e-6,
                           c_event.enabled_datetime + timedelta(seconds=delay, microseconds=1), is_inter_event=True)
        due.race, due.armed_event, due.duration_sec = c_event.race, c_event, delay
        self.events_queue.append_event(due)
        return True

    def _branch_with_nothing_to_collect(self, p_case, gateway_id):
        """The branch with nothing to collect that wins at once, the first listed; None if there is none."""
        branches = [branch for branch in self._race_branches[gateway_id]
                    if branch in self._consume_points and self._needed(p_case, branch) == 0]
        if not branches:
            return None
        branches.sort(key=lambda branch: self._consume_order[id(self._consume_points[branch][0])])
        if len(branches) > 1 and gateway_id not in self._warned_races_won_at_once:
            self._warned_races_won_at_once.add(gateway_id)
            warning_logger.add_warning(
                f"several branches of the race at {gateway_id} have nothing to collect "
                f"({', '.join(sorted(branches))}); {branches[0]}, whose 'consume' entry comes first, wins")
        return branches[0]

    def _win_race(self, race, winner, p_state):
        """Cancels every branch of race but winner: their tokens are taken off, and a queued timer is skipped later."""
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

    def _end_terminated(self, p_case):
        """Ends a case a terminate end event ended: its tokens and races go; parked_events() then drops its waits."""
        p_state = self.all_process_states[p_case]
        p_state.remove_all_tokens_on_terminate()
        self._tokens_moved(p_case, p_state)
        for key in [key for key in self._races if key[0] == p_case]:
            self._races.pop(key).won = True

    def _tokens_moved(self, p_case, p_state):
        """Keeps _unfinished up to date after the tokens of the case p_case moved."""
        if any(p_state.tokens.values()):
            self._unfinished[p_case] = None
        else:
            self._unfinished.pop(p_case, None)

    def create_case(self, now, set_values=None, start_event=None):
        """Starts a new case at now, as a planned arrival would, at start_event (by default the scheduled start), and
        returns its id. set_values(p_case, case_values), if given, changes the drawn values first."""
        p_case = len(self.log_info.trace_list)
        case_values = self.sim_setup.case_attributes.get_values_calculated()
        case_values.update(self._starting_values())
        if set_values is not None:
            set_values(p_case, case_values)
        self.case_prioritisation.all_case_attributes[p_case] = case_values
        self.case_prioritisation.all_case_priorities[p_case] = \
            self.sim_setup.prioritisation_rules.get_priority(case_values)
        self.sim_setup.bpmn_graph.all_attributes[p_case] = case_values
        self._update_initial_event_info(self.sim_setup, p_case, self.simulation_at_from_datetime(now), start_event)
        return p_case

    def receive_catalogue_values(self, received):
        """Takes the catalogue values for the 'with' of references, once, when the engine is built: (attribute,
        source attribute, values, function, initial, catalogue); values.at(id, when) is an object's value then."""
        self._received = list(received)

    def _apply_received(self, case_values, when, sources=None):
        """Gives a case the values 'with' brings, as valid at 'when', and returns the attributes it set: all for a
        new case, at a claim only those whose source is in sources."""
        given = []
        for name, source, values, function, initial, catalogue in self._received:
            if sources is not None and source not in sources:
                continue
            value = case_values.get(source)
            named = value if isinstance(value, list) else [value]
            if value is None:
                continue
            unknown = [object_id for object_id in named if object_id not in values]
            if unknown:
                self._warn_unknown_received(source, catalogue, value, unknown)
                continue
            if function is None:
                if not isinstance(value, list):  # a plain copy names one object
                    case_values[name] = values.at(value, when)
                    given.append(name)
            elif named or function not in ("min", "max") or initial is not None:
                case_values[name] = aggregate_of(function, [values.at(object_id, when) for object_id in named], initial)
                given.append(name)
        return given

    def _warn_unknown_received(self, source, catalogue, value, unknown):
        if (source, catalogue) in self._warned_unknown_received:
            return
        self._warned_unknown_received.add((source, catalogue))
        unset = [name for name, other_source, *_, other_catalogue in self._received
                 if (other_source, other_catalogue) == (source, catalogue)]
        named = f"whose entries {unknown} are" if isinstance(value, list) else "which is"
        warning_logger.add_warning(f"object attribute {source} has the value {value!r}, {named} not in catalogue "
                                   f"{catalogue}; 'with' leaves {', '.join(unset)} unset")

    def _started_received(self, case_values):
        """The 'with' values over a list still empty, which a new case only starts with (see only_started_values)."""
        return tuple(name for name, source, *_ in self._received
                     if source in self._appended and case_values.get(source) == [] and name in case_values)

    def _starting_values(self):
        """What appended and aggregated attributes start with: [], or an aggregate's initial, else 0 for a sum or
        count."""
        values = {target: [] for target in self._appended}
        values.update((target, initial if initial is not None else 0) for target, (function, initial)
                      in self._aggregated.items() if initial is not None or function in ("sum", "count"))
        return values

    def _apply_aggregate(self, point, values, p_case, case_values):
        """Adds the message's values to the case's aggregates; returns the attributes it wrote."""
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
            if function != "count" and not is_number(value):
                if (point.event_id, source_name) not in self._warned_not_numbers:
                    self._warned_not_numbers.add((point.event_id, source_name))
                    warning_logger.add_warning(
                        f"{point.type} message accepted at {point.event_id} has {source_name} {value!r}, which isn't "
                        f"a number to {function} into {target}; {target} is left unchanged")
                continue
            aggregated = self._aggregated_values.setdefault((p_case, target), [])
            aggregated.append(value)
            case_values[target] = aggregate_of(function, aggregated, initial)
            written.append(target)
        return written

    def _apply_copy(self, point, values, case_values):
        for target, source_name in point.copy:
            if source_name in values:
                case_values[target] = values[source_name]
            elif (point.event_id, source_name) not in self._warned_missing_copies:
                self._warned_missing_copies.add((point.event_id, source_name))
                warning_logger.add_warning(
                    f"{point.type} message accepted at {point.event_id} has no {source_name} to copy into "
                    f"{target}; {target} is left unchanged")

    def _apply_append(self, point, values, case_values):
        """Appends the message's values to the case's lists, as new lists so that earlier records keep theirs."""
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
        """Whether an unfinished case, or one a start message may still create, could accept the message. Only
        terms on fixed values and object_id are decided now; other case attributes can still change."""
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
        """For a case a start message may still create: '=' on object_id holds only for an id numbered at least
        the cases so far; any other comparison might hold."""
        if term.object_attribute is None:
            return self._term_holds(term, values, None)
        if term.attribute not in values:
            return False
        if term.comparison != "=":
            return True
        prefix, _, number = str(values[term.attribute]).rpartition("-")
        return (prefix == self.id_prefix and number.isdigit() and str(int(number)) == number
                and int(number) >= len(self.log_info.trace_list))

    def _fan_out(self, point, p_case):
        """How many messages point publishes for a case now: 1, its fan_out, or one per entry of its list."""
        if not point.fans_out:
            return 1
        if point.fan_out_list is not None:
            return len(self._fan_out_list(point, p_case))
        if point.fan_out_attribute is None:
            return point.fan_out
        number = self.sim_setup.bpmn_graph.get_all_attributes(p_case).get(point.fan_out_attribute)
        if is_whole_number(number, minimum=0):
            return int(number)
        if (point.event_id, point.fan_out_attribute) not in self._warned_fan_outs:
            self._warned_fan_outs.add((point.event_id, point.fan_out_attribute))
            warning_logger.add_warning(
                f"case {p_case} passes {point.event_id} with no valid {point.fan_out_attribute} (got {number!r}); "
                f"it publishes one {point.type} message")
        return 1

    def _fan_out_list(self, point, p_case):
        entries = self.sim_setup.bpmn_graph.get_all_attributes(p_case).get(point.fan_out_list)
        if isinstance(entries, (list, tuple)) and entries:
            return entries
        if point.event_id not in self._warned_fan_out_lists:
            self._warned_fan_out_lists.add(point.event_id)
            kind = "an empty" if isinstance(entries, (list, tuple)) else "no"
            warning_logger.add_warning(
                f"case {p_case} passes {point.event_id} with {kind} list {point.fan_out_list} (got {entries!r}); "
                f"it publishes no {point.type} message")
        return ()

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
                            pile.element_id, point))

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

    def object_id(self, p_case):
        """How a case's object is named in messages, both when publishing and in conditions: e.g. order-7."""
        return f"{self.id_prefix}-{p_case}"

    def _message_attributes(self, point, p_case, index):
        """The attributes point's message carries, from the case's current values; index numbers it in this pass."""
        values = self.sim_setup.bpmn_graph.get_all_attributes(p_case)
        attributes = dict()
        for name in point.attributes:
            if name == "object_id":
                attributes[name] = self.object_id(p_case)
            elif name == "index":
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
        if point.fan_out_list is not None:  # only called for an index within the list
            attributes[point.fan_out_as] = values[point.fan_out_list][index - 1]
        return attributes

    def tasks_next_to(self, element_id, forward):
        """The tasks the token reaches first after element_id (forward) or came from before it, through gateways and
        events: task id -> whether it may run alongside the others (a parallel or inclusive split or join between)."""
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
        """Case id -> (attribute, value, time): its values at creation, then each value a message wrote later."""
        history = {p_case: [(name, value, createdAt) for name, value in values.items()]
                   for p_case, (createdAt, values) in self._case_created.items()}
        for p_case, name, value, copiedAt in self._copied_later:
            history[p_case].append((name, value, copiedAt))
        return history

    def case_creation_times(self):
        """Case id -> when it was created."""
        return {p_case: createdAt for p_case, (createdAt, _) in self._case_created.items()}

    def only_started_values(self):
        """Case id -> the attributes whose value at creation is only a start no message wrote: an aggregate's 0 or
        initial, or a 'with' over a list still empty."""
        return dict(self._only_started)

    def stalled_at(self, parked_event):
        """Where a parked case is reported stalled: the race gateway if it waits in a race, else its event."""
        return self._race_gateway_of.get(parked_event.task_id, parked_event.task_id)

    def stalled_waits(self):
        """The cases still waiting for a message, as (parked event, place, message types), the one waiting longest
        first; a case in a race is reported once, at the gateway."""
        self._close_collections()
        waits = dict()
        for parked_event in self.parked_events():
            place = self.stalled_at(parked_event)
            _, _, types = waits.setdefault((parked_event.p_case, place), (parked_event, place, set()))
            types.update(self.waiting_for(parked_event.task_id))
        return [(parked_event, place, sorted(types)) for parked_event, place, types in waits.values()]

    def parked_events(self):
        """The catch events where cases wait for a message, the one waiting longest first (ties by case id)."""
        for key, parked_event in list(self._parked_events.items()):
            if not any(parked_event.p_state.has_token(flow) for flow in self.sim_setup.bpmn_graph.element_info[parked_event.task_id].incoming_flows):
                del self._parked_events[key]  # the case moved on without it (e.g. a terminate end event)
                del self._collected[key]
                self._closing.pop(key, None)
        return [self._parked_events[key] for key in sorted(self._parked_events, key=lambda key: (self._parked_events[key].enabled_datetime, key))]

    def waiting_for(self, event_id):
        """The message types a case waiting at event_id accepts."""
        return sorted({point.type for point in self._consume_points[event_id]})


def _hashable(value):
    # a group's values key its pile; a list value becomes a tuple
    return tuple(map(_hashable, value)) if isinstance(value, list) else value


def aggregate_of(function, values, initial=None):
    """function (count, sum, min or max) over values and initial, if given. A sum of whole numbers stays whole;
    any other is rounded once (math.fsum), so it doesn't depend on the order of the messages."""
    if function == "count":
        return len(values) + (initial if initial is not None else 0)
    if initial is not None:
        values = [initial, *values]
    if function == "sum":
        return sum(values) if all(isinstance(value, numbers.Integral) for value in values) else math.fsum(values)
    return min(values) if function == "min" else max(values)
