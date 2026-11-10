"""How the engine steps: arrivals, next_event_time and step, held effects and terminate end events."""
from heapq import heappop, heappush

from prosimos.messaging.engine.fan_in import Pile


class Stepping:
    """Arrivals, next_event_time and step, and the effects held until their time."""

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
        self._drop_lost_timers()
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

    def _drop_lost_timers(self):
        """Takes the timers of races another branch already won off the head of the queue: when due, they would do
        nothing, and announcing them would keep the run going until then."""
        next_event = self.events_queue.peek()
        while next_event is not None and next_event.armed_event is not None and next_event.race.won:
            self.events_queue.pop_next_event()
            next_event = self.events_queue.peek()

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
            if element_id in self._watched_ends:  # also a terminate end event, whose own pass is skipped below
                self._ended_at[p_case] = element_id
            if element_id is None:
                self._end_terminated(p_case)
                self._cancel_open_quotes(p_case, due, self._ended_at.get(p_case))
            if self.sim_setup.bpmn_graph.terminated_by(p_case, due):
                continue
            for point in self._publish_points.get(element_id, []):
                if point.fans_in:
                    self._hand_in(point, p_case, element_id, due)
                    continue
                for index in range(1, self._fan_out(point, p_case) + 1):
                    self.outbox.append((point.type, self._message_attributes(point, p_case, index), ((p_case, due),),
                                        element_id, point, self._quote_of(point, p_case)))
                self._track_quote(point, p_case)
            if element_id in self._watched_ends and not any(self.all_process_states[p_case].tokens.values()) \
                    and not any(held[1] == p_case for held in self._held):
                self._cancel_open_quotes(p_case, due, element_id)  # the case ended now

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
