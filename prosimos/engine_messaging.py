"""
The messaging part of the Prosimos engine (docs/engine-internals.md): the methods SimBPMEnv inherits to step one
event at a time, receive and publish messages, race the branches of event-based gateways, set case values from
messages, and tell the output what happened.
"""
from heapq import heappop, heappush

from prosimos import simulation_engine


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
            self._event_generator = simulation_engine.execute_full_process(self)
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
