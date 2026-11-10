"""What the output asks the engine: attribute history, stalled waits."""


class Reporting:
    """Answers what the output asks once the run is over."""

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
