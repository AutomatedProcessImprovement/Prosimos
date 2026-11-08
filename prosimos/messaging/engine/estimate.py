"""Estimates: when this engine's items could be ready for a quote's lines."""


class Estimating:
    """Answers estimate(): per line of a quote, the times its matching items not yet promised would be ready."""

    def slots_for(self, quote_type, requester, lines):
        """line id -> the times, earliest first, the items matching each of lines (line id, attributes) from
        requester would be at a catch event taking the quote type: waiting there since, or queued to reach it.
        Items holding a line, and those the open lines will take first, are promised and skipped."""
        self._ensure_arrivals_generated()
        free = self._quote_candidates(quote_type)
        for line in self._open_lines:
            taken = next((item for item in free if self._line_matches(line["type"], line["values"], item)), None)
            if taken is not None:
                free.remove(taken)
        return {line_id: [time for time, *item in free
                          if self._line_matches(quote_type, dict(attributes, source=requester), (time, *item))]
                for line_id, attributes in lines}

    def _quote_candidates(self, quote_type):
        """(time, case id, event id) of each item not holding a line at a catch event taking quote_type, waiting
        there or queued to reach it, in time order."""
        events = {point.event_id for points in self._consume_points.values() for point in points
                  if point.type == quote_type and not point.starts_case}
        waiting = [(parked_event.enabled_datetime, parked_event.p_case, parked_event.task_id)
                   for parked_event in self.parked_events() if parked_event.task_id in events
                   and (parked_event.p_case, parked_event.task_id) not in self._set_aside]
        queued = [(event.enabled_datetime, event.p_case, event.task_id)
                  for event in self.events_queue.enabled_events.entry_finder
                  if event.task_id in events and event.parked_event is None]
        return sorted(waiting + queued)

    def _line_matches(self, quote_type, values, item):
        """Whether the item's catch event's consume entry for quote_type takes a line with values."""
        _, p_case, event_id = item
        point = next((point for point in self._consume_points.get(event_id, []) if point.type == quote_type), None)
        return point is not None and self._accepts(point, values, p_case)
