"""Reservations: the lines a quote reserved at this engine, and the waiting items that take them."""


class Reservations:
    """Opens a quote's reserved lines and lets waiting and arriving items take them, first come, first served."""

    def open_lines(self, quote_id, quote_type, requester, lines):
        """Opens the lines (line id, attributes) of the quote message quote_id from requester, in reservation
        order; items already waiting take them at once."""
        for line_id, attributes in lines:
            self._open_lines.append({"quote": quote_id, "line": line_id, "type": quote_type,
                                     "values": dict(attributes, source=requester)})
        for parked_event in self.parked_events():
            self._take_open_line(parked_event)

    def _take_open_line(self, parked_event):
        """The waiting item takes the oldest open line its catch event's consume entry for the line's quote type
        accepts, like a message of that type; the line is set aside on the item."""
        key = (parked_event.p_case, parked_event.task_id)
        if key in self._set_aside:
            return
        for line in self._open_lines:
            point = next((point for point in self._consume_points.get(parked_event.task_id, [])
                          if point.type == line["type"]), None)
            if point is not None and self._accepts(point, line["values"], parked_event.p_case):
                self._open_lines.remove(line)
                self._set_aside[key] = line
                return
