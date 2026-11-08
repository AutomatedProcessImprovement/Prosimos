"""Reservations: the lines a quote reserved at this engine, and the waiting items that take them."""


class Reservations:
    """Opens a quote's reserved lines and lets waiting and arriving items take them, first come, first served; a
    quote's items here move on together once all its lines here are taken. A quote is its requester, request, and
    quote type, so the lines its modifies added here belong to the same share."""

    def open_lines(self, message_id, quote_type, requester, request, lines, now):
        """Opens the lines (line id, attributes) the quote or modify message message_id from requester reserved here
        at now, in reservation order; items already waiting take them at once."""
        for line_id, attributes in lines:
            self._open_lines.append({"quote": message_id, "share": (requester, request, quote_type), "line": line_id,
                                     "type": quote_type, "values": dict(attributes, source=requester)})
        for parked_event in self.parked_events():
            self._take_open_line(parked_event, now)

    def has_moved(self, message_id):
        """Whether the share holding the lines the quote or modify message message_id reserved here has moved
        on."""
        return message_id in self._moved_quotes

    def _take_open_line(self, parked_event, now):
        """The waiting item takes the oldest open line its catch event's consume entry for the line's quote type
        accepts, like a message of that type; the line is set aside on the item."""
        key = (parked_event.p_case, parked_event.task_id)
        if key in self._set_aside or key not in self._parked_events:
            return
        for line in self._open_lines:
            point = next((point for point in self._consume_points.get(parked_event.task_id, [])
                          if point.type == line["type"]), None)
            if point is not None and self._accepts(point, line["values"], parked_event.p_case):
                self._open_lines.remove(line)
                self._set_aside[key] = (line, point)
                self._move_if_complete(line["share"], now)
                return

    def _move_if_complete(self, share, now):
        """Once no line of the quote's share is open here, each of its items claims its line at now, and they move
        on together; every message that reserved one of them has moved from then on."""
        if any(line["share"] == share for line in self._open_lines):
            return
        for key, (line, point) in list(self._set_aside.items()):
            if line["share"] == share:
                del self._set_aside[key]
                self._claim(self._parked_events[key], point, line["values"], now)
                self._moved_quotes[line["quote"]] = now
