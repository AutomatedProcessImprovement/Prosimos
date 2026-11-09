"""The messages a case publishes: how many (fan-out), their attributes, its quotes, and its object id."""
from datetime import timedelta

from prosimos.messaging.parser.common import is_number, is_whole_number
from prosimos.warning_logger import warning_logger


class Publishing:
    """Builds the messages a case publishes."""

    def _quote_of(self, point, p_case):
        """What point's message asks as a quote, modify, or cancel, for the orchestration's Quote: its kind, the quote
        type, the requester's object id, and for a quote or modify its lines (attribute dicts), and reply types, and
        for a quote its limit; None for an ordinary message."""
        kind, quoted = ("quote", point) if point.quote is not None else self._quote_kinds.get(point.type, (None, None))
        if kind is None:
            return None
        asked = {"kind": kind, "quote_type": quoted.type, "request": self.object_id(p_case)}
        if kind != "cancel":
            asked.update(lines=self._quote_lines(quoted, p_case, point.event_id), accepted=quoted.quote.accepted,
                         declined=quoted.quote.declined)
        if kind == "quote":
            asked["limit"] = self._quote_limit(point, p_case)
        return asked

    def _track_quote(self, point, p_case):
        """Notes that the case published a quote with point, or canceled it."""
        if point.quote is not None:
            self._open_quotes.setdefault(p_case, {})[point.type] = point
        elif self._quote_kinds.get(point.type, (None, None))[0] == "cancel":
            self._open_quotes.get(p_case, {}).pop(self._quote_kinds[point.type][1].type, None)

    def _cancel_open_quotes(self, p_case, due, ended_at):
        """Publishes the cancel of each quote the ended case left open, at due, as if the model had thrown it at the
        cancel's own event; its record names ended_at, the end event where the case ended, which its links attach
        to (the case never passed the cancel's event)."""
        for quoted in self._open_quotes.pop(p_case, {}).values():
            cancel = next(point for point in self._publish_points.get(self._cancel_events[quoted.quote.cancel], [])
                          if point.type == quoted.quote.cancel)
            self.outbox.append((cancel.type, self._message_attributes(cancel, p_case, 1), ((p_case, due),),
                                ended_at or cancel.event_id, cancel, self._quote_of(cancel, p_case)))

    def _quote_lines(self, quoted, p_case, event_id):
        """The lines of the quote entry quoted, read from the case now, as a case passing event_id asks them: the
        entry's attributes, and over a list each line's entry under its as name."""
        settings, values = quoted.quote, self.sim_setup.bpmn_graph.get_all_attributes(p_case)
        carried = self._message_attributes(quoted, p_case, 1)
        if settings.quantity_list is not None:
            entries = values.get(settings.quantity_list)
            if isinstance(entries, (list, tuple)):
                return [{**carried, settings.quantity_as: entry} for entry in entries]
            self._warn_quote(event_id, settings.quantity_list, f"case {p_case} passes {event_id} with no list "
                             f"{settings.quantity_list} (got {entries!r}); its {quoted.type} has no lines")
            return []
        number = settings.quantity
        if settings.quantity_attribute is not None:
            number = values.get(settings.quantity_attribute)
            if not is_whole_number(number, minimum=0):
                self._warn_quote(event_id, settings.quantity_attribute, f"case {p_case} passes {event_id} with no "
                                 f"valid {settings.quantity_attribute} (got {number!r}); its {quoted.type} asks one "
                                 f"line")
                number = 1
        return [dict(carried) for _ in range(int(number))]

    def _quote_limit(self, point, p_case):
        """The quote's within: fixed, or a number of its unit read from the case now; None without a limit."""
        settings = point.quote
        if settings.within_attribute is None:
            return settings.within
        number = self.sim_setup.bpmn_graph.get_all_attributes(p_case).get(settings.within_attribute)
        if is_number(number) and number > 0:
            return timedelta(**{settings.within_unit: number})
        self._warn_quote(point.event_id, settings.within_attribute, f"case {p_case} passes {point.event_id} with no "
                         f"valid {settings.within_attribute} (got {number!r}); its {point.type} has no limit")
        return None

    def _warn_quote(self, event_id, attribute, warning):
        if (event_id, attribute) not in self._warned_quotes:
            self._warned_quotes.add((event_id, attribute))
            warning_logger.add_warning(warning)

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
