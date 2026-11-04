"""The messages a case publishes: how many (fan-out), their attributes and its object id."""
from prosimos.messaging.parser import is_whole_number
from prosimos.warning_logger import warning_logger


class Publishing:
    """Builds the messages a case publishes."""

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
