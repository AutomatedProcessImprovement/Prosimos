"""The case values messages and the catalogue write: copy, append, aggregate and received values."""
import math
import numbers

from prosimos.messaging.parser.common import is_number
from prosimos.warning_logger import warning_logger


class Values:
    """Writes the values messages and the catalogue give a case."""

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

    def _start_fresh(self, p_case, event_id, when):
        """A case reaching a catch event it reached before, back through a loop, starts its entries' appended and
        aggregated values over, and the 'with' values over those lists follow; the output writes each as cleared at
        when, as a starting value no message wrote."""
        if (p_case, event_id) not in self._reached:
            self._reached.add((p_case, event_id))
            return
        case_values = self.sim_setup.bpmn_graph.all_attributes[p_case]
        appended = {target for point in self._consume_points[event_id] for target, _ in point.append}
        aggregated = {target for point in self._consume_points[event_id] for target, *_ in point.aggregate}
        starting = self._starting_values()
        for target in appended | aggregated:
            case_values[target] = starting.get(target)
            self._aggregated_values.pop((p_case, target), None)
        following = {name for name, source, *_ in self._received if source in appended}
        for name in following:
            case_values[name] = None
        self._apply_received(case_values, when, appended)
        self._copied_later.extend((p_case, name, None, when) for name in sorted(appended | aggregated | following))

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
