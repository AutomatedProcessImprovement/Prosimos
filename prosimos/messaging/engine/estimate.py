"""Estimates: when this engine's items could be ready for a quote's lines."""
from datetime import timedelta

from pix_framework.statistics.distribution import DistributionType

from prosimos.control_flow_manager import BPMN


class Estimating:
    """Answers estimate(): per line of a quote, the times its matching items not yet promised would be ready."""

    def slots_for(self, quote_type, requester, lines, now):
        """line id -> the times, earliest first, the items matching each of lines (line id, attributes) from
        requester would be at a catch event taking the quote type at now: waiting there since, queued to reach it,
        or forecast through the tasks ahead. Items holding a line, and those the open lines will take first, are
        promised and skipped."""
        self._ensure_arrivals_generated()
        free = self._quote_candidates(quote_type, now)
        for line in self._open_lines:
            taken = next((item for item in free if self._line_matches(line["type"], line["values"], item)), None)
            if taken is not None:
                free.remove(taken)
        return {line_id: [time for time, *item in free
                          if self._line_matches(quote_type, dict(attributes, source=requester), (time, *item))]
                for line_id, attributes in lines}

    def _quote_candidates(self, quote_type, now):
        """(time, case id, event id) of each item not holding a line at a catch event taking quote_type, waiting
        there, queued to reach it, or queued at a task on a way to it with only tasks and throw events, in time
        order."""
        events = {point.event_id for points in self._consume_points.values() for point in points
                  if point.type == quote_type and not point.starts_case}
        waiting = [(parked_event.enabled_datetime, parked_event.p_case, parked_event.task_id)
                   for parked_event in self.parked_events() if parked_event.task_id in events
                   and (parked_event.p_case, parked_event.task_id) not in self._set_aside]
        queued, waits = [], {}
        for event in self.events_queue.enabled_events.entry_finder:
            if event.task_id in events and event.parked_event is None:
                queued.append((event.enabled_datetime, event.p_case, event.task_id))
                continue
            way = self._way_to(event.task_id, events)
            if way is not None:
                tasks, catch_event = way
                ready = event.enabled_datetime
                for task_id in tasks:
                    if task_id not in waits:
                        waits[task_id] = self._average_wait(task_id, now)
                    ready = forecast_end(ready, self._staff_calendars(task_id),
                                         waits[task_id] + self._average_duration(task_id))
                queued.append((ready, event.p_case, catch_event))
        return sorted(waiting + queued)

    def _way_to(self, element_id, events):
        """The tasks from the task element_id on to a catch event in events, and that event; None if element_id
        isn't a task or the way has a gateway or any other element (forks and choices aren't forecast)."""
        graph = self.sim_setup.bpmn_graph
        if graph.element_info[element_id].type != BPMN.TASK:
            return None
        tasks, passed = [], set()
        while element_id not in events:
            info = graph.element_info[element_id]
            if info.type not in (BPMN.TASK, BPMN.INTERMEDIATE_THROW_EVENT) or len(info.outgoing_flows) != 1 \
                    or element_id in passed:
                return None
            passed.add(element_id)
            if info.type == BPMN.TASK:
                tasks.append(element_id)
            element_id = graph.flow_arcs[info.outgoing_flows[0]][1]
        return tasks, element_id

    def _average_wait(self, task_id, now):
        """The average working seconds from ready to started of task_id, on each performer's calendar, over the
        times it started by now; 0 while none has."""
        setup = self.sim_setup
        waits = [setup.calendars_map[setup.resources_map[event.resource_id].calendar_id].find_working_time(
                     event.enabled_datetime, event.started_datetime)
                 for trace in self.log_info.trace_list for event in trace.event_list
                 if event.type == BPMN.TASK and event.task_id == task_id and event.started_datetime <= now]
        return sum(waits) / len(waits) if waits else 0

    def _average_duration(self, task_id):
        """The mean of task_id's duration distributions over the staff who may do it, in working seconds."""
        distributions = self.sim_setup.task_resource[task_id].values()
        return sum(_mean(distribution) for distribution in distributions) / len(distributions)

    def _staff_calendars(self, task_id):
        setup = self.sim_setup
        return [setup.calendars_map[setup.resources_map[resource_id].calendar_id]
                for resource_id in setup.task_resource[task_id]]

    def _line_matches(self, quote_type, values, item):
        """Whether the item's catch event's consume entry for quote_type takes a line with values."""
        _, p_case, event_id = item
        point = next((point for point in self._consume_points.get(event_id, []) if point.type == quote_type), None)
        return point is not None and self._accepts(point, values, p_case)


def forecast_end(start, calendars, working_seconds):
    """When working_seconds of work from start end on the earliest of calendars, skipping their time off."""
    return min(start + timedelta(seconds=calendar.find_idle_time(start, working_seconds)) for calendar in calendars)


def _mean(distribution):
    # a uniform distribution has only its bounds
    if distribution.type == DistributionType.UNIFORM:
        return (distribution.min + distribution.max) / 2
    return distribution.mean
