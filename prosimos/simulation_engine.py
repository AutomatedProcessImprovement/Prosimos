import csv
import datetime
import itertools
import json
import os
import numpy as np
from datetime import timedelta
from typing import List
import random

import pytz

from prosimos.control_flow_manager import (
    BPMN,
    CustomDatetimeAndSeconds,
)
from prosimos.engine_messaging import EngineMessaging
from prosimos.execution_info import EnabledEvent, TaskEvent, Trace
from prosimos.file_manager import FileManager
from prosimos.messaging_parser import is_whole_number
from prosimos.prioritisation import CasePrioritisation
from prosimos.simulation_properties_parser import parse_datetime
from prosimos.simulation_queues_ds import (
    DiffResourceQueue,
    EventQueue,
)
from prosimos.simulation_setup import SimDiffSetup
from prosimos.simulation_stats_calculator import LogInfo
from prosimos.warning_logger import warning_logger


class SimResource:
    def __init__(self):
        self.switching_time = 0
        self.allocated_tasks = 0
        self.worked_time = 0
        self.available_time = 0
        self.last_released = 0


class SimBPMEnv(EngineMessaging):
    def __init__(self, sim_setup: SimDiffSetup, stat_fwriter, log_fwriter, id_prefix=None):
        self.sim_setup = sim_setup
        self.sim_resources = dict()
        self.stat_fwriter = stat_fwriter
        self.additional_columns = self.sim_setup.all_attributes.get_all_columns_generated()
        # attributes copied, appended or aggregated from messages get a column too; it stays empty until a value is
        # copied, an appended list starts empty, and a sum or count starts at 0, or at its initial
        self._appended = sorted({target for point in self.sim_setup.messaging.consume for target, _ in point.append})
        self._aggregated = {target: (function, initial) for point in self.sim_setup.messaging.consume
                            for target, function, _, initial in point.aggregate}
        self.additional_columns.extend(sorted({target for point in self.sim_setup.messaging.consume
                                               for target, _ in point.copy} | set(self._appended)
                                              | set(self._aggregated) - set(self.additional_columns)))
        if self.sim_setup.batch_processing not in [None, {}]:
            self.additional_columns.append("batch_id")
        self.log_writer = FileManager(10000, log_fwriter, self.additional_columns)
        self.log_info = LogInfo(sim_setup)
        self.executed_events = 0
        self._arrivals_generated = False
        self._event_generator = None
        self.time_update_process_state = 0

        r_first_available = dict()
        for r_id in sim_setup.resources_map:
            self.sim_resources[r_id] = SimResource()
            r_first_available[r_id] = self.sim_setup.next_resting_time(r_id, self.sim_setup.start_datetime)

        self.resource_queue = DiffResourceQueue(self.sim_setup.task_resource, r_first_available)
        self.events_queue = EventQueue()
        self.all_process_states = dict()  # store all process states with a case_id as a key
        # the cases that still hold a token (as keys, in case id order), kept up to date as tokens move: when a
        # case is created, executes an event or wins a race; _could_match reads it for every message offered
        self._unfinished = dict()

        self.case_prioritisation = CasePrioritisation(
            self.sim_setup.total_num_cases or 0,  # None for a process started by messages: no planned cases
            self.sim_setup.case_attributes,
            self.sim_setup.prioritisation_rules,
        )

        all_attributes = {
            "global": self.sim_setup.all_attributes.global_attribute_initial_values,
            **self.case_prioritisation.all_case_attributes
        }

        self.sim_setup.bpmn_graph.all_attributes = all_attributes

        # Held effects. A case passes events that take no time (e.g. a message throw event) during
        # the step of the task before them, ahead of their real time, or even while arrivals are
        # generated. What passing them does is therefore held here and done by step() when it is
        # due: a heap of (due datetime, case id, pass order, element id).
        self.id_prefix = id_prefix or sim_setup.process_name  # the prefix of the object ids in its messages
        self._publish_points = dict()  # event id -> the 'publish' entries for it
        for point in sim_setup.messaging.publish:
            self._publish_points.setdefault(point.event_id, []).append(point)
        self.sim_setup.bpmn_graph.watched_elements = set(self._publish_points)
        self._held = []
        self._pass_order = itertools.count()
        self.outbox = []  # (message type, attributes, case id, element id, publish entry) released by step()
        self.logged_rows = []  # (case id, element id, enabled, completed) per row of the log, in order
        self._tasks_next_to = dict()  # (element id, forward) -> the tasks next to it, see tasks_next_to()
        self._warned_missing_values = set()
        self._warned_counts = set()
        self._warned_count_lists = set()

        # Waiting. A case reaching a catch event listed under 'consume' doesn't get a delay: when
        # the queued catch event is executed (at the time the case really reaches it), the case is parked
        # there, its token left on the event's incoming flow, until deliver() resumes it.
        self._consume_points = dict()  # event id -> the 'consume' entries for it
        for point in sim_setup.messaging.consume:
            self._consume_points.setdefault(point.event_id, []).append(point)
        self._parked_events = dict()  # (case id, event id) -> the parked EnabledEvent
        self.claimed_by = []  # (case id, event id, consume entry) that took the message deliver() last claimed
        # case attributes over time: the declared ones and those copied from messages, as each case was
        # created, and every later copy at a catch event
        self._case_attribute_names = ({attribute.name for attribute in sim_setup.case_attributes.attributes}
                                      | {target for point in sim_setup.messaging.consume for target, _ in point.copy}
                                      | set(self._appended) | set(self._aggregated))
        self._case_created = dict()  # case id -> (when it was created, its case attribute values then)
        # case id -> its sums and counts no message had written when it was created: their value then is only the
        # starting 0
        self._only_started = dict()
        self._copied_later = []  # (case id, attribute, value, when copied)
        self._warned_missing_copies = set()
        self._warned_missing_appends = set()
        self._aggregated_values = dict()  # (case id, case attribute) -> the message values aggregated so far
        self._warned_missing_aggregates = set()
        self._warned_not_numbers = set()
        self._warned_capacities = set()
        # (case id, event id) -> [messages claimed so far, min, max, the values of 'same' the first message set]
        self._collected = dict()
        # (case id, event id) -> when it got min: resumed then, but still taking messages offered at that instant, up
        # to max, until the orchestrator's offers for that step are done
        self._closing = dict()
        self._warned_collects = set()
        self._warned_missing_same = set()

        # Races. At an event-based gateway with a branch that waits for a message, every branch is armed:
        # timers are queued for when they fire, message branches park; the first to happen wins and
        # the other branches are canceled. Gateways without such a branch keep drawing a duration per branch.
        graph = self.sim_setup.bpmn_graph
        self._race_branches = dict()  # race gateway id -> the catch events after it
        for gateway_id, element in graph.element_info.items():
            if element.type is BPMN.EVENT_BASED_GATEWAY:
                branches = [graph.flow_arcs[flow][1] for flow in element.outgoing_flows]
                if any(branch in self._consume_points for branch in branches):
                    self._race_branches[gateway_id] = branches
        graph.race_gateways = set(self._race_branches)
        self._race_gateway_of = {branch: gateway_id for gateway_id, branches in self._race_branches.items()
                                 for branch in branches}
        self._races = dict()  # (case id, gateway id) -> the Race running for that case
        # a message matching several branches of one race goes to the branch whose entry comes first
        self._consume_order = {id(point): index for index, point in enumerate(sim_setup.messaging.consume)}
        self._warned_overlapping_races = set()
        self._warned_races_won_at_once = set()

    def _full_process(self):
        """The generator that runs this engine's events one by one (execute_full_process), for step()."""
        return execute_full_process(self)

    # Queue priorities (case priority first, timers at 0) were removed; last version with them: 5f40913

    def _count(self, point, p_case):
        """How many messages point publishes for a case passing it now: 1 without count, else its fixed count
        or the case attribute holding it, read now; one that isn't a whole number of at least 0 counts as 1,
        with one warning per event and attribute. Over a list, one per entry; a missing or empty list publishes
        none, with one warning per event."""
        if not point.counted:
            return 1
        if point.count_list is not None:
            return len(self._count_list(point, p_case))
        if point.count_attribute is None:
            return point.count
        count = self.sim_setup.bpmn_graph.get_all_attributes(p_case).get(point.count_attribute)
        if is_whole_number(count, minimum=0):
            return int(count)
        if (point.event_id, point.count_attribute) not in self._warned_counts:
            self._warned_counts.add((point.event_id, point.count_attribute))
            warning_logger.add_warning(
                f"case {p_case} passes {point.event_id} with no valid {point.count_attribute} (got {count!r}); "
                f"it publishes one {point.type} message")
        return 1

    def _count_list(self, point, p_case):
        entries = self.sim_setup.bpmn_graph.get_all_attributes(p_case).get(point.count_list)
        if isinstance(entries, (list, tuple)) and entries:
            return entries
        if point.event_id not in self._warned_count_lists:
            self._warned_count_lists.add(point.event_id)
            kind = "an empty" if isinstance(entries, (list, tuple)) else "no"
            warning_logger.add_warning(
                f"case {p_case} passes {point.event_id} with {kind} list {point.count_list} (got {entries!r}); "
                f"it publishes no {point.type} message")
        return ()

    def object_id(self, p_case):
        """How a case's object is named in messages, both when publishing and in conditions: e.g. order-7."""
        return f"{self.id_prefix}-{p_case}"

    def tasks_next_to(self, element_id, forward):
        """The tasks the token reaches first after element_id (forward) or came from last before it: the flows
        are followed through gateways and events, and stop at tasks. Task id -> whether some path to it passes
        a parallel or inclusive gateway that joins (or splits, forward) several flows, i.e. whether it may run
        alongside the others."""
        key = (element_id, forward)
        if key not in self._tasks_next_to:
            graph = self.sim_setup.bpmn_graph
            tasks, seen, to_visit = {}, {(element_id, False)}, [(element_id, False)]
            while to_visit:
                current, concurrent = to_visit.pop()
                element = graph.element_info[current]
                if element.type in (BPMN.PARALLEL_GATEWAY, BPMN.INCLUSIVE_GATEWAY) and current != element_id:
                    concurrent = concurrent or (element.is_split() if forward else element.is_join())
                for flow in element.outgoing_flows if forward else element.incoming_flows:
                    neighbour = graph.flow_arcs[flow][1 if forward else 0]
                    if (neighbour, concurrent) in seen:
                        continue
                    seen.add((neighbour, concurrent))
                    if graph.element_info[neighbour].type is BPMN.TASK:
                        tasks[neighbour] = tasks.get(neighbour, False) or concurrent
                    else:
                        to_visit.append((neighbour, concurrent))
            self._tasks_next_to[key] = tasks
        return self._tasks_next_to[key]

    def case_attribute_history(self):
        """Case id -> its case attributes over time, as (attribute, value, time): every value the case had
        when it was created, then every value a message copied into it later, in the order they came."""
        history = {p_case: [(name, value, createdAt) for name, value in values.items()]
                   for p_case, (createdAt, values) in self._case_created.items()}
        for p_case, name, value, copiedAt in self._copied_later:
            history[p_case].append((name, value, copiedAt))
        return history

    def case_creation_times(self):
        """Case id -> when it was created."""
        return {p_case: createdAt for p_case, (createdAt, _) in self._case_created.items()}

    def only_started_values(self):
        """Case id -> the aggregated attributes whose value when it was created is only their start, 0 or their
        initial: no message had written them yet (a message that starts a case writes its own aggregates into the
        new case)."""
        return dict(self._only_started)

    def stalled_at(self, parked_event):
        """Where a parked case is reported stalled: the race gateway if it waits in a race, else its event."""
        return self._race_gateway_of.get(parked_event.task_id, parked_event.task_id)

    def stalled_waits(self):
        """The cases still waiting for a message, once per case and place where they are reported stalled:
        (the parked event, the place, the message types accepted there), the one waiting longest first.
        A case waiting in a race is reported once, at the gateway, with the types of all its message
        branches."""
        self._close_collections()
        waits = dict()  # (case id, place) -> (parked event, place, message types)
        for parked_event in self.parked_events():
            place = self.stalled_at(parked_event)
            _, _, types = waits.setdefault((parked_event.p_case, place), (parked_event, place, set()))
            types.update(self.waiting_for(parked_event.task_id))
        return [(parked_event, place, sorted(types)) for parked_event, place, types in waits.values()]

    def parked_events(self):
        """The queued catch events (EnabledEvents) at which cases are parked, waiting for a message: one
        per waiting case and event, the one waiting longest first (ties by case id)."""
        for key, parked_event in list(self._parked_events.items()):
            if not any(parked_event.p_state.has_token(flow) for flow in self.sim_setup.bpmn_graph.element_info[parked_event.task_id].incoming_flows):
                del self._parked_events[key]  # the case moved on without it (e.g. a terminate end event)
                del self._collected[key]
                self._closing.pop(key, None)
        return [self._parked_events[key] for key in sorted(self._parked_events, key=lambda key: (self._parked_events[key].enabled_datetime, key))]

    def waiting_for(self, event_id):
        """The message types a case waiting at event_id accepts."""
        return sorted({point.type for point in self._consume_points[event_id]})

    def _message_attributes(self, point, p_case, index):
        """The listed attributes, copied from the case's current values; index is the message's number among
        the ones point publishes for this pass."""
        values = self.sim_setup.bpmn_graph.get_all_attributes(p_case)
        attributes = dict()
        for name in point.attributes:
            if name == "object_id":
                attributes[name] = self.object_id(p_case)
            elif name == "index":  # only listed on an entry with count (checked at load time)
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
        if point.count_list is not None:  # only called for an index within the list
            attributes[point.count_as] = values[point.count_list][index - 1]
        return attributes

    def _update_initial_event_info(self, sim_setup, p_case, arrival_time):
        for e_id in sim_setup.bpmn_graph.last_datetime:
            sim_setup.bpmn_graph.last_datetime[e_id][p_case] = None
        p_state = sim_setup.initial_state()
        enabled_datetime = self.simulation_datetime_from(arrival_time)
        enabled_time = CustomDatetimeAndSeconds(arrival_time, enabled_datetime)
        enabled_tasks, visited_at = sim_setup.update_process_state(
            p_case, sim_setup.bpmn_graph.starting_event, p_state, enabled_time
        )
        self._log_passed_throw_events(p_case, p_state, visited_at)
        self.all_process_states[p_case] = p_state
        self._tokens_moved(p_case, p_state)
        self.log_info.trace_list.append(Trace(p_case, enabled_datetime))
        if self._appended or self._aggregated:  # a planned case's lists start empty too, and its sums and counts at 0
            case_values = sim_setup.bpmn_graph.all_attributes.setdefault(p_case, {})
            for target, value in self._starting_values().items():
                case_values.setdefault(target, value)
        case_values = sim_setup.bpmn_graph.all_attributes.get(p_case, {})
        self._case_created[p_case] = (enabled_datetime, {name: value for name, value in case_values.items()
                                                         if name in self._case_attribute_names})
        self._only_started[p_case] = tuple(
            target for target in self._starting_values()
            if target in self._aggregated and (p_case, target) not in self._aggregated_values)
        for task in enabled_tasks:
            task_id = task.task_id
            self.events_queue.append_event(
                EnabledEvent(
                    p_case,
                    p_state,
                    task_id,
                    arrival_time,
                    enabled_datetime,
                    task.batch_info_exec,
                    task.duration_sec,
                    task.is_event,
                ),
            )
        return enabled_datetime

    def execute_enabled_event(self, c_event: EnabledEvent):
        self.executed_events += 1

        event_element_info = self.sim_setup.bpmn_graph.element_info[c_event.task_id]

        if event_element_info.type == BPMN.TASK and c_event.batch_info_exec is not None:
            # execute batched task
            executed_tasks = self.execute_task_batch(c_event)

            for task in executed_tasks:
                completed_at, completed_datetime, p_case = task
                p_state = self.all_process_states[p_case]
                enabled_time = CustomDatetimeAndSeconds(completed_at, completed_datetime)
                enabled_tasks, visited_at = self.sim_setup.update_process_state(
                    p_case,
                    c_event.task_id,
                    self.all_process_states[p_case],
                    enabled_time,
                )
                self._log_passed_throw_events(p_case, p_state, visited_at)
                self._tokens_moved(p_case, p_state)

                for next_task in enabled_tasks:
                    self.events_queue.append_event(
                        EnabledEvent(
                            p_case,
                            p_state,
                            next_task.task_id,
                            visited_at[next_task.task_id].seconds_from_start,
                            visited_at[next_task.task_id].datetime,
                            next_task.batch_info_exec,
                            next_task.duration_sec,
                            next_task.is_event,
                        ),
                    )
        else:
            if (c_event.task_id in self._race_gateway_of and c_event.parked_event is None
                    and self._race_step(c_event)):
                return
            if c_event.task_id in self._consume_points and c_event.parked_event is None:
                needed = self._needed(c_event.p_case, c_event.task_id)
                most = self._most(c_event.p_case, c_event.task_id, needed)
                if most > 0:
                    self._parked_events[(c_event.p_case, c_event.task_id)] = c_event
                    self._collected[(c_event.p_case, c_event.task_id)] = [0, needed, most, None]
                    if needed == 0:  # it goes on now, taking up to max of the messages offered at this instant
                        self._resume(c_event, c_event.enabled_datetime, keep_open=True)
                    return
                c_event.complete_at_once = True  # nothing to collect: the case passes straight on
                if c_event.race is not None:
                    self._win_race(c_event.race, c_event.task_id, c_event.p_state)
            if event_element_info.type == BPMN.TASK:
                # execute not batched task
                completed_at, completed_datetime = self.execute_task(c_event)
            else:
                completed_at, completed_datetime = self.execute_event(c_event)

            # Updating the process state. Retrieving/enqueuing enabled tasks, it also schedules the corresponding event
            # s_t = datetime.datetime.now()
            enabled_time = CustomDatetimeAndSeconds(completed_at, completed_datetime)
            enabled_tasks, visited_at = self.sim_setup.update_process_state(
                c_event.p_case, c_event.task_id, c_event.p_state, enabled_time
            )
            self._log_passed_throw_events(c_event.p_case, c_event.p_state, visited_at)
            self._tokens_moved(c_event.p_case, c_event.p_state)
            # self.time_update_process_state += (datetime.datetime.now() - s_t).total_seconds()

            for next_task in enabled_tasks:
                self.events_queue.append_event(
                    EnabledEvent(
                        c_event.p_case,
                        c_event.p_state,
                        next_task.task_id,
                        visited_at[next_task.task_id].seconds_from_start,
                        visited_at[next_task.task_id].datetime,
                        next_task.batch_info_exec,
                        next_task.duration_sec,
                        next_task.is_event,
                    ),
                )

    def pop_and_allocate_resource(self, task_id: str, num_allocated_tasks: int):
        r_id, r_avail_at = self.resource_queue.pop_resource_for(task_id)
        self.sim_resources[r_id].allocated_tasks += num_allocated_tasks
        return r_id, r_avail_at

    def execute_task(self, c_event: EnabledEvent):
        if self.sim_setup.multitask_info is None:
            r_id, r_avail_at = self.pop_and_allocate_resource(c_event.task_id, 1)
            r_avail_at = max(c_event.enabled_at, r_avail_at)
            avail_datetime = self._datetime_from(r_avail_at)
            is_working, _ = self.sim_setup.get_resource_calendar(r_id).is_working_datetime(avail_datetime)
            if not is_working:
                r_avail_at = r_avail_at + self.sim_setup.next_resting_time(r_id, avail_datetime)
        else:
            r_id, r_avail_at = self.allocate_multitasking_resource(c_event)

        full_evt = TaskEvent(
            c_event.p_case,
            c_event.task_id,
            r_id,
            r_avail_at,
            c_event.enabled_at,
            c_event.enabled_datetime,
            self,
        )

        self.log_info.add_event_info(c_event.p_case, full_evt, self.sim_setup.resources_map[r_id].cost_per_hour)

        if self.sim_setup.multitask_info is None:
            r_next_available = full_evt.completed_at

            if self.sim_resources[r_id].switching_time > 0:
                r_next_available += self.sim_setup.next_resting_time(r_id, full_evt.completed_datetime)

            self.resource_queue.update_resource_availability(r_id, r_next_available)
            self.sim_resources[r_id].worked_time += full_evt.ideal_duration
        else:
            self.release_multitasking_resource(r_id, full_evt, r_avail_at)

        self.update_attributes(c_event)
        self._log_row(full_evt)

        completed_at = full_evt.completed_at
        completed_datetime = full_evt.completed_datetime

        return completed_at, completed_datetime

    def allocate_multitasking_resource(self, c_event: EnabledEvent):
        r_id, r_avail_at = self.resource_queue.pop_resource_for(c_event.task_id)

        candidates = [[r_id, r_avail_at]]
        while r_avail_at is not None and r_avail_at <= c_event.enabled_at:
            r_id, r_avail_at = self.resource_queue.pop_resource_for(c_event.task_id)
            if r_id is not None:
                candidates.append([r_id, r_avail_at])

        if len(candidates) > 1:
            i = random.randint(0, len(candidates) - 1)
            [r_id, r_avail_at] = candidates[i]
            for j in range(0, len(candidates)):
                if j != i:
                    self.resource_queue.update_resource_availability(candidates[j][0], candidates[j][1])
        elif r_id is None and len(candidates) == 1:
            [r_id, r_avail_at] = candidates[0]

        # best_r, best_avail = r_id, r_avail_at
        # while r_avail_at is not None and r_avail_at <= c_event.enabled_at:
        #     c_workload = self.sim_setup.multitask_info.workload_diff(r_id, c_event.task_id)
        #     if c_workload > max_workload_diff:
        #         candidates.append([best_r, best_avail])
        #         best_r, best_avail = r_id, r_avail_at
        #         max_workload_diff = c_workload
        #     else:
        #         candidates.append([r_id, r_avail_at])
        #     r_id, r_avail_at = self.resource_queue.pop_resource_for(c_event.task_id)

        # print(f'Selected: {best_r, best_avail}')
        # if 'Loan Officer' in best_r and len(candidates) > 1:
        #     print("hola")
        # if len(candidates) > 0:
        #     # for r in candidates:
        #     #     print(r)
        #     # print("-------------------------------------")
        #     for j in range(0, len(candidates)):
        #         self.resource_queue.update_resource_availability(candidates[j][0], candidates[j][1])
        #
        # r_id, r_avail_at = best_r, best_avail
        # self.sim_resources[r_id].allocated_tasks += 1
        if c_event.enabled_at > r_avail_at:
            next_avail_at = c_event.enabled_at
            avail_datetime = self._datetime_from(next_avail_at)
            is_working, _ = self.sim_setup.get_resource_calendar(r_id).is_working_datetime(avail_datetime)
            if not is_working:
                r_avail_at = r_avail_at + self.sim_setup.next_resting_time(r_id, avail_datetime)

        return r_id, r_avail_at

    def release_multitasking_resource(self, r_id: str, full_evt: TaskEvent, r_init_avail):
        completed_dt = self._datetime_from(full_evt.completed_at)
        self.sim_setup.multitask_info.allocate_task_to(r_id, full_evt.task_id, completed_dt)
        r_next_avail = r_init_avail
        if not self.sim_setup.multitask_info.can_get_new_tasks(r_id, completed_dt):
            last_time = self.sim_setup.multitask_info.release_tasks_from(r_id, completed_dt)
            r_next_avail += self.sim_setup.next_resting_time(r_id, last_time)

        self.resource_queue.update_resource_availability(r_id, r_next_avail)
        self.sim_resources[r_id].worked_time += full_evt.ideal_duration

    def update_attributes(self, current_event):
        event_attributes = self.sim_setup.all_attributes.event_attributes.attributes
        global_event_attributes = self.sim_setup.all_attributes.global_event_attributes.attributes

        all_attribute_values = {
            **self.sim_setup.bpmn_graph.all_attributes["global"],
            **self.sim_setup.bpmn_graph.all_attributes[current_event.p_case]
        }

        new_global_attr_values = self._extract_attributes_for_event(current_event.task_id, global_event_attributes, all_attribute_values)
        new_event_attr_values = self._extract_attributes_for_event(current_event.task_id, event_attributes, all_attribute_values)

        self.sim_setup.bpmn_graph.all_attributes["global"].update(new_global_attr_values)
        self.sim_setup.bpmn_graph.all_attributes[current_event.p_case].update(new_event_attr_values)

    def _extract_attributes_for_event(self, task_id, source_attributes, all_attribute_values):
        new_attributes = {}

        if task_id in source_attributes:
            task_attributes = source_attributes[task_id]
            for key, value in task_attributes.items():
                new_attributes[key] = value.get_next_value(all_attribute_values)

        return new_attributes

    def _log_row(self, full_event):
        """Writes one row of the log and remembers which case and element it belongs to, and when it was
        enabled and completed: logged_rows[k] is the k-th row this engine wrote."""
        self.log_writer.add_csv_row(self.get_csv_row_data(full_event))
        self.logged_rows.append((full_event.p_case, full_event.task_id,
                                 full_event.enabled_datetime, full_event.completed_datetime))

    def get_csv_row_data(self, full_event: TaskEvent):
        """
        Return array of values for one line of the csv file based on full_event information.
        In case we have defined case attributes setup, we will have additional columns besides the basic ones.
        """

        resource_name = (
            self.sim_setup.resources_map[full_event.resource_id].resource_name
            if (hasattr(full_event, "resource_id"))
            else "No assigned resource"
        )

        row_basic_info = verify_miliseconds(
            [
                full_event.p_case,
                self.sim_setup.bpmn_graph.element_info[full_event.task_id].name,
                full_event.enabled_datetime,
                full_event.started_datetime,
                full_event.completed_datetime,
                resource_name,
            ]
        )

        all_attrs = self.sim_setup.bpmn_graph.get_all_attributes(full_event.p_case)

        # values = ["" if all_attrs.get(col) is None else all_attrs.get(col) for col in self.additional_columns]
        values = ["" if all_attrs.get(col) is None else _logged(all_attrs.get(col)) for col in self.additional_columns if col != "batch_id"]
        if self.log_writer.has_batch:
            values.append(str(full_event.batch_id))
        return [*row_basic_info, *values]

    def append_any_enabled_batch_tasks(self, current_event: EnabledEvent) -> List[EnabledEvent]:
        enabled_datetime = CustomDatetimeAndSeconds(current_event.enabled_at, current_event.enabled_datetime)
        enabled_batch_task_ids = self.sim_setup.is_any_batch_enabled(enabled_datetime)

        if enabled_batch_task_ids is not None:
            for batch_task_id, batch_info in enabled_batch_task_ids.items():
                start_time_from_rule = batch_info.start_time_from_rule

                # TODO: cover with additional test cases
                # when start_time_from_rule > current_event.enabled_datetime

                if start_time_from_rule < current_event.enabled_datetime:
                    # get needed value in seconds according to the
                    # already existing pair of seconds and datetime
                    timedelta_sec = (current_event.enabled_datetime - start_time_from_rule).total_seconds()
                    enabled_at = current_event.enabled_at - timedelta_sec
                    enabled_datetime = start_time_from_rule
                else:
                    enabled_at = current_event.enabled_at
                    enabled_datetime = current_event.enabled_datetime

                c_event = EnabledEvent(
                    current_event.p_case,
                    current_event.p_state,
                    batch_task_id,
                    enabled_at,
                    enabled_datetime,
                    batch_info,
                )
                self.events_queue.append_event(c_event)

    def execute_if_any_unexecuted_batch(self, last_task_enabled_time: CustomDatetimeAndSeconds):
        for case_id, enabled_datetime in self.sim_setup.is_any_unexecuted_batch(last_task_enabled_time):
            if not enabled_datetime:
                return

            enabled_batch_task_ids = self.sim_setup.is_any_batch_enabled(enabled_datetime)

            if not len(enabled_batch_task_ids):
                # no rules were satisfied
                # check whether there are some invalid rules
                invalid_batches = self.sim_setup.get_invalid_batches_if_any(last_task_enabled_time)
                if invalid_batches is not None:
                    for key, item in invalid_batches.items():
                        if key not in enabled_batch_task_ids:
                            enabled_batch_task_ids[key] = item

            if enabled_batch_task_ids is not None:
                for batch_task_id, batch_info in enabled_batch_task_ids.items():
                    c_event = EnabledEvent(
                        case_id,
                        self.all_process_states[case_id],
                        batch_task_id,
                        self.simulation_at_from_datetime(batch_info.start_time_from_rule),
                        batch_info.start_time_from_rule,
                        batch_info,
                    )
                    self.events_queue.append_event(c_event)

    def _get_chunk(self, batch_spec, curr_index, all_case_ids):
        """Return only the part of the all_case_ids that will be executed as a batch"""
        acc_tasks_in_batch = 0
        for i in range(0, curr_index):
            acc_tasks_in_batch = acc_tasks_in_batch + batch_spec[i]
        num_tasks_in_batch = batch_spec[curr_index]
        return all_case_ids[acc_tasks_in_batch : acc_tasks_in_batch + num_tasks_in_batch]

    def execute_task_batch(self, c_event: EnabledEvent):
        all_tasks_waiting = len(c_event.batch_info_exec.case_ids)

        if all_tasks_waiting == 0:
            print("WARNING: Number of tasks in the enabled batch is 0.")

        all_case_ids = list(c_event.batch_info_exec.case_ids.items())
        ordered_case_ids = self.case_prioritisation.get_ordered_case_ids_by_priority(all_case_ids)
        batch_spec = c_event.batch_info_exec.batch_spec
        chunks = [self._get_chunk(batch_spec, i, ordered_case_ids) for i in range(0, len(batch_spec))]

        if c_event.batch_info_exec.is_sequential():
            return self.execute_seq_task_batch(c_event, chunks)
        elif c_event.batch_info_exec.is_parallel():
            return self.execute_parallel_task_batch(c_event, chunks)
        else:
            print(f"WARNING: {c_event.batch_info_exec.task_batch_info.type} not supported")

    def execute_seq_task_batch(self, c_event: EnabledEvent, chunks):
        start_time_from_rule_seconds = (
            c_event.batch_info_exec.start_time_from_rule - self.sim_setup.start_datetime
        ).total_seconds()

        for batch_item in chunks:
            num_tasks_in_batch = len(batch_item)

            r_id, r_avail_at = self.pop_and_allocate_resource(c_event.task_id, num_tasks_in_batch)

            completed_at = 0

            for case_id, enabled_time in batch_item:
                p_case = case_id
                task_id = c_event.task_id
                enabled_at = enabled_time.seconds_from_start
                enabled_datetime = enabled_time.datetime
                enabled_batch = c_event.enabled_at

                r_avail_at = max(
                    enabled_at,
                    r_avail_at,
                    enabled_batch,
                    completed_at,
                    start_time_from_rule_seconds,
                )
                avail_datetime = self._datetime_from(r_avail_at)
                is_working, _ = self.sim_setup.get_resource_calendar(r_id).is_working_datetime(avail_datetime)
                if not is_working:
                    r_avail_at = r_avail_at + self.sim_setup.next_resting_time(r_id, avail_datetime)

                full_evt = TaskEvent(
                    p_case,
                    task_id,
                    r_id,
                    r_avail_at,
                    enabled_at,
                    enabled_datetime,
                    self,
                    num_tasks_in_batch,
                    c_event.batch_info_exec.batch_id
                )

                self.sim_resources[r_id].worked_time += full_evt.ideal_duration
                (
                    completed_at,
                    completed_datetime,
                ) = self._update_logs_and_resource_availability(full_evt, r_id)

                yield completed_at, completed_datetime, p_case

    def execute_parallel_task_batch(self, c_event: EnabledEvent, chunks):
        task_id = c_event.task_id

        start_time_from_rule_datetime = c_event.batch_info_exec.start_time_from_rule
        if start_time_from_rule_datetime == None:
            start_time_from_rule_seconds = 0
            enabled_batch = c_event.enabled_at
        else:
            # edge case: start_time_from_rule overwrite the enabled time from the execution
            # happens when we entered the day (e.g., Monday) during the time
            # waiting for the task execution in the queue
            start_time_from_rule_seconds = (
                c_event.batch_info_exec.start_time_from_rule - self.sim_setup.start_datetime
            ).total_seconds()
            enabled_batch = 0

        for batch_item in chunks:
            num_tasks_in_batch = len(batch_item)

            r_id, r_avail_at = self.pop_and_allocate_resource(c_event.task_id, num_tasks_in_batch)

            r_avail_at = max(r_avail_at, enabled_batch, start_time_from_rule_seconds)
            avail_datetime = self._datetime_from(r_avail_at)
            is_working, _ = self.sim_setup.get_resource_calendar(r_id).is_working_datetime(avail_datetime)
            if not is_working:
                r_avail_at = r_avail_at + self.sim_setup.next_resting_time(r_id, avail_datetime)

            for case_id, enabled_time in batch_item:
                p_case = case_id
                enabled_at = enabled_time.seconds_from_start
                enabled_datetime = enabled_time.datetime

                full_evt = TaskEvent(
                    p_case,
                    task_id,
                    r_id,
                    r_avail_at,
                    enabled_at,
                    enabled_datetime,
                    self,
                    num_tasks_in_batch,
                    c_event.batch_info_exec.batch_id
                )

                (
                    completed_at,
                    completed_datetime,
                ) = self._update_logs_and_resource_availability(full_evt, r_id)

                yield completed_at, completed_datetime, p_case

            # since the tasks are executed in parallel
            # we add their duration only once cause they were happening at the same time
            self.sim_resources[r_id].worked_time += full_evt.ideal_duration

    def _update_logs_and_resource_availability(self, full_evt: TaskEvent, r_id):
        self.log_info.add_event_info(full_evt.p_case, full_evt, self.sim_setup.resources_map[r_id].cost_per_hour)

        r_next_available = full_evt.completed_at

        if self.sim_resources[r_id].switching_time > 0:
            r_next_available += self.sim_setup.next_resting_time(r_id, full_evt.completed_datetime)

        self.resource_queue.update_resource_availability(r_id, r_next_available)

        self._log_row(full_evt)

        completed_at = full_evt.completed_at
        completed_datetime = full_evt.completed_datetime

        return completed_at, completed_datetime

    def _log_passed_throw_events(self, p_case, p_state, visited_at):
        # throw events pass their token straight on inside update_process_state, so they never
        # reach execute_event; like other intermediate events, they are logged only on request
        if not self.sim_setup.is_event_added_to_log:
            return
        for e_id, passed_at in visited_at.items():
            if self.sim_setup.bpmn_graph.element_info[e_id].type is BPMN.INTERMEDIATE_THROW_EVENT:
                c_event = EnabledEvent(p_case, p_state, e_id, passed_at.seconds_from_start, passed_at.datetime)
                full_evt = TaskEvent.create_event_entity(c_event, passed_at.seconds_from_start, passed_at.datetime)
                self._log_row(full_evt)

    def execute_event(self, c_event):
        # Handle event types separately (they don't need assigned resource)
        if c_event.parked_event is not None or c_event.armed_event is not None or c_event.complete_at_once:
            # resumed by a message: the event lasted from when the case reached it until the message;
            # a timer that won a race: from when it was armed until it fired; nothing to collect: no time
            reached = c_event.parked_event or c_event.armed_event or c_event
            ended_at, ended_datetime = c_event.enabled_at, c_event.enabled_datetime
            if c_event.armed_event is not None:  # logged at the timer's real time, not a microsecond later
                ended_at = c_event.armed_event.enabled_at + c_event.duration_sec
                ended_datetime = c_event.armed_event.enabled_datetime + timedelta(seconds=c_event.duration_sec)
            full_evt = TaskEvent.create_event_entity(reached, ended_at, ended_datetime)
            self.log_info.add_event_info(c_event.p_case, full_evt, 0)
            if self.sim_setup.is_event_added_to_log:
                self._log_row(full_evt)
            return c_event.enabled_at, c_event.enabled_datetime
        event_duration_seconds = None
        event_element = self.sim_setup.bpmn_graph.element_info[c_event.task_id]
        [event_duration_seconds] = self.sim_setup.bpmn_graph.event_duration(event_element.id)

        completed_at = c_event.enabled_at + event_duration_seconds
        completed_datetime = c_event.enabled_datetime + timedelta(seconds=event_duration_seconds)

        full_evt = TaskEvent.create_event_entity(c_event, completed_at, completed_datetime)

        self.log_info.add_event_info(c_event.p_case, full_evt, 0)

        if self.sim_setup.is_event_added_to_log:
            self._log_row(full_evt)

        return completed_at, completed_datetime

    def _datetime_from(self, in_seconds):
        return self.simulation_datetime_from(in_seconds) if in_seconds is not None else None

    def simulation_datetime_from(self, simpy_time):
        return self.sim_setup.start_datetime + timedelta(seconds=simpy_time)

    def simulation_at_from_datetime(self, datetime):
        td = datetime - self.sim_setup.start_datetime
        return td.total_seconds()

    def get_utilization_for(self, resource_id):
        if self.sim_resources[resource_id].available_time == 0:
            return -1
        return self.sim_resources[resource_id].worked_time / self.sim_resources[resource_id].available_time

    def _find_worked_times(self, event_info, completed_events):
        i = len(completed_events) - 1
        resource_calendar = self.sim_setup.get_resource_calendar(event_info.resource_id)
        current_end = event_info.completed_at
        duration = 0
        while i >= 0:
            prev_event = completed_events[i]
            if event_info.started_at >= prev_event.completed_at:
                break
            else:
                if prev_event.completed_at < current_end:
                    duration += resource_calendar.find_working_time(prev_event.completed_at, current_end)
                if event_info.started_at < prev_event.started_at:
                    current_end = prev_event.started_at
                else:
                    return duration
            i -= 1
        return duration + resource_calendar.find_working_time(event_info.started_at, current_end)


def execute_full_process(bpm_env: SimBPMEnv, fixed_starting_times=None):
    # Initialize event queue with the arrival times of all the cases to simulate,
    # i.e., all the initial events are enqueued and sorted by their arrival times
    # s_t = datetime.datetime.now()
    bpm_env._ensure_arrivals_generated(fixed_starting_times)

    # print("Generation of all cases: %s" %
    #       str(datetime.timedelta(seconds=(datetime.datetime.now() - s_t).total_seconds())))
    current_event = bpm_env.events_queue.pop_next_event()
    executed_cases = set()

    while current_event is not None:
        if current_event.p_case not in executed_cases:
            executed_cases.add(current_event.p_case)
            global_case_attributes = bpm_env.sim_setup.all_attributes.global_case_attributes.attributes
            new_attributes = {attr.name: attr.get_next_value() for attr in global_case_attributes}
            bpm_env.sim_setup.bpmn_graph.all_attributes["global"].update(new_attributes)

        bpm_env.execute_enabled_event(current_event)

        # the enabled time of the event we just executed, used below as the reference point
        # to look for any batched task that should fire once the queue runs dry
        last_event_datetime = CustomDatetimeAndSeconds(current_event.enabled_at, current_event.enabled_datetime)

        # find the next event to be executed
        # double-check whether there are elements that need to be executed before the start of the event
        # add found elements to the queue, if any
        intermediate_event = bpm_env.events_queue.peek()
        if intermediate_event is not None:
            bpm_env.append_any_enabled_batch_tasks(intermediate_event)
        else:
            # we reached the point where all tasks enabled for the execution were executed
            # add to the events_queue batched tasks if any
            bpm_env.execute_if_any_unexecuted_batch(last_event_datetime)

        # the queue is now fully settled (any due batched work has been flushed into it),
        # so it is safe for next_event_time() to read it via a plain peek()
        yield current_event

        current_event = bpm_env.events_queue.pop_next_event()


def run_simulation(
    bpmn_path,
    json_path,
    total_cases,
    stat_out_path=None,
    log_out_path=None,
    starting_at=None,
    is_event_added_to_log=False,
    fixed_arrival_times=None,
):
    starting_at_datetime = (
        parse_datetime(starting_at, True) if starting_at else pytz.utc.localize(datetime.datetime.now())
    )
    diffsim_info = SimDiffSetup(bpmn_path, json_path, is_event_added_to_log, total_cases, starting_at_datetime)

    if not diffsim_info:
        return None

    if stat_out_path is None and log_out_path is None:
        return run_simpy_simulation(diffsim_info, None, None, fixed_arrival_times)

    csv_writer_config = {
        'delimiter': ',',
        'quotechar': '"',
        'quoting': csv.QUOTE_MINIMAL
    }

    stat_csv_file = open(stat_out_path, mode="w", newline="", encoding="utf-8") if stat_out_path else None
    log_csv_file = open(log_out_path, mode="w", newline="", encoding="utf-8") if log_out_path else None

    try:
        stat_writer = csv.writer(stat_csv_file, **csv_writer_config) if stat_csv_file else None
        log_writer = csv.writer(log_csv_file, **csv_writer_config) if log_csv_file else None

        result = run_simpy_simulation(diffsim_info, stat_writer, log_writer, fixed_arrival_times)
    finally:
        if stat_csv_file:
            stat_csv_file.close()
        if log_csv_file:
            log_csv_file.close()

    warning_file_name = "simulation_warnings.txt"
    if stat_out_path:
        warning_file_path = os.path.join(os.path.dirname(stat_out_path), warning_file_name)
    elif log_out_path:
        warning_file_path = os.path.join(os.path.dirname(log_out_path), warning_file_name)
    else:
        warning_file_path = warning_file_name

    with open(warning_file_path, "w") as warning_file:
        for warning in warning_logger.get_all_warnings():
            warning_file.write(f"{warning}\n")

    return result


def run_simpy_simulation(diffsim_info, stat_fwriter, log_fwriter, fixed_starting_times=None):
    bpm_env = SimBPMEnv(diffsim_info, stat_fwriter, log_fwriter)
    for _ in execute_full_process(bpm_env, fixed_starting_times):
        pass
    if fixed_starting_times is not None:
        return bpm_env
    if log_fwriter is None and stat_fwriter is None:
        return bpm_env.log_info.compute_process_kpi(bpm_env), bpm_env.log_info
    if log_fwriter:
        bpm_env.log_writer.force_write()
    if stat_fwriter:
        bpm_env.log_info.save_joint_statistics(bpm_env)

    warning_logger.add_warnings(bpm_env.sim_setup.bpmn_graph.simulation_execution_stats.find_issues())

    return None


def verify_miliseconds(array):
    """
    In case of datetime.microsecond = 0, standard converter does not print microseconds
    So we force the conversion, so that the datetime format is the same for every datetime in the final file
    Indexes correspond to the next values:
        2 - enabled_datetime
        3 - start_datetime
        4 - end_datetime
    """
    for i in range(2, 5):
        if array[i].microsecond == 0:
            array[i] = _get_string_from_datetime(array[i])

    return array


def _get_string_from_datetime(datetime):
    datetime_without_colon = datetime.strftime("%Y-%m-%d %H:%M:%S.%f%z")
    return "{0}:{1}".format(datetime_without_colon[:-2], datetime_without_colon[-2:])


def _logged(value):
    """A list (an appended attribute) as JSON text, e.g. ["Warehouse-12", "Warehouse-17"]; anything else as it is."""
    return json.dumps(value) if isinstance(value, list) else value
