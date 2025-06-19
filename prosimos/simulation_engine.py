import csv
import datetime
import os
import numpy as np
from datetime import timedelta
from typing import List

import pytz

from prosimos.control_flow_manager import (
    BPMN,
    CustomDatetimeAndSeconds,
)
from prosimos.execution_info import EnabledEvent, TaskEvent, Trace
from prosimos.file_manager import FileManager
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


class SimBPMEnv:
    def __init__(self, sim_setup: SimDiffSetup, stat_fwriter, log_fwriter, process_state=None, simulation_horizon=None):
        self._last_case_arrival_dt = None
        self.sim_setup = sim_setup
        self.sim_resources = dict()
        self.stat_fwriter = stat_fwriter
        self.additional_columns = self.sim_setup.all_attributes.get_all_columns_generated()
        self.log_writer = FileManager(10000, log_fwriter, self.additional_columns)
        self.log_info = LogInfo(sim_setup)
        self.executed_events = 0
        self.time_update_process_state = 0
        self.all_process_states = dict()
        self.simulation_horizon = simulation_horizon
        self.cases_first_start = {}
        self.cases_skip = {}

        self.case_prioritisation = CasePrioritisation(
            self.sim_setup.total_num_cases,
            self.sim_setup.case_attributes,
            self.sim_setup.prioritisation_rules,
        )

        all_attributes = {
            "global": self.sim_setup.all_attributes.global_attribute_initial_values,
            **self.case_prioritisation.all_case_attributes
        }

        self.sim_setup.bpmn_graph.all_attributes = all_attributes

        if process_state:
            self.initialize_from_process_state(process_state)
        else:
            r_first_available = dict()
            for r_id in sim_setup.resources_map:
                self.sim_resources[r_id] = SimResource()
                r_first_available[r_id] = self.sim_setup.next_resting_time(r_id, self.sim_setup.start_datetime)

            self.resource_queue = DiffResourceQueue(self.sim_setup.task_resource, r_first_available)
            self.events_queue = EventQueue()
            self.generate_all_arrival_events()

    def filter_event_log(self):
        if self.simulation_horizon is None:
            return
        filtered_traces = {}
        for case_id, trace in self.log_info.trace_list.items():
            if not trace.event_list:
                continue
            case_start_time = trace.event_list[0].started_datetime
            if case_start_time < self.sim_setup.simulation_horizon:
                # Keep entire case
                filtered_traces[case_id] = trace
            # else skip
        self.log_info.trace_list = filtered_traces

    def initialize_from_process_state(self, process_state):
        """
        Initializes resources and partial states WITHOUT using resource_last_end_times.
        Instead, we compute resource availability from the ongoing tasks in the partial state.
        """
        # remember when the last case in the snapshot arrived
        self._last_case_arrival_dt = None
        if "last_case_arrival" in process_state:
            raw = process_state["last_case_arrival"]
            if isinstance(raw, str):
                self._last_case_arrival_dt = parse_datetime(raw, has_date=True)
            else:
                self._last_case_arrival_dt = raw
        
        resource_name_to_id = self.sim_setup.resource_name_to_id

        # ---------------------------------------------------------
        # 1) Compute resource availability from ongoing activities
        # ---------------------------------------------------------
        r_first_available = {}
        for r_id in self.sim_setup.resources_map:
            # Default each resource’s next availability to start_datetime or next_resting_time
            self.sim_resources[r_id] = SimResource()
            default_avail = self.sim_setup.next_resting_time(r_id, self.sim_setup.start_datetime)
            r_first_available[r_id] = default_avail

        for r_id in self.sim_setup.resources_map:
            if r_id == '94bb2af225d701db1038d35bc62a1563b0403c76a3eb217ca1e46fb27fc86c35':
                print(f"[R-INIT] {r_id} default_avail={r_first_available[r_id]} s")
        # We'll keep track of the maximum time each resource is occupied by any ongoing task
        # so that we can update r_first_available properly.
        resource_end_times_map = {r_id: r_first_available[r_id] for r_id in self.sim_setup.resources_map}

        # ------------------------------------------------------
        # 2) Set up event queue, process states, ongoing tasks
        # ------------------------------------------------------
        self.resource_queue = DiffResourceQueue(self.sim_setup.task_resource, r_first_available)
        self.events_queue = EventQueue()

        for case_id_str, case_data in process_state.get('cases', {}).items():
            case_id = int(case_id_str)
            p_state = self.sim_setup.initial_state()

            # Set tokens based on control_flow_state
            flows = case_data.get('control_flow_state', {}).get('flows', [])
            tokens = {flow_id: 1 for flow_id in flows}
            p_state.set_tokens(tokens)
            # print(f"Set tokens for case {case_id}: {tokens}")

            self.all_process_states[case_id] = p_state
            self.log_info.trace_list[case_id] = Trace(case_id, self.sim_setup.start_datetime)

            # --------------------------------------
            # 2a) Handle ongoing activities
            # --------------------------------------
            for activity in case_data.get("ongoing_activities", []):
                task_id = activity["id"]
                if case_id == '58':
                    print("processing case 58")
                # task_id = self.sim_setup.bpmn_graph.get_task_id_by_name(task_name)
                resource_name = activity.get("resource")

                start_time = activity.get("start_time")  # Real datetime
                if isinstance(start_time, str):
                    start_time = parse_datetime(start_time)

                # If the partial state includes an explicit 'enabled_time', use that;
                # otherwise, fallback to 'start_time'.
                enabled_time_dt = activity.get("enabled_time")  # e.g., "2024-03-01T12:00:00+00:00"
                if enabled_time_dt is None:
                    enabled_time_dt = start_time  # fallback
                elif isinstance(enabled_time_dt, str):
                    enabled_time_dt = parse_datetime(enabled_time_dt)

                # Convert that to simulation seconds
                enabled_at = (enabled_time_dt - self.sim_setup.start_datetime).total_seconds()
                # if enabled_at < 0:
                #     print(f"Adjusted enabled_at for case {case_id} from {enabled_at} to 0.")
                #     enabled_at = 0

                # Also convert the real start_time to simulation seconds
                started_at = (start_time - self.sim_setup.start_datetime).total_seconds()
                # if started_at < 0:
                #     print(f"Adjusted started_at for case {case_id} from {started_at} to 0.")
                #     started_at = 0

                # Map resource name to ID
                resource_id = resource_name_to_id.get(resource_name)
                if int(case_id) == 58:
                    print(f"[DEBUG] Mapped resource ID: {resource_id}")

                if resource_id is None:
                    # External resource
                    # print(
                    #     f"Resource '{resource_name}' not found in simulation parameters. Treating as external resource.")
                    resource_in_pool = False
                    resource_id = resource_name  # just store the raw name
                else:
                    resource_in_pool = True
                    # If the resource cannot do that task, pick a capable one
                    if (task_id not in self.sim_setup.task_resource or
                            resource_id not in self.sim_setup.task_resource[task_id]):
                        # print(
                        #     f"Resource '{resource_id}' cannot perform task '{task_id}'. Assigning a capable resource.")
                        possible_resources = self.sim_setup.task_resource.get(task_id, {})
                        if possible_resources:
                            # Just pick one
                            resource_id = np.random.choice(list(possible_resources.keys()))
                            resource_in_pool = resource_id in self.sim_setup.resources_map
                        else:
                            print(f"No resources available for task '{task_id}'.")
                            continue
                if int(case_id) == 58:
                    print(f"[DEBUG] Ongoing activity for case 58:")
                    print(f"        Task ID       : {task_id}")
                    print(f"        Resource      : {resource_name}")
                    print(f"        Start time    : {start_time}")
                    print(f"        Enabled time  : {enabled_time_dt}")

                # --- BEGIN replacement block ------------------------------------------
                remaining_duration = activity.get("remaining_duration")
                if remaining_duration is None and "remaining_time" in activity:
                    remaining_duration = float(activity["remaining_time"])

                if remaining_duration is None:
                    # 1️⃣  pick a distribution to sample the *total* ideal working time
                    possible = self.sim_setup.task_resource.get(task_id, {})
                    dist_res_id = (
                        resource_id if resource_id in possible
                        else next(iter(possible)) if possible else None
                    )
                    if dist_res_id is None:
                        print(f"[WARN] No duration distribution for task '{task_id}'.")
                        continue
                    total_working_sec = (
                        self.sim_setup.task_resource[task_id][dist_res_id]
                        .generate_sample(1)[0]
                    )

                    # 2️⃣  measure how many *working* seconds have already happened
                    if resource_in_pool:
                        calendar = self.sim_setup.get_resource_calendar(resource_id)
                        worked_before_anchor = calendar.find_working_time(
                            start_time, self.sim_setup.start_datetime
                        )
                    else:
                        # external resource – fallback to wall-clock subtraction
                        worked_before_anchor = max(
                            (self.sim_setup.start_datetime - start_time).total_seconds(), 0
                        )

                    # 3️⃣  compute the residual working time (at least 1 s)
                    remaining_duration = max(total_working_sec - worked_before_anchor, 1)
                # --- END replacement block --------------------------------------------

                if int(case_id) == 58:
                    print(f"[DEBUG] Computed heuristic duration for task {task_id}: {remaining_duration}")

                # Create the event in the queue
                enabled_event = EnabledEvent(
                    p_case=case_id,
                    p_state=p_state,
                    task_id=task_id,
                    enabled_at=enabled_at,
                    enabled_datetime=enabled_time_dt,
                    duration_sec=remaining_duration,
                    assigned_resource_id=resource_id
                )
                if int(case_id == 58):
                    print(f"[DEBUG] Scheduling EnabledEvent for case 58:")
                    print(f"        Enabled at     : {enabled_at}")
                    print(f"        Started at     : {started_at}")
                    print(f"        Duration       : {remaining_duration}")
                    print(f"        Assigned Res ID: {resource_id}")


                enabled_event.started_at = started_at
                enabled_event.started_datetime = start_time
                enabled_event.from_process_state = True
                self.calc_priority_and_append_to_queue(enabled_event, is_arrival_event=False)

                # Update the resource's next availability time if it's in the pool
                if resource_in_pool and resource_id in resource_end_times_map:
                    # The activity will end at started + remaining
                    # started = enabled_at in seconds from sim start
                    # ends_at = started + remaining
                    activity_ends_at = started_at + remaining_duration
                    if activity_ends_at < 0:
                        activity_ends_at = 0
                    # Keep the maximum
                    if activity_ends_at > resource_end_times_map[resource_id]:
                        resource_end_times_map[resource_id] = activity_ends_at
                    if resource_id == '94bb2af225d701db1038d35bc62a1563b0403c76a3eb217ca1e46fb27fc86c35':
                        print(f"[R-END]  ends_at={activity_ends_at}  current={resource_end_times_map[resource_id]}")

                # print(
                #     f"Scheduled ongoing '{task_id}' for case {case_id} at sim time {enabled_at} with remaining {remaining_duration}.")

                # Update tokens for the ongoing activity
                task = self.sim_setup.bpmn_graph.element_info[task_id]
                incoming_flows = task.incoming_flows
                tokens_for_task = {flow_id: 1 for flow_id in incoming_flows}
                p_state.set_tokens(tokens_for_task)
                # print(f"Set tokens for case {case_id}: {tokens_for_task}")

            # --------------------------------------
            # 2b) Handle enabled (but not started) activities
            # --------------------------------------
            for activity in case_data.get('enabled_activities', []):
                task_id = activity['id']
                # task_id = self.sim_setup.bpmn_graph.get_task_id_by_name(task_name)
                enabled_time = activity['enabled_time']
                if isinstance(enabled_time, str):
                    enabled_time = parse_datetime(enabled_time)
                enabled_at = (enabled_time - self.sim_setup.start_datetime).total_seconds()

                enabled_event = EnabledEvent(
                    case_id,
                    p_state,
                    task_id,
                    enabled_at,
                    enabled_time
                )
                self.calc_priority_and_append_to_queue(enabled_event, is_arrival_event=False)
                # print(f"Scheduling enabled '{task_id}' for case {case_id} at {enabled_time}.")

            # --------------------------------------
            # 2c) Fire enabled gateways
            # --------------------------------------
            for gateway_info in case_data.get("enabled_gateways", []):
                gateway_id = gateway_info["id"]

                gw_enabled_time_dt = gateway_info.get("enabled_time")
                if gw_enabled_time_dt:
                    if isinstance(gw_enabled_time_dt, str):
                        gw_enabled_time_dt = parse_datetime(gw_enabled_time_dt, True)
                else:
                    gw_enabled_time_dt = self.sim_setup.start_datetime

                gw_enabled_at = (gw_enabled_time_dt - self.sim_setup.start_datetime).total_seconds()

                # Instead of putting it in the event queue, we "execute" the gateway logic right now:
                # print(f"Immediately firing gateway '{gateway_id}' for case={case_id} at {gw_enabled_time_dt}.")

                # We treat the gateway as if it "completed" at its enabled time
                dummy_time = CustomDatetimeAndSeconds(gw_enabled_at, gw_enabled_time_dt)
                print(f"gateway enabled in case {case_id} at {dummy_time.datetime} with remaining {dummy_time.seconds_from_start} seconds.")

                # Force the BFS update in control_flow_manager:
                enabled_tasks, visited_at = self.sim_setup.update_process_state(
                    case_id, gateway_id, p_state, dummy_time
                )

                # Now any tasks enabled by that gateway can be scheduled
                for next_task in enabled_tasks:
                    # print(f"Next task for case ={case_id} after firing gateway: {next_task.task_id}")
                    # next_task.task_id is an ID that must be scheduled
                    visited_time = visited_at[next_task.task_id]
                    print(f"in case {case_id} next task is: {next_task.task_id} and enabled time at {visited_time.datetime}.")
                    new_evt = EnabledEvent(
                        p_case=case_id,
                        p_state=p_state,
                        task_id=next_task.task_id,
                        enabled_at=visited_time.seconds_from_start,
                        enabled_datetime=visited_time.datetime
                    )
                    new_evt.from_process_state = True
                    self.calc_priority_and_append_to_queue(new_evt, is_arrival_event=False)

            # --------------------------------------
            # 2d) Handle enabled events
            # --------------------------------------
            for event in case_data.get("enabled_events", []):
                event_id = event["id"]
                enabled_time = event["enabled_time"]
                if isinstance(enabled_time, str):
                    enabled_time = parse_datetime(enabled_time)
                enabled_at = (enabled_time - self.sim_setup.start_datetime).total_seconds()
                enabled_event = EnabledEvent(
                    p_case=case_id,
                    p_state=p_state,
                    task_id=event_id,
                    enabled_at=enabled_at,
                    enabled_datetime=enabled_time,
                )
                # Immediately execute the enabled event using our new logic.
                self.execute_event_from_process_state(enabled_event)

            # print(f"Initialized case {case_id} with partial process state.")
            # if 58 in self.all_process_states:
            #     print("[DEBUG] Event queue after initialization (filtered for case 58):")
            #     for priority, count, evt in self.events_queue.enabled_events.pq:
            #         if evt != self.events_queue.enabled_events.REMOVED and evt.p_case == 58:
            #             print(
            #                 f"    Task {evt.task_id} at {evt.enabled_datetime} (start={evt.started_at}, duration={evt.duration_sec})")

            # print("[DEBUG] Final resource availability times:")
            # for r_id, avail_time in r_first_available.items():
            #     print(f"    Resource {r_id}: available at {avail_time:.2f}")


        # ------------------------------------------------------
        # 3) Update resource_queue with final computed avail.
        # ------------------------------------------------------
        for r_id, end_time_sec in resource_end_times_map.items():
            r_first_available[r_id] = max(r_first_available[r_id], end_time_sec)
            self.sim_resources[r_id].available_time = r_first_available[r_id]
        self.resource_queue = DiffResourceQueue(self.sim_setup.task_resource, r_first_available)


        # print("Events in event queue after initialization:", self.events_queue.enabled_events)

        # ------------------------------------------------------
        # 4) Generate any additional arrival events if needed
        # ------------------------------------------------------
        self.generate_remaining_arrival_events(process_state)

    def generate_remaining_arrival_events(self, process_state):
        """
        Updated: Generate remaining arrival events using the simulation start datetime
        as the base time for new arrivals instead of using the earliest case start times.
        """
        existing_case_ids = set(int(cid) for cid in process_state['cases'].keys())
        total_existing_cases = len(existing_case_ids)
        process_state_cases = total_existing_cases
        total_cases_needed = self.sim_setup.total_num_cases

        # If we already have enough cases in partial state, do nothing
        if total_existing_cases >= total_cases_needed:
            return

        # -------------------------------------------------------------
        # Use simulation start datetime as the baseline for new arrivals
        # -------------------------------------------------------------
        if hasattr(self, "_last_case_arrival_dt") and self._last_case_arrival_dt:
            current_time = self._last_case_arrival_dt
        else:
            current_time = self.sim_setup.start_datetime

        # Determine the next case id. If there are existing cases, start from the next one;
        # otherwise, start at 0.
        case_id = max(existing_case_ids) + 1 if existing_case_ids else 0

        # -------------------------------------------------------------
        # Schedule new arrivals until we reach total_num_cases
        # -------------------------------------------------------------
        while total_existing_cases < total_cases_needed:
            inter_arrival_seconds = self.sim_setup.next_arrival_time(current_time)
            next_arrival_time = current_time + timedelta(seconds=inter_arrival_seconds)

            enabled_at = (next_arrival_time - self.sim_setup.start_datetime).total_seconds()
            arrival_event = EnabledEvent(
                p_case=case_id,
                p_state=self.sim_setup.initial_state(),
                task_id=self.sim_setup.bpmn_graph.starting_event,
                enabled_at=enabled_at,
                enabled_datetime=next_arrival_time
            )
            self.calc_priority_and_append_to_queue(arrival_event, is_arrival_event=True)

            current_time = next_arrival_time
            case_id += 1
            total_existing_cases += 1

        print(f"Added {process_state_cases} existing cases + new arrivals up to {total_cases_needed}.")

    def calc_priority_and_append_to_queue(self, enabled_event: EnabledEvent, is_arrival_event: bool):
        if enabled_event.is_inter_event:
            # append with the highest priority
            highest_priority = 0
            self.append_enabled_event_to_queue(enabled_event, is_arrival_event, highest_priority)
            return

        case_priority = self.calc_priority_for_task_or_batch(enabled_event)

        self.append_enabled_event_to_queue(enabled_event, is_arrival_event, case_priority)

    def calc_priority_for_task_or_batch(self, enabled_event):
        """
        Calculate case priority by following one of two path:
        1) no batching  - use current case's priority
        2) batching     - find case id with the highest priority
        """
        if enabled_event.batch_info_exec is not None:
            # batched task
            multiple_cases_dict = enabled_event.batch_info_exec.case_ids
            multiple_cases_arr = [(k, v) for k, v in multiple_cases_dict.items()]
            case_priority = self.case_prioritisation.calculate_max_priority(multiple_cases_arr)
        else:
            case_priority = self.case_prioritisation.get_priority_by_case_id(enabled_event.p_case)

        return case_priority

    def append_enabled_event_to_queue(self, enabled_event: EnabledEvent, is_arrival_event: bool, case_priority):
        "Append as either an arrival event or enabled intermediate/end event"
        if enabled_event.p_case == 58:
            print(f"[QUEUE-ADD] {enabled_event.task_id} at t={enabled_event.enabled_at}")
        if is_arrival_event:
            self.events_queue.append_arrival_event(enabled_event, case_priority)
        else:
            self.events_queue.append_enabled_event(enabled_event, case_priority)

    def generate_all_arrival_events(self):
        sim_setup = self.sim_setup
        arrival_time = 0
        for p_case in range(0, sim_setup.total_num_cases):
            enabled_datetime = self._update_initial_event_info(self.sim_setup, p_case, arrival_time)
            arrival_time += sim_setup.next_arrival_time(enabled_datetime)

    def generate_fixed_arrival_events(self, starting_times):
        p_case = 0
        for arrival_time in starting_times:
            self._update_initial_event_info(self.sim_setup, p_case, arrival_time)
            p_case += 1

    def _update_initial_event_info(self, sim_setup, p_case, arrival_time):
        for e_id in sim_setup.bpmn_graph.last_datetime:
            sim_setup.bpmn_graph.last_datetime[e_id][p_case] = None
        p_state = sim_setup.initial_state()
        enabled_datetime = self.simulation_datetime_from(arrival_time)
        enabled_time = CustomDatetimeAndSeconds(arrival_time, enabled_datetime)
        enabled_tasks, _ = sim_setup.update_process_state(
            p_case, sim_setup.bpmn_graph.starting_event, p_state, enabled_time
        )
        self.all_process_states[p_case] = p_state
        self.log_info.trace_list[p_case] = Trace(p_case, enabled_datetime)
        for task in enabled_tasks:
            task_id = task.task_id
            self.calc_priority_and_append_to_queue(
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
                True,
            )
        return enabled_datetime

    def execute_enabled_event(self, c_event: EnabledEvent, resource_in_pool=True):
        # ─────── debug helper ───────
        if c_event.p_case == 58:
            print(f"[EXECUTE] {c_event.task_id} "
                  f"resource={c_event.assigned_resource_id}")

        self.executed_events += 1
        e_info = self.sim_setup.bpmn_graph.element_info[c_event.task_id]

        # ------------------------------------------------------------------
        # 1) FAST-PATH for activities captured as *already running* in the
        #    snapshot.  We only take it when we **really** have `started_at`.
        # ------------------------------------------------------------------
        if (getattr(c_event, "from_process_state", False)
                and e_info.type == BPMN.TASK
                and getattr(c_event, "started_at", None) is not None):

            completed_at, completed_dt = self._execute_ongoing_partial_task(
                c_event, resource_in_pool
            )

        # ------------------------------------------------------------------
        # 2) FAST-PATH for enabled timer/message events from the snapshot
        # ------------------------------------------------------------------
        elif getattr(c_event, "from_process_state", False) and e_info.is_event():
            completed_at, completed_dt = self.execute_event_from_process_state(c_event)

        # ------------------------------------------------------------------
        # 3) NORMAL execution path for everything else
        # ------------------------------------------------------------------
        else:
            if e_info.type == BPMN.TASK and c_event.batch_info_exec is not None:
                # … (unchanged batch handling) …
                for completed_at, completed_dt, p_case in self.execute_task_batch(c_event):
                    p_state = self.all_process_states[p_case]
                    self._fire_bfs_after_completion(
                        p_case, c_event.task_id, p_state, completed_at, completed_dt
                    )
                return  # BFS done inside the loop

            elif e_info.type == BPMN.TASK:
                completed_at, completed_dt = self.execute_task(c_event, resource_in_pool)
            else:
                completed_at, completed_dt = self.execute_event(c_event)

        # ------------------------------------------------------------------
        # 4) BFS update for single-task / single-event completions
        # ------------------------------------------------------------------
        self._fire_bfs_after_completion(
            c_event.p_case, c_event.task_id, c_event.p_state,
            completed_at, completed_dt
        )

    def _fire_bfs_after_completion(self, p_case, finished_task_id,
                                   p_state, completed_at, completed_dt):
        enabled_time = CustomDatetimeAndSeconds(completed_at, completed_dt)
        enabled_tasks, visited_at = self.sim_setup.update_process_state(
            p_case, finished_task_id, p_state, enabled_time
        )
        for nxt in enabled_tasks:
            vt = visited_at[nxt.task_id]
            self.calc_priority_and_append_to_queue(
                EnabledEvent(
                    p_case,
                    p_state,
                    nxt.task_id,
                    vt.seconds_from_start,
                    vt.datetime,
                    nxt.batch_info_exec,
                    nxt.duration_sec,
                    nxt.is_event
                ),
                is_arrival_event=False
            )

    def _execute_ongoing_partial_task(self, c_event: EnabledEvent, resource_in_pool=True):
        """
        Use partial-state's started_at and do not override it.
        Then log it as normal. BFS is only done AFTER it completes.
        """
        task_id = c_event.task_id
        p_case = c_event.p_case

        # Resource assignment is still relevant if needed for logging or cost.
        if c_event.assigned_resource_id:
            resource_id = c_event.assigned_resource_id
        else:
            # If partial-state didn't specify resource, pick one or keep it external.
            resource_id = None

        duration = c_event.duration_sec if c_event.duration_sec is not None else 0
        # partial-state says "it started at X"
        # We trust partial-state. (Ensure it's non-negative if you prefer.)
        started_at = max(c_event.started_at, 0) if c_event.started_at is not None else 0
        started_datetime = c_event.started_datetime or self.simulation_datetime_from(started_at)

        # If there's a resource pool, we don't re-check resource availability for partial-state
        # because we assume it's already started.
        if resource_in_pool and resource_id in self.sim_setup.resources_map:
            real_duration = self.sim_setup.real_task_duration(duration, resource_id, started_datetime)
        else:
            real_duration = duration

        completed_at = started_at + real_duration
        completed_datetime = self.simulation_datetime_from(completed_at)

        # Build the TaskEvent
        full_evt = TaskEvent(
            p_case=p_case,
            task_id=task_id,
            resource_id=resource_id,
            started_at=started_at,
            started_datetime=started_datetime,
            enabled_at=c_event.enabled_at,  # partial-state
            enabled_datetime=c_event.enabled_datetime,
            real_duration=real_duration,
            ideal_duration=duration,
            bpm_env=self
        )

        # Update logs, resource usage
        if resource_id:
            self._update_logs_and_resource_availability(full_evt, resource_id, resource_in_pool)
        else:
            # If no resource, still log it
            self.log_info.add_event_info(p_case, full_evt, 0)
            row_data = self.get_csv_row_data(full_evt)
            if row_data:
                self.log_writer.add_csv_row(row_data)

        return completed_at, completed_datetime

    def pop_and_allocate_resource(self, task_id: str, num_allocated_tasks: int):
        r_id, r_avail_at = self.resource_queue.pop_resource_for(task_id)
        self.sim_resources[r_id].allocated_tasks += num_allocated_tasks
        return r_id, r_avail_at

    def execute_task(self, c_event: EnabledEvent, resource_in_pool=True):
        task_id = c_event.task_id
        p_case = c_event.p_case

        # Decide resource
        if c_event.assigned_resource_id:
            resource_id = c_event.assigned_resource_id
            resource_available_at = self.resource_queue.get_resource_availability(resource_id)
        else:
            resource_id, resource_available_at = self.pop_and_allocate_resource(task_id, num_allocated_tasks=1)
            self.sim_resources[resource_id].allocated_tasks += 1

        # Determine the duration
        if c_event.duration_sec is not None:
            duration = c_event.duration_sec
        else:
            duration = self.sim_setup.ideal_task_duration(task_id, resource_id, num_tasks_in_batch=0)

        # ---------------------------------------------
        # 1) If partial-state says "it started at X", use it
        # 2) Otherwise, do max(enabled_at, resource_avail)
        # ---------------------------------------------
        if hasattr(c_event, "started_at") and c_event.started_at is not None:
            started_at = max(c_event.started_at, resource_available_at)
            # If you want partial state to override resource availability, do:
            #   started_at = c_event.started_at
            #
            # If you want to ensure resource can't start before it's available,
            # you'd do: started_at = max(c_event.started_at, resource_available_at)
        else:
            started_at = max(c_event.enabled_at, resource_available_at)

        started_datetime = self.simulation_datetime_from(started_at)
        # print(f"Started at {started_datetime} and started {started_at}")

        # Calculate real duration (accounting for resource calendar)
        if resource_in_pool:
            real_duration = self.sim_setup.real_task_duration(duration, resource_id, started_datetime)
        else:
            real_duration = duration
        # print(f"DEBUG: started_at={started_at}, real_duration={real_duration}")

        completed_at = started_at + real_duration
        completed_datetime = self.simulation_datetime_from(completed_at)

        if c_event.p_case == 58:
            print(f"[START-CLAMP] started_at(raw)={started_datetime} "
                  f"resource_avail={resource_available_at}")

        # Create the TaskEvent
        full_evt = TaskEvent(
            p_case=p_case,
            task_id=task_id,
            resource_id=resource_id,
            started_at=started_at,
            started_datetime=started_datetime,
            enabled_at=c_event.enabled_at,
            enabled_datetime=c_event.enabled_datetime,
            real_duration=real_duration,
            ideal_duration=duration,
            bpm_env=self
        )

        # Update logs and resource availability
        self._update_logs_and_resource_availability(full_evt, resource_id, resource_in_pool)

        return completed_at, completed_datetime

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

    def get_csv_row_data(self, full_event: TaskEvent):
        """
        Return array of values for one line of the CSV file based on full_event information.
        Ensures all date/time fields are converted to strings.
        """

        # Check if the case started after the simulation horizon
        case_id = full_event.p_case

        # Get the start time of the case
        # trace = self.log_info.trace_list.get(case_id)
        # if trace and trace.event_list:
        #     case_start_time = trace.event_list[0].started_datetime
        # else:
        #     # If no events yet, use the start time of the current event
        #     case_start_time = full_event.started_datetime

        # If this case starts AFTER the horizon, skip it
        # if self.simulation_horizon is not None:
        #     if case_start_time >= self.simulation_horizon:
        #         return None

        # print(
        #     f"CSV Row Data: Case {full_event.p_case} started at {case_start_time} comparing to simulation_horizon {self.simulation_horizon}"
        # )

        # Determine resource name
        if hasattr(full_event, "resource_id"):
            if full_event.resource_id in self.sim_setup.resources_map:
                resource_name = self.sim_setup.resources_map[full_event.resource_id].resource_name
            else:
                # Resource is external or not in the pool
                resource_name = full_event.resource_id
        else:
            resource_name = "No assigned resource"

        # Build row with the basic info
        row_basic_info = [
            full_event.p_case,
            self.sim_setup.bpmn_graph.element_info[full_event.task_id].name,
            full_event.enabled_datetime,
            full_event.started_datetime,
            full_event.completed_datetime,
            resource_name,
        ]

        # -------------------------------------------------------------------
        # Convert each datetime in row_basic_info into a string (indexes 2-4)
        # -------------------------------------------------------------------
        for idx in [2, 3, 4]:  # enabled_datetime, started_datetime, completed_datetime
            if isinstance(row_basic_info[idx], datetime.datetime):
                row_basic_info[idx] = _get_string_from_datetime(row_basic_info[idx])

        # Add any additional attributes
        all_attrs = self.sim_setup.bpmn_graph.get_all_attributes(full_event.p_case)
        values = ["" if all_attrs.get(col) is None else all_attrs.get(col) for col in self.additional_columns]

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
                self.calc_priority_and_append_to_queue(c_event, False)

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
                    self.calc_priority_and_append_to_queue(c_event, False)

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
                )

                (
                    completed_at,
                    completed_datetime,
                ) = self._update_logs_and_resource_availability(full_evt, r_id)

                yield completed_at, completed_datetime, p_case

            # since the tasks are executed in parallel
            # we add their duration only once cause they were happening at the same time
            self.sim_resources[r_id].worked_time += full_evt.ideal_duration


    def _update_logs_and_resource_availability(self, full_evt: TaskEvent,
                                               r_id, resource_in_pool=True):
        # 0) cost
        resource_cost = (self.sim_setup.resources_map[r_id].cost_per_hour
                         if resource_in_pool and r_id in self.sim_setup.resources_map
                         else 0)
        self.log_info.add_event_info(full_evt.p_case, full_evt, resource_cost)

        # 1) update resource calendar / utilisation
        if resource_in_pool and r_id in self.sim_setup.resources_map:
            r_next_available = full_evt.completed_at
            if self.sim_resources[r_id].switching_time > 0:
                r_next_available += self.sim_setup.next_resting_time(
                    r_id, self.simulation_datetime_from(r_next_available)
                )
            self.resource_queue.update_resource_availability(r_id, r_next_available)
            self.sim_resources[r_id].worked_time += full_evt.real_duration

        # 2) horizon-based **case-skip** – decide once per case
        cid = full_evt.p_case
        first_start = self.cases_first_start.setdefault(cid, full_evt.started_datetime)
        if cid not in self.cases_skip:
            self.cases_skip[cid] = (self.simulation_horizon is not None and
                                    first_start >= self.sim_setup.simulation_horizon)

        # 3) actually write the CSV row if the case is not skipped
        if not self.cases_skip[cid]:
            row = self.get_csv_row_data(full_evt)
            if row:
                self.log_writer.add_csv_row(row)

        # 4) If not skipping, then proceed with writing the row to CSV
        row_data = self.get_csv_row_data(full_evt)
        if full_evt.p_case == 58:
            print(f"[LOG] row_data is {'None' if row_data is None else 'OK'}")

        if row_data:
            self.log_writer.add_csv_row(row_data)

            try:
                with open("../output.txt", "a", encoding="utf-8") as fh:
                    fh.write(f"{row_data}\n")
            except OSError as err:
                warning_logger.add_warning(
                    f"Could not write to ../output.txt → {err}"
                )

        return full_evt.completed_at, full_evt.completed_datetime

    def execute_event(self, c_event):
        # Handle event types separately (they don't need assigned resource)
        event_duration_seconds = None
        event_element = self.sim_setup.bpmn_graph.element_info[c_event.task_id]
        [event_duration_seconds] = self.sim_setup.bpmn_graph.event_duration(event_element.id)

        completed_at = c_event.enabled_at + event_duration_seconds
        completed_datetime = c_event.enabled_datetime + timedelta(seconds=event_duration_seconds)

        full_evt = TaskEvent.create_event_entity(c_event, completed_at, completed_datetime)

        self.log_info.add_event_info(c_event.p_case, full_evt, 0)

        if self.sim_setup.is_event_added_to_log:
            # Get the CSV row data
            row_data = self.get_csv_row_data(full_evt)
            if row_data:
                # Write event to log file
                self.log_writer.add_csv_row(row_data)
            else:
                # Event is not included due to filtering
                pass

        return completed_at, completed_datetime

    def execute_event_from_process_state(self, c_event):
        """
        Executes an enabled event that originates from the partial state.

        In this case, we assume that c_event.enabled_at is expressed as seconds relative
        to the simulation start (which is 0). It may be negative if the event was enabled
        before simulation start. We then compute the elapsed time as (current_sim_time - enabled_at),
        where for process state we take current_sim_time = 0.

        If the event’s duration (from the BPMN distribution) is less than or equal to the elapsed time,
        then its timer expired in the past and we set its completion time to
        c_event.enabled_at + duration (which may be negative), thereby preserving the past offset.
        Otherwise, we subtract the elapsed time from the duration and schedule it to complete
        after the remaining time.

        After firing, we update the process state (via sim_setup.update_process_state) so that subsequent
        activities become enabled.
        """
        # Retrieve event duration (in seconds) from the BPMN distribution.
        event_element = self.sim_setup.bpmn_graph.element_info[c_event.task_id]
        [duration] = self.sim_setup.bpmn_graph.event_duration(event_element.id)

        # For process state events, we assume the current simulation time is 0 (i.e. at simulation start)
        current_sim_time = 0
        elapsed = current_sim_time - c_event.enabled_at

        if duration <= elapsed:
            # The event’s timer expired in the past.
            completed_at = c_event.enabled_at + duration
            completed_datetime = c_event.enabled_datetime + timedelta(seconds=duration)
            print(f"Event {c_event.task_id} from process state (either enabled event or coming after gateway) expired in the past and completed at {completed_datetime}.")
        else:
            effective_duration = duration - elapsed
            completed_at = current_sim_time + effective_duration
            completed_datetime = self.simulation_datetime_from(completed_at)
            print(f"Event {c_event.task_id} from process state (either enabled event or coming after gateway) completed after simulation start at {effective_duration} seconds and completed at {completed_datetime}.")

        full_evt = TaskEvent.create_event_entity(c_event, completed_at, completed_datetime)
        self.log_info.add_event_info(c_event.p_case, full_evt, 0)
        if self.sim_setup.is_event_added_to_log:
            row_data = self.get_csv_row_data(full_evt)
            if row_data:
                self.log_writer.add_csv_row(row_data)

        # Update the process state for the case.
        if c_event.p_case in self.all_process_states:
            p_state = self.all_process_states[c_event.p_case]
            updated_time = CustomDatetimeAndSeconds(completed_at, completed_datetime)
            enabled_tasks, visited_at = self.sim_setup.update_process_state(
                c_event.p_case, c_event.task_id, p_state, updated_time)
            for next_task in enabled_tasks:
                vt = visited_at[next_task.task_id]
                new_evt = EnabledEvent(
                    p_case=c_event.p_case,
                    p_state=p_state,
                    task_id=next_task.task_id,
                    enabled_at=vt.seconds_from_start,
                    enabled_datetime=vt.datetime
                )
                self.calc_priority_and_append_to_queue(new_evt, is_arrival_event=False)
        return completed_at, completed_datetime

    def _datetime_from(self, in_seconds):
        return self.simulation_datetime_from(in_seconds) if in_seconds is not None else None

    def simulation_datetime_from(self, simpy_time):
        return self.sim_setup.start_datetime + timedelta(seconds=simpy_time)

    def simulation_at_from_datetime(self, datetime):
        td = datetime - self.sim_setup.start_datetime
        return td.seconds

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
    # Only generate arrival events if the events_queue is empty
    if bpm_env.events_queue.is_empty():
        if fixed_starting_times is None:
            bpm_env.generate_all_arrival_events()
        else:
            bpm_env.generate_fixed_arrival_events(fixed_starting_times)

    # print("Generation of all cases: %s" %
    #       str(datetime.timedelta(seconds=(datetime.datetime.now() - s_t).total_seconds())))
    current_event = bpm_env.events_queue.pop_next_event()
    if current_event and current_event.p_case == 58:
        print(f"[QUEUE-POP] case 58 → {current_event.task_id} "
              f"enabled@{current_event.enabled_at}")
    executed_cases = set()

    while current_event is not None:
        if current_event.p_case not in executed_cases:
            executed_cases.add(current_event.p_case)
            global_case_attributes = bpm_env.sim_setup.all_attributes.global_case_attributes.attributes
            new_attributes = {attr.name: attr.get_next_value() for attr in global_case_attributes}
            bpm_env.sim_setup.bpmn_graph.all_attributes["global"].update(new_attributes)

        # print(f"Processing event at simulation time: {current_event}")
        bpm_env.execute_enabled_event(current_event)

        # find the next event to be executed
        # double-check whether there are elements that need to be executed before the start of the event
        # add founded elements to the queue, if any
        intermediate_event = bpm_env.events_queue.peek()
        if intermediate_event is not None:
            bpm_env.append_any_enabled_batch_tasks(intermediate_event)

        current_event = bpm_env.events_queue.pop_next_event()
        if current_event is not None:
            # save the datetime of the last executed task in the flow
            last_event_datetime = CustomDatetimeAndSeconds(current_event.enabled_at, current_event.enabled_datetime)
        else:
            # we reached the point where all tasks enabled for the execution were executed
            # add to the events_queue batched tasks if any
            bpm_env.execute_if_any_unexecuted_batch(last_event_datetime)

            # verifying whether we still have (batched) tasks to be executed in the future
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
    process_state=None,
    simulation_horizon=None
):
    diffsim_info = SimDiffSetup(bpmn_path, json_path, is_event_added_to_log, total_cases, process_state=process_state, simulation_horizon=simulation_horizon)

    if not diffsim_info:
        return None

    # When no process_state, set start_datetime based on starting_at or current time
    starting_at_datetime = (
        parse_datetime(starting_at, True) if starting_at else pytz.utc.localize(datetime.datetime.now())
    )
    diffsim_info.set_starting_datetime(starting_at_datetime)
    diffsim_info.setup_horizon()

    if stat_out_path is None and log_out_path is None:
        return run_simpy_simulation(diffsim_info, None, None, process_state=process_state, simulation_horizon=simulation_horizon)

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

        result = run_simpy_simulation(diffsim_info, stat_writer, log_writer, fixed_starting_times=fixed_arrival_times, process_state=process_state, simulation_horizon=simulation_horizon)
        # print("run_simulation: result =", result)
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


def run_simpy_simulation(diffsim_info, stat_fwriter, log_fwriter, fixed_starting_times=None, process_state=None, simulation_horizon=None):
    bpm_env = SimBPMEnv(diffsim_info, stat_fwriter, log_fwriter, process_state=process_state, simulation_horizon=simulation_horizon)
    execute_full_process(bpm_env, fixed_starting_times)

    # bpm_env.filter_event_log()
    if fixed_starting_times is not None:
        return bpm_env
    if log_fwriter is None and stat_fwriter is None:
        return bpm_env.log_info.compute_process_kpi(bpm_env), bpm_env.log_info
    if log_fwriter:
        bpm_env.log_writer.force_write()
    if stat_fwriter:
        bpm_env.log_info.save_joint_statistics(bpm_env)

    warning_logger.add_warnings(bpm_env.sim_setup.bpmn_graph.simulation_execution_stats.find_issues())

    if 58 in bpm_env.log_info.trace_list:
        print(f"[TRACE] case 58 produced "
              f"{len(bpm_env.log_info.trace_list[58].event_list)} events")
    else:
        print("[TRACE] case 58 never reached log_info")
    # print("run_simpy_simulation: bpm_env =", bpm_env)
    return bpm_env


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
