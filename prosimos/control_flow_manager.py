import copy
import random
import secrets
import sys
from collections import deque
from enum import Enum
from typing import List

from pix_framework.statistics.distribution import DurationDistribution

from prosimos.batch_processing import (BATCH_TYPE, AndFiringRule,
                                       BatchConfigPerTask)
from prosimos.exceptions import InvalidBpmnModelException
from prosimos.weekday_helper import CustomDatetimeAndSeconds

seconds_per_unit = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}

class BatchInfoForExecution:
    def __init__(self, all_case_ids, task_batch_info, curr_task_id, batch_spec, start_time_from_rule):
        self.case_ids = all_case_ids[curr_task_id].copy()
        self.task_batch_info = task_batch_info[curr_task_id]
        self.batch_spec = batch_spec
        self.start_time_from_rule = start_time_from_rule

    def is_sequential(self):
        return self.task_batch_info.type == BATCH_TYPE.SEQUENTIAL
    
    def is_concurrent(self):
        return self.task_batch_info.type == BATCH_TYPE.CONCURRENT

    def is_parallel(self):
        return self.task_batch_info.type == BATCH_TYPE.PARALLEL


class EnabledTask:
    def __init__(self, task_id, batch_info_exec: BatchInfoForExecution = None, duration_sec = None, is_event = False):
        self.task_id = task_id
        self.batch_info_exec = batch_info_exec
        
        # is filled only for event-based gateways
        # (when we already know the duration at the control flow step)
        self.duration_sec = duration_sec
        self.is_event = is_event


class BPMN(Enum):
    TASK = 'TASK'
    START_EVENT = 'START-EVENT'
    END_EVENT = 'END-EVENT',
    INTERMEDIATE_EVENT = 'INTERMEDIATE_EVENT',
    EXCLUSIVE_GATEWAY = 'EXCLUSIVE-GATEWAY'
    INCLUSIVE_GATEWAY = 'INCLUSIVE-GATEWAY'
    PARALLEL_GATEWAY = 'PARALLEL-GATEWAY'
    EVENT_BASED_GATEWAY = 'EVENT-BASED-GATEWAY'
    UNDEFINED = 'UNDEFINED'

    @classmethod
    def is_event(cls, type):
        if (type in [cls.START_EVENT, cls.END_EVENT, cls.INTERMEDIATE_EVENT]):
            return True
        else:
            return False

class EVENT_TYPE(Enum):
    MESSAGE = 'MESSAGE'
    TIMER = 'TIMER'
    LINK = 'LINK'
    SIGNAL = 'SIGNAL'
    TERMINATE = 'TERMINATE'
    UNDEFINED = 'UNDEFINED'


class ElementInfo:
    def __init__(self, element_type, element_id, element_name, event_type):
        self.id = element_id
        self.name = element_name
        self.type = element_type
        self.event_type = event_type
        self.incoming_flows = list()
        self.outgoing_flows = list()

    def is_split(self):
        return len(self.outgoing_flows) > 1

    def is_join(self):
        return len(self.incoming_flows) > 1

    def is_gateway(self):
        return self.type in [BPMN.EXCLUSIVE_GATEWAY, BPMN.PARALLEL_GATEWAY, BPMN.INCLUSIVE_GATEWAY, BPMN.EVENT_BASED_GATEWAY]

    def is_start_or_end_event(self):
        return self.type in [BPMN.START_EVENT, BPMN.END_EVENT]
    
    def is_event(self):
        return self.type in [BPMN.START_EVENT, BPMN.END_EVENT, BPMN.INTERMEDIATE_EVENT]


class ProcessState:
    def __init__(self, bpmn_graph):
        self.arcs_bitset = bpmn_graph.arcs_bitset
        self.tokens = dict()
        self.flow_date = dict()
        self.state_mask = 0
        for flow_arc in bpmn_graph.flow_arcs:
            self.tokens[flow_arc] = 0

    def add_token(self, flow_id):
        if flow_id in self.tokens:
            self.tokens[flow_id] += 1
            self.state_mask |= self.arcs_bitset[flow_id]

    def remove_token(self, flow_id):
        if self.has_token(flow_id):
            self.tokens[flow_id] = 0
            self.state_mask &= ~self.arcs_bitset[flow_id]

    def remove_all_tokens_on_terminate(self):
        for token in self.tokens:
            self.remove_token(token)

    def has_token(self, flow_id):
        return flow_id in self.tokens and self.tokens[flow_id] > 0

    def pending_tokens(self):
        marked_flows = list()
        for flow_id in self.tokens:
            if self.tokens[flow_id] > 0:
                marked_flows.append(flow_id)
        return marked_flows


class BPMNGraph:
    def __init__(self):
        self.starting_event = None
        self.end_event = None
        self.end_events_count = 0
        self.element_info = dict()
        self.from_name = dict()
        self.flow_arcs = dict()
        self.concurrent_enablers = dict()
        self.nodes_bitset = dict()
        self.arcs_bitset = dict()
        self.or_join_pred = dict()  # or_id -> [0 = node predecesors bitset, 1 = predecesors flow arcs]
        self.or_join_conflicting_pred = dict()
        self.decision_successors = dict()
        self.element_probability = None
        self.task_resource_probability = None
        self.closest_distance = None
        self.decision_flows_sortest_path = None
        self._c_trace = None
        self.event_distribution = None
        self.batch_info = dict()
        self.batch_count = dict()
        self.batch_waiting_processes = dict()
        self.last_datetime = dict()

    def set_additional_fields_from_json(self, element_probability, task_resource_probability, \
        event_distribution, batch_processing):
        self.element_probability = element_probability
        self.task_resource_probability = task_resource_probability
        self.event_distribution = event_distribution
        self.batch_info = batch_processing

    def add_bpmn_element(self, element_id, element_info):
        if element_info.type == BPMN.START_EVENT:
            self.starting_event = element_id
        if element_info.type == BPMN.END_EVENT:
            self.end_event = element_id
            self.end_events_count += 1
        self.element_info[element_id] = element_info
        self.from_name[element_info.name] = element_id
        self.nodes_bitset[element_id] = (1 << len(self.element_info))
        self.last_datetime[element_id] = dict()

    def add_flow_arc(self, flow_id, source_id, target_id):
        for node_id in [source_id, target_id]:
            if node_id not in self.element_info:
                self.element_info[node_id] = ElementInfo(BPMN.UNDEFINED, node_id, node_id, None)
        self.element_info[source_id].outgoing_flows.append(flow_id)
        self.element_info[target_id].incoming_flows.append(flow_id)
        self.flow_arcs[flow_id] = [source_id, target_id]
        self.arcs_bitset[flow_id] = (1 << len(self.flow_arcs))

    def encode_or_join_predecesors(self):
        for e_id in self.element_info:
            element = self.element_info[e_id]
            if element.type is BPMN.INCLUSIVE_GATEWAY and element.is_join():
                self.or_join_pred[e_id] = [0, 0]
                self._find_or_conflicting_predecesors(e_id)
                pred_queue = deque([e_id])
                while len(pred_queue) > 0:
                    element = self.element_info[pred_queue.popleft()]
                    for flow_id in element.incoming_flows:
                        prev_id = self.flow_arcs[flow_id][0]
                        if self.or_join_pred[e_id][0] & self.nodes_bitset[prev_id] == 0:
                            pred_queue.append(prev_id)
                        self.or_join_pred[e_id][0] |= self.nodes_bitset[prev_id]
                        if self.flow_arcs[flow_id][1] != e_id:
                            self.or_join_pred[e_id][1] |= self.arcs_bitset[flow_id]
            if element.type in [BPMN.EXCLUSIVE_GATEWAY, BPMN.INCLUSIVE_GATEWAY] and element.is_split():
                self._find_decision_successors(element)

    def _find_decision_successors(self, split_info):
        self.decision_successors[split_info.id] = set()
        visited = {split_info.id}
        suc_queue = deque([split_info])
        while suc_queue:
            e_info = suc_queue.popleft()
            for out_flow in e_info.outgoing_flows:
                next_info = self._get_successor(out_flow)
                if next_info.id not in visited:
                    visited.add(next_info.id)
                    next_info = self.element_info[next_info.id]
                    if next_info.type is BPMN.TASK:
                        self.decision_successors[split_info.id].add(next_info.id)
                    elif next_info.is_gateway():
                        suc_queue.append(next_info)

    def validate_model(self):
        if self.end_events_count == 0:
            raise InvalidBpmnModelException("At least one end event is required")
        if self.end_events_count > 1:
            raise InvalidBpmnModelException("Temporarily not supporting multiple end events")

    def _find_or_conflicting_predecesors(self, or_join_id):
        visited = {or_join_id}
        self.or_join_conflicting_pred[or_join_id] = set()
        for in_flow in self.element_info[or_join_id].incoming_flows:
            self._dfs_from_or_join(or_join_id, in_flow, self._get_predecessor(in_flow), visited)

    def _dfs_from_or_join(self, or_id, flow_id, e_info, visited):
        visited.add(e_info.id)
        if e_info.type in [BPMN.INCLUSIVE_GATEWAY, BPMN.EXCLUSIVE_GATEWAY] and e_info.is_split():
            self.or_join_conflicting_pred[or_id].add(e_info.id)
        for in_flow in e_info.incoming_flows:
            prev_info = self._get_predecessor(in_flow)
            if prev_info.id not in visited and prev_info.is_gateway():
                self._dfs_from_or_join(or_id, flow_id, prev_info, visited)

    def discover_path(self, from_e_id, to_e_id):
        if from_e_id not in self.element_info or to_e_id not in self.element_info:
            return None
        visited_elements = dict()
        visited_elements[from_e_id] = None
        elements_queue = deque()
        elements_queue.append(from_e_id)
        while len(elements_queue) > 0:
            from_e_id = elements_queue.popleft()
            out_flows = self.element_info[from_e_id].outgoing_flows
            for flow_id in out_flows:
                next_e = self.element_info[self.flow_arcs[flow_id][1]]
                if next_e.id in visited_elements:
                    continue
                visited_elements[next_e.id] = flow_id
                if next_e == to_e_id:
                    return visited_elements
                if next_e.type in [BPMN.EXCLUSIVE_GATEWAY, BPMN.INCLUSIVE_GATEWAY, BPMN.PARALLEL_GATEWAY]:
                    elements_queue.append(next_e.id)
        return None

    def is_enabled(self, e_id, p_state):
        if e_id not in self.element_info:
            return False
        if e_id == self.starting_event:
            return True
        e_info = self.element_info[e_id]
        if e_info.type == BPMN.TASK:
            # not enabled in case there is no token before the task
            # we assume that there might be only one incoming flow to the task
            for f_arc in e_info.incoming_flows:
                if p_state.tokens[f_arc] < 1:
                    return False

            return True
        if e_info.type in [BPMN.END_EVENT, BPMN.PARALLEL_GATEWAY, BPMN.INTERMEDIATE_EVENT]:
            for f_arc in e_info.incoming_flows:
                if p_state.tokens[f_arc] < 1:
                    return False
            return True
        elif e_info.type in [BPMN.EXCLUSIVE_GATEWAY, BPMN.EVENT_BASED_GATEWAY]:
            for f_arc in e_info.incoming_flows:
                if p_state.tokens[f_arc] > 0:
                    return True
            return False
        elif e_info.type == BPMN.INCLUSIVE_GATEWAY:
            if e_info.is_split():
                if p_state.has_token(e_info.incoming_flows[0]):
                    return True
                for flow_id in e_info.outgoing_flows:
                    if p_state.has_token(flow_id):
                        return True
                return False
            else:
                count_tokens = 0
                for flow_id in e_info.incoming_flows:
                    if p_state.tokens[flow_id] > 0:
                        count_tokens += 1
                if count_tokens == len(e_info.incoming_flows):
                    return True
                if count_tokens > 0 and self.or_join_pred[e_id][1] & p_state.state_mask == 0:
                    return True
                return False
        return False

    def update_process_state(self, case_id, e_id, p_state, completed_datetime_prev_event: CustomDatetimeAndSeconds):
        """
        :param completed_time_prev_event: datetime of the completion of the previous event
                                          which equals to enabled time for current event
        """
        if not self.is_enabled(e_id, p_state):
            return []
        enabled_tasks = list()
        to_execute = [e_id]
        current = 0
        visited_at = {e_id: self._check_and_update_enabling_time(case_id, e_id, completed_datetime_prev_event)}
        while current < len(to_execute):
            e_info: ElementInfo = self.element_info[to_execute[current]]
            last_enabled = visited_at[e_info.id]
            if e_info.type == BPMN.END_EVENT and e_info.event_type == EVENT_TYPE.TERMINATE:
                """ Terminate End Event detected"""
                p_state.remove_all_tokens_on_terminate()
            else:
                for in_flow in e_info.incoming_flows:
                    if p_state.tokens[in_flow] > 0:
                        p_state.tokens[in_flow] -= 1
                        p_state.state_mask &= ~self.arcs_bitset[in_flow]
            flows = e_info.outgoing_flows
            f_arcs = [(flow, None) for flow in flows]

            if len(f_arcs) > 1:
                if e_info.type is BPMN.EXCLUSIVE_GATEWAY:
                    f_arcs = [(self.element_probability[e_info.id].get_outgoing_flow(), None)]
                elif e_info.type is BPMN.EVENT_BASED_GATEWAY:
                    f_arcs = [self.get_event_gateway_choice(e_info, last_enabled.datetime)]
                else:
                    if e_info.type in \
                        [BPMN.TASK, BPMN.PARALLEL_GATEWAY, BPMN.START_EVENT, BPMN.INTERMEDIATE_EVENT]:
                        flows = copy.deepcopy(e_info.outgoing_flows)
                        f_arcs = [(flow, None) for flow in flows]
                    elif e_info.type is BPMN.INCLUSIVE_GATEWAY:
                        f_arcs = self.element_probability[e_info.id].get_multiple_flows()
                random.shuffle(f_arcs)
            for f_arc in f_arcs:
                self._find_next(f_arc, case_id, p_state, enabled_tasks, to_execute, last_enabled, visited_at)
            current += 1
        if len(enabled_tasks) > 1:
            random.shuffle(enabled_tasks)
        return enabled_tasks, visited_at

    def is_task_batched(self, task_id):
        return self.batch_info.get(task_id, None) != None


    def is_batched_task_enabled(self, task_id: str, enabled_at: CustomDatetimeAndSeconds):
        """
        Get whether the task could be enabled for the execution.
        :param task_id: id of the task
        :return: (is batched task enabled, chunks info, start time of the batch)
        """
        task_batch_info = self.batch_info.get(task_id, None)

        if task_batch_info == None:
            print(f"WARNING: task {task_id} does not have config for batching processing")
            return False, None, None

        firing_rules: List[AndFiringRule] = task_batch_info.firing_rules

        size_count = self.batch_count[task_id] if self.batch_count.get(task_id, None) != None else 0

        if not firing_rules.is_batch_size_enough_for_exec(size_count): #size_count < 2:
            # not enough items for batch execution
            return False, None, None

        waiting_time = [ (enabled_at.datetime - v.datetime).total_seconds() for (_, v) in self.batch_waiting_processes[task_id].items() ] 
        
        spec = {
            "size": size_count,
            "waiting_times": waiting_time,
            "enabled_datetimes": [ v.datetime for (_, v) in self.batch_waiting_processes[task_id].items() ],
            "curr_enabled_at": enabled_at.datetime,
            "is_triggered_by_batch": True, # specify where from we checking the rule. If not triggered by batch - then we move to midnight time
            "is_only_one_batch_return": False
        }

        if task_batch_info.are_rules_discovered:
            # if the rules were discovered,
            # we proceed with checking whether the rule is satisfied and
            # information for executing the batch 
            return firing_rules.is_true(spec)
        else:
            return self.get_batch_size_no_rules_defined(spec, firing_rules, task_batch_info)

    def get_batch_size_no_rules_defined(self, spec, firing_rules, task_batch_info):
        """
        In case no rules were discovered, size distribution is being used to define how batches will be formed.
        For this, we regenerate the rule associated with the batch after each batch creation.

        """
        spec["is_only_one_batch_return"] = True
        
        final_is_true, final_batch_spec, final_start_time = False, None, None

        while True:
            is_true, batch_spec, start_time = firing_rules.is_true(spec)

            if not is_true:
                # size rule could not be satisfied
                # return what we found so far
                break

            # size rule satisfied, remember the number of the items inside this batch
            # continue searching for next batches if any
            if (final_batch_spec == None): final_batch_spec = []
            num_items = batch_spec[0]
            final_batch_spec.append(num_items)
            final_is_true = is_true
            final_start_time = start_time
            
            # regenerate the rule based on the size distribution
            task_batch_info.update_firing_rules_from_distr()

            # update spec by removing the items we already included in the batch
            spec["waiting_times"] = spec["waiting_times"][num_items:]
            spec["enabled_datetimes"] = spec["enabled_datetimes"][num_items:]
            new_num_tasks_in_queue = spec["size"] - num_items

            if new_num_tasks_in_queue == 0:
                # all items were included in the batches
                break
            
            spec["size"] = spec["size"] - num_items

        return final_is_true, final_batch_spec, final_start_time


    def get_start_time(self, task_id, last_task_enabled_time) -> tuple([int, int, CustomDatetimeAndSeconds]):
        task_batch_info = self.batch_info.get(task_id, None)
        firing_rules: List[AndFiringRule] = task_batch_info.firing_rules
        enabled_times = list(self.batch_waiting_processes[task_id].items())
        batch_enabled_time = firing_rules.get_enabled_time(
            enabled_times, 
            last_task_enabled_time,
        )
        
        if batch_enabled_time != None:
            enabled_time = batch_enabled_time

            enabled_time_obj = CustomDatetimeAndSeconds(0, enabled_time)
            case_id, _ = enabled_times[0]

            return case_id, enabled_time_obj

        return None


    def get_event_gateway_choice(self, gateway_element_info: ElementInfo, completed_datetime_prev_event):
        """
        Find which flow will be executed next based on gateway element info

        :param gateway_element_info: object of type ElementInfo which is gateway
        :param completed_datetime_prev_event: datetime of the previously finished event
        :return: flow name that will be executed
        """
        all_gateway_choices = dict()
        for outgoing_flow in gateway_element_info.outgoing_flows:
            event_id = self.flow_arcs[outgoing_flow][1]
            [duration] = self.event_duration(event_id)
            all_gateway_choices[outgoing_flow] = duration
        
        min_value = min(all_gateway_choices.values())
        res = [(key, value) for key, value in all_gateway_choices.items() if value == min_value]

        # return randomly selected outgoing flow
        # in case of same value for multiple flows
        return secrets.choice(res)


    def event_duration(self, event_id):
        """
        Find event duration of all types except TIMER

        :event_id: id of the event element
        :return: duration in seconds 
        """
        distribution: DurationDistribution = self.event_distribution[event_id]
        return distribution.generate_sample(1)


    def reply_trace(self, task_sequence, f_arcs_frequency, post_p=True, trace=None):
        self._c_trace = trace
        task_enabling = list()
        p_state = ProcessState(self)
        fired_tasks = list()
        fired_or_splits = set()
        for flow_id in self.element_info[self.starting_event].outgoing_flows:
            p_state.flow_date[flow_id] = self._c_trace[0].started_at if self._c_trace is not None else None
            p_state.add_token(flow_id)
        self.update_flow_dates(self.element_info[self.starting_event], p_state, self._c_trace[0].started_at)
        pending_tasks = dict()
        for current_index in range(len(task_sequence)):
            # print(task_sequence[current_index])
            el_id = self.from_name.get(task_sequence[current_index])
            fired_tasks.append(False)
            in_flow = self.element_info[el_id].incoming_flows[0]
            task_enabling.append(p_state.flow_date[in_flow] if in_flow in p_state.flow_date else None)
            if self._c_trace:
                self.update_flow_dates(self.element_info[el_id], p_state,
                                       self._c_trace[current_index].completed_at if self._c_trace is not None else None)

            self.try_firing(current_index, current_index, task_sequence, fired_tasks, pending_tasks,
                            p_state, f_arcs_frequency, fired_or_splits)

            if el_id is None:  # NOTE: skipping if no such element in self.from_name
                continue
            p_state.add_token(self.element_info[el_id].outgoing_flows[0])
            if current_index in pending_tasks:
                for pending_index in pending_tasks[current_index]:
                    self.try_firing(pending_index, current_index, task_sequence, fired_tasks, pending_tasks,
                                    p_state, f_arcs_frequency, fired_or_splits)

            # if fired_tasks[current_index] and task_enabling[current_index] is None:
            #     task_enabling[current_index] = self.fix_missing_flow_enabled_dates(el_id, p_state)

        # Firing End Event
        enabled_end, or_fired, path_decisions = self._find_enabled_predecessors(
            self.element_info[self.end_event], p_state)
        self.fire_enabled_predecessors(enabled_end, p_state, or_fired, path_decisions, f_arcs_frequency,
                                       fired_or_splits)
        end_flow = self.element_info[self.end_event].incoming_flows[0]
        if p_state.has_token(end_flow):
            p_state.tokens[end_flow] = 0

        is_correct = True
        for i in range(0, len(task_sequence)):
            if not fired_tasks[i]:
                is_correct = False
                break

        self.check_unfired_or_splits(fired_or_splits, f_arcs_frequency, p_state)
        if post_p:
            self.postprocess_unfired_tasks(task_sequence, fired_tasks, f_arcs_frequency, task_enabling)
        self._c_trace = None
        return is_correct, fired_tasks, p_state.pending_tokens(), task_enabling

    def postprocess_unfired_tasks(self, task_sequence: list, fired_tasks: list, f_arcs_frequency: dict,
                                  task_enablement: list):
        if self.closest_distance is None:
            self._sort_by_closest_predecesors()
        task_sequence = [task_name for task_name in task_sequence if task_name in self.from_name]
        for i in range(0, len(fired_tasks)):
            if not fired_tasks[i]:
                e_info = self.element_info[self.from_name.get(task_sequence[i])]
                fix_from = [self.starting_event, self.closest_distance[e_info.id][self.starting_event]]
                j = i - 1
                while j >= 0:
                    p_info = self.element_info[self.from_name.get(task_sequence[j])]
                    if p_info.id in self.closest_distance[e_info.id] and self.closest_distance[e_info.id][p_info.id] < \
                            fix_from[1]:
                        fix_from = [p_info.id, self.closest_distance[e_info.id][p_info.id]]
                        if fix_from[1] == 1:
                            break
                    j -= 1
                if fix_from[0] is not None:
                    if task_enablement[i] is None:
                        task_enablement[i] = self._c_trace[j].completed_at if j >= 0 else self._c_trace[0].completed_at
                    for flow_id in self.decision_flows_sortest_path[e_info.id][fix_from[0]]:
                        if flow_id not in f_arcs_frequency:
                            f_arcs_frequency[flow_id] = 0
                        f_arcs_frequency[flow_id] += 1

    def _sort_by_closest_predecesors(self):
        self.closest_distance = dict()
        self.decision_flows_sortest_path = dict()
        for e_id in self.element_info:
            self.closest_distance[e_id] = dict()
            pred_seq = dict()
            distance_map = {e_id: 0}
            pred_queue = deque([self.element_info[e_id]])
            while pred_queue:
                e_info = pred_queue.popleft()
                for flow_id in e_info.incoming_flows:
                    pred_info = self._get_predecessor(flow_id)
                    if pred_info.id not in distance_map:
                        pred_seq[pred_info.id] = flow_id
                        dist = distance_map[e_info.id]
                        if pred_info.type in [BPMN.TASK, BPMN.START_EVENT]:
                            dist += 1
                            self.closest_distance[e_id][pred_info.id] = dist
                        distance_map[pred_info.id] = dist
                        pred_queue.append(pred_info)
            self.decision_flows_sortest_path[e_id] = dict()
            for p_id in self.element_info:
                self.decision_flows_sortest_path[e_id][p_id] = list()
                if p_id is not e_id and p_id in self.closest_distance[e_id]:
                    p_info = self.element_info[p_id]
                    while p_info.id is not e_id:
                        if p_info.type in [BPMN.INCLUSIVE_GATEWAY, BPMN.EXCLUSIVE_GATEWAY] and p_info.is_split():
                            self.decision_flows_sortest_path[e_id][p_id].append(pred_seq[p_info.id])
                        p_info = self._get_successor(pred_seq[p_info.id])

    def try_firing(self, task_index, from_index, task_sequence, fired_tasks, pending_tasks, p_state,
                   f_arcs_frequency, fired_or_splits):
        el_id = self.from_name.get(task_sequence[task_index])
        if el_id is None:
            return
        task_info = self.element_info[el_id]
        if not p_state.has_token(task_info.incoming_flows[0]):
            enabled_pred, or_fired, path_decisions = self._find_enabled_predecessors(task_info, p_state)
            firing_index = self.find_firing_index(task_index, from_index, task_sequence, path_decisions, enabled_pred)
            if firing_index == from_index:
                self.fire_enabled_predecessors(enabled_pred, p_state, or_fired, path_decisions, f_arcs_frequency,
                                               fired_or_splits)
            elif firing_index not in pending_tasks:
                pending_tasks[firing_index] = [task_index]
            else:
                pending_tasks[firing_index].append(task_index)
        if p_state.has_token(task_info.incoming_flows[0]):
            p_state.remove_token(task_info.incoming_flows[0])
            fired_tasks[task_index] = True

    def closer_enabled_predecessors(self, e_info, flow_id, enabled_pred, or_firing, path_split, visited, p_state, dist,
                                    min_dist):
        # Verificar que hacer cuando el camino más corto termina en el Start event,
        # pero hay una tarea habilitada en un camino más largo
        if self.is_enabled(e_info.id, p_state):
            # if e_info.type is BPMN.START_EVENT:
            #     dist = len(self.flow_arcs)
            if dist not in enabled_pred:
                enabled_pred[dist] = list()
            enabled_pred[dist].append([e_info, flow_id])
            min_dist[0] = max(min_dist[0], dist)
            return dist, enabled_pred, or_firing, path_split
        elif e_info.type is BPMN.INCLUSIVE_GATEWAY and e_info.is_join():
            for in_or in e_info.incoming_flows:
                if p_state.has_token(in_or):
                    or_firing[e_info.id] = dist
                    break
        if e_info.type in [BPMN.INCLUSIVE_GATEWAY, BPMN.EXCLUSIVE_GATEWAY]:
            path_split[e_info.id] = flow_id
        visited.add(e_info.id)
        if e_info.is_gateway():
            if e_info.type is BPMN.EXCLUSIVE_GATEWAY and e_info.is_join():
                closer_pred, temp_path, or_f = dict(), dict(), dict()
                c_min = sys.maxsize
                for in_flow in e_info.incoming_flows:
                    pr_info = self._get_predecessor(in_flow)
                    if pr_info.id not in visited:
                        d, e_p, o_f, t_path = self.closer_enabled_predecessors(pr_info, in_flow, dict(), dict(), dict(),
                                                                               visited, p_state, dist + 1, min_dist)

                        if d < c_min or (BPMNGraph._has_start_event(closer_pred) and d != sys.maxsize):
                            c_min, closer_pred, or_f, temp_path = d, e_p, o_f, t_path

                for e_id in closer_pred:
                    enabled_pred[e_id] = closer_pred[e_id]
                for e_id in temp_path:
                    path_split[e_id] = temp_path[e_id]
                for e_id in or_f:
                    or_firing[e_id] = dist
                return c_min, enabled_pred, or_firing, path_split
            else:
                c_min = dist if e_info.id in or_firing else sys.maxsize
                for in_flow in e_info.incoming_flows:
                    pred_info = self._get_predecessor(in_flow)
                    if pred_info.id not in visited and pred_info.is_gateway():
                        res = self.closer_enabled_predecessors(pred_info, in_flow, enabled_pred, or_firing, path_split,
                                                               visited, p_state, dist + 1, min_dist)
                        c_min = min(res[0], c_min)

                return c_min, enabled_pred, or_firing, path_split
        return sys.maxsize, enabled_pred, or_firing, path_split

    @staticmethod
    def _has_start_event(closer_pred):
        has_start_event = False
        for e_id in closer_pred:
            for e_info in closer_pred[e_id]:
                if e_info[0].type is BPMN.START_EVENT:
                    has_start_event = True
                    break
        return has_start_event

    def _find_enabled_predecessors(self, from_task_info, p_state):
        pred_info = self._get_predecessor(from_task_info.incoming_flows[0])
        max_dist = [0]
        closer_pred = self.closer_enabled_predecessors(pred_info, from_task_info.incoming_flows[0], dict(),
                                                       dict(), dict(), set(), p_state, 0,
                                                       max_dist)
        enabled_pred = deque()
        for i in range(0, max_dist[0] + 1):
            if i in closer_pred[1]:
                for pred_id in closer_pred[1][i]:
                    enabled_pred.appendleft(pred_id)
        return enabled_pred, closer_pred[2], closer_pred[3]

    def find_firing_index(self, task_index, from_index, task_sequence, path_decisions, enabled_pred):
        is_conflicting, conflicting_gateways = self.is_conflicting_task(path_decisions, enabled_pred)
        if is_conflicting:
            firing_index = from_index
            for i in range(from_index + 1, len(task_sequence)):
                if task_sequence[i] != task_sequence[task_index]:
                    for or_id in conflicting_gateways:
                        for split_id in conflicting_gateways[or_id]:
                            if task_sequence[i] in self.decision_successors[split_id]:
                                return i
            return firing_index
        return from_index

    def is_conflicting_task(self, path_decisions, enabled_pred):
        conflicting_gateways = dict()
        is_conflicting = False
        for or_id in path_decisions:
            if self.element_info[or_id].type is BPMN.INCLUSIVE_GATEWAY and self.element_info[or_id].is_join():
                conflicting_gateways[or_id] = set()
                for enabled in enabled_pred:
                    e_info = enabled[0]
                    if e_info.id in self.or_join_conflicting_pred[or_id]:
                        conflicting_gateways[or_id].add(e_info.id)
                    if len(conflicting_gateways[or_id]) > 1:
                        is_conflicting = True
        return is_conflicting, conflicting_gateways

    def fire_enabled_predecessors(self, enabled_pred, p_state, or_firing, path_decisions, f_arcs_frequency,
                                  fired_or_split):
        visited_elements = set()
        if not enabled_pred:
            self.try_firing_or_join(enabled_pred, p_state, or_firing, path_decisions, f_arcs_frequency)
        while enabled_pred:
            [e_info, e_flow] = enabled_pred.popleft()
            if self.is_enabled(e_info.id, p_state):
                visited_elements.add(e_info.id)
                if e_info.type is BPMN.PARALLEL_GATEWAY:
                    for out_flow in e_info.outgoing_flows:
                        self._update_next(out_flow, enabled_pred, p_state, or_firing, path_decisions, f_arcs_frequency)
                elif e_info.type is BPMN.EXCLUSIVE_GATEWAY:
                    self._update_next(e_flow, enabled_pred, p_state, or_firing, path_decisions, f_arcs_frequency)
                elif e_info.type is BPMN.INCLUSIVE_GATEWAY:
                    self._update_next(e_flow, enabled_pred, p_state, or_firing, path_decisions, f_arcs_frequency)
                    if e_info.is_split():
                        fired_or_split.add(e_info.id)
                        for flow_id in e_info.outgoing_flows:
                            if flow_id != e_flow:
                                self._update_next(flow_id, enabled_pred, p_state, or_firing, path_decisions,
                                                  f_arcs_frequency, True)
            for in_flow in e_info.incoming_flows:
                p_state.remove_token(in_flow)
            self.try_firing_or_join(enabled_pred, p_state, or_firing, path_decisions, f_arcs_frequency)

    def update_flow_dates(self, e_info: ElementInfo, p_state: ProcessState, last_date):
        visited_elements = set()
        suc_queue = deque([e_info])
        visited_elements.add(e_info.id)
        while suc_queue:
            e_info = suc_queue.popleft()
            for out_flow in e_info.outgoing_flows:
                next_info = self._get_successor(out_flow)
                p_state.flow_date[out_flow] = last_date
                if next_info.is_gateway() and next_info.id not in visited_elements:
                    suc_queue.append(next_info)
                    visited_elements.add(next_info.id)

    def try_firing_or_join(self, enabled_pred, p_state, or_firing, path_decisions, f_arcs_frequency):
        fired = set()
        or_firing_list = list()
        for or_join_id in or_firing:
            or_firing_list.append(or_join_id)
        for or_join_id in or_firing_list:
            if self.is_enabled(or_join_id, p_state) or not enabled_pred:
                fired.add(or_join_id)
                e_info = self.element_info[or_join_id]
                self._update_next(e_info.outgoing_flows[0], enabled_pred, p_state, or_firing, path_decisions,
                                  f_arcs_frequency)
                for in_flow in e_info.incoming_flows:
                    p_state.remove_token(in_flow)
                if enabled_pred:
                    break
                if len(or_firing_list) != len(or_firing):
                    for e_id in or_firing:
                        if e_id not in or_firing_list:
                            or_firing_list.append(e_id)
        for or_id in fired:
            del or_firing[or_id]

    def check_unfired_or_splits(self, or_splits, f_arcs_frequency, p_state):
        for or_id in or_splits:
            for flow_id in self.element_info[or_id].outgoing_flows:
                if p_state.tokens[flow_id] > 0:
                    f_arcs_frequency[flow_id] -= p_state.tokens[flow_id]
                    p_state.tokens[flow_id] = 0

    def _update_next(self, flow_id, enabled_pred, p_state, or_firing, path_decisions, f_arcs_frequency, from_or=False):
        if flow_id not in f_arcs_frequency:
            f_arcs_frequency[flow_id] = 1
        else:
            f_arcs_frequency[flow_id] += 1
        p_state.add_token(flow_id)
        if not from_or:
            next_info = self._get_successor(flow_id)
            if next_info.type is BPMN.PARALLEL_GATEWAY and self.is_enabled(next_info.id, p_state):
                enabled_pred.appendleft([next_info, None])
            elif next_info.id in path_decisions:
                if next_info.type is BPMN.INCLUSIVE_GATEWAY:
                    if next_info.is_split():
                        enabled_pred.appendleft([next_info, path_decisions[next_info.id]])
                    else:
                        if next_info.id not in or_firing:
                            or_firing[next_info.id] = 1
                else:
                    enabled_pred.appendleft([next_info, path_decisions[next_info.id]])

    def _get_predecessor(self, flow_id):
        return self.element_info[self.flow_arcs[flow_id][0]]

    def _get_successor(self, flow_id):
        return self.element_info[self.flow_arcs[flow_id][1]]

    def compute_branching_probability(self, flow_arcs_frequency):
        gateways_branching = dict()
        for e_id in self.element_info:
            if self.element_info[e_id].type in [BPMN.EXCLUSIVE_GATEWAY, BPMN.INCLUSIVE_GATEWAY] and len(
                    self.element_info[e_id].outgoing_flows) > 1:
                total_frequency = 0
                for flow_id in self.element_info[e_id].outgoing_flows:
                    if flow_id not in flow_arcs_frequency:
                        flow_arcs_frequency[flow_id] = 0
                    total_frequency += flow_arcs_frequency[flow_id]
                flow_arc_probability = dict()
                for flow_id in self.element_info[e_id].outgoing_flows:
                    flow_arc_probability[flow_id] = flow_arcs_frequency[
                                                        flow_id] / total_frequency if total_frequency > 0 else 0
                gateways_branching[e_id] = flow_arc_probability
        return gateways_branching

    def _find_next(self, f_arc_and_duration, case_id, p_state, enabled_tasks, to_execute, enabled_time: CustomDatetimeAndSeconds, visited_at):
        # verify that there is no batch waiting to be executed
        f_arc, duration_sec = f_arc_and_duration
        p_state.tokens[f_arc] += 1
        p_state.state_mask |= self.arcs_bitset[f_arc]
        next_e = self.flow_arcs[f_arc][1]
        enabled_time = self._check_and_update_enabling_time(case_id, next_e, enabled_time)
        visited_at[next_e] = enabled_time
        if self.is_enabled(next_e, p_state):
            next_el_type = self.element_info[next_e].type 
            if next_el_type == BPMN.TASK and self.is_task_batched(next_e):
                self.enable_batch_or_task(next_e, case_id, enabled_time, enabled_tasks, duration_sec)           
            elif next_el_type == BPMN.TASK:
                enabled_tasks.append(EnabledTask(next_e, None, duration_sec))
            elif next_el_type == BPMN.INTERMEDIATE_EVENT:
                enabled_tasks.append(EnabledTask(next_e, None, duration_sec, True))
            else:
                to_execute.append(next_e)

    def _check_and_update_enabling_time(self, p_case, e_id, enabled_at: CustomDatetimeAndSeconds):
        if self.last_datetime[e_id][p_case] is None or self.last_datetime[e_id][p_case].datetime < enabled_at.datetime:
            self.last_datetime[e_id][p_case] = enabled_at
        elif self.last_datetime[e_id][p_case].datetime > enabled_at.datetime:
            enabled_at = self.last_datetime[e_id][p_case]
        return enabled_at

    def is_executing_alone(self, task_id: str):
        task_batch_info: BatchConfigPerTask = self.batch_info.get(task_id, None)

        if task_batch_info == None:
            print(f"WARNING: task {task_id} does not have config for batching processing")
            return False, None, None

        return task_batch_info.is_batch_exec_alone()

    def is_open_batch(self, task_id):
        """ True if there are open batch waiting for the batch execution
            else False"""
        return task_id in self.batch_waiting_processes and \
            len(self.batch_waiting_processes[task_id]) > 0

    def enable_batch_or_task(self, task_id: str, case_id, enabled_time: CustomDatetimeAndSeconds, \
        enabled_tasks: List[EnabledTask], duration_sec):
        """ 
        Checking the compliance with the size_distr specified in the input
        In case the probability returns 1, the task is being executed alone
        In case we already have open batch, we continue allocate tasks there 
        till the moment it could be executed
        """
        if not self.is_open_batch(task_id) and \
            self.is_executing_alone(task_id):
            # execute alone as a simple task
            enabled_tasks.append(EnabledTask(task_id, None, duration_sec))
            return

        # moving batch to list of enabled tasks if the batch is enabled
        self.move_batch_if_enabled(task_id, case_id, enabled_time, enabled_tasks)

    def move_batch_if_enabled(self, task_id: str, case_id, enabled_time: CustomDatetimeAndSeconds, \
        enabled_tasks: List[EnabledTask]):
        """
        Moving batch to list of enabled tasks if the batch is enabled
        Before, checking whether it complies with the size_distr specified in the input
        In case the probability returns 1, the task is being executed alone
        In case we already have open batch, we continue allocate tasks there 
        till the moment it could be executed
        """
        self.increase_task_count(task_id, case_id, enabled_time)
        is_enabled, batch_spec, start_time_from_rule = self.is_batched_task_enabled(task_id, enabled_time)

        if is_enabled:
            batch_info = BatchInfoForExecution(
                self.batch_waiting_processes,
                self.batch_info,
                task_id,
                batch_spec,
                start_time_from_rule)
            enabled_tasks.append((EnabledTask(task_id, batch_info)))
            self._clear_batch(task_id, batch_spec)


    def _clear_batch(self, next_e, batch_spec):
        """
        When we passed on the information about the batch for the execution,
        clear that data from here to avoid multiple execution
        """
        for batch_size in batch_spec:
            if batch_size != None:
                # remove first batch_size-element since they are being executed
                # the rest stays in the queue for being enabled for batch execution
                curr_index = 0
                for item_key in list(self.batch_waiting_processes[next_e].keys()):
                    del self.batch_waiting_processes[next_e][item_key]
                    curr_index = curr_index + 1

                    if curr_index == batch_size:
                        # all waiting processes regarding the selected batch was removed
                        break

                self.batch_count[next_e] = self.batch_count[next_e] - batch_size


    def increase_task_count(self, task_id, case_id, enabled_time):
        """ Keep track of the batched tasks in the queue """

        # keep track of case_id which are waiting for the firing rule to be true
        if task_id not in self.batch_waiting_processes:
            self.batch_waiting_processes[task_id] = dict()
        
        # in case we execute the batch, there is no need to increase the total num
        if case_id in self.batch_waiting_processes[task_id]:
            return
        
        self.batch_waiting_processes[task_id][case_id] = enabled_time
        
        if task_id in self.batch_count:
            self.batch_count[task_id] += 1
        else:
            self.batch_count[task_id] = 1

    def is_any_batch_enabled(self, started_datetime: CustomDatetimeAndSeconds):
        enabled_task_batch = dict()
        for (task_id, waiting_tasks) in self.batch_waiting_processes.items():
            if len(waiting_tasks) == 0:
                continue

            # finding the date in the future when rule will be satisfied
            is_enabled, batch_spec, start_time_from_rule = self.is_batched_task_enabled(task_id, started_datetime)
            if is_enabled:
                enabled_task_batch[task_id] = self.create_batch_info_and_clear_from_queue(
                    task_id, batch_spec, start_time_from_rule
                )

        return enabled_task_batch

    def is_any_unexecuted_batch(self, last_task_enabled_time: CustomDatetimeAndSeconds):
        is_any = self.batch_waiting_processes.items()

        if is_any == None:
            return None

        for (task_id, waiting_tasks) in self.batch_waiting_processes.items():
            if not self.batch_info[task_id].firing_rules.is_batch_size_enough_for_exec(len(waiting_tasks)):
                # either no or not enough tasks waiting for batch execution
                # minimum number of tasks to be executed: 2
                continue

            case_id_and_start_time = self.get_start_time(task_id, last_task_enabled_time)

            if (case_id_and_start_time != None):
                yield case_id_and_start_time


    def get_invalid_batches_if_any(self, current_point_of_time):
        enabled_task_batch = dict()
        for (task_id, waiting_tasks) in self.batch_waiting_processes.items():
            num_tasks_wait_batch = len(waiting_tasks)
            if num_tasks_wait_batch == 0:
                continue
            
            if self.is_or_rule_invalid(waiting_tasks, task_id, num_tasks_wait_batch, current_point_of_time):
                # if the rule become invlaid, we execute everything we have
                # invalid = the one that cannot be satisfied in the future anymore
                # this method is being called only in case it is the end of all simulated cases
                waiting_tasks_len = len(waiting_tasks)
                batch_spec = [waiting_tasks_len]

                # in case last executed task was the one that triggers batch exec
                # we need to take the enabled_time of the last task in this batch
                # as start_time for the whole batch
                last_added_key = list(waiting_tasks.keys())[-1]
                last_task_in_batch_start = waiting_tasks[last_added_key].datetime
                start_time_from_rule = max(current_point_of_time.datetime, last_task_in_batch_start)
                enabled_task_batch[task_id] = self.create_batch_info_and_clear_from_queue(
                    task_id, batch_spec, start_time_from_rule
                )
                continue
        
        return enabled_task_batch

    def create_batch_info_and_clear_from_queue(self, task_id: str, batch_spec, start_time_from_rule):
        batch_info = BatchInfoForExecution(
            self.batch_waiting_processes,
            self.batch_info,
            task_id,
            batch_spec,
            start_time_from_rule)

        self._clear_batch(task_id, batch_spec)
        return batch_info

    def is_or_rule_invalid(self, waiting_tasks, task_id: str, num_tasks_wait_batch: int, current_point_of_time: CustomDatetimeAndSeconds):
        all_keys = list(waiting_tasks.keys())
        
        first_key = all_keys[0]
        first_wt = (current_point_of_time.datetime - waiting_tasks[first_key].datetime).seconds

        last_key = all_keys[-1]
        last_wt = (current_point_of_time.datetime - waiting_tasks[last_key].datetime).seconds

        return self.batch_info[task_id].firing_rules.is_invalid_end(num_tasks_wait_batch, first_wt, last_wt)
