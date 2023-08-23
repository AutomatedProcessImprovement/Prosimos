import itertools
from collections import deque
from heapq import heappop, heappush


class PriorityQueue:
    def __init__(self):
        self.pq = []  # list of entries arranged in a heap
        self.entry_finder = {}  # mapping of tasks to entries
        self.REMOVED = "<removed-task>"  # placeholder for a removed task
        self.counter = itertools.count()  # unique sequence count

    def is_empty(self):
        return len(self.entry_finder) == 0

    def size(self):
        return len(self.entry_finder)

    def contains(self, element):
        return element in self.entry_finder

    def get_priority(self, element):
        if element in self.entry_finder:
            return self.entry_finder[element][0]
        return None

    def insert(self, element, priority=0):
        """Add a new task or update the priority of an existing task"""
        if element in self.entry_finder:
            self.remove_element(element)
        count = next(self.counter)
        entry = [priority, count, element]
        self.entry_finder[element] = entry
        heappush(self.pq, entry)

    def pop_min(self):
        """Remove and return the lowest priority task. Raise KeyError if empty."""
        while self.pq:
            priority, count, task = heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task, priority
        return None, None

    def remove_element(self, element):
        """Mark an existing task as REMOVED.  Raise KeyError if not found."""
        entry = self.entry_finder.pop(element)
        entry[-1] = self.REMOVED

    def peek(self):
        """Return the lowest priority task (without removing it from the queue)"""
        for heap_item in self.pq:
            priority, _, task = heap_item
            if task is not self.REMOVED:
                return task, priority

        return None, None


class DiffResourceQueue:
    # Two tasks share a resource queue iff the share all the resources. If the two tasks share only a set of resources,
    # then they will point to different resource queues. Therefore, a resource may be repeated in many queues.
    def __init__(self, task_resource_map, r_initial_availability):
        self._resource_queues = list()  # List of (shared) resource queues, i.e., many tasks may share a resource queue
        self._resource_queue_map = dict()  # Map relating the indexes of the queues where a resource r_id is contained
        self._task_queue_map = dict()  # Map with the index of the resource queue that can perform a task r_id

        self._init_simulation_queues(task_resource_map, r_initial_availability)

    def pop_resource_for(self, task_id):
        return self._resource_queues[self._task_queue_map[task_id]].pop_min()

    def update_resource_availability(self, resource_id, released_at):
        for q_index in self._resource_queue_map[resource_id]:
            self._resource_queues[q_index].insert(resource_id, released_at)

    def _init_simulation_queues(self, task_resource_map, r_initial_availability):
        joint_sets = list()
        taken = set()
        # Grouping the tasks that share all the resources
        for task_id_1 in task_resource_map:
            if task_id_1 not in taken:
                joint_tasks = set()
                for task_id_2 in task_resource_map:
                    is_joint = True
                    if len(task_resource_map[task_id_2]) != len(task_resource_map[task_id_1]):
                        continue
                    for r_id in task_resource_map[task_id_2]:
                        if r_id not in task_resource_map[task_id_1]:
                            is_joint = False
                            break
                    if is_joint:
                        taken.add(task_id_2)
                        joint_tasks.add(task_id_2)
                joint_sets.append(joint_tasks)

        index = 0
        for j_set in joint_sets:
            resource_queue = PriorityQueue()
            for task_id in j_set:
                self._task_queue_map[task_id] = index
                if resource_queue.is_empty():
                    for r_id in task_resource_map[task_id]:
                        if r_id not in self._resource_queue_map:
                            self._resource_queue_map[r_id] = list()
                        resource_queue.insert(r_id, r_initial_availability[r_id])
                        self._resource_queue_map[r_id].append(index)
            self._resource_queues.append(resource_queue)
            index += 1


class EventQueue:
    def __init__(self):
        self.enabled_events = PriorityQueue()

    def append_arrival_event(self, event_info, case_priority):
        self.enabled_events.insert(event_info, (case_priority, event_info.enabled_at))

    def append_enabled_event(self, event_info, case_priority):
        self.enabled_events.insert(event_info, (case_priority, event_info.enabled_at))

    def pop_next_event(self):
        if self.enabled_events:
            event_info, _ = self.enabled_events.pop_min()
            return event_info
        else:
            return None

    def peek(self):
        if self.enabled_events:
            event_info, _ = self.enabled_events.peek()
            return event_info
        else:
            return None


class EventQueue1:
    def __init__(self):
        self.arrival_events = deque()
        self.enabled_events = deque()

    def append_arrival_event(self, event_info):
        self.arrival_events.append(event_info)

    def append_enabled_event(self, event_info):
        self.enabled_events.append(event_info)

    def pop_next_event(self):
        if self.arrival_events and self.enabled_events:
            if self.arrival_events[0].enabled_at < self.enabled_events[0].enabled_at:
                return self.arrival_events.popleft()
            else:
                return self.enabled_events.popleft()
        elif self.enabled_events:
            return self.enabled_events.popleft()
        elif self.arrival_events:
            return self.arrival_events.popleft()
        else:
            return None
