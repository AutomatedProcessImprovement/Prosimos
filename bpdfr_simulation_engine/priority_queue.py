import itertools
from heapq import heappush
from heapq import heappop


class PriorityQueue:
    def __init__(self):
        self.pq = []  # list of entries arranged in a heap
        self.entry_finder = {}  # mapping of tasks to entries
        self.REMOVED = '<removed-task>'  # placeholder for a removed task
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
        """ Add a new task or update the priority of an existing task """
        if element in self.entry_finder:
            self.remove_element(element)
        count = next(self.counter)
        entry = [priority, count, element]
        self.entry_finder[element] = entry
        heappush(self.pq, entry)

    def pop_min(self):
        """ Remove and return the lowest priority task. Raise KeyError if empty. """
        while self.pq:
            priority, count, task = heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task, priority
        return None, None

    def remove_element(self, element):
        """ Mark an existing task as REMOVED.  Raise KeyError if not found. """
        entry = self.entry_finder.pop(element)
        entry[-1] = self.REMOVED