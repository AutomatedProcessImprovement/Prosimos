class PoolInfo:
    def __init__(self, pool_id, pool_name):
        self.pool_id = pool_id
        self.pool_name = pool_name


class ResourceProfile:
    def __init__(self, resource_id, resource_name, calendar_id=None, cost_per_hour=None):
        self.resource_id = resource_id
        self.resource_name = resource_name
        self.cost_per_hour = cost_per_hour
        self.calendar_id = calendar_id
        self.resource_amount = 1
        self.pool_info = None


class Node:
    def __init__(self, task_id=None, enabled_at=None):
        self.task_id = task_id
        self.enabled_at = enabled_at
        self.next = None
        self.prev = None


class PendingTaskList:
    def __init__(self):
        self.sise = 0
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def insert_tasks(self, enabled_at, task_list):
        current = self.tail.prev
        while current is not self.head and enabled_at < current.enabled_at:
            current = current.prev
        temp_node = current.next
        for task_id in task_list:
            current.next = Node(task_id, enabled_at)
            current.next.prev = current
            current = current.next
        current.next = temp_node
        temp_node.prev = current
        self.sise += len(task_list)