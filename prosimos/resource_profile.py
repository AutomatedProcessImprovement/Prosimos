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