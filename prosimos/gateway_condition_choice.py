class GatewayConditionChoice:
    def __init__(self, candidates_list, rules_list, default_path=None):
        self.candidates_list = candidates_list
        self.rules_list = rules_list
        self.default_path = default_path

    def set_default(self, default_path):
        self.default_path = default_path

    def get_outgoing_flow(self, attributes):
        return [candidate for candidate, rule in zip(self.candidates_list, self.rules_list)
                if rule.is_true(attributes)]

    def get_default_path(self):
        return self.default_path
