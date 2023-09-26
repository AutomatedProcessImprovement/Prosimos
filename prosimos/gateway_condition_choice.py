class GatewayConditionChoice:
    def __init__(self, candidates_list, rules_list):
        self.candidates_list = candidates_list
        self.rules_list = rules_list

    def get_outgoing_flow(self, case_attributes):
        return [candidate for candidate, rule in zip(self.candidates_list, self.rules_list)
                if rule.is_true(case_attributes)]


