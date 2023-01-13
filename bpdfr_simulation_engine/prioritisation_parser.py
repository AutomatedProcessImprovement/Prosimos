from typing import List
from bpdfr_simulation_engine.prioritisation import AllPriorityRules, AndPrioritisationRule, OrPrioritisationRule, PrioritisationRule, PriorityWithRule

class PrioritisationParser:
    def __init__(self, json_data_with_prioritization):
        self.data = json_data_with_prioritization

    def parse(self):
        priority_rules: List[PriorityWithRule] = []
        all_rules: List[OrPrioritisationRule] = []
        for curr_priority_rule in self.data:
            or_rules_json = curr_priority_rule["rules"]
            priority = curr_priority_rule["priority_level"]

            or_rules: List[AndPrioritisationRule] = []
            for or_rule in or_rules_json:
                and_rules: List[PrioritisationRule] = []
                for and_rule_json in or_rule:
                    rule: PrioritisationRule = PrioritisationRule(
                        and_rule_json["attribute"],
                        and_rule_json['condition'],
                        and_rule_json["value"]
                    )
                    and_rules.append(rule)
            
                or_rules.append(AndPrioritisationRule(and_rules))
            
            all_rules.append(OrPrioritisationRule(or_rules))

            priority_rules.append(PriorityWithRule(all_rules, priority))

        return AllPriorityRules(priority_rules)
