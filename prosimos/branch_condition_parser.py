from typing import List
from prosimos.branch_condition_rules import (
    AllBranchConditionRules,
    AndBranchConditionRule,
    OrBranchConditionRule,
    BranchConditionRule,
    BranchConditionWithRule
)


class BranchConditionParser:
    def __init__(self, json_data_with_branching_conditions):
        self.data = json_data_with_branching_conditions

    def parse(self):
        branching_conditions: List[BranchConditionWithRule] = []
        for curr_branch_condition_rule in self.data:
            or_rules_json = curr_branch_condition_rule["rules"]
            branch_id = curr_branch_condition_rule["id"]

            or_rules: List[AndBranchConditionRule] = []
            for or_rule in or_rules_json:
                and_rules: List[BranchConditionRule] = []
                for and_rule_json in or_rule:
                    rule: BranchConditionRule = BranchConditionRule(
                        and_rule_json["attribute"],
                        and_rule_json["comparison"],
                        and_rule_json["value"],
                    )
                    and_rules.append(rule)
                all_and_rules = AndBranchConditionRule(and_rules)
                or_rules.append(all_and_rules)

            all_or_rules = OrBranchConditionRule(or_rules)

            branching_conditions.append(BranchConditionWithRule(all_or_rules, branch_id))

        return AllBranchConditionRules(branching_conditions)
