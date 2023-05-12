from typing import List, Tuple, Union
from prosimos.case_attributes import AllCaseAttributes
from prosimos.prioritisation_rules import AllPriorityRules

ArrayOfTuplesOrStrings = Union[List[Tuple[str, any]], List[str]]


class CasePrioritisation:
    def __init__(
        self,
        total_num_cases,
        case_attributes: AllCaseAttributes,
        prioritisation_rules: AllPriorityRules,
    ):
        self.total_num_cases = total_num_cases
        self.case_attributes = case_attributes
        self.prioritisation_rules = prioritisation_rules
        (
            self.all_case_attributes,
            self.all_case_priorities,
        ) = self.calculate_case_attr_and_priorities()

    def get_priority_by_case_id(self, case_id):
        return self.all_case_priorities[case_id]

    def get_case_attr_values(self, case_id):
        "Return all case attributes assigned to the specific case id"
        return self.all_case_attributes[case_id].values()

    def calculate_case_attr_and_priorities(self):
        "Calculate values of each case attribute for every case that will be executed"
        total_num_cases = self.total_num_cases
        all_case_attr_dict = dict()
        all_case_priorities = dict()
        for case_id in range(0, total_num_cases):
            curr_case_attributes = self.case_attributes.get_values_calculated()
            all_case_attr_dict[case_id] = curr_case_attributes
            all_case_priorities[case_id] = self.prioritisation_rules.get_priority(
                curr_case_attributes
            )

        return all_case_attr_dict, all_case_priorities

    @staticmethod
    def _get_sorting_func(value: tuple):
        # if input array is tuple - sort by priority, then by datetime
        # if simple type (e.g., string) - sort by priority
        is_tuple = type(value[1]) is tuple
        return value[0], value[1][1].datetime if is_tuple else value[0]

    def get_ordered_case_ids_by_priority(self, case_ids: ArrayOfTuplesOrStrings):
        "Sort list of case_ids based on their previously calculated priority"
        if not self.all_case_priorities:
            "Return the original case_ids in case no priorities were defined"
            return case_ids

        # check whether it's array of tuples
        is_tuple = type(case_ids[0]) is tuple
        # get only array of case ids
        only_case_ids = [case_id for (case_id, _) in case_ids] if is_tuple else case_ids
        # calculate the priority of each case id in the list
        priority = [self.all_case_priorities[case_id] for case_id in only_case_ids]
        # sort the input list based on the calculated priorities
        return [
            x
            for _, x in sorted(
                zip(priority, case_ids), key=CasePrioritisation._get_sorting_func
            )
        ]

    def calculate_max_priority(self, case_ids: List[Tuple[str, any]]):
        "Calculate the maximum priority for the list of case ids"
        ordered_case_ids = self.get_ordered_case_ids_by_priority(case_ids)
        case_highest_priority = ordered_case_ids[0][0]
        return self.all_case_priorities[case_highest_priority]
