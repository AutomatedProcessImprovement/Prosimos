from typing import List, Tuple, Union
from prosimos.case_attributes import AllCaseAttributes
from prosimos.prioritisation_rules import AllPriorityRules

ArrayOfTuplesOrStrings = Union[List[Tuple[str, any]], List[str]]

class CasePrioritisation:
    """
    Dynamically assigns priorities to any case ID,
    so there's no KeyError when a new or large ID appears.
    """
    def __init__(
        self,
        total_num_cases: int,
        case_attributes: AllCaseAttributes,
        prioritisation_rules: AllPriorityRules,
    ):
        self.total_num_cases = total_num_cases
        self.case_attributes = case_attributes
        self.prioritisation_rules = prioritisation_rules

        self.all_case_attributes, self.all_case_priorities = self._precompute_initial_priorities()

    def _precompute_initial_priorities(self):
        attr_dict = {}
        priority_dict = {}
        for case_id in range(self.total_num_cases):
            new_attrs = self.case_attributes.get_values_calculated()
            attr_dict[case_id] = new_attrs
            priority_dict[case_id] = self.prioritisation_rules.get_priority(new_attrs)
        return attr_dict, priority_dict

    def get_priority_by_case_id(self, case_id: int):
        """
        Return the priority for 'case_id'. If we never saw this ID,
        dynamically compute a new set of attributes and a new priority.
        """
        if case_id not in self.all_case_priorities:
            new_attrs = self.case_attributes.get_values_calculated()
            self.all_case_attributes[case_id] = new_attrs
            self.all_case_priorities[case_id] = self.prioritisation_rules.get_priority(new_attrs)
        return self.all_case_priorities[case_id]

    def get_case_attr_values(self, case_id: int):
        if case_id not in self.all_case_attributes:
            new_attrs = self.case_attributes.get_values_calculated()
            self.all_case_attributes[case_id] = new_attrs
            self.all_case_priorities[case_id] = self.prioritisation_rules.get_priority(new_attrs)
        return self.all_case_attributes[case_id].values()

    def get_ordered_case_ids_by_priority(self, case_ids: ArrayOfTuplesOrStrings):
        if not self.all_case_priorities:
            return case_ids
        # Distinguish array of tuples vs. array of IDs
        is_tuple = isinstance(case_ids[0], tuple)
        only_case_ids = [cid for (cid, _) in case_ids] if is_tuple else case_ids

        # Ensure each ID is in the dictionary
        for cid in only_case_ids:
            self.get_priority_by_case_id(cid)

        # Retrieve priorities now that they're guaranteed to exist
        priority = [self.all_case_priorities[cid] for cid in only_case_ids]

        def _get_sorting_func(value: tuple):
            # if input array is tuple => sort by priority, then datetime
            # otherwise just priority
            is_tup = isinstance(value[1], tuple)
            return value[0], value[1][1].datetime if is_tup else value[0]

        return [
            x
            for _, x in sorted(zip(priority, case_ids), key=_get_sorting_func)
        ]

    def calculate_max_priority(self, case_ids: List[Tuple[str, any]]):
        ordered = self.get_ordered_case_ids_by_priority(case_ids)
        highest_pid = ordered[0][0]
        return self.all_case_priorities[highest_pid]
