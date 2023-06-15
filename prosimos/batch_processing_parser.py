from datetime import time
from typing import List
from prosimos.batch_processing import (
    BATCH_TYPE,
    RULE_TYPE,
    AndFiringRule,
    BatchConfigPerTask,
    FiringSubRule,
    OrFiringRule,
)
from prosimos.exceptions import InvalidRuleDefinitionException


class BatchProcessingParser:
    def __init__(self, json_data_with_batch_info):
        self.data = json_data_with_batch_info

    def parse(self):
        """
        Parse "batch_processing" section of json data
        """
        batch_config = dict()

        for batch_processing in self.data:
            t_id = batch_processing["task_id"]
            batch_type = BATCH_TYPE(batch_processing["type"])

            parsed_or_rules: List[OrFiringRule] = []
            for or_rules in batch_processing["firing_rules"]:
                parsed_and_rules: List[FiringSubRule] = []

                for and_rule in or_rules:
                    subrule = BatchProcessingParser.create_subrule(
                        and_rule["attribute"], and_rule["comparison"], and_rule["value"]
                    )

                    parsed_and_rules.append(subrule)

                BatchProcessingParser._move_size_to_end(parsed_and_rules)

                firing_rule = AndFiringRule(parsed_and_rules)

                firing_rule.validate()
                firing_rule.init_boundaries()

                parsed_or_rules.append(firing_rule)

            firing_rules = OrFiringRule(parsed_or_rules)

            duration_distibution = dict()
            for item in batch_processing["duration_distrib"]:
                key = int(item["key"])
                value = float(item["value"])
                duration_distibution[key] = value

            # add the default scale coefficient for the activity's duration
            # if it was not provided by the input
            if duration_distibution.get(1) is None:
                duration_distibution[1] = 1.0

            possible_options, probabilities = BatchProcessingParser.parse_size_distrib(
                batch_processing["size_distrib"]
            )

            batch_config[t_id] = BatchConfigPerTask(
                batch_type,
                duration_distibution,
                firing_rules,
                possible_options,
                probabilities,
            )

        return batch_config

    @staticmethod
    def parse_size_distrib(size_distrib):
        if len(size_distrib) == 0:
            return {}, {}

        possible_options = []
        probabilities = []

        for item in size_distrib:
            option = int(item["key"])
            possible_options.append(option)
            probabilities.append(item["value"])

        return possible_options, probabilities

    @staticmethod
    def create_subrule(attribute, comparison, value):
        if attribute == RULE_TYPE.DAILY_HOUR.value:
            formatted_value = time(int(value), 0, 0, 0)
        elif attribute == RULE_TYPE.WEEK_DAY.value:
            # string is accepted for the WEEK_DAY
            formatted_value = value
        elif attribute == RULE_TYPE.SIZE.value:
            formatted_value = int(value)
        else:
            # all others should have the number as the value
            formatted_value = float(value)

        if attribute == RULE_TYPE.WEEK_DAY.value and comparison != "=":
            # only "=" operator is allowed to be used with the week_day type of rule
            raise InvalidRuleDefinitionException(
                f"'{comparison}' is not allowed operator for the week_day type of rule."
            )

        return FiringSubRule(attribute, comparison, formatted_value)

    @staticmethod
    def _move_size_to_end(list: List[FiringSubRule]):
        """
        Re-order the elements inside list so that:
            1) size rule is the last one in the sequence
            2) week_day rule is the second item from the end
        """

        BatchProcessingParser._move_to_end("daily_hour", list)
        BatchProcessingParser._move_to_end("size", list)

    @staticmethod
    def _move_to_end(rule_name: str, list):
        rule_index = next(
            (i for i, item in enumerate(list) if item.variable1 == rule_name), -1
        )
        last_index = len(list) - 1

        if rule_index == -1:
            # size rule is not on the list of all rules
            return

        list.insert(last_index, list.pop(rule_index))
