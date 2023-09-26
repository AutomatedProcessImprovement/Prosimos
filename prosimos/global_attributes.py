from enum import Enum
from functools import reduce
from random import choices
from typing import Dict

from pix_framework.statistics.distribution import DurationDistribution

from prosimos.exceptions import InvalidGlobalAttributeException


class GLOBAL_ATTR_TYPE(Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


def parse_discrete_value(value_info_arr):
    prob_arr = []
    options_arr = []
    for item in value_info_arr:
        options_arr.append(item["key"])
        prob_arr.append(float(item["value"]))

    return {
        "options": options_arr,
        "probabilities": prob_arr
    }


def parse_continuous_value(value_info) -> "DurationDistribution":
    return DurationDistribution.from_dict(value_info)


class GlobalAttribute:
    def __init__(self, name, global_attr_type, value):
        self.name: str = name
        self.global_attr_type: GLOBAL_ATTR_TYPE = GLOBAL_ATTR_TYPE(global_attr_type)

        if self.global_attr_type == GLOBAL_ATTR_TYPE.DISCRETE:
            self.value = parse_discrete_value(value)
        elif self.global_attr_type == GLOBAL_ATTR_TYPE.CONTINUOUS:
            self.value = parse_continuous_value(value)
        else:
            raise Exception(f"Not supported global attribute {type}")

        self.validate()

    def get_next_value(self):
        if self.global_attr_type == GLOBAL_ATTR_TYPE.DISCRETE:
            one_choice_arr = choices(self.value["options"], self.value["probabilities"])
            return one_choice_arr[0]
        else:
            return self.value.generate_sample(1)[0]

    def validate(self):
        if self.global_attr_type == GLOBAL_ATTR_TYPE.DISCRETE:
            actual_sum_probabilities = reduce(lambda acc, item: acc + item, self.value["probabilities"], 0)

            if actual_sum_probabilities != 1:
                raise InvalidGlobalAttributeException(
                    f"Global attribute ${self.name}: probabilities' sum should be equal to 1")

        return True


class AllGlobalAttributes:
    def __init__(self, global_attr_dict: Dict[str, GlobalAttribute]):
        self.attributes = global_attr_dict

    def get_columns_generated(self):
        return list({attr_name for attr_name in self.attributes.keys()})

    def get_values_calculated(self):
        return {attr_name: attr.get_next_value() for attr_name, attr in self.attributes.items()}
