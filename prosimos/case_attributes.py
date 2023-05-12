from enum import Enum
from functools import reduce
from typing import List
from prosimos.exceptions import InvalidCaseAttributeException
from prosimos.probability_distributions import generate_number_from
from random import choices

class CASE_ATTR_TYPE(Enum):
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

def parse_continuous_value(value_info):
    dist_params = []
    for param_info in value_info["distribution_params"]:
        dist_params.append(float(param_info["value"]))
    
    return {
        "distribution_name": value_info["distribution_name"],
        "distribution_params": dist_params
    }


class CaseAttribute():
    def __init__(self, name, case_atrr_type, value):
        self.name: str = name
        self.case_atrr_type: CASE_ATTR_TYPE = CASE_ATTR_TYPE(case_atrr_type)

        if self.case_atrr_type == CASE_ATTR_TYPE.DISCRETE:
            self.value = parse_discrete_value(value)
        elif self.case_atrr_type == CASE_ATTR_TYPE.CONTINUOUS:
            self.value = parse_continuous_value(value)
        else:
            raise Exception(f"Not supported case attribute {type}")

        self.validate()

    def get_next_value(self):
        if self.case_atrr_type == CASE_ATTR_TYPE.DISCRETE:
            one_choice_arr = choices(self.value["options"], self.value["probabilities"])
            return one_choice_arr[0]
        else:
            return generate_number_from(self.value["distribution_name"], self.value["distribution_params"])

    def validate(self):
        if self.case_atrr_type == CASE_ATTR_TYPE.DISCRETE:
            actual_sum_probabilities = reduce(lambda acc, item: acc + item, self.value["probabilities"], 0) 
            
            if actual_sum_probabilities != 1:
                raise InvalidCaseAttributeException(f"Case attribute ${self.name}: probabilities' sum should be equal to 1") 
        
        return True


class AllCaseAttributes():
    def __init__(self, case_attr_arr: List[CaseAttribute]):
        self.attributes = case_attr_arr

    def get_columns_generated(self):
        return [attr.name for attr in self.attributes]

    def get_values_calculated(self):
        # return the list of calculated values specified
        # the order should reflect the one with headers

        return {attr.name: attr.get_next_value() for attr in self.attributes}
