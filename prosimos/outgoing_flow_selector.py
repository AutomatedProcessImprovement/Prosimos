import copy
from prosimos.control_flow_manager import BPMN
from prosimos.probability_distributions import Choice
from prosimos.gateway_condition_choice import GatewayConditionChoice
from prosimos.warning_logger import warning_logger


class OutgoingFlowSelector:
    @staticmethod
    def choose_outgoing_flow(e_info, element_probability, all_attributes, gateway_conditions):
        if e_info.type is BPMN.EXCLUSIVE_GATEWAY:
            return OutgoingFlowSelector._handle_exclusive_gateway(e_info, element_probability, all_attributes,
                                                                  gateway_conditions)
        elif e_info.type is BPMN.INCLUSIVE_GATEWAY:
            return OutgoingFlowSelector._handle_inclusive_gateway(e_info, element_probability, all_attributes,
                                                                  gateway_conditions)
        elif e_info.type in [BPMN.TASK, BPMN.PARALLEL_GATEWAY, BPMN.START_EVENT, BPMN.INTERMEDIATE_EVENT]:
            return OutgoingFlowSelector._handle_parallel_events(e_info)

    @staticmethod
    def _handle_exclusive_gateway(e_info, element_probability, all_attributes, gateway_conditions):
        curr_gateway_conditions = gateway_conditions[e_info.id]
        candidates_list = curr_gateway_conditions.candidates_list

        # No conditions (use probabilities)
        if not candidates_list:
            return OutgoingFlowSelector._use_probabilities(e_info, element_probability)
        
        condition_choice = GatewayConditionChoice(candidates_list, curr_gateway_conditions.rules_list)
        passed_arcs_ids = condition_choice.get_outgoing_flow(all_attributes)

        # One true condition
        if len(passed_arcs_ids) == 1:
            return [(passed_arcs_ids[0], None)]

        # All true or all false
        if len(passed_arcs_ids) in [0, len(candidates_list)]:
            warning_logger.add_warning(f"{e_info.id} all XOR gateway conditions evaluated to the same result ")
            return OutgoingFlowSelector._use_probabilities(e_info, element_probability)

        # More than 1 true (use scaled probabilities)
        curr_candidates = element_probability[e_info.id].candidates_list
        curr_probabilities = element_probability[e_info.id].probability_list
        passed_arcs_probs = OutgoingFlowSelector._get_probabilities(passed_arcs_ids, curr_candidates,
                                                                    curr_probabilities)

        scaled_probabilities = OutgoingFlowSelector._scale_to_one(passed_arcs_probs)
        choice = Choice(passed_arcs_ids, scaled_probabilities)

        warning_logger.add_warning(f"{e_info.id} more than 1 XOR gateway conditions evaluated to positive result. Scaled probabilities were used.")
        return [(choice.get_outgoing_flow(), None)]


    @staticmethod
    def _handle_inclusive_gateway(e_info, element_probability, all_attributes, gateway_conditions):
        curr_gateway_conditions = gateway_conditions[e_info.id]
        candidates_list = curr_gateway_conditions.candidates_list

        # No conditions (use probabilities)
        if not candidates_list:
            return element_probability[e_info.id].get_multiple_flows()

        condition_choice = GatewayConditionChoice(candidates_list, curr_gateway_conditions.rules_list)
        passed_arcs_ids = condition_choice.get_outgoing_flow(all_attributes)

        # All false (use probabilities)
        if not passed_arcs_ids:
            warning_logger.add_warning(f"{e_info.id} all OR gateway conditions evaluated to negative result. Probabilities were used.")
            return element_probability[e_info.id].get_multiple_flows()
        else:
            return [(flow, None) for flow in passed_arcs_ids]
        
    @staticmethod
    def _handle_parallel_events(e_info):
        flows = copy.deepcopy(e_info.outgoing_flows)
        return [(flow, None) for flow in flows]

    @staticmethod
    def _scale_to_one(values):
        total = sum(values)
        return [value / total for value in values]

    @staticmethod
    def _get_probabilities(passed_flows, candidates_list, probability_list):
        return [probability_list[candidates_list.index(flow)] for flow in passed_flows]

    @staticmethod
    def _use_probabilities(e_info, element_probability):
        return [(element_probability[e_info.id].get_outgoing_flow(), None)]