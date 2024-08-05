import copy
import pprint

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
        curr_gateway_conditions = gateway_conditions.get(e_info.id, None)

        # No conditions (use probabilities)
        if not curr_gateway_conditions or not curr_gateway_conditions.candidates_list:
            return OutgoingFlowSelector._use_probabilities(e_info, element_probability)

        candidates_list = curr_gateway_conditions.candidates_list
        passed_arcs_ids = []

        # Evaluate conditions
        for candidate in candidates_list:
            condition = curr_gateway_conditions.rules_list[candidates_list.index(candidate)]
            if not condition or condition.is_true(all_attributes):
                passed_arcs_ids.append(candidate)

        # All conditions evaluated to false
        if not passed_arcs_ids:
            default_path = curr_gateway_conditions.get_default_path()
            if default_path:
                warning_logger.add_warning(f"[XOR] {e_info.id} all conditions evaluated to false. Using default path {default_path}.")
                return [(default_path, None)]
            else:
                warning_logger.add_warning(f"[XOR] {e_info.id} all conditions evaluated to false and no default path. Using probabilities.")
                return OutgoingFlowSelector._use_probabilities(e_info, element_probability)

        # One true condition
        if len(passed_arcs_ids) == 1:
            return [(passed_arcs_ids[0], None)]

        # More than 1 true (use scaled probabilities)
        curr_candidates = element_probability[e_info.id].candidates_list
        curr_probabilities = element_probability[e_info.id].probability_list
        passed_arcs_probs = OutgoingFlowSelector._get_probabilities(passed_arcs_ids, curr_candidates, curr_probabilities)

        scaled_probabilities = OutgoingFlowSelector._scale_to_one(passed_arcs_probs)
        choice = Choice(passed_arcs_ids, scaled_probabilities)

        warning_logger.add_warning(f"{e_info.id} more than 1 XOR gateway conditions evaluated to positive result. Scaled probabilities were used.")
        return [(choice.get_outgoing_flow(), None)]

    @staticmethod
    def _handle_inclusive_gateway(e_info, element_probability, all_attributes, gateway_conditions):
        curr_gateway_conditions = gateway_conditions.get(e_info.id, None)

        # No conditions (use probabilities)
        if not curr_gateway_conditions or not curr_gateway_conditions.candidates_list:
            return element_probability[e_info.id].get_multiple_flows()

        candidates_list = curr_gateway_conditions.candidates_list
        passed_arcs_ids = []
        evaluated_arcs_ids = []

        # Evaluate conditions
        for candidate in candidates_list:
            condition = curr_gateway_conditions.rules_list[candidates_list.index(candidate)]
            if condition:
                evaluated_arcs_ids.append(candidate)
                if condition.is_true(all_attributes):
                    passed_arcs_ids.append(candidate)

        # All conditions evaluated to false
        if not passed_arcs_ids:
            default_path = curr_gateway_conditions.get_default_path()
            if default_path:
                warning_logger.add_warning(f"[OR] {e_info.id} all conditions evaluated to false. Using default path {default_path}.")
                return [(default_path, None)]
            else:
                warning_logger.add_warning(f"[OR] {e_info.id} all conditions evaluated to false and no default path. Using probabilities.")
                return element_probability[e_info.id].get_multiple_flows()

        # Execute true conditions
        result_flows = [(flow, None) for flow in passed_arcs_ids]

        # Use probabilities for the rest of the flows where conditions are missing
        missing_conditions_flows = [flow for flow in candidates_list if flow not in evaluated_arcs_ids]
        if missing_conditions_flows:
            probabilities = element_probability[e_info.id].get_multiple_flows()
            result_flows.extend([(flow, None) for flow in probabilities if flow in missing_conditions_flows])

        return result_flows
        
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