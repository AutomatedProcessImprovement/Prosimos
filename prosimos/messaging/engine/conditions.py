"""Whether a message meets a consume entry's condition."""
from prosimos.branch_condition_rules import BranchConditionRule


class Conditions:
    """Conditions and their terms, for a waiting case or a future one."""

    def _condition_holds(self, condition, values, p_case):
        if condition is None:
            return True
        return any(all(self._term_holds(term, values, p_case) for term in alternative) for alternative in condition)

    def _term_holds(self, term, values, p_case):
        if term.attribute not in values:
            return False
        if term.object_attribute is None:
            compared_with = term.value
        elif term.object_attribute == "object_id":
            compared_with = self.object_id(p_case)
        else:
            case_values = self.sim_setup.bpmn_graph.get_all_attributes(p_case)
            if term.object_attribute not in case_values:
                return False
            compared_with = case_values[term.object_attribute]
        # the same comparison as branch rules
        return BranchConditionRule(term.attribute, term.comparison, compared_with).is_rule_true(
            {term.attribute: values[term.attribute]})

    def _could_match(self, condition, values):
        """Whether an unfinished case, or one a start message may still create, could accept the message. Only
        terms on fixed values and object_id are decided now; other case attributes can still change."""
        unfinished = self._unfinished
        future_cases = self.sim_setup.messaging.started_by_messages
        if condition is None:
            return bool(unfinished) or future_cases
        for alternative in condition:
            decided = [term for term in alternative if term.object_attribute in (None, "object_id")]
            if any(all(self._term_holds(term, values, p_case) for term in decided) for p_case in unfinished):
                return True
            if future_cases and all(self._term_could_hold_for_a_future_case(term, values) for term in decided):
                return True
        return False

    def _term_could_hold_for_a_future_case(self, term, values):
        """For a case a start message may still create: '=' on object_id holds only for an id numbered at least
        the cases so far; any other comparison might hold."""
        if term.object_attribute is None:
            return self._term_holds(term, values, None)
        if term.attribute not in values:
            return False
        if term.comparison != "=":
            return True
        prefix, _, number = str(values[term.attribute]).rpartition("-")
        return (prefix == self.id_prefix and number.isdigit() and str(int(number)) == number
                and int(number) >= len(self.log_info.trace_list))
