"""Races at event-based gateways: the first branch to happen wins."""
from datetime import timedelta

from prosimos.execution_info import EnabledEvent
from prosimos.warning_logger import warning_logger


class Race:
    """The branches of one event-based gateway, armed for one case: the first to happen wins."""

    def __init__(self, p_case, gateway_id):
        self.p_case = p_case
        self.gateway_id = gateway_id
        self.won = False
        self.wins_at_once = None  # the message branch with nothing to collect that wins at once, if any


class Races:
    """The methods of the races at event-based gateways."""

    def _first_listed_branch(self, race, message_type, values):
        """The parked branch and entry of a race that take a message several branches accept: the first listed."""
        accepting = []
        for branch in self._race_branches[race.gateway_id]:
            parked_event = self._parked_events.get((race.p_case, branch))
            if parked_event is None:
                continue
            point = next((point for point in self._consume_points[branch] if point.type == message_type
                          and self._accepts(point, values, race.p_case)), None)
            if point is not None:
                accepting.append((parked_event, point))
        parked_event, point = min(accepting, key=lambda accepted: self._consume_order[id(accepted[1])])
        if len(accepting) > 1 and race.gateway_id not in self._warned_overlapping_races:
            self._warned_overlapping_races.add(race.gateway_id)
            branches = ", ".join(sorted(accepted[0].task_id for accepted in accepting))
            warning_logger.add_warning(
                f"a {message_type} message matches several branches of the race at {race.gateway_id} ({branches}); "
                f"it goes to {point.event_id}, whose 'consume' entry comes first")
        return parked_event, point

    def _race_step(self, c_event):
        """A branch of a race comes off the queue. Returns True if nothing more is to be done with it now."""
        if c_event.race is not None:  # a timer branch due to fire
            if c_event.race.won:
                return True
            incoming = self.sim_setup.bpmn_graph.element_info[c_event.task_id].incoming_flows
            if not any(c_event.p_state.has_token(flow) for flow in incoming):
                # the case left the race without it, e.g. ended by a terminate end event on another branch
                c_event.race.won = True
                self._races.pop((c_event.race.p_case, c_event.race.gateway_id), None)
                return True
            self._win_race(c_event.race, c_event.task_id, c_event.p_state)
            return False  # it completes now; execute_event logs it from when it was armed
        incoming = self.sim_setup.bpmn_graph.element_info[c_event.task_id].incoming_flows
        if not any(c_event.p_state.has_token(flow) for flow in incoming):
            return True  # another branch already won at this instant, e.g. one with nothing to collect
        key = (c_event.p_case, self._race_gateway_of[c_event.task_id])
        if key not in self._races:
            self._races[key] = Race(*key)
            self._races[key].wins_at_once = self._branch_with_nothing_to_collect(*key)
        c_event.race = self._races[key]
        if c_event.race.wins_at_once not in (None, c_event.task_id):
            return True  # a branch with nothing to collect wins at once; this one is never armed
        if c_event.task_id in self._consume_points:
            return False
        # due a microsecond late, so a message other engines publish for the same instant wins
        [delay] = self.sim_setup.bpmn_graph.event_duration(c_event.task_id)
        due = EnabledEvent(c_event.p_case, c_event.p_state, c_event.task_id, c_event.enabled_at + delay + 1e-6,
                           c_event.enabled_datetime + timedelta(seconds=delay, microseconds=1), is_inter_event=True)
        due.race, due.armed_event, due.duration_sec = c_event.race, c_event, delay
        self.events_queue.append_event(due)
        return True

    def _branch_with_nothing_to_collect(self, p_case, gateway_id):
        """The branch with nothing to collect that wins at once, the first listed; None if there is none."""
        branches = [branch for branch in self._race_branches[gateway_id]
                    if branch in self._consume_points and self._needed(p_case, branch) == 0]
        if not branches:
            return None
        branches.sort(key=lambda branch: self._consume_order[id(self._consume_points[branch][0])])
        if len(branches) > 1 and gateway_id not in self._warned_races_won_at_once:
            self._warned_races_won_at_once.add(gateway_id)
            warning_logger.add_warning(
                f"several branches of the race at {gateway_id} have nothing to collect "
                f"({', '.join(sorted(branches))}); {branches[0]}, whose 'consume' entry comes first, wins")
        return branches[0]

    def _win_race(self, race, winner, p_state):
        """Cancels every branch of race but winner: their tokens are taken off, and a queued timer is skipped later."""
        race.won = True
        del self._races[(race.p_case, race.gateway_id)]
        graph = self.sim_setup.bpmn_graph
        for branch in self._race_branches[race.gateway_id]:
            if branch == winner:
                continue
            for flow in graph.element_info[branch].incoming_flows:
                if p_state.tokens[flow] > 0:
                    p_state.tokens[flow] -= 1
                    if p_state.tokens[flow] == 0:
                        p_state.state_mask &= ~graph.arcs_bitset[flow]
            self._parked_events.pop((race.p_case, branch), None)
            self._collected.pop((race.p_case, branch), None)
        self._tokens_moved(race.p_case, p_state)
