"""Prosimos as an engine: a Prosimos simulation behind the engine interface (docs/orchestrator.md)."""
import json
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from prosimos.control_flow_manager import BPMN
from prosimos.messaging.model import ConsumePoint, PublishPoint
from prosimos.messaging.parser.common import unknown_keys
from prosimos.messaging.parser.reading import writing_message_events
from prosimos.orchestration.catalogue.model import QuantityFrom
from prosimos.orchestration.catalogue.picks import Brought, Pick, Received
from prosimos.orchestration.config import ProcessSpec
from prosimos.orchestration.protocol import (CaseObject, EngineReport, Line, Message, MessageSide, Quote, QuoteKind,
                                           SimulationEngine, Slots, StalledCase, Verdict)
from prosimos.orchestration.random_generators import engine_random_states, generators
from prosimos.simulation_engine import SimBPMEnv
from prosimos.simulation_setup import SimDiffSetup
from prosimos.warning_logger import warning_logger


# a process's JSON sections: Prosimos's own, with object_attributes for its case_attributes, and messages
PROCESS_SETTINGS_KEYS = {"resource_profiles", "resource_calendars", "task_resource_distribution",
                         "arrival_time_distribution", "arrival_time_calendar", "gateway_branching_probabilities",
                         "branch_rules", "event_distribution", "batch_processing", "prioritisation_rules",
                         "object_attributes", "event_attributes", "global_attributes", "gateway_execution_limit",
                         "multitask", "model_type", "granule_size", "messages"}


def load_process(bpmn_path, json_path, arrivals, start_datetime, picks=()) -> SimDiffSetup:
    """Prosimos's setup of one process from its JSON settings in the orchestrator's names, handed to Prosimos in its
    own names in a temporary copy. picks: attributes given from outside the settings (see Pick)."""
    with open(json_path) as file:
        settings = json.load(file)
    problem = unknown_keys("top level", settings, PROCESS_SETTINGS_KEYS)
    if problem:
        raise ValueError(f"{json_path}: {problem}")
    if "object_attributes" in settings:
        settings["case_attributes"] = settings.pop("object_attributes")
    with tempfile.TemporaryDirectory() as folder:
        translated = Path(folder) / Path(json_path).name
        translated.write_text(json.dumps(settings))
        return SimDiffSetup(bpmn_path, str(translated), False, arrivals, start_datetime, picks, "arrivals")


def _quote(asked):
    """The Quote a message carries, from what the messaging engine says it asks; None for an ordinary message."""
    if asked is None:
        return None
    return Quote(QuoteKind(asked["kind"]), asked["quote_type"], asked["request"],
                 [Line(dict(attributes)) for attributes in asked.get("lines", [])], asked.get("limit"),
                 asked.get("accepted"), asked.get("declined"))


class ProsimosEngine(SimulationEngine):
    """A Prosimos simulation (SimBPMEnv) behind the SimulationEngine interface. Prosimos shares one warning list
    and the global generators between engines, so each call hands it this engine's own: their warnings never mix,
    and one engine's draws don't depend on the others. With a seed, its generators are seeded from the seed and its
    name; without one, from the global generators."""

    # it puts its own states into the global generators for each call, so the run loop needn't give them back
    manages_global_generators = True

    def __init__(self, spec: ProcessSpec, start_datetime: datetime, log_writer=None, seed: Optional[int] = None,
                 picks: Tuple["Pick", ...] = ()):
        self._spec = spec
        self._warnings: List[str] = []
        # (message, case id, element id, publish entry, when); the orchestrator stamps the message's id later
        self._published: List[Tuple[Message, int, str, PublishPoint, datetime]] = []
        # (message id, case id, element id, consume entry, when, since when the case waited there)
        self._claimed: List[Tuple[str, int, str, ConsumePoint, datetime, datetime]] = []
        self._python_state, self._numpy_state = engine_random_states(seed, spec.name)
        with self._own_globals():
            # picked attributes join the declared ones: a column of the log, and messages may carry them
            sim_setup = load_process(spec.bpmn_path, spec.json_path, spec.arrivals, start_datetime, picks)
            for pick in picks:
                if isinstance(pick, Pick) and isinstance(pick.quantity, QuantityFrom):
                    pick.read_quantity_from(next(attribute for attribute in sim_setup.case_attributes.attributes
                                              if attribute.name == pick.quantity.object_attribute))
            self._env = SimBPMEnv(sim_setup, None, log_writer, id_prefix=spec.id_prefix)
            # its organization's catalogue values, for the 'with' of its references (see Received and Brought)
            self._env.receive_catalogue_values([(pick.name, pick.source, pick.values, pick.function, pick.initial,
                                                 pick.catalogue) for pick in picks
                                                if isinstance(pick, (Received, Brought))])
        tasks = {element_id for element_id, element in sim_setup.bpmn_graph.element_info.items()
                 if element.type is BPMN.TASK}
        with open(spec.json_path) as file:
            events = writing_message_events(spec.bpmn_path, json.load(file).get("messages"))
        named = [element_id for element_id, _ in spec.qualifier_by_activity if element_id not in tasks]
        # a message event by its id only, as a task: events also holds their names
        unknown = sorted(element_id for element_id in named
                         if element_id not in events or element_id not in sim_setup.bpmn_graph.element_info)
        if unknown:
            what = "is not a task or message event" if len(unknown) == 1 else "are not tasks or message events"
            raise ValueError(f"{spec.name}: qualifier_by_activity names {', '.join(unknown)}, which {what} of its "
                             f"model")
        silent = sorted(element_id for element_id in named if not events[element_id])
        if silent:
            raise ValueError(f"{spec.name}: qualifier_by_activity names {silent[0]}, which writes no event: its entry "
                             f"has no qualifier")

    @contextmanager
    def _own_globals(self):
        shared_warnings = warning_logger.warnings_queue
        warning_logger.warnings_queue = self._warnings
        generators.take(self)
        try:
            yield
        finally:
            warning_logger.warnings_queue = shared_warnings
            if not generators.keep:
                generators.give_back()

    def subscriptions(self) -> List[str]:
        with self._own_globals():
            return self._env.sim_setup.messaging.subscriptions()

    def next_event_time(self) -> Optional[datetime]:
        # None while only waiting cases are left: nothing to do now, but not finished
        with self._own_globals():
            return self._env.next_event_time()

    def step(self) -> List[Message]:
        with self._own_globals():
            now = self._env.next_event_time()  # the time the orchestrator stamps on what this step publishes
            self._env.step()
            # hand the buffered log rows over now, so nobody outside the engine reaches into it at the end
            self._env.log_writer.force_write()
            released = []
            for message_type, attributes, publishers, element_id, point, asked in self._env.outbox:
                message = Message(message_type, attributes, quote=_quote(asked))
                # the orchestrator stamps the id on this same object, so it can be read at the end; a fan-in
                # message has one publish record per member, at its own time
                self._published.extend((message, p_case, element_id, point, when) for p_case, when in publishers)
                released.append(message)
            self._env.outbox.clear()
            return released

    def deliver(self, message: Message, now: datetime) -> Verdict:
        with self._own_globals():
            verdict = Verdict(self._env.deliver(message.type, message.attributes, message.source, now))
            if verdict is Verdict.CLAIMED:
                self._claimed.extend((message.id, p_case, element_id, point, now, enabled)
                                     for p_case, element_id, point, enabled in self._env.claimed_by)
            return verdict

    def estimate(self, message: Message, now: datetime) -> Slots:
        with self._own_globals():
            return self._env.slots_for(message.quote.quote_type, message.source,
                                       [(line.id, line.attributes) for line in message.quote.lines], now)

    def moved(self, message: Message, now: datetime) -> bool:
        return self._env.has_moved(message.id)

    def reserve(self, message: Message, line_ids: List[str], now: datetime) -> None:
        with self._own_globals():
            lines = [(line.id, line.attributes) for line in message.quote.lines if line.id in line_ids]
            self._env.open_lines(message.id, message.quote.quote_type, message.source, message.quote.request, lines,
                                 now)

    def release(self, message: Message, line_ids: List[str], now: datetime) -> None:
        with self._own_globals():
            self._env.release_lines(message.id, line_ids, now)

    def finish(self) -> EngineReport:
        with self._own_globals():
            self._env.warn_unsent_fan_ins()
            stalled = [StalledCase(self._env.object_id(parked_event.p_case), place, message_types,
                                   parked_event.enabled_datetime, *self._env.collected(parked_event))
                       for parked_event, place, message_types in self._env.stalled_waits()]
            created, only_started = self._env.case_creation_times(), self._env.only_started_values()
            objects = [CaseObject(self._env.object_id(p_case), created[p_case], history, only_started[p_case])
                       for p_case, history in sorted(self._env.case_attribute_history().items())]
            self._claimed.extend(self._env.line_claims)  # an item that took a line claimed its quote or modify
            published, claimed = self._attached_records()
            logged_elements = [element_id for _, element_id, _, _ in self._env.logged_rows]
        return EngineReport(stalled, list(self._warnings), published, claimed, objects, logged_elements,
                            dict(self._env._moved_quotes))

    def _attached_records(self):
        """The publish and claim records; an end with a qualifier, of a process with an object type, also says when
        its OCEL event happened and when its case reached the element (see MessageSide)."""
        attach = self._spec.object_type is not None
        graph = self._env.sim_setup.bpmn_graph

        def event(element_id, point, at, enabled):
            if not attach or not point.qualifier:
                return None, None, None
            return graph.element_info[element_id].name, at, enabled

        published = [MessageSide(self._env.object_id(p_case), element_id, point.qualifier or "",
                                 *event(element_id, point, when, when), point.o2o, point.carry, message_id=message.id)
                     for message, p_case, element_id, point, when in self._published]
        claimed = [MessageSide(self._env.object_id(p_case), element_id, point.qualifier or "",
                               *event(element_id, point, when, enabled), point.o2o, point.carry, message_id=message_id)
                   for message_id, p_case, element_id, point, when, enabled in self._claimed]
        return published, claimed
