import itertools
from heapq import heappop, heappush

from prosimos.orchestrator import EngineReport, Message, SimulationEngine, Verdict


class ScriptedEngine(SimulationEngine):
    """
    A test double implementing the engine interface, so the orchestrator can't tell it from a real
    engine. Actions run at scheduled times and may publish messages. Consuming points decide, for
    each offered message, whether it is CLAIMED (and may schedule a reaction), DISCARDED (it can
    never match) or PENDING. A message type with no consuming point here is discarded.
    """

    def __init__(self, name):
        self.name = name
        self._agenda = []
        self._sequence = itertools.count()
        self._consumers = {}
        self.offered = []  # (message, time) every offer this engine received
        self.claimed = []  # (message, time) this engine claimed
        self.discarded = []  # (message, time) this engine discarded
        self.report = EngineReport()  # what finish() returns; tests may fill it in

    def at(self, time, action):
        """Run action(now) at time; it may return messages to publish."""
        heappush(self._agenda, (time, next(self._sequence), action))

    def publish_at(self, time, message_type, **attributes):
        self.at(time, lambda now: [Message(message_type, dict(attributes))])

    def consume(self, message_type, decide):
        """decide(message, now) returns a Verdict."""
        self._consumers.setdefault(message_type, []).append(decide)

    def subscriptions(self):
        return sorted(self._consumers)

    def next_event_time(self):
        return self._agenda[0][0] if self._agenda else None

    def step(self):
        now, _, action = heappop(self._agenda)
        return list(action(now) or [])

    def deliver(self, message, now):
        self.offered.append((message, now))
        verdicts = []
        for decide in self._consumers.get(message.type, []):
            verdict = decide(message, now)
            verdicts.append(verdict)
            if verdict is Verdict.CLAIMED:
                self.claimed.append((message, now))
                return verdict
        if all(verdict is Verdict.DISCARDED for verdict in verdicts):
            self.discarded.append((message, now))
            return Verdict.DISCARDED
        return Verdict.PENDING

    def finish(self):
        return self.report
