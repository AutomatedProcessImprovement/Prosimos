import itertools
from heapq import heappop, heappush

from prosimos.orchestration.protocol import EngineReport, Message, SimulationEngine, Verdict


class ScriptedEngine(SimulationEngine):
    """
    A test double implementing the engine interface, so the orchestrator can't tell it from a real
    engine. Actions run at scheduled times and may publish messages. Consuming points decide, for
    each offered message, whether it is CLAIMED (and may schedule a reaction), DISCARDED (it can
    never match) or PENDING. A message type with no consuming point here is discarded. Quotes are
    answered from a script too: slots give each line's slot times, and moved_when whether some line
    of the quote has moved on.
    """

    def __init__(self, name):
        self.name = name
        self._agenda = []
        self._sequence = itertools.count()
        self._consumers = {}
        self.offered = []  # (message, time) every offer this engine received
        self.claimed = []  # (message, time) this engine claimed
        self.discarded = []  # (message, time) this engine discarded
        self._slots = {}
        self._moved = lambda message, now: False
        self.estimated = []  # (quote message, time) every estimate() call
        self.moved_checks = []  # (quote message, time) every moved() call
        self.reserved = []  # (quote message, line ids, time) every reserve() call
        self.released = []  # (quote message, line ids, time) every release() call
        self.report = EngineReport()  # what finish() returns; tests may fill it in

    def at(self, time, action):
        """Run action(now) at time; it may return messages to publish."""
        heappush(self._agenda, (time, next(self._sequence), action))

    def publish_at(self, time, message_type, **attributes):
        self.at(time, lambda now: [Message(message_type, dict(attributes))])

    def consume(self, message_type, decide):
        """decide(message, now) returns a Verdict."""
        self._consumers.setdefault(message_type, []).append(decide)

    def slots(self, quote_type, decide):
        """decide(line, now) returns that line's slot times for the quotes of quote_type, which this engine then
        subscribes to; without it, a quote gets no slots."""
        self._slots[quote_type] = decide

    def subscriptions(self):
        return sorted({*self._consumers, *self._slots})

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

    def estimate(self, message, now):
        self.estimated.append((message, now))
        decide = self._slots.get(message.quote.quote_type)
        slots = {line.id: list(decide(line, now)) for line in message.quote.lines} if decide is not None else {}
        return slots

    def moved_when(self, decide):
        """decide(message, now) returns whether some line of the quote message has moved on here; without it,
        nothing ever has."""
        self._moved = decide

    def moved(self, message, now):
        self.moved_checks.append((message, now))
        return self._moved(message, now)

    def reserve(self, message, line_ids, now):
        self.reserved.append((message, list(line_ids), now))

    def release(self, message, line_ids, now):
        self.released.append((message, list(line_ids), now))

    def finish(self):
        return self.report
