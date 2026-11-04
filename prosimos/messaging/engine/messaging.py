"""The messaging methods SimBPMEnv inherits (docs/engine-internals.md, docs/messaging.md)."""
from prosimos.messaging.engine.conditions import Conditions
from prosimos.messaging.engine.fan_in import FanIn
from prosimos.messaging.engine.publishing import Publishing
from prosimos.messaging.engine.races import Races
from prosimos.messaging.engine.receiving import Receiving
from prosimos.messaging.engine.reporting import Reporting
from prosimos.messaging.engine.stepping import Stepping
from prosimos.messaging.engine.values import Values


class EngineMessaging(Stepping, Receiving, Races, Conditions, Values, Publishing, FanIn, Reporting):
    """The methods the messaging adds to SimBPMEnv, which inherits them as they are (same names, same self)."""
