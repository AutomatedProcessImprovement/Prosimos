"""The quote of a publish entry: its lines, its limit, and its message types."""
from collections import Counter

from prosimos.messaging.model import QuoteSettings
from prosimos.messaging.parser.common import PERIODS, check_keys, fail, read_multiplicity, read_period

QUOTE_KEYS = {"quantity", "within", "accepted", "declined", "cancel", "modify"}
PORTION_SIZE = "portion_size"
NEEDED = ("quantity", "accepted", "declined", "cancel")
TYPES = ("accepted", "declined", "cancel", "modify")


def read_quote(entry, where, declared):
    if "quote" not in entry:
        return None
    for key in ("fan_out", "fan_in"):
        if key in entry:
            fail(f"{where}: quote and {key} can't be combined on one entry: the quote's lines say how many items "
                 f"it asks for, and its replies come from the orchestrator")
    if isinstance(entry.get("attributes"), list) and "index" in entry["attributes"]:
        fail(f"{where}: a quote's lines can't carry index: modify matches lines by their attributes")
    quote, in_quote = entry["quote"], f"{where}.quote"
    if not isinstance(quote, dict):
        fail(f"{where}: 'quote' must be an object with quantity, accepted, declined, and cancel, got {quote!r}")
    quantity = quote.get("quantity")
    if (isinstance(entry.get("attributes"), list) and PORTION_SIZE in entry["attributes"]) or (
            isinstance(quantity, dict) and quantity.get("as") == PORTION_SIZE):
        fail(f"{where}: a quote's lines can't carry {PORTION_SIZE}, which is reserved: an item claims its line with "
             f"the number of items in its supplier's portion under that name")
    check_keys(in_quote, quote, QUOTE_KEYS)
    missing = [key for key in NEEDED if key not in quote]
    if missing:
        fail(f"{where}: 'quote' needs quantity, accepted, declined, and cancel; it has no {', '.join(missing)}")
    names = {key: quote[key] for key in TYPES if key in quote}
    for key, name in names.items():
        if not isinstance(name, str) or not name.strip():
            fail(f"{where}: the quote's {key} must be a message type, a non-empty string, got {name!r}")
    used = [entry["type"], *names.values()]
    repeated = sorted(name for name, times in Counter(used).items() if times > 1)
    if repeated:
        fail(f"{where}: {', '.join(map(repr, repeated))} is used more than once among the entry's type and the "
             f"quote's accepted, declined, cancel, and modify: each needs a type of its own")
    return QuoteSettings(quote["accepted"], quote["declined"], quote["cancel"], quote.get("modify"),
                         *read_multiplicity(quote, in_quote, declared, "quantity"), *_within(quote, in_quote, declared))


def _within(quote, in_quote, declared):
    """(the limit as a timedelta, the case attribute holding its number, its unit); all None without within."""
    within = quote.get("within")
    if not (isinstance(within, dict) and any(isinstance(number, dict) for number in within.values())):
        return read_period(quote, in_quote, "within"), None, None
    check_keys(f"{in_quote}.within", within, set(PERIODS))
    if len(within) != 1:
        fail(f"{in_quote}: 'within' has one unit, weeks, days, or hours, got {within!r}")
    [(unit, number)] = within.items()
    check_keys(f"{in_quote}.within.{unit}", number, {"object_attribute"})
    attribute = number.get("object_attribute")
    if not isinstance(attribute, str) or attribute not in declared:
        fail(f"{in_quote}: 'within' {unit} object_attribute {attribute!r} is not a declared object, global or event "
             f"attribute")
    return None, attribute, unit


def check_quote_replies(consume, publish, names):
    """A consume entry for a quote's accepted or declined type has a term on request in every alternative of its
    condition: else it would take the replies to other objects' quotes. names: event id -> its name in the model,
    or its id."""
    replies = {name for _, point in publish if point.quote is not None
               for name in (point.quote.accepted, point.quote.declined)}
    for index, point in enumerate(consume):
        if point.type in replies and not (point.condition and all(
                any(term.attribute == "request" for term in alternative) for alternative in point.condition)):
            fail(f"messages.consume[{index}]: {point.type} at {names[point.event_id]!r} has no condition on request: "
                 f"it would take the replies to other objects' quotes")


def check_quote_types(publish):
    """Each quote's cancel and modify type has a publish entry of its own: the requester throws it at its own event."""
    published = {point.type for _, point in publish}
    for where, point in publish:
        for key in ("cancel", "modify"):
            name = getattr(point.quote, key) if point.quote is not None else None
            if name is not None and name not in published:
                fail(f"{where}: the quote's {key} type {name!r} has no publish entry of its own in this process: "
                     f"the requester throws it at its own event")
