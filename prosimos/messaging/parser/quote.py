"""The quote of a publish entry: its lines, its limit, and its message types."""
from prosimos.messaging.model import QuoteSettings
from prosimos.messaging.parser.common import read_multiplicity, read_period


def read_quote(entry, where, declared):
    if "quote" not in entry:
        return None
    quote, in_quote = entry["quote"], f"{where}.quote"
    within = quote.get("within")
    if isinstance(within, dict) and len(within) == 1 and isinstance(next(iter(within.values())), dict):
        [(unit, number)] = within.items()
        limit = None, number.get("object_attribute"), unit
    else:
        limit = read_period(quote, in_quote, "within"), None, None
    return QuoteSettings(quote.get("accepted"), quote.get("declined"), quote.get("cancel"), quote.get("modify"),
                         *read_multiplicity(quote, in_quote, declared, "quantity"), *limit)
