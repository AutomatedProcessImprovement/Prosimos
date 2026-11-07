"""What the parser's files share with each other and with other code: failing, keys, number checks,
multiplicities, and periods."""
import numbers
from datetime import timedelta

from prosimos.exceptions import InvalidSimScenarioException

PERIODS = ("weeks", "days", "hours")  # of a deadline, and of a catalogue change
# what each multiplicity counts, its least number, and whose object_id the reserved name means
MULTIPLICITIES = {"fan_out": ("message", 0, "publishing"),
                  "quantity": ("line", 1, "requesting")}


def is_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def is_whole_number(value, minimum):
    return (not isinstance(value, bool) and isinstance(value, (int, float)) and value >= minimum
            and float(value).is_integer())


def is_whole_number_of_at_least_one(value):
    return is_whole_number(value, minimum=1)


def unknown_keys(where, mapping, allowed):
    """The load error naming the keys of mapping that aren't allowed; None if there are none. A misspelled key
    would otherwise change the run silently."""
    unknown = [key for key in mapping if key not in allowed]
    if not unknown:
        return None
    return (f"{where}: unknown key{'s' if len(unknown) > 1 else ''} {', '.join(map(repr, unknown))}; "
            f"allowed: {', '.join(sorted(allowed))}")


def check_keys(where, mapping, allowed):
    problem = unknown_keys(where, mapping, allowed)
    if problem:
        fail(problem)


def fail(reason):
    raise InvalidSimScenarioException(f"Invalid 'messages' section: {reason}")


def read_multiplicity(entry, where, declared, key):
    """(number, its case attribute, a list attribute, the name each message or line carries its entry under) of
    entry[key], fan_out or a quote's quantity; all None without it."""
    if key not in entry:
        return None, None, None, None
    multiplicity, (_, least, _) = entry[key], MULTIPLICITIES[key]
    if is_whole_number(multiplicity, minimum=least):
        return int(multiplicity), None, None, None
    if isinstance(multiplicity, dict):
        check_keys(f"{where}.{key}", multiplicity, {"object_attribute", "list", "as"})
    if isinstance(multiplicity, dict) and "list" in multiplicity:
        return (None, None) + _multiplicity_over_list(multiplicity, where, declared, key)
    if not isinstance(multiplicity, dict) or set(multiplicity) != {"object_attribute"}:
        fail(f"{where}: '{key}' must be either a whole number of at least {least}, {{\"object_attribute\": <object "
              f"attribute>}} or {{\"list\": <object attribute>, \"as\": <attribute name>}}, got {multiplicity!r}")
    if not isinstance(multiplicity["object_attribute"], str) or multiplicity["object_attribute"] not in declared:
        fail(f"{where}: '{key}' object_attribute {multiplicity['object_attribute']!r} is not a declared object, "
              f"global or event attribute")
    return None, multiplicity["object_attribute"], None, None


def _multiplicity_over_list(multiplicity, where, declared, key):
    each, _, whose = MULTIPLICITIES[key]
    if not set(multiplicity) <= {"list", "as"}:
        fail(f"{where}: '{key}' over a list has only 'list' and 'as', got "
             f"{', '.join(map(repr, sorted(multiplicity)))}")
    if not isinstance(multiplicity["list"], str) or multiplicity["list"] not in declared:
        fail(f"{where}: '{key}' list {multiplicity['list']!r} is not a declared object, global or event attribute")
    if "as" not in multiplicity:
        fail(f"{where}: '{key}' over a list needs 'as', the attribute name under which each {each} carries its "
              f"entry")
    if not isinstance(multiplicity["as"], str) or not multiplicity["as"].strip():
        fail(f"{where}: '{key}' as must be a non-empty string")
    if multiplicity["as"] in ("object_id", "index"):
        fail(f"{where}: '{key}' as can't be {multiplicity['as']}, which is reserved: object_id is the {whose} "
              f"object's id and index the {each}'s number")
    return multiplicity["list"], multiplicity["as"]


def read_period(entry, where, key):
    """entry[key], a fan-in's deadline or a quote's within, as a timedelta; None without it."""
    if key not in entry:
        return None
    period = entry[key]
    if isinstance(period, dict):
        check_keys(f"{where}.{key}", period, set(PERIODS))
    if not (isinstance(period, dict) and len(period) == 1 and is_number(next(iter(period.values())))
            and next(iter(period.values())) > 0):
        fail(f"{where}: '{key}' must be {{\"weeks\" | \"days\" | \"hours\": n}}, n a number above 0, got "
             f"{period!r}")
    [(unit, amount)] = period.items()
    return timedelta(**{unit: amount})
