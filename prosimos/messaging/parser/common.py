"""What the parser's files share with each other and with other code: failing, keys and number checks."""
import numbers

from prosimos.exceptions import InvalidSimScenarioException

PERIODS = ("weeks", "days", "hours")  # of a deadline, and of a catalogue change


def is_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def is_whole_number(value, minimum):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value >= minimum and float(value).is_integer()


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
