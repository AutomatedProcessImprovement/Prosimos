"""The load errors of a publish entry's quote (docs/messaging.md), one test each."""
import pytest

from prosimos.exceptions import InvalidSimScenarioException
from prosimos.messaging.parser.reading import parse_messages
from testing_scripts.multiprocess.messaging.test_messaging_parser import bpmn  # noqa: F401 (a fixture)
from testing_scripts.multiprocess.messaging.test_quote_settings import DECLARED, section

QUOTE = {"quantity": {"list": "products", "as": "product"}, "within": {"days": 3}, "accepted": "QuoteAccepted",
         "declined": "QuoteDeclined", "cancel": "QuoteCanceled", "modify": "QuoteModified"}
WHERE = r"^Invalid 'messages' section: messages\.publish\[0\]"


def _rejected(bpmn, message, messages=None, **quote):
    with pytest.raises(InvalidSimScenarioException, match=WHERE + message):
        parse_messages(messages or section(**quote), bpmn, DECLARED)


def test_an_unknown_key(bpmn):
    _rejected(bpmn, r"\.quote: unknown key 'withn'; allowed: accepted, cancel, declined, modify, quantity, within$",
              **QUOTE, withn={"days": 3})


@pytest.mark.parametrize("missing", ["quantity", "accepted", "declined", "cancel"])
def test_a_missing_key(bpmn, missing):
    _rejected(bpmn, rf": 'quote' needs quantity, accepted, declined, and cancel; it has no {missing}$",
              **{key: value for key, value in QUOTE.items() if key != missing})


def test_a_quote_that_isnt_an_object(bpmn):
    messages = section()
    messages["publish"][0]["quote"] = "ItemQuote"
    _rejected(bpmn, r": 'quote' must be an object with quantity, accepted, declined, and cancel, got 'ItemQuote'$",
              messages)


def test_a_type_that_isnt_a_name(bpmn):
    _rejected(bpmn, r": the quote's declined must be a message type, a non-empty string, got ''$",
              **{**QUOTE, "declined": ""})


@pytest.mark.parametrize("change, repeated", [({"declined": "QuoteAccepted"}, "QuoteAccepted"),
                                              ({"accepted": "ItemQuote"}, "ItemQuote")])
def test_the_same_type_twice(bpmn, change, repeated):
    _rejected(bpmn, rf": '{repeated}' is used more than once among the entry's type and the quote's accepted, "
                    rf"declined, cancel, and modify: each needs a type of its own$", **{**QUOTE, **change})


@pytest.mark.parametrize("key, name", [("cancel", "QuoteCanceled"), ("modify", "QuoteModified")])
def test_a_cancel_or_modify_type_without_its_own_publish_entry(bpmn, key, name):
    messages = section(**QUOTE)
    messages["publish"] = [entry for entry in messages["publish"] if entry["type"] != name]
    _rejected(bpmn, rf": the quote's {key} type '{name}' has no publish entry of its own in this process: the case "
                    rf"throws it at its own event$", messages)


@pytest.mark.parametrize("quantity, message", [
    (0, r": 'quantity' must be either a whole number of at least 1, \{\"object_attribute\": <object attribute>\} or "
        r"\{\"list\": <object attribute>, \"as\": <attribute name>\}, got 0$"),
    ("2", r": 'quantity' must be either a whole number of at least 1, .*, got '2'$"),
    ({"object_attribute": "lines"}, r": 'quantity' object_attribute 'lines' is not a declared object, global or "
                                    r"event attribute$"),
    ({"list": "products"}, r": 'quantity' over a list needs 'as', the attribute name under which each line carries "
                           r"its entry$"),
    ({"list": "products", "as": "index"}, r": 'quantity' as can't be index, which is reserved: object_id is the "
                                          r"requesting object's id and index the line's number$"),
    ({"items": 2}, r"\.quantity: unknown key 'items'; allowed: as, list, object_attribute$"),
])
def test_a_bad_quantity(bpmn, quantity, message):
    _rejected(bpmn, r"\.quote" + message, **{**QUOTE, "quantity": quantity})


@pytest.mark.parametrize("within, message", [
    ({"days": 0}, r": 'within' must be \{\"weeks\" \| \"days\" \| \"hours\": n\}, n a number above 0, got "
                  r"\{'days': 0\}$"),
    ({"minutes": 30}, r"\.within: unknown key 'minutes'; allowed: days, hours, weeks$"),
    ({"minutes": {"object_attribute": "patience"}}, r"\.within: unknown key 'minutes'; allowed: days, hours, weeks$"),
    ({"days": {"object_attribute": "patience"}, "hours": {"object_attribute": "patience"}},
     r": 'within' has one unit, weeks, days, or hours, got \{'days': \{'object_attribute': 'patience'\}, "
     r"'hours': \{'object_attribute': 'patience'\}\}$"),
    ({"days": {"object_attribute": "mood"}}, r": 'within' days object_attribute 'mood' is not a declared object, "
                                             r"global or event attribute$"),
    ({"days": {"attribute": "patience"}}, r"\.within\.days: unknown key 'attribute'; allowed: object_attribute$"),
])
def test_a_bad_within(bpmn, within, message):
    _rejected(bpmn, r"\.quote" + message, **{**QUOTE, "within": within})


@pytest.mark.parametrize("key, value", [("fan_out", 2), ("fan_in", 3)])
def test_a_quote_with_fan_out_or_fan_in(bpmn, key, value):
    messages = section(**QUOTE)
    messages["publish"][0][key] = value
    _rejected(bpmn, rf": quote and {key} can't be combined on one entry: the quote's lines say how many items it "
                    rf"asks for, and its replies come from the orchestrator$", messages)


def test_fan_out_still_allows_0(bpmn):
    messages = {"publish": [{"event_id": "Throw_OrderPlaced", "type": "ItemOrdered", "fan_out": 0}]}

    assert parse_messages(messages, bpmn, DECLARED).publish[0].fan_out == 0
