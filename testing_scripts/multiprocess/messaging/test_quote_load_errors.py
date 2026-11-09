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
    _rejected(bpmn, rf": the quote's {key} type '{name}' has no publish entry of its own in this process: the "
                    rf"requester throws it at its own event$", messages)


@pytest.mark.parametrize("quantity, message", [
    (0, r": 'quantity' must be either a whole number of at least 1, \{\"object_attribute\": <object attribute>\} or "
        r"\{\"list\": <object attribute>, \"as\": <attribute name>\}, got 0$"),
    ("2", r": 'quantity' must be either a whole number of at least 1, .*, got '2'$"),
    ({"object_attribute": "lines"}, r": 'quantity' object_attribute 'lines' is not a declared object, global or "
                                    r"event attribute$"),
    ({"list": "products"}, r": 'quantity' over a list needs 'as', the attribute name under which each line carries "
                           r"its entry$"),
    ({"list": "products", "as": "index"}, r": 'quantity' as can't be index, which is reserved: object_id is the "
                                          r"requesting object's id, and index isn't available on quotes$"),
    ({"list": "products", "as": "object_id"}, r": 'quantity' as can't be object_id, which is reserved: object_id is "
                                              r"the requesting object's id, and index isn't available on quotes$"),
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


def test_a_quote_entry_listing_index(bpmn):
    messages = section(**QUOTE)
    messages["publish"][0]["attributes"] = ["object_id", "index"]
    _rejected(bpmn, r": a quote's lines can't carry index: modify matches lines by their attributes$", messages)


@pytest.mark.parametrize("key, value", [("fan_out", 2), ("fan_in", 3)])
def test_a_quote_with_fan_out_or_fan_in(bpmn, key, value):
    messages = section(**QUOTE)
    messages["publish"][0][key] = value
    _rejected(bpmn, rf": quote and {key} can't be combined on one entry: the quote's lines say how many items it "
                    rf"asks for, and its replies come from the orchestrator$", messages)


def test_fan_out_still_allows_0(bpmn):
    messages = {"publish": [{"event_id": "Throw_OrderPlaced", "type": "ItemOrdered", "fan_out": 0}]}

    assert parse_messages(messages, bpmn, DECLARED).publish[0].fan_out == 0


# the replies: a catch or start event taking a quote's accepted or declined type needs a condition on request

NAMED = {"Throw_Quote": ("intermediateThrowEvent", "Ask quote"), "Throw_Cancel": ("intermediateThrowEvent", None),
         "Catch_Accepted": ("intermediateCatchEvent", "Quote accepted"),
         "Catch_Declined": ("intermediateCatchEvent", None)}
BY_REQUEST = [[{"attribute": "request", "comparison": "=", "object_attribute": "object_id"}]]


@pytest.fixture
def named(tmp_path):
    """Sales with a quote's events, all after one task; only some of them have a name."""
    elements = "".join(f'<bpmn:{kind} id="{element}"{f" name={chr(34)}{name}{chr(34)}" if name else ""}>'
                       f'<bpmn:messageEventDefinition/></bpmn:{kind}>' for element, (kind, name) in NAMED.items())
    flows = "".join(f'<bpmn:sequenceFlow id="Flow_{index}" sourceRef="Ask" targetRef="{element}"/>'
                    for index, element in enumerate(NAMED))
    path = tmp_path / "sales.bpmn"
    path.write_text('<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">'
                    f'<bpmn:process id="Sales"><bpmn:task id="Ask"/>{elements}{flows}</bpmn:process>'
                    '</bpmn:definitions>')
    return path


def _replies(*consume):
    """The quote at Throw_Quote, without modify, its cancel at Throw_Cancel, and the consume entries."""
    quote = {**{key: value for key, value in QUOTE.items() if key != "modify"}, "quantity": 1}
    return {"publish": [{"event_id": "Throw_Quote", "type": "ItemQuote", "attributes": ["object_id"], "quote": quote},
                        {"event_id": "Throw_Cancel", "type": "QuoteCanceled"}],
            "consume": list(consume)}


def _without_request(bpmn, index, reply, messages):
    message = rf"^Invalid 'messages' section: messages\.consume\[{index}\]: {reply} has no condition on request: it " \
              rf"would take the replies to other objects' quotes$"
    with pytest.raises(InvalidSimScenarioException, match=message):
        parse_messages(messages, bpmn, DECLARED)


def _reply(event_id, type, condition=None):
    return {"event_id": event_id, "type": type, **({} if condition is None else {"condition": condition})}


def test_the_replies_with_a_condition_on_request_load(named):
    model = parse_messages(_replies(_reply("Catch_Accepted", "QuoteAccepted", BY_REQUEST),
                                    _reply("Catch_Declined", "QuoteDeclined", BY_REQUEST)), named, DECLARED)

    assert [point.type for point in model.consume] == ["QuoteAccepted", "QuoteDeclined"]


def test_an_accepted_reply_without_a_condition(named):
    _without_request(named, 0, "QuoteAccepted at 'Quote accepted'",
                     _replies(_reply("Catch_Accepted", "QuoteAccepted"),
                              _reply("Catch_Declined", "QuoteDeclined", BY_REQUEST)))


def test_a_declined_reply_with_a_condition_on_another_attribute(named):
    # the event has no name, so the error names its id
    _without_request(named, 1, "QuoteDeclined at 'Catch_Declined'",
                     _replies(_reply("Catch_Accepted", "QuoteAccepted", BY_REQUEST),
                              _reply("Catch_Declined", "QuoteDeclined",
                                     [[{"attribute": "source", "comparison": "=", "value": "orchestrator"}]])))


def test_a_reply_with_an_alternative_without_request(named):
    either = [*BY_REQUEST, [{"attribute": "source", "comparison": "=", "value": "orchestrator"}]]

    _without_request(named, 0, "QuoteAccepted at 'Quote accepted'", _replies(_reply("Catch_Accepted", "QuoteAccepted",
                                                                                     either)))


def test_a_message_start_event_taking_a_reply_without_a_condition(bpmn):
    # the rule covers start events too; the model's start event has no name, so the error names its id
    messages = section(**QUOTE)
    messages["consume"] = [_reply("Start_Message", "QuoteAccepted")]

    _without_request(bpmn, 0, "QuoteAccepted at 'Start_Message'", messages)


def test_another_process_may_take_the_types_without_a_condition(named):
    # only the process that asks the quote takes its replies by request
    messages = {"consume": [_reply("Catch_Accepted", "QuoteAccepted")]}

    assert parse_messages(messages, named, DECLARED).consume[0].condition is None


@pytest.mark.parametrize("where", ["attributes", "as"])
def test_portion_size_is_reserved(bpmn, where):
    messages = section(**{**QUOTE, "quantity": {"list": "products", "as": "portion_size"}} if where == "as" else QUOTE)
    if where == "attributes":
        messages["publish"][0]["attributes"] = ["object_id", "portion_size"]
    with pytest.raises(InvalidSimScenarioException, match=WHERE + r": a quote's lines can't carry portion_size, which "
                       r"is reserved: an item claims its line with the number of items in its supplier's portion under "
                       r"that name$"):
        parse_messages(messages, bpmn, DECLARED | {"portion_size"})
