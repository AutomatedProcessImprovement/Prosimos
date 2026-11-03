"""
Fan-in settings (docs/messaging.md, "Fan-in"): a publish entry's fan_in makes one message of the parts of
several cases with the same values of 'same'. Loading and load errors only.
"""
import pytest

from prosimos.exceptions import InvalidSimScenarioException
from prosimos.messaging.model import PublishPoint
from prosimos.messaging.parser import parse_messages
from testing_scripts.multiprocess.messaging.test_messaging_parser import (  # noqa: F401 (bpmn is a fixture)
    bpmn, consume, publish)

DECLARED = {"order_id", "items", "weight"}
ORDER_PICKED = {"fan_in": {"object_attribute": "items"}, "same": ["order_id"],
                "attributes": ["order_id", {"item_ids": "object_id"}, {"total_weight": {"sum": "weight"}}]}


def _parsed(bpmn, **entry):
    return parse_messages(publish(**entry), bpmn, DECLARED).publish[0]


def _rejected(bpmn, message, section=None, **entry):
    with pytest.raises(InvalidSimScenarioException, match=message):
        parse_messages(section or publish(**entry), bpmn, DECLARED)


def test_the_items_of_an_order_make_one_order_picked(bpmn):
    assert _parsed(bpmn, **ORDER_PICKED) == PublishPoint(
        "Throw_OrderPlaced", "OrderPlaced", ("order_id",), fan_in_attribute="items", same=("order_id",),
        gathered=(("item_ids", "object_id"),), computed=(("total_weight", "sum", "weight", None),))


def test_a_fixed_size_and_a_function_with_initial(bpmn):
    point = _parsed(bpmn, **{**ORDER_PICKED, "fan_in": 3,
                             "attributes": [{"lines": {"count": "weight"}, "heaviest": {"max": "weight"}},
                                            {"total_weight": {"sum": "weight", "initial": 0.5}}]})

    assert (point.fan_in, point.fan_in_attribute, point.fans_in, point.attributes) == (3, None, True, ())
    assert point.computed == (("lines", "count", "weight", None), ("heaviest", "max", "weight", None),
                              ("total_weight", "sum", "weight", 0.5))


def test_an_entry_without_fan_in_doesnt_fan_in(bpmn):
    assert not _parsed(bpmn, attributes=["order_id"]).fans_in


@pytest.mark.parametrize("fan_in, message", [
    (0, r"'fan_in' must be either a whole number of at least 1 or \{\"object_attribute\": <object attribute>\}, "
        r"got 0"),
    (1.5, r"'fan_in' must be either a whole number of at least 1 .*, got 1.5"),
    ("3", r"'fan_in' must be either a whole number of at least 1 .*, got '3'"),
    ({}, r"'fan_in' must be either a whole number of at least 1 .*, got \{\}"),
    ({"object_attribute": "lines"}, r"'fan_in' object_attribute 'lines' is not a declared object, global or event "
                                    r"attribute"),
    ({"object_atribute": "items"}, r"messages.publish\[0\].fan_in: unknown key 'object_atribute'; allowed: "
                                   r"object_attribute$"),
])
def test_an_invalid_fan_in_is_rejected(bpmn, fan_in, message):
    _rejected(bpmn, message, **{**ORDER_PICKED, "fan_in": fan_in})


def test_without_same_any_parts_make_a_group(bpmn):
    point = _parsed(bpmn, fan_in=5, attributes=[{"parcel_ids": "object_id"}, {"total_weight": {"sum": "weight"}}])

    assert (point.fan_in, point.same, point.attributes) == (5, (), ())
    assert point.gathered == (("parcel_ids", "object_id"),)
    assert point.computed == (("total_weight", "sum", "weight", None),)


def test_without_same_a_plain_name_is_rejected(bpmn):
    # without same, the members share no attribute that could be sent once
    _rejected(bpmn, r"messages.publish\[0\]: 'order_id' is sent by its plain name, but without 'same' the members of "
                    r"a group may differ in every attribute: send it as a list, \{<name>: \"order_id\"\}, or a "
                    r"function over the members", fan_in=5, attributes=["order_id"])


@pytest.mark.parametrize("same, message", [
    (["customer"], r"'same' names 'customer', not declared as an object, global or event attribute"),
    ([], r"'same' must be a non-empty list of object attribute names, got \[\]"),
    (["order_id", "order_id"], r"'same' lists order_id more than once"),
])
def test_an_invalid_same_is_rejected(bpmn, same, message):
    _rejected(bpmn, message, **{**ORDER_PICKED, "same": same})


def test_a_plain_name_not_in_same_is_rejected(bpmn):
    # each member has its own weight, so the message must say how to send them all
    _rejected(bpmn, r"messages.publish\[0\]: 'weight' isn't in 'same', so the members of a group may differ in it: "
                    r"send it as a list, \{<name>: \"weight\"\}, or a function over the members, "
                    r"\{<name>: \{\"sum\": \"weight\"\}\}", **{**ORDER_PICKED, "attributes": ["order_id", "weight"]})


@pytest.mark.parametrize("attributes", [["order_id", "index"], [{"numbers": "index"}]])
def test_index_is_rejected(bpmn, attributes):
    _rejected(bpmn, r"'index' numbers the messages of a fan_out, but a fan-in entry sends one message per group",
              **{**ORDER_PICKED, "attributes": attributes})


def test_object_id_by_its_name_is_rejected(bpmn):
    _rejected(bpmn, r"a fan-in message can't carry object_id: it has several publishers. Send their ids as a list, "
                    r"e.g. \{\"item_ids\": \"object_id\"\}", **{**ORDER_PICKED, "attributes": ["object_id"]})


def test_fan_in_together_with_fan_out_is_rejected(bpmn):
    _rejected(bpmn, r"messages.publish\[0\]: fan_in and fan_out can't be combined on one entry",
              **{**ORDER_PICKED, "fan_out": 2})


@pytest.mark.parametrize("attributes, message", [
    ([{"item_ids": "item_id"}], r"'item_id' not declared as an object, global or event attribute"),
    ([{"total_weight": {"avg": "weight"}}], r"messages.publish\[0\].attributes.total_weight: unknown key 'avg'; "
                                            r"allowed: count, initial, max, min, sum$"),
    ([{"total_weight": {"sum": "weight", "max": "weight"}}], r"attribute 'total_weight' must be one function and an "
                                                             r"object attribute"),
    ([{"total_weight": {"sum": "weight", "initial": "5"}}], r"attribute 'total_weight' initial must be a number, "
                                                            r"got '5'"),
    ([{"total_weight": 3}], r"attribute 'total_weight' must be an object attribute, whose values go into a list, "
                            r"or \{<function>: <object attribute>\}, got 3"),
    (["order_id", {"order_id": "object_id"}], r"'attributes' sends 'order_id' more than once"),
    ([3], r"'attributes' of a fan-in entry must list attribute names"),
])
def test_invalid_attributes_are_rejected(bpmn, attributes, message):
    _rejected(bpmn, message, **{**ORDER_PICKED, "attributes": attributes})


def test_same_without_fan_in_is_rejected(bpmn):
    _rejected(bpmn, r"messages.publish\[0\]: 'same' on a publish entry needs a fan_in", same=["order_id"])


def test_a_list_or_a_function_without_fan_in_is_rejected(bpmn):
    _rejected(bpmn, r"messages.publish\[0\]: a list or a function in 'attributes' needs a fan_in",
              attributes=[{"item_ids": "object_id"}])


def test_fan_in_on_a_consume_entry_is_rejected(bpmn):
    _rejected(bpmn, r"messages.consume\[0\]: fan_in is only for publish entries, where several cases make one "
                    r"message; to take several messages at a catch event, use collect", section=consume(fan_in=3))
