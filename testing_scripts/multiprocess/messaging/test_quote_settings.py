"""
A publish entry's quote (docs/messaging.md): the lines it asks for (quantity), by when (within) and the types of its
replies, modify and cancel. Loading only; nothing uses it yet.
"""
from datetime import timedelta

import pytest

from prosimos.messaging.model import QuoteSettings
from prosimos.messaging.parser.reading import parse_messages
from testing_scripts.multiprocess.messaging.test_messaging_parser import bpmn, publish  # noqa: F401 (a fixture)

DECLARED = {"products", "items", "patience"}
TYPES = {"accepted": "QuoteAccepted", "declined": "QuoteDeclined", "cancel": "QuoteCanceled"}


def section(**quote):
    """A quote entry at Throw_OrderPlaced, with the publish entries of its cancel and modify types."""
    return {"publish": [{"event_id": "Throw_OrderPlaced", "type": "ItemQuote", "attributes": ["object_id"],
                         "quote": quote},
                        {"event_id": "End_Closed", "type": "QuoteCanceled"},
                        {"event_id": "End_Merged", "type": "QuoteModified"}]}


def _quote(bpmn, **quote):
    return parse_messages(section(**quote), bpmn, DECLARED).publish[0].quote


def test_the_sheets_entry_loads(bpmn):
    assert _quote(bpmn, quantity={"list": "products", "as": "product"}, within={"days": 3}, modify="QuoteModified",
                  **TYPES) == QuoteSettings("QuoteAccepted", "QuoteDeclined", "QuoteCanceled", "QuoteModified",
                                            quantity_list="products", quantity_as="product",
                                            within=timedelta(days=3))


@pytest.mark.parametrize("quantity, read", [
    (2, (2, None, None, None)),
    ({"object_attribute": "items"}, (None, "items", None, None)),
    ({"list": "products", "as": "product"}, (None, None, "products", "product")),
])
def test_quantity_takes_the_three_forms(bpmn, quantity, read):
    quote = _quote(bpmn, quantity=quantity, **TYPES)

    assert (quote.quantity, quote.quantity_attribute, quote.quantity_list, quote.quantity_as) == read


@pytest.mark.parametrize("within, read", [
    ({"hours": 12}, (timedelta(hours=12), None, None)),
    ({"weeks": 1}, (timedelta(weeks=1), None, None)),
    ({"days": {"object_attribute": "patience"}}, (None, "patience", "days")),
    ({"hours": {"object_attribute": "patience"}}, (None, "patience", "hours")),
    (None, (None, None, None)),  # no limit: a backorder
])
def test_within_is_a_period_with_a_number_or_an_object_attribute_or_left_out(bpmn, within, read):
    quote = _quote(bpmn, quantity=1, **TYPES, **({} if within is None else {"within": within}))

    assert (quote.within, quote.within_attribute, quote.within_unit) == read


def test_modify_is_optional(bpmn):
    assert _quote(bpmn, quantity=1, **TYPES).modify is None


def test_an_entry_without_quote_has_none(bpmn):
    assert parse_messages(publish(), bpmn, DECLARED).publish[0].quote is None
