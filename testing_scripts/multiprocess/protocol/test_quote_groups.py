"""
A quote type reaches exactly one group (docs/orchestrator.md): the orchestrator checks the first quote of each
type against the groups whose members subscribe to it, and stops the run if there are several.
"""
from datetime import datetime, timedelta

import pytest
import pytz

from prosimos.orchestration.orchestrator import orchestrate
from prosimos.orchestration.protocol import Line, Message, Quote, QuoteKind
from testing_scripts.multiprocess.scripted_engine import ScriptedEngine

START = pytz.utc.localize(datetime(2024, 1, 1, 9))


def _engines(*suppliers):
    """Sales asks a quote for an iPhone at 09:00; each supplier answers ItemQuote from a script."""
    sales = ScriptedEngine("Sales")
    sales.at(START, lambda now: [Message("ItemQuote", {"object_id": "order-7"}, quote=Quote(
        QuoteKind.QUOTE, "ItemQuote", "order-7", [Line({"product": "iPhone"})], timedelta(days=3), "QuoteAccepted",
        "QuoteDeclined"))])
    engines = {"Sales": sales}
    for name in suppliers:
        engines[name] = ScriptedEngine(name)
        engines[name].slots("ItemQuote", lambda line, now: [now])
    return engines


def test_a_quote_type_subscribed_in_two_groups_stops_the_run():
    groups = {"Sales": ["Sales"], "Warehouses": ["Tartu", "Tallinn"], "Partners": ["Riga"]}

    with pytest.raises(ValueError, match=r"^ItemQuote m1 from Sales asks a quote of type ItemQuote, which the groups "
                                         r"\['Partners', 'Warehouses'\] subscribe to: a quote type must reach exactly "
                                         r"one group$"):
        orchestrate(_engines("Tartu", "Tallinn", "Riga"), groups, 1)


def test_without_consumer_groups_two_suppliers_are_two_groups():
    with pytest.raises(ValueError, match=r"which the groups \['Riga', 'Tartu'\] subscribe to"):
        orchestrate(_engines("Tartu", "Riga"), None, 1)


def test_a_quote_type_subscribed_in_one_group_runs():
    report = orchestrate(_engines("Tartu", "Tallinn"), {"Sales": ["Sales"], "Warehouses": ["Tartu", "Tallinn"]}, 1)

    assert [(message.type, message.source) for message in report.published] == [("ItemQuote", "Sales"),
                                                                                  ("QuoteAccepted", "orchestrator")]
