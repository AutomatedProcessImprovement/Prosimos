"""
The load balancer (docs/orchestrator.md, "Message routing"): a group's policy decides the order its members are
offered a message, at every offer round; the round itself is unchanged. Scripted Sales and warehouses, many orders.
"""
import random
from collections import Counter
from datetime import datetime, timedelta

import pytz

from prosimos.orchestration.orchestrator import orchestrate
from prosimos.orchestration.protocol import Verdict
from testing_scripts.multiprocess.scripted_engine import ScriptedEngine

START = pytz.utc.localize(datetime(2024, 1, 1, 9))
SEED = 1
ORDERS = 1000


def _run(warehouses, policy=None, orders=ORDERS, seed=SEED, ticks=0):
    """Sales publishes orders 0, 1, ... a minute apart to a group of warehouses, each deciding with
    warehouses[name](order, now); the report and every offer, (order, warehouse), in the order made. The first
    warehouse also acts every minute for ticks minutes, which offers it and its group the copies in the pool again."""
    sales = ScriptedEngine("Sales")
    for number in range(orders):
        sales.publish_at(START + timedelta(minutes=number), "Order", number=number)
    offers = []
    engines = {"Sales": sales}
    for name, decide in warehouses.items():
        engine = ScriptedEngine(name)

        def offered(message, now, name=name, decide=decide):
            offers.append((message.attributes["number"], name))
            return decide(message.attributes["number"], now)

        engine.consume("Order", offered)
        engines[name] = engine
    first = engines[next(iter(warehouses))]
    for minute in range(ticks):
        first.at(START + timedelta(minutes=minute, seconds=30), lambda now: [])
    groups = {"Sales": ["Sales"], "Warehouses": list(warehouses)}
    load_balancing = None if policy is None else {"Warehouses": policy}
    return orchestrate(engines, groups, seed, load_balancing), offers


def _always(verdict):
    return lambda order, now: verdict


def _claims(report):
    return Counter(member for _, member, _ in report.claims)


def _first_asked(offers):
    """The warehouse asked first in each round, by order: a round starts where an order is offered again."""
    firsts, previous = [], None
    for order, member in offers:
        if order != previous:
            firsts.append(member)
        previous = order
    return firsts


def test_weighted_8_2_asks_tartu_first_in_about_80_percent_of_the_orders():
    report, _ = _run({"Tartu": _always(Verdict.CLAIMED), "Tallinn": _always(Verdict.CLAIMED)},
                     {"weighted": {"Tartu": 8, "Tallinn": 2}})

    # 1,000 orders: the standard deviation of the share is 0.013, so 0.04 is three of them
    assert abs(_claims(report)["Tartu"] / ORDERS - 0.8) < 0.04
    assert sum(_claims(report).values()) == ORDERS


def test_the_same_seed_gives_the_same_split_and_another_seed_another():
    warehouses = {"Tartu": _always(Verdict.CLAIMED), "Tallinn": _always(Verdict.CLAIMED)}
    policy = {"weighted": {"Tartu": 8, "Tallinn": 2}}

    first, _ = _run(warehouses, policy)
    again, _ = _run(warehouses, policy)
    other, _ = _run(warehouses, policy, seed=2)

    assert first.claims == again.claims
    assert first.claims != other.claims


def test_a_weight_is_a_preference_not_a_quota():
    # Tartu can't take every other order yet: Tallinn, asked next, takes it
    def tartu(order, now):
        return Verdict.PENDING if order % 2 else Verdict.CLAIMED

    report, offers = _run({"Tartu": tartu, "Tallinn": _always(Verdict.CLAIMED)},
                          {"weighted": {"Tartu": 8, "Tallinn": 2}})

    assert abs(_first_asked(offers).count("Tartu") / ORDERS - 0.8) < 0.04  # still asked first as often
    assert abs(_claims(report)["Tartu"] / ORDERS - 0.4) < 0.04  # but handles half of that
    assert report.unclaimed == []


def test_preferred_is_a_fallback_chain_through_discards():
    # order kinds by number: Tartu takes kind 0, Riga kinds 0 and 1, Tallinn every kind
    report, offers = _run({"Tartu": lambda order, now: Verdict.CLAIMED if order % 3 == 0 else Verdict.DISCARDED,
                           "Riga": lambda order, now: Verdict.CLAIMED if order % 3 < 2 else Verdict.DISCARDED,
                           "Tallinn": _always(Verdict.CLAIMED)},
                          {"preferred": ["Tartu", "Riga", "Tallinn"]}, orders=30)

    number = {message.id: message.attributes["number"] for message in report.published}
    assert [(number[message_id], member) for message_id, member, _ in report.claims] == [
        (order, ["Tartu", "Riga", "Tallinn"][order % 3]) for order in range(30)]
    assert offers == [(order, member) for order in range(30)
                      for member in ["Tartu", "Riga", "Tallinn"][:order % 3 + 1]]


def test_a_copy_back_from_the_pool_is_ordered_again_weighted_by_a_new_draw():
    # one order nobody can take yet: each minute Tartu acts and the copy is offered again
    report, offers = _run({"Tartu": _always(Verdict.PENDING), "Tallinn": _always(Verdict.PENDING)},
                          {"weighted": {"Tartu": 8, "Tallinn": 2}}, orders=1, ticks=500)

    rounds = [offers[start:start + 2] for start in range(0, len(offers), 2)]
    assert len(rounds) == 501 and all(sorted(member for _, member in round_) == ["Tallinn", "Tartu"]
                                      for round_ in rounds)
    firsts = [round_[0][1] for round_ in rounds]
    assert abs(firsts.count("Tartu") / len(firsts) - 0.8) < 0.06
    assert [message.attributes["number"] for _, message in report.unclaimed] == [0]


def test_a_copy_back_from_the_pool_is_ordered_again_preferred_by_the_same_ranking():
    report, offers = _run({"Tallinn": _always(Verdict.PENDING), "Riga": _always(Verdict.PENDING),
                           "Tartu": _always(Verdict.PENDING)},
                          {"preferred": ["Tartu", "Riga", "Tallinn"]}, orders=1, ticks=20)

    assert offers == [(0, member) for _ in range(21) for member in ["Tartu", "Riga", "Tallinn"]]


def test_without_load_balancing_the_order_is_todays_shuffle():
    members = ["Tartu", "Riga", "Tallinn"]
    report, offers = _run({name: _always(Verdict.CLAIMED) for name in members}, orders=200)

    # one shuffle of the members, in the group's order, per offer round, from the orchestrator's generator
    rng = random.Random(f"orchestrator-{SEED}")
    expected = []
    for _ in range(200):
        order = list(members)
        rng.shuffle(order)
        expected.append(order[0])
    assert [member for _, member, _ in report.claims] == expected
