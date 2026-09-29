"""The running example (docs/running-example.md) built from scripted engines."""
from datetime import datetime, timedelta

import pytz

from prosimos.orchestrator import Message, Verdict, run_engines
from testing_scripts.scripted_engine import ScriptedEngine

DAY = pytz.utc.localize(datetime(2024, 1, 1))
TRUCK_CAPACITY = 2

ORDERS = [  # (time, order_id, city, when Sales starts waiting for the shipment)
    ("09:00", "ord1", "Tartu", "12:00"),
    ("09:10", "ord2", "Tallinn", None),  # canceled at 10:00
    ("09:20", "ord3", "Tapa", None),  # never waits
    ("09:30", "ord4", "Pärnu", "09:30"),
    ("09:40", "ord5", "Tartu", "09:40"),
]
TRUCKS = [  # (time, truck_id, dock)
    ("09:45", "T0", "Narva"),
    ("10:30", "T1", "Tartu"),
    ("12:00", "T2", "Tallinn"),
    ("13:00", "T3", "Tartu"),
    ("16:00", "T4", "Tartu"),
]
CONSUMER_GROUPS = {
    "Sales": ["Sales"],
    "Billing": ["Billing"],
    "Carrier": ["Carrier"],
    "Warehouses": ["TartuWarehouse", "TallinnWarehouse"],
}


def at(hh_mm):
    hours, minutes = map(int, hh_mm.split(":"))
    return DAY.replace(hour=hours, minute=minutes)


def sales():
    engine = ScriptedEngine("Sales")
    engine.cases = {}  # order_id -> "open", "waiting", "closed" or "canceled"
    engine.closed_at = {}

    def place(order_id, city, waits_now):
        def action(now):
            engine.cases[order_id] = "waiting" if waits_now else "open"
            return [Message("OrderPlaced", {"order_id": order_id, "city": city})]
        return action

    def set_state(order_id, state):
        def action(now):
            engine.cases[order_id] = state
        return action

    for time, order_id, city, waits_from in ORDERS:
        engine.at(at(time), place(order_id, city, waits_from == time))
        if waits_from is not None and waits_from != time:
            engine.at(at(waits_from), set_state(order_id, "waiting"))
    engine.publish_at(at("09:05"), "Newsletter")
    engine.at(at("10:00"), set_state("ord2", "cancelled"))

    def shipment(message, now):
        order_id = message.attributes["order_id"]
        state = engine.cases.get(order_id)
        if state == "waiting":
            engine.cases[order_id] = "closed"
            engine.closed_at[order_id] = now
            return Verdict.CLAIMED
        if state == "open":
            return Verdict.PENDING
        return Verdict.DISCARDED  # canceled, already closed, or not an order of ours

    engine.consume("Shipment", shipment)
    engine.waiting = lambda: sorted(order for order, state in engine.cases.items() if state == "waiting")
    return engine


def billing():
    engine = ScriptedEngine("Billing")
    engine.consume("OrderPlaced", lambda message, now: Verdict.CLAIMED)
    return engine


def warehouse(name, cities, dock, packing_hours):
    engine = ScriptedEngine(name)
    engine.packed = []  # orders packed and waiting for a truck, oldest first
    engine.loads = {}  # truck_id -> orders it took

    def order_placed(message, now):
        if message.attributes["city"] not in cities:
            return Verdict.DISCARDED
        order_id = message.attributes["order_id"]
        engine.at(now + timedelta(hours=packing_hours), lambda _: engine.packed.append(order_id))
        return Verdict.CLAIMED

    def truck(message, now):
        if message.attributes["dock"] != dock:
            return Verdict.DISCARDED
        if not engine.packed:
            return Verdict.PENDING
        load = engine.packed[:message.attributes["capacity"]]
        del engine.packed[:len(load)]
        engine.loads[message.attributes["truck_id"]] = load
        engine.at(now, lambda _: [Message("Shipment", {"order_id": order_id}) for order_id in load])
        return Verdict.CLAIMED

    engine.consume("OrderPlaced", order_placed)
    engine.consume("Truck", truck)
    return engine


def carrier():
    engine = ScriptedEngine("Carrier")
    for time, truck_id, dock in TRUCKS:
        engine.publish_at(at(time), "Truck", truck_id=truck_id, dock=dock, capacity=TRUCK_CAPACITY)
    return engine


def build_engines():
    engines = [
        sales(),
        billing(),
        warehouse("TartuWarehouse", {"Tartu", "Tapa"}, "Tartu", 1),
        warehouse("TallinnWarehouse", {"Tallinn", "Tapa"}, "Tallinn", 1.5),
        carrier(),
    ]
    return {engine.name: engine for engine in engines}


def run_scenario(seed):
    engines = build_engines()
    report = run_engines(engines, CONSUMER_GROUPS, seed)
    return engines, report
