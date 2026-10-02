"""
The first real-engine version of the running example (docs/running-example.md): Sales is a real
Prosimos model, Billing and the two warehouses are scripted fake engines.
"""
from datetime import timedelta

from prosimos.orchestrator import Message, ProcessSpec, SimulationConfig, Verdict, run_orchestrator
from testing_scripts.protocol_scenario import DAY, billing
from testing_scripts.scripted_engine import ScriptedEngine

SALES_BPMN = "testing_scripts/assets/running_example/sales.bpmn"
SALES_JSON = "testing_scripts/assets/running_example/sales.json"
START = DAY.replace(hour=9)
CONSUMER_GROUPS = {
    "Sales": ["Sales"],
    "Billing": ["Billing"],
    "Warehouses": ["TartuWarehouse", "TallinnWarehouse"],
}


def shipping_warehouse(name, cities, shipping_hours):
    """A fake warehouse that claims the orders for its cities and ships each one a fixed time after
    claiming it. (The scripted warehouses of the protocol scenario ship on trucks instead.)"""
    engine = ScriptedEngine(name)

    def order_placed(message, now):
        if message.attributes["city"] not in cities:
            return Verdict.DISCARDED
        order_id = message.attributes["case_id"]
        engine.at(now + timedelta(hours=shipping_hours), lambda _: [Message("Shipment", {"order_id": order_id})])
        return Verdict.CLAIMED

    engine.consume("OrderPlaced", order_placed)
    return engine


def run_with_real_sales(seed, cases, log_path):
    """Runs Sales (real) with Billing and the warehouses (fake); writes the merged log to log_path."""
    fake_engines = {
        "Billing": billing(),
        "TartuWarehouse": shipping_warehouse("TartuWarehouse", {"Tartu", "Tapa"}, 1),
        "TallinnWarehouse": shipping_warehouse("TallinnWarehouse", {"Tallinn", "Tapa"}, 1.5),
    }
    config = SimulationConfig([ProcessSpec("Sales", SALES_BPMN, SALES_JSON, cases)], START, seed, CONSUMER_GROUPS,
                              extra_processes=list(fake_engines))
    return run_orchestrator(config, log_path, extra_engines=fake_engines)
