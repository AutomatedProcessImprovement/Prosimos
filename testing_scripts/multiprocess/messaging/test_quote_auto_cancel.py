"""
The automatic cancel (docs/messaging.md): a case that ends with an open quote publishes the quote's cancel at its
end, as if the model had thrown it, so the lines it reserved are freed.
"""
from datetime import datetime, timedelta

import pytest
import pytz

from prosimos.orchestration.config import Organization, ProcessSpec, SimulationConfig
from prosimos.orchestration.protocol import QuoteKind
from prosimos.orchestration.simulation import run_multiprocess_simulation
from testing_scripts.multiprocess.scripted_engine import ScriptedEngine

START = pytz.utc.localize(datetime(2024, 1, 1, 9))
MINUTE, HOUR = timedelta(minutes=1), timedelta(hours=1)
ASSETS = "testing_scripts/assets/quotes"
# sales_auto_cancel: Sales places the order (10 minutes) and quotes an iPhone; then it waits for the shipment, and
# after it cancels the quote itself, or gives up after 2 hours and ends with the quote still open


def _run(tmp_path, shipped_at=None, terminate=False):
    """One order against a scripted warehouse that has every line 3 hours after it is asked, and ships the order
    at shipped_at, if given; with terminate, giving up ends at a terminate end event. The report and the warehouse."""
    bpmn = f"{ASSETS}/sales_auto_cancel.bpmn"
    if terminate:
        with open(bpmn) as file:
            model = file.read()
        gave_up = '<bpmn:endEvent id="End_Gave_Up"><bpmn:incoming>F7</bpmn:incoming>'
        bpmn = tmp_path / "sales.bpmn"
        bpmn.write_text(model.replace(f"{gave_up}</bpmn:endEvent>",
                                      f"{gave_up}<bpmn:terminateEventDefinition/></bpmn:endEvent>"))
    sales = ProcessSpec("Sales", str(bpmn), f"{ASSETS}/sales_auto_cancel.json", 1, "order")
    warehouse = ScriptedEngine("Warehouse")
    warehouse.slots("ItemQuote", lambda line, now: [now + 3 * HOUR])
    if shipped_at is not None:
        warehouse.publish_at(shipped_at, "Shipped", order_id="order-0")
    config = SimulationConfig([Organization("Shop", [sales])], START, 1, extra_processes=["Warehouse"])
    return run_multiprocess_simulation(config, None, {"Warehouse": warehouse}), warehouse


def _cancels(report):
    return [(message.time, message.source, message.quote.kind) for message in report.published
            if message.type == "QuoteCanceled"]


@pytest.mark.parametrize("terminate", [False, True])
def test_an_order_ended_by_a_timer_cancels_its_quote_and_frees_its_lines(tmp_path, terminate):
    # the quote at 09:10 is accepted; the order gives up at 11:10 and ends with it open, at a plain or terminate end
    report, warehouse = _run(tmp_path, terminate=terminate)

    # after a race timer wins, its branch goes on a microsecond later: the order ends then
    gave_up = START + 2 * HOUR + 10 * MINUTE + timedelta(microseconds=1)
    assert _cancels(report) == [(gave_up, "Sales", QuoteKind.CANCEL)]
    assert [(message.id, line_ids, now) for message, line_ids, now in warehouse.released] == [("m1", ["l1"], gave_up)]


def test_an_order_that_canceled_its_quote_itself_isnt_canceled_again(tmp_path):
    # shipped at 10:00, the order throws its own cancel and ends
    report, warehouse = _run(tmp_path, shipped_at=START + HOUR)

    assert _cancels(report) == [(START + HOUR, "Sales", QuoteKind.CANCEL)]
    assert len(warehouse.released) == 1
