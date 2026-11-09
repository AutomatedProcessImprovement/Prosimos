"""
Broken promises (docs/orchestrator.md, "Output"): a member's share of a quote that moves on after the ready_by of
its accepted reply breaks the promise; the report counts them per quote type, with how late on average and at most.
"""
import json
from datetime import datetime, timedelta

import pytest
import pytz

from prosimos.orchestration.config import Organization, ProcessSpec, SimulationConfig
from prosimos.orchestration.simulation import run_multiprocess_simulation

START = pytz.utc.localize(datetime(2024, 1, 1, 9))
MINUTE, HOUR = timedelta(minutes=1), timedelta(hours=1)
ASSETS = "testing_scripts/assets/quotes"
# sales_quote: one order is placed (10 minutes) and quotes its items at 09:10. make_and_ship: an item arrives every
# 30 minutes from 09:00 and is made (1 hour, by one maker) for the shelf. At 09:10 item-0 is being made until 10:00,
# and item-1, arriving at 09:30, is forecast with no wait measured yet: 10:30. It waits for the maker, though, and
# reaches the shelf at 11:00


@pytest.mark.parametrize("items, ready_by, broken", [
    (1, START + HOUR, {"count": 0, "average_late": None, "most_late": None, "undelivered": 0, "most_overdue": None}),
    (2, START + 90 * MINUTE, {"count": 1, "average_late": 30 * MINUTE, "most_late": 30 * MINUTE, "undelivered": 0,
                              "most_overdue": None})],
    ids=["kept", "broken"])
def test_a_share_moving_on_after_the_promise_breaks_it(tmp_path, items, ready_by, broken):
    with open(f"{ASSETS}/sales_quote.json") as file:
        settings = json.load(file)
    settings["messages"]["publish"][0]["quote"]["quantity"] = items
    (tmp_path / "sales.json").write_text(json.dumps(settings))
    sales = ProcessSpec("Sales", f"{ASSETS}/sales_quote.bpmn", str(tmp_path / "sales.json"), 1, "order")
    warehouse = ProcessSpec("Warehouse", f"{ASSETS}/make_and_ship.bpmn", f"{ASSETS}/make_and_ship.json", 2, "item")

    report = run_multiprocess_simulation(SimulationConfig([Organization("Shop", [sales, warehouse])], START, 1))

    [accepted] = [message for message in report.published if message.type == "QuoteAccepted"]
    assert accepted.attributes["ready_by"] == ready_by
    assert report.broken_promises == {"ItemQuote": broken}
    assert report.to_dict()["broken_promises"] == [{"type": "ItemQuote", **{
        name: value.total_seconds() if isinstance(value, timedelta) else value for name, value in broken.items()}}]
