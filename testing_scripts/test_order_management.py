"""
The Order Management running example (docs/order-management.md): four processes after the OCEL 2.0 Order
Management log, run together on about 100 orders with a fixed seed.
"""
import dataclasses
import json
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from cli.diff_res_bpsim import cli
from prosimos.orchestrator import SimulationConfig, run_orchestrator
from prosimos.simulation_engine import SimBPMEnv
from testing_scripts.order_management_compare import package_contents

CONFIG = "testing_scripts/assets/order_management/simulation.json"  # 2,000 orders; seed 1
ORDERS = 100
# a race timer's branch continues one microsecond after its time, so each reminder comes 20 days and one
# microsecond after the confirmation or the previous reminder (the reminder itself takes no time)
REMINDER_GAP = timedelta(days=20, microseconds=1)


def _config(orders=ORDERS):
    config = SimulationConfig.from_json(CONFIG)
    return dataclasses.replace(config, processes=[
        dataclasses.replace(spec, total_cases=orders) if spec.name == "Sales" else spec for spec in config.processes])


class _PackageContents:
    """Records which ItemPicked each package took, as the engine hands them out: the one that started it, then
    the ones its catch event collected."""

    def __init__(self, monkeypatch):
        self.items = defaultdict(list)  # package case -> [(order_id, item_index, customer)]
        original_copy, original_claim = SimBPMEnv._apply_copy, SimBPMEnv._claim
        record = self

        def apply_copy(env, point, values, case_values):
            if env.process_name == "Packaging" and point.starts_case:
                # called while the case is created, before its trace is added: its id is the trace count
                record.add(len(env.log_info.trace_list), values)
            return original_copy(env, point, values, case_values)

        def claim(env, parked_event, point, values, now):
            if env.process_name == "Packaging":
                record.add(parked_event.p_case, values)
            return original_claim(env, parked_event, point, values, now)

        monkeypatch.setattr(SimBPMEnv, "_apply_copy", apply_copy)
        monkeypatch.setattr(SimBPMEnv, "_claim", claim)

    def add(self, package, values):
        self.items[package].append((values["order_id"], values["item_index"], values["customer"]))


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """One 100-order run: the report, the merged log, and what each package took."""
    monkeypatch = pytest.MonkeyPatch()
    contents = _PackageContents(monkeypatch)
    log_path = tmp_path_factory.mktemp("order_management") / "log.csv"
    try:
        report = run_orchestrator(_config(), log_path)
    finally:
        monkeypatch.undo()
    log = pd.read_csv(log_path, parse_dates=["enable_time", "start_time", "end_time"])
    return report, log, contents.items


def _rows(log, process, activity=None):
    rows = log[log["process"] == process]
    return rows if activity is None else rows[rows["activity"] == activity]


def _orders(log):
    """Sales case -> (order id, items, customer)."""
    first = _rows(log, "Sales").groupby("case_id").first()
    return {case: (f"Sales-{case}", int(row["items"]), row["customer"]) for case, row in first.iterrows()}


def test_every_order_publishes_one_item_ordered_per_item(run):
    report, log, _ = run

    published = defaultdict(list)
    for message in report.published:
        if message.type == "ItemOrdered":
            published[message.attributes["case_id"]].append(
                (message.attributes["index"], message.attributes["customer"]))
    orders = _orders(log)
    assert len(orders) == ORDERS
    assert all(published[order] == [(index, customer) for index in range(1, items + 1)]
               for order, items, customer in orders.values())


def test_every_item_case_has_its_order_id_index_and_customer(run):
    _, log, _ = run

    items = _rows(log, "Warehouse").groupby("case_id").first()
    customers = {order: customer for order, _, customer in _orders(log).values()}
    expected = sorted((order, index) for order, count, _ in _orders(log).values() for index in range(1, count + 1))
    assert sorted(zip(items["order_id"], items["item_index"].astype(int))) == expected
    assert all(customer == customers[order] for order, customer in zip(items["order_id"], items["customer"]))


def test_every_package_holds_items_of_one_customer_and_no_item_is_in_two_packages(run):
    report, log, contents = run

    assert all(len({customer for _, _, customer in items}) == 1 for items in contents.values())
    taken = [(order, index) for items in contents.values() for order, index, _ in items]
    assert len(taken) == len(set(taken))
    # every picked item is in a package, maybe one still collecting at the end
    picked = [(m.attributes["order_id"], m.attributes["item_index"]) for m in report.published if m.type == "ItemPicked"]
    assert sorted(taken) == sorted(picked)


def test_the_comparison_script_reconstructs_the_packages_from_the_log(run):
    _, log, contents = run

    reconstructed, still_collecting = package_contents(log)
    assert reconstructed == {package: [(order, index) for order, index, _ in items]
                             for package, items in contents.items() if package in reconstructed}
    created = set(_rows(log, "Packaging", "create package")["case_id"])
    assert set(reconstructed) == created
    assert {items[0][2]: [(order, index) for order, index, _ in items]
            for package, items in contents.items() if package not in created} == still_collecting


def test_every_order_ends_paid(run):
    report, log, _ = run

    assert set(_rows(log, "Sales", "pay order")["case_id"]) == set(_orders(log))
    assert not any(process == "Sales" for process, _ in report.stalled)


def test_reminders_come_exactly_20_days_apart_and_only_while_unpaid(run):
    _, log, _ = run

    confirmed = _rows(log, "Sales", "confirm order").groupby("case_id")["end_time"].first()
    paid = _rows(log, "Sales", "pay order").groupby("case_id")["enable_time"].first()  # when the payment came
    reminders = _rows(log, "Sales", "payment reminder").groupby("case_id")["enable_time"].apply(sorted)
    assert len(reminders) > 0  # some orders were reminded
    for order in _orders(log):
        # the race starts when the order is confirmed, and again after each reminder
        times = [confirmed[order]] + reminders.get(order, [])
        assert all(later - earlier == REMINDER_GAP for earlier, later in zip(times, times[1:]))
        # every reminder came before the payment, and the payment came before the next one was due
        assert times[-1] < paid[order] < times[-1] + REMINDER_GAP


def test_at_the_end_the_only_stalled_cases_are_open_packages_at_most_one_per_customer(run):
    report, _, contents = run

    assert report.stalled  # with this seed, some packages are still collecting
    assert all(process == "Packaging" and case.event_id == "Catch_More_Items" for process, case in report.stalled)
    customers = [contents[int(case.case_id.split("-")[1])][0][2] for _, case in report.stalled]
    assert len(customers) == len(set(customers))
    assert report.unclaimed == [] and report.discards == []


def test_the_same_seed_gives_the_same_run(tmp_path):
    first = run_orchestrator(_config(), tmp_path / "first.csv")
    second = run_orchestrator(_config(), tmp_path / "second.csv")

    assert first.to_dict() == second.to_dict()
    assert (tmp_path / "first.csv").read_text() == (tmp_path / "second.csv").read_text()


def test_the_same_seed_gives_the_same_run_through_start_orchestration(tmp_path):
    # the committed configuration has 2,000 orders; this copy has 100, with the models' absolute paths
    with open(CONFIG) as file:
        settings = json.load(file)
    folder = Path(CONFIG).parent.resolve()
    for process in settings["processes"]:
        process["bpmn_path"] = str(folder / process["bpmn_path"])
        process["json_path"] = str(folder / process["json_path"])
        if process["name"] == "Sales":
            process["total_cases"] = ORDERS
    config_path = tmp_path / "simulation.json"
    config_path.write_text(json.dumps(settings))

    for name in ("first", "second"):
        result = CliRunner().invoke(cli, ["start-orchestration", "--config", str(config_path),
                                          "--log_out_path", str(tmp_path / f"{name}.csv"),
                                          "--report_out_path", str(tmp_path / f"{name}.json")])
        assert result.exit_code == 0, result.output
    run_orchestrator(_config(), tmp_path / "python.csv")

    assert (tmp_path / "first.csv").read_text() == (tmp_path / "second.csv").read_text()
    assert (tmp_path / "first.json").read_text() == (tmp_path / "second.json").read_text()
    assert (tmp_path / "first.csv").read_text() == (tmp_path / "python.csv").read_text()
