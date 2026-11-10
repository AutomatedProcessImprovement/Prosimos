"""
Runs of the Stock Reservation example (docs/stock-reservation.md) with a few settings changed, so that a handful of
orders and items tell one story, what the tests read from them, and edits of the Sales model. By default there is
one order, of one MacBook, which never changes and is patient for 5 days; no item is scrapped; TartuWarehouse makes
three MacBooks, one every 2 hours from 09:00, each done an hour later, and TallinnWarehouse makes one iPhone, which
nobody asks.
"""
import csv
import json
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import pytz

from prosimos.orchestration.config import SimulationConfig
from prosimos.orchestration.simulation import run_multiprocess_simulation

EXAMPLE = "testing_scripts/assets/stock_reservation"
BPMN = "http://www.omg.org/spec/BPMN/20100524/MODEL"
MINUTE, HOUR = timedelta(minutes=1), timedelta(hours=1)
AFTER_TIES = timedelta(microseconds=1)  # a race timer's branch goes on 1 µs late


def at(clock):
    """09:00 on the run's first day, 2024-01-01, at clock, UTC."""
    return pytz.utc.localize(datetime.fromisoformat(f"2024-01-01T{clock}"))


def run(tmp_path, change, seed=1, model=None):
    """The report and log rows of the example with its settings changed: change(settings) changes the files'
    contents, by name (simulation, sales, tartu_warehouse, tallinn_warehouse), after the defaults; model, if given,
    changes the text of sales.bpmn."""
    folder = tmp_path / "stock_reservation"
    shutil.copytree(EXAMPLE, folder)
    settings = {name: json.loads((folder / f"{name}.json").read_text())
                for name in ("simulation", "sales", "tartu_warehouse", "tallinn_warehouse")}
    _defaults(settings)
    change(settings)
    settings["simulation"]["seed"] = seed
    for name, content in settings.items():
        (folder / f"{name}.json").write_text(json.dumps(content))
    if model is not None:
        (folder / "sales.bpmn").write_text(model((folder / "sales.bpmn").read_text()))
    report = run_multiprocess_simulation(SimulationConfig.from_json(folder / "simulation.json"), folder / "log.csv")
    with open(folder / "log.csv") as file:
        return report, list(csv.DictReader(file))


def _defaults(settings):
    orders(settings, 1, "MacBook")
    always(settings["sales"], "patience", 5)
    branches(settings["sales"], "Change", Keep=1.0, Add=0.0)
    for warehouse in ("tartu_warehouse", "tallinn_warehouse"):
        branches(settings[warehouse], "Check", Good=1.0, Bad=0.0)
    makes(settings, "TartuWarehouse", "MacBook", 3)
    makes(settings, "TallinnWarehouse", "iPhone", 1)


def _process(settings, name):
    return next(spec for spec in settings["simulation"]["organizations"][0]["processes"] if spec["name"] == name)


def orders(settings, count, product, items=1):
    """count orders, each of items of the product."""
    sales = _process(settings, "Sales")
    sales["arrivals"] = count
    sales["references"][0]["pick"] = {"weights": {product: 1}}
    always(settings["sales"], "items", items)


def makes(settings, warehouse, product, count):
    _process(settings, warehouse)["arrivals"] = count
    _process(settings, warehouse)["references"][0]["pick"] = {"weights": {product: 1}}


def always(process_settings, attribute, value):
    next(entry for entry in process_settings["object_attributes"] if entry["name"] == attribute)["values"] = [
        {"key": value, "value": 1.0}]


def branches(process_settings, gateway, **shares):
    next(entry for entry in process_settings["gateway_branching_probabilities"] if entry["gateway_id"] == gateway)[
        "probabilities"] = [{"path_id": flow, "value": share} for flow, share in shares.items()]


def every(process_settings, interval):
    process_settings["arrival_time_distribution"] = {"distribution_name": "fix",
                                                     "distribution_params": [{"value": interval.total_seconds()}]}


def takes(process_settings, task_id, duration):
    next(entry for entry in process_settings["task_resource_distribution"] if entry["task_id"] == task_id)[
        "resources"][0].update(distribution_name="fix", distribution_params=[{"value": duration.total_seconds()}])


def gives_up_after(sales, delay):
    sales["event_distribution"][0]["distribution_params"] = [{"value": delay.total_seconds()}]


def messages(report, order):
    """The order's messages, in time order: (time, type, what): the products of a quote or modify, the ready_by or
    earliest of a reply, the warehouse and items of a Shipment, nothing for a cancel."""
    story = []
    for message in report.published:
        if message.type == "Shipment" and message.attributes["order_id"] == order:
            story.append((message.time, "Shipment", (message.source, message.attributes["item_ids"])))
        elif message.source == "orchestrator" and message.attributes["request"] == order:
            story.append((message.time, message.type,
                          message.attributes.get("ready_by", message.attributes.get("earliest"))))
        elif message.source == "Sales" and message.attributes.get("object_id") == order:
            products = [line.attributes["product"] for line in message.quote.lines] if message.quote.lines else None
            story.append((message.time, message.type, products))
    return story


def taken(report):
    """Each line an item took: (time, warehouse, item, the quote or modify message's type, its order), in time
    order."""
    sent = {message.id: message for message in report.published}
    return sorted(((claimer.at, claimer.process, claimer.object_id, sent[record.message_id].type,
                    sent[record.message_id].attributes["object_id"])
                   for record in report.message_records if sent[record.message_id].quote is not None
                   for claimer in record.claimers), key=lambda taken: taken[0])


def ended(rows, process, case, activity):
    return next(datetime.fromisoformat(row["end_time"]) for row in rows
                if row["process"] == process and row["case_id"] == case and row["activity"] == activity)


def outcomes(report, **counts):
    outcomes = dict.fromkeys(("accepted", "declined", "canceled", "cancel_too_late", "modified_accepted",
                              "modified_declined", "declined_as_second"), 0)
    return report.quotes == {"ItemQuote": {**outcomes, **counts}}


ON_TIME = {"count": 0, "average_late": None, "most_late": None, "undelivered": 0, "most_overdue": None}


class Model:
    """Edits of the Sales model's text, keeping each element's incoming and outgoing flows in step."""

    def __init__(self, text):
        ET.register_namespace("bpmn", BPMN)
        self.root = ET.fromstring(text)
        self.process = self.root.find(f"{{{BPMN}}}process")

    def _element(self, element_id):
        return next(element for element in self.process if element.get("id") == element_id)

    def add(self, tag, element_id, name, definition=None):
        element = ET.SubElement(self.process, f"{{{BPMN}}}{tag}", id=element_id, name=name)
        if definition:
            ET.SubElement(element, f"{{{BPMN}}}{definition}EventDefinition", id=f"{element_id}_definition")
        return self

    def remove(self, element_id):
        self.process.remove(self._element(element_id))
        return self

    def flow(self, flow_id, source, target):
        ET.SubElement(self.process, f"{{{BPMN}}}sequenceFlow", id=flow_id, sourceRef=source, targetRef=target)
        self._link(source, "outgoing", flow_id)
        self._link(target, "incoming", flow_id)
        return self

    def retarget(self, flow_id, target):
        flow = self._element(flow_id)
        old = self._element(flow.get("targetRef"))
        old.remove(next(link for link in old if link.tag == f"{{{BPMN}}}incoming" and link.text == flow_id))
        flow.set("targetRef", target)
        self._link(target, "incoming", flow_id)
        return self

    def _link(self, element_id, kind, flow_id):
        link = ET.Element(f"{{{BPMN}}}{kind}")
        link.text = flow_id
        self._element(element_id).insert(0, link)  # before an event definition

    def text(self):
        return ET.tostring(self.root, encoding="unicode")
