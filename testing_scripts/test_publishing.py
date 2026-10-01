"""
Publishing (docs/messaging-model.md): a case passing an event listed under 'publish' produces a
message, held until the time the case really passes the event and then returned by step().
"""
import csv
import json
import random
from datetime import datetime

import numpy as np
import pytest
import pytz

from prosimos.orchestrator import ProcessSpec, ProsimosEngine
from prosimos.simulation_engine import run_simulation
from prosimos.warning_logger import warning_logger

ASSETS = "testing_scripts/assets/messaging"
SALES = f"{ASSETS}/sales.bpmn"  # start -> Take order -> throw OrderPlaced -> catch Shipment -> end
SALES_ANNOUNCED_AT_START = f"{ASSETS}/sales_announced_at_start.bpmn"  # throw OrderPlaced right after the start
SALES_JSON = f"{ASSETS}/sales.json"
START = pytz.utc.localize(datetime(2024, 1, 1, 9))


class _Log:
    """Stands in for the csv writer the engine writes its log to."""

    def __init__(self):
        self.rows = []

    def writerow(self, header):
        self.header = header

    def writerows(self, rows):
        self.rows.extend([str(value) for value in row] for row in rows)


def _time(logged):
    return datetime.fromisoformat(logged)


def _step_by_hand(name, bpmn, json_path, cases, seed=1):
    """Steps one engine until it is finished: (time announced, messages returned, log rows written) per step."""
    random.seed(seed)
    np.random.seed(seed)
    log = _Log()
    engine = ProsimosEngine(ProcessSpec(name, bpmn, json_path, cases), START, log)
    steps = []
    while (now := engine.next_event_time()) is not None:
        written = len(log.rows)
        steps.append((now, engine.step(), log.rows[written:]))
    return steps, log.rows


def _published(steps):
    return [(now, message) for now, messages, _ in steps for message in messages]


def _without_publishing(tmp_path, json_path):
    with open(json_path) as file:
        settings = json.load(file)
    del settings["messages"]["publish"]
    path = tmp_path / "no_publishing.json"
    path.write_text(json.dumps(settings))
    return str(path)


def test_sales_publishes_one_order_placed_per_case_when_the_case_passes_the_throw_event():
    steps, log = _step_by_hand("Sales", SALES, SALES_JSON, 10)
    take_order = {row[0]: row for row in log if row[1] == "Take order"}  # case -> case_id, activity, enable, start, end, resource, city

    published = _published(steps)
    assert [message.type for _, message in published] == ["OrderPlaced"] * 10
    for now, message in published:
        case = message.attributes["case_id"].removeprefix("Sales-")
        # the throw event follows Take order, so the case passes it when Take order ends
        assert now == _time(take_order[case][4])
        assert message.attributes == {"case_id": f"Sales-{case}", "city": take_order[case][6]}
    assert sorted(m.attributes["case_id"] for _, m in published) == [f"Sales-{case}" for case in range(10)]


def test_a_message_is_not_released_before_its_time():
    # the case passes the throw event during the step of Take order, at its start; the message
    # comes out of a later step, the one announced for the time Take order ends
    steps, _ = _step_by_hand("Sales", SALES, SALES_JSON, 10)

    assert all(messages == [] for _, messages, written in steps if written)
    assert all(written == [] for _, messages, written in steps if messages)


def test_a_throw_event_right_after_the_start_publishes_at_the_case_arrival_before_its_first_task():
    steps, log = _step_by_hand("Sales", SALES_ANNOUNCED_AT_START, SALES_JSON, 10)

    published = _published(steps)
    assert len(published) == 10
    assert steps[0][1] != []  # nothing is published before the simulation starts
    for now, message in published:
        case = message.attributes["case_id"].removeprefix("Sales-")
        index = next(i for i, (_, messages, _) in enumerate(steps) if message in messages)
        take_order = next(i for i, (_, _, written) in enumerate(steps) if ["Take order", case] in
                          [[row[1], row[0]] for row in written])
        # arrival = when Take order is enabled; at equal times the message comes first
        assert now == _time(next(row[2] for row in log if row[0] == case))
        assert index < take_order and steps[take_order][0] == now


def test_publishing_leaves_the_log_unchanged(tmp_path):
    no_publishing = _without_publishing(tmp_path, SALES_JSON)

    for bpmn in (SALES, SALES_ANNOUNCED_AT_START):
        assert _step_by_hand("Sales", bpmn, SALES_JSON, 20)[1] == _step_by_hand("Sales", bpmn, no_publishing, 20)[1]

        logs = []
        for json_path in (SALES_JSON, no_publishing):
            random.seed(1)
            np.random.seed(1)
            run_simulation(bpmn, json_path, 20, None, tmp_path / "log.csv", "2024-01-01T09:00:00+00:00")
            with open(tmp_path / "log.csv") as file:
                logs.append(list(csv.reader(file)))
        assert logs[0] == logs[1]


@pytest.fixture
def bursts():
    # three cases arriving together at 09:00; each passes First, Second, then Left and Right on two
    # parallel branches, all at once; then Work (10 min, one worker) and the message end event Done
    warning_logger.clear_warnings()
    steps, _ = _step_by_hand("Bursts", f"{ASSETS}/bursts.bpmn", f"{ASSETS}/bursts.json", 3)
    yield steps
    warning_logger.clear_warnings()


def test_messages_due_together_come_out_of_one_step_by_case_then_order_passed(bursts):
    first_step = [(m.attributes["case_id"], m.type) for m in bursts[0][1]]

    assert bursts[0][0] == START
    assert [case for case, _ in first_step] == ["Bursts-0"] * 4 + ["Bursts-1"] * 4 + ["Bursts-2"] * 4
    for case in range(3):
        types = [message_type for case_id, message_type in first_step if case_id == f"Bursts-{case}"]
        # Left and Right are on parallel branches, passed in the order the engine chose
        assert types[:2] == ["First", "Second"] and set(types[2:]) == {"Left", "Right"}


def test_an_engine_holding_messages_is_not_finished_even_with_an_empty_queue(bursts):
    # the three Work events all leave the queue at 09:00; the Done messages are held until each
    # case really ends, so the engine keeps announcing steps after its queue is empty
    assert [(now.strftime("%H:%M"), [(m.attributes["case_id"], m.type) for m in messages])
            for now, messages, _ in bursts[-3:]] == [
        ("09:10", [("Bursts-0", "Done")]),
        ("09:20", [("Bursts-1", "Done")]),
        ("09:30", [("Bursts-2", "Done")]),
    ]


def test_an_attribute_without_a_value_yet_is_sent_as_none_with_one_warning_per_event_and_attribute(bursts):
    # weight is an event attribute of Work: unset when a case passes First, set when it passes Done
    published = [message for _, messages, _ in bursts for message in messages]

    assert [m.attributes["weight"] for m in published if m.type == "First"] == [None, None, None]
    assert [m.attributes["weight"] for m in published if m.type == "Done"] == ["heavy", "heavy", "heavy"]
    assert [w for w in warning_logger.get_all_warnings() if "weight" in w] == [
        "Attribute weight has no value when case 0 passes Throw_First; its First message carries None"
    ]
