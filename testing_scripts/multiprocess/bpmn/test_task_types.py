"""
Task types: a user, service, send, receive, script, manual or business rule task runs exactly like a plain task:
same durations, resources and log row.
"""
from prosimos.control_flow_manager import TASK_TAGS
from prosimos.messaging.parser import parse_messages
from testing_scripts.multiprocess.bpmn.test_end_events import _log, _model

FLOWS = [("start", "Start", "Approve"), ("approved", "Approve", "Notify"), ("notified", "Notify", "End")]


def test_user_and_service_tasks_run_and_log_like_plain_tasks(tmp_path):
    (tmp_path / "typed").mkdir()
    (tmp_path / "plain").mkdir()
    typed = _model(tmp_path / "typed", {"Start": "startEvent", "Approve": "userTask", "Notify": "serviceTask",
                                        "End": "endEvent"}, FLOWS, hours={"Notify": 2})
    plain = _model(tmp_path / "plain", {"Start": "startEvent", "Approve": "task", "Notify": "task", "End": "endEvent"},
                   FLOWS, hours={"Notify": 2})

    rows = _log(*typed, 5, tmp_path / "typed")

    assert rows == _log(*plain, 5, tmp_path / "plain")
    assert [(row["activity"], row["start_time"][11:16], row["end_time"][11:16]) for row in rows[:2]] == [
        ("Approve", "09:00", "10:00"), ("Notify", "10:00", "12:00")]


def test_every_task_type_runs(tmp_path):
    elements = {"Start": "startEvent", **{f"A_{tag}": tag for tag in TASK_TAGS}, "End": "endEvent"}
    chain = ["Start", *[f"A_{tag}" for tag in TASK_TAGS], "End"]
    flows = [(f"f{number}", source, target) for number, (source, target) in enumerate(zip(chain, chain[1:]))]

    rows = _log(*_model(tmp_path, elements, flows), 1, tmp_path)

    assert [row["activity"] for row in rows] == [f"A_{tag}" for tag in TASK_TAGS]


def test_carry_may_name_a_user_task(tmp_path):
    bpmn, _ = _model(tmp_path, {"Start": "startEvent", "Approve": "userTask",
                                "Notified": "intermediateThrowEvent:message", "Archive": "userTask", "End": "endEvent"},
                     [("start", "Start", "Approve"), ("approved", "Approve", "Notified"),
                      ("notified", "Notified", "Archive"), ("archived", "Archive", "End")])

    model = parse_messages({"publish": [{"event_id": "Notified", "type": "Notified", "qualifier": "approval",
                                         "carry": ["Archive"]}]}, bpmn, set())

    assert model.publish[0].carry == ("Archive",)
