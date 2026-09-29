"""The run report: one test per warning and per count. Stalled cases are left for Sprint 3."""
from prosimos.orchestrator import Verdict, run_engines
from testing_scripts.protocol_scenario import at, run_scenario
from testing_scripts.scripted_engine import ScriptedEngine

SEED = 1


def _publisher(*message_types):
    engine = ScriptedEngine("P")
    for minute, message_type in enumerate(message_types):
        engine.publish_at(at(f"09:{minute:02d}"), message_type)
    return engine


def _consumer(name, message_type, verdict):
    engine = ScriptedEngine(name)
    engine.consume(message_type, lambda message, now: verdict)
    return engine


def test_warning_for_a_message_type_nobody_subscribes_to():
    report = run_engines({"P": _publisher("Newsletter")}, None, SEED)

    assert report.warnings == ["Newsletter m1 from P has no subscribers"]
    assert report.copies == [] and report.unclaimed == []


def test_warning_for_a_message_every_recipient_discarded():
    # the Narva truck from the running example: both warehouses discard it, nobody else subscribes
    _, report = run_scenario(SEED)
    narva = next(m for m in report.published if m.attributes.get("dock") == "Narva")

    assert f"Truck {narva.id} from Carrier was discarded by every recipient" in report.warnings


def test_no_warning_when_another_group_claims_the_message():
    engines = {"P": _publisher("X"), "A": _consumer("A", "X", Verdict.CLAIMED), "B": _consumer("B", "X", Verdict.DISCARDED)}

    report = run_engines(engines, {"P": ["P"], "GroupA": ["A"], "GroupB": ["B"]}, SEED)

    assert report.warnings == []


def test_no_warning_while_a_copy_is_still_pending():
    engines = {"P": _publisher("X"), "A": _consumer("A", "X", Verdict.DISCARDED), "B": _consumer("B", "X", Verdict.PENDING)}

    report = run_engines(engines, {"P": ["P"], "Pair": ["A", "B"]}, SEED)

    assert report.warnings == []


def test_discarded_counts_per_type_and_process():
    engines = {
        "P": _publisher("X", "X", "Y"),
        "A": _consumer("A", "X", Verdict.DISCARDED),
        "B": _consumer("B", "Y", Verdict.DISCARDED),
    }

    report = run_engines(engines, None, SEED)

    assert report.discarded_counts == {("X", "A"): 2, ("Y", "B"): 1}


def test_unclaimed_copies_are_the_ones_left_in_the_pool():
    engines = {
        "P": _publisher("X", "Y"),
        "A": _consumer("A", "X", Verdict.PENDING),
        "B": _consumer("B", "X", Verdict.PENDING),
        "C": _consumer("C", "Y", Verdict.CLAIMED),
    }

    report = run_engines(engines, None, SEED)

    # X has one copy per subscribed group, both still pending; Y was claimed, so no copy is left
    assert sorted((group, message.type) for group, message in report.unclaimed) == [("A", "X"), ("B", "X")]


def test_warnings_are_collected_not_printed(capsys, caplog):
    _, report = run_scenario(SEED)

    assert len(report.warnings) == 3
    printed = capsys.readouterr()
    assert printed.out == "" and printed.err == ""
    assert caplog.records == []
