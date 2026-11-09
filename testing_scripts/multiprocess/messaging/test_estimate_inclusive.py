"""
Inclusive splits in estimates (docs/messaging.md): each branch is taken with its own chance, and if none is, Prosimos
takes one drawn by the chances; every combination goes on from the join when its longest branch gets there,
weighted by its chance. A split whose chances don't add up to 1 gives no slot. A branch with a condition is taken
exactly when its data is known, by its history when the data isn't known yet, and always when there's no history
yet.
"""
import json
from datetime import timedelta

import pytest

from prosimos.orchestration.config import ProcessSpec
from prosimos.orchestration.prosimos_engine import ProsimosEngine
from prosimos.orchestration.protocol import Line, Message, Quote, QuoteKind
from testing_scripts.multiprocess.bpmn.test_end_events import START, _model

MINUTE, HOUR = timedelta(minutes=1), timedelta(hours=1)

# an item is received (1 hour), then polished (1 hour), tested (30 minutes), or both, and packed (30 minutes) for the
# shelf
ELEMENTS = {"Start": "startEvent", "Receive": "task", "Split": "inclusiveGateway", "Polish": "task", "Test": "task",
            "Join": "inclusiveGateway", "Pack": "task", "Shelf": "intermediateCatchEvent:message", "End": "endEvent"}
FLOWS = [("start", "Start", "Receive"), ("received", "Receive", "Split"), ("to_polish", "Split", "Polish"),
         ("to_test", "Split", "Test"), ("polished", "Polish", "Join"), ("tested", "Test", "Join"),
         ("joined", "Join", "Pack"), ("packed", "Pack", "Shelf"), ("shelved", "Shelf", "End")]
# the same, with the item cleaned (10 minutes) before it is tested
CLEANED = ({**ELEMENTS, "Clean": "task"}, [flow for flow in FLOWS if flow[0] != "to_test"]
           + [("to_test", "Split", "Clean"), ("cleaned", "Clean", "Test")])


def _warehouse(tmp_path, polish, test, model=(ELEMENTS, FLOWS), seed=1):
    """The warehouse's engine for one item arriving at 09:00; the split polishes with chance polish and tests with
    chance test."""
    bpmn, json_path = _model(tmp_path, *model, hours={"Clean": 1 / 6, "Test": 0.5, "Pack": 0.5},
                             probabilities={"Split": [("to_polish", polish), ("to_test", test)]},
                             messages={"consume": [{"event_id": "Shelf", "type": "ItemQuote"}]})
    return ProsimosEngine(ProcessSpec("Warehouse", str(bpmn), str(json_path), 1, "item"), START, seed=seed)


def _slots(engine, now=START):
    """Steps the engine through everything due before now, then its slots for a one-line quote."""
    while engine.next_event_time() is not None and engine.next_event_time() < now:
        engine.step()
    asked = Message("ItemQuote", {"object_id": "order-9"}, "m9", "Sales", now,
                    Quote(QuoteKind.QUOTE, "ItemQuote", "order-9", [Line({"object_id": "order-9"}, "l1")],
                          timedelta(days=3), "QuoteAccepted", "QuoteDeclined"))
    return engine.estimate(asked, now)["l1"]


def test_each_combination_goes_on_from_the_join_when_its_longest_branch_gets_there(tmp_path):
    # 70% / 30%: only Polish 63.7% (49% + 70% of the 21% that take none), only Test 15.3%, both 21%. Received by
    # 10:00, the join at 11:00, 10:30, and 11:00: 55.41 minutes on average, then packed
    [slot] = _slots(_warehouse(tmp_path, 0.7, 0.3))

    expected = START + timedelta(minutes=60 + 0.637 * 60 + 0.153 * 30 + 0.21 * 60 + 30)
    assert abs(slot - expected) < timedelta(milliseconds=1)


@pytest.mark.parametrize("polish, test", [(0.7, 0.4), (0.5, 0.3)])
def test_a_split_whose_chances_dont_add_up_to_1_gives_no_slot(tmp_path, polish, test):
    # Prosimos itself can't run it once no branch is taken
    assert _slots(_warehouse(tmp_path, polish, test)) == []


def test_an_item_inside_the_split_goes_on_when_its_latest_branch_gets_there(tmp_path):
    # with seed 3 the split takes both branches at 10:00: at 10:05 Polish runs until 11:00, the join's time for it,
    # and Test is queued at 10:10 (done by 10:40): the join at 11:00, packed by 11:30
    engine = _warehouse(tmp_path, 0.7, 0.3, CLEANED, seed=3)

    assert _slots(engine, START + 65 * MINUTE) == [START + 150 * MINUTE]
    env = engine._env
    assert [(event.task_id, event.enabled_datetime) for event in env.events_queue.enabled_events.entry_finder] == [
        ("Test", START + 70 * MINUTE)]
    assert env.sim_setup.bpmn_graph.last_datetime["Join"][0].datetime == START + 2 * HOUR


# an item is inspected (10 minutes), which sets its quality, good or bad, and washed (10 minutes); then a good one
# is polished (1 hour) and a bad one labeled (20 minutes) for the shelf. One item arrives a day; with seed 4, items
# 0, 1, and 2 are bad, good, and bad
GRADED = ({"Start": "startEvent", "Inspect": "task", "Wash": "task", "Split": "inclusiveGateway", "Polish": "task",
           "Label": "task", "Join": "inclusiveGateway", "Shelf": "intermediateCatchEvent:message",
           "End": "endEvent"},
          [("start", "Start", "Inspect"), ("inspected", "Inspect", "Wash"), ("washed", "Wash", "Split"),
           ("to_polish", "Split", "Polish"), ("to_label", "Split", "Label"), ("polished", "Polish", "Join"),
           ("labeled", "Label", "Join"), ("joined", "Join", "Shelf"), ("shelved", "Shelf", "End")])
DAY = timedelta(days=1)


def _graded(tmp_path):
    bpmn, json_path = _model(tmp_path, *GRADED, hours={"Inspect": 1 / 6, "Wash": 1 / 6, "Label": 1 / 3},
                             probabilities={"Split": [("to_polish", 0.5), ("to_label", 0.5)]},
                             messages={"consume": [{"event_id": "Shelf", "type": "ItemQuote"}]})
    settings = json.loads(json_path.read_text())
    for probability in settings["gateway_branching_probabilities"][0]["probabilities"]:
        probability["condition_id"] = "good" if probability["path_id"] == "to_polish" else "bad"
    settings["branch_rules"] = [{"id": quality, "rules": [[{"attribute": "quality", "comparison": "=",
                                                             "value": quality}]]} for quality in ("good", "bad")]
    settings["event_attributes"] = [{"event_id": "Inspect", "attributes": [{"name": "quality", "type": "discrete",
                                                                            "values": [{"key": "good", "value": 0.5},
                                                                                       {"key": "bad", "value": 0.5}]}]}]
    json_path.write_text(json.dumps(settings))
    return ProsimosEngine(ProcessSpec("Warehouse", str(bpmn), str(json_path), 4, "item"), START, seed=4)


def test_branches_with_conditions_on_data_not_known_yet_and_no_history_are_taken(tmp_path):
    # both branches: polished by 10:20
    assert _slots(_graded(tmp_path)) == [START + index * DAY + 80 * MINUTE for index in range(4)]


def test_branches_with_conditions_go_by_known_data_or_their_history(tmp_path):
    # on day 3 at 09:05 item-2 is inspected, bad: labeled by 09:40. Item-3's quality isn't known: each branch is taken
    # with chance 1/2 as so far, both 25%, polishing alone 25% + 12.5% (half of the 25% that take neither), labeling
    # alone 37.5%: the join on average 45 minutes after washing, at 10:05
    engine = _graded(tmp_path)

    assert _slots(engine, START + 2 * DAY + 5 * MINUTE)[2:] == [START + 2 * DAY + 40 * MINUTE,
                                                                 START + 3 * DAY + 65 * MINUTE]
    assert engine._env._history("Split") == ({"to_label": 1, "to_polish": 1}, 2)
