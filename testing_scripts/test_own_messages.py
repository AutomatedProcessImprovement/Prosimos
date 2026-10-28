"""
A process never receives its own messages (docs/orchestrator.md, "Message routing"): a copy is never offered to the
process whose name is the message's source, and a group whose only member subscribed to the type is the publisher
gets no copy.
"""
import pytest

from prosimos.orchestration.orchestrator import orchestrate
from prosimos.orchestration.protocol import Verdict
from testing_scripts.protocol_scenario import at
from testing_scripts.scripted_engine import ScriptedEngine

SEED = 1


def _workshop(name="Workshop", verdict=Verdict.CLAIMED):
    """Frees a part at 09:00 and catches every PartFreed."""
    workshop = ScriptedEngine(name)
    workshop.publish_at(at("09:00"), "PartFreed", part="p1")
    workshop.consume("PartFreed", lambda message, now: verdict)
    return workshop


def _warehouse(name, order):
    """Forwards an order at 09:00 and claims every order forwarded to it."""
    warehouse = ScriptedEngine(name)
    warehouse.publish_at(at("09:00"), "ItemForwarded", order=order)
    warehouse.consume("ItemForwarded", lambda message, now: Verdict.CLAIMED)
    return warehouse


def test_a_workshop_alone_never_claims_its_own_message_and_gets_a_warning():
    workshop = _workshop()

    report = orchestrate({"Workshop": workshop}, None, SEED)

    assert workshop.offered == [] and report.claims == []
    assert report.copies == [] and report.unclaimed == []
    assert report.warnings == ["PartFreed m1 from Workshop: no subscriber other than its publisher"]


def test_a_workshop_s_copy_goes_only_to_the_other_members_of_its_group_and_ends_unclaimed():
    # a second workshop in the same group leaves it pending: the copy is never offered back to its publisher
    workshop, other = _workshop(), ScriptedEngine("Other")
    other.consume("PartFreed", lambda message, now: Verdict.PENDING)

    report = orchestrate({"Workshop": workshop, "Other": other}, {"Workshops": ["Workshop", "Other"]}, SEED)

    assert workshop.offered == [] and report.claims == []
    assert [message.attributes["part"] for message, _ in other.offered] == ["p1"]
    assert [(group, message.id) for group, message in report.unclaimed] == [("Workshops", "m1")]
    assert report.warnings == []


def test_a_copy_every_member_but_the_publisher_discarded_is_resolved():
    workshop, other = _workshop(), ScriptedEngine("Other")
    other.consume("PartFreed", lambda message, now: Verdict.DISCARDED)

    report = orchestrate({"Workshop": workshop, "Other": other}, {"Workshops": ["Workshop", "Other"]}, SEED)

    assert report.unclaimed == []
    assert report.warnings == ["PartFreed m1 from Workshop was discarded by every recipient"]


@pytest.mark.parametrize("groups", [{"Warehouses": ["Tallinn", "Tartu"]},  # one group
                                    None])  # each its own group
def test_two_warehouses_forwarding_to_each_other_each_get_the_other_s_order(groups):
    tartu, tallinn = _warehouse("Tartu", 1), _warehouse("Tallinn", 2)

    report = orchestrate({"Tartu": tartu, "Tallinn": tallinn}, groups, SEED)

    assert [message.attributes["order"] for message, _ in tartu.claimed] == [2]
    assert [message.attributes["order"] for message, _ in tallinn.claimed] == [1]
    assert [message.source for message, _ in tartu.offered + tallinn.offered] == ["Tallinn", "Tartu"]
    assert len(report.copies) == 2  # one per forwarded order
    assert report.unclaimed == [] and report.warnings == []


def test_a_workshop_s_message_still_reaches_another_group_that_subscribes():
    # the publisher subscribes to the type too, but Other, in its own group, takes the copy
    workshop, other = _workshop(), ScriptedEngine("Other")
    other.consume("PartFreed", lambda message, now: Verdict.CLAIMED)

    report = orchestrate({"Workshop": workshop, "Other": other}, None, SEED)

    assert report.claims == [("m1", "Other", at("09:00"))]
    assert report.copies == [("m1", "Other")] and report.warnings == []
