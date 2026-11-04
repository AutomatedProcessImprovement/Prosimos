"""
Unknown keys (docs/orchestrator.md, docs/messaging.md): a key the configuration readers don't know is a load error,
naming the key, where it is and the allowed keys, since ignored, a misspelled key would change the run without any
message. One test per place, each with a misspelled key.
"""
import json
import time
from datetime import datetime

import pytest
import pytz

from prosimos.exceptions import InvalidSimScenarioException
from prosimos.messaging.parser import parse_messages
from prosimos.orchestration.catalogue.checks import checked_catalogue
from prosimos.orchestration.catalogue.model import CataloguePool
from prosimos.orchestration.config import ProcessSpec, SimulationConfig
from prosimos.orchestration.prosimos_engine import load_process
from testing_scripts.multiprocess.catalogue.test_catalogue import config_with
from testing_scripts.multiprocess.messaging.test_messaging_parser import (  # noqa: F401 (bpmn is a fixture)
    bpmn, consume, publish)
from testing_scripts.multiprocess.orchestrator_settings import one_organization

START = pytz.utc.localize(datetime(2024, 1, 1, 9))


def _config_rejected(tmp_path, message, change):
    path = config_with(tmp_path)
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        SimulationConfig.from_json(path)


# simulation.json

def test_the_top_level(tmp_path):
    def misspelled(data):
        data["sead"] = data.pop("seed")

    _config_rejected(tmp_path, r"simulation.json: top level: unknown key 'sead'; allowed: consumer_groups, "
                               r"organizations, seed, start_time$", misspelled)


def test_processes_belong_to_an_organization_not_to_the_top_level(tmp_path):
    def at_the_top_level(data):
        data["processes"] = data.pop("organizations")[0]["processes"]

    _config_rejected(tmp_path, r"simulation.json: top level: unknown key 'processes'; allowed: consumer_groups, "
                               r"organizations, seed, start_time$", at_the_top_level)


def test_an_organization_entry(tmp_path):
    def misspelled(data):
        data["organizations"][0]["proceses"] = data["organizations"][0].pop("processes")

    _config_rejected(tmp_path, r"simulation.json: organizations\[0\]: unknown key 'proceses'; allowed: "
                               r"catalogue_changes, catalogues, name, processes$", misspelled)


def test_a_process_entry(tmp_path):
    def misspelled(data):
        data["organizations"][0]["processes"][2]["object_typ"] = data["organizations"][0]["processes"][2].pop("object_type")

    _config_rejected(tmp_path, r"processes\[2\]: unknown key 'object_typ'; allowed: arrivals, bpmn_path, "
                               r"json_path, name, object_type, ocel_attributes, qualifier, "
                               r"qualifier_by_activity, references$",
                     misspelled)


def test_an_object_of_a_listed_catalogue(tmp_path):
    def misspelled(data):
        customers = data["organizations"][0]["catalogues"]["customer"]
        customers[2]["attribute"] = {}

    _config_rejected(tmp_path, r"organizations\[0\].catalogues.customer\[2\]: unknown key 'attribute'; allowed: attributes, id$",
                     misspelled)


def test_a_pool(tmp_path):
    def misspelled(data):
        data["organizations"][0]["catalogues"]["customer"] = {"size": 10, "atributes": []}

    _config_rejected(tmp_path, r"organizations\[0\].catalogues.customer: unknown key 'atributes'; allowed: attributes, size$",
                     misspelled)


def test_count_is_no_key_of_a_pool(tmp_path):
    # size says how many objects a pool has
    def counted(data):
        data["organizations"][0]["catalogues"]["customer"] = {"count": 10, "attributes": []}

    _config_rejected(tmp_path, r"organizations\[0\].catalogues.customer: unknown key 'count'; allowed: attributes, "
                               r"size$", counted)


def test_a_growing_catalogue(tmp_path):
    # a misspelled size would otherwise make the pool a growing catalogue
    def misspelled(data):
        data["organizations"][0]["catalogues"]["customer"] = {"cont": 10, "attributes": []}

    _config_rejected(tmp_path, r"organizations\[0\].catalogues.customer: unknown key 'cont'; allowed: attributes, size$",
                     misspelled)


def test_a_catalogue_change(tmp_path):
    def misspelled(data):
        data["organizations"][0]["catalogue_changes"] = [{"catalogue": "product", "attribute": "price",
                                                          "every": {"days": 60}, "multiply": 1.03, "rond": 2}]

    _config_rejected(tmp_path, r"organizations\[0\].catalogue_changes\[0\]: unknown key 'rond'; allowed: add, at, "
                               r"attribute, catalogue, every, multiply, objects, round, set$", misspelled)


def test_the_period_of_a_catalogue_change(tmp_path):
    def misspelled(data):
        data["organizations"][0]["catalogue_changes"] = [{"catalogue": "product", "attribute": "price",
                                                          "every": {"day": 60}, "multiply": 1.03}]

    _config_rejected(tmp_path, r"organizations\[0\].catalogue_changes\[0\].every: unknown key 'day'; allowed: days, "
                               r"hours, weeks$", misspelled)


def test_a_reference(tmp_path):
    def misspelled(data):
        reference = data["organizations"][0]["processes"][0]["references"][0]
        reference["link_event"] = reference.pop("link_events")
        reference["revers"] = reference.pop("reverse")

    _config_rejected(tmp_path, r"processes\[0\].references\[0\]: unknown keys 'link_event', 'revers'; allowed: "
                               r"catalogue, link_events, o2o, object_attribute, pick, quantity, reverse, tag_along, "
                               r"with$", misspelled)


def test_count_is_no_key_of_a_reference(tmp_path):
    # quantity says how many objects a pick chooses
    def counted(data):
        data["organizations"][0]["processes"][0]["references"][0]["count"] = 2

    _config_rejected(tmp_path, r"processes\[0\].references\[0\]: unknown key 'count'; allowed: ", counted)


def test_a_pick_object(tmp_path):
    def misspelled(data):
        data["organizations"][0]["processes"][0]["references"][0]["pick"] = {"weight": {"c1": 1}}

    _config_rejected(tmp_path, r"processes\[0\].references\[0\].pick: unknown key 'weight'; allowed: new_share, weights$",
                     misspelled)


def test_catalogues_and_references_are_keys_of_a_process_not_of_the_top_level(tmp_path):
    def at_the_top_level(data):
        data["references"] = data["organizations"][0]["processes"][0].pop("references")

    _config_rejected(tmp_path, r"simulation.json: top level: unknown key 'references'; allowed: ", at_the_top_level)


def test_a_reference_names_no_process_its_entry_is_its_process(tmp_path):
    def naming_its_process(data):
        data["organizations"][0]["processes"][0]["references"][0]["process"] = "Sales"

    _config_rejected(tmp_path, r"processes\[0\].references\[0\]: unknown key 'process'; allowed: ", naming_its_process)


# the messages section

def _messages_rejected(section, bpmn_path, message):
    with pytest.raises(InvalidSimScenarioException, match=message):
        parse_messages(section, bpmn_path, {"items", "city"})


def test_a_publish_entry(bpmn):
    _messages_rejected(publish(atributes=["object_id"]), bpmn,
                       r"messages.publish\[0\]: unknown key 'atributes'; allowed: attributes, carry, deadline, "
                       r"event_id, fan_in, fan_out, o2o, qualifier, same, type$")


@pytest.mark.parametrize("section, where", [(publish(count=3), r"messages.publish\[0\]"),
                                            (consume(count=3), r"messages.consume\[0\]")])
def test_count_is_no_key_of_a_message_entry(bpmn, section, where):
    # fan_out says how many messages a publish entry sends
    _messages_rejected(section, bpmn, rf"{where}: unknown key 'count'; allowed: ")


def test_a_consume_entry(bpmn):
    _messages_rejected(consume(colect="all"), bpmn,
                       r"messages.consume\[0\]: unknown key 'colect'; allowed: aggregate, append, capacity, carry, "
                       r"collect, condition, copy, event_id, o2o, qualifier, same, type$")


def test_a_fan_out_object(bpmn):
    _messages_rejected(publish(fan_out={"object_atribute": "items"}), bpmn,
                       r"messages.publish\[0\].fan_out: unknown key 'object_atribute'; allowed: as, list, object_attribute$")


def test_a_collect_object(bpmn):
    _messages_rejected(consume(collect={"vale": 3}), bpmn,
                       r"messages.consume\[0\].collect: unknown key 'vale'; allowed: max, min, object_attribute$")


def test_a_capacity_object(bpmn):
    _messages_rejected(consume(capacity={"atribute": "seats"}), bpmn,
                       r"messages.consume\[0\].capacity: unknown key 'atribute'; allowed: attribute, max, min$")


def test_fan_out_on_a_consume_entry_keeps_its_own_reason(bpmn):
    _messages_rejected(consume(fan_out=2), bpmn, r"fan_out is only for publish entries")


# a process's JSON settings

SALES = "testing_scripts/assets/order_management/sales"


def _settings_rejected(tmp_path, message, change):
    with open(f"{SALES}.json") as file:
        settings = json.load(file)
    change(settings)
    (tmp_path / "sales.json").write_text(json.dumps(settings))
    with pytest.raises(ValueError, match=message):
        load_process(f"{SALES}.bpmn", str(tmp_path / "sales.json"), 1, START)


def test_the_top_level_of_a_processs_settings(tmp_path):
    def misspelled(settings):
        settings["mesages"] = settings.pop("messages")

    _settings_rejected(tmp_path, r"sales.json: top level: unknown key 'mesages'; allowed: arrival_time_calendar, "
                                 r"arrival_time_distribution, .*, messages, .*, object_attributes, .*$", misspelled)


# Prosimos's own names for what the configuration names after objects are no keys of it

def test_total_cases_is_no_key_of_a_process_entry(tmp_path):
    def prosimos_name(data):
        data["organizations"][0]["processes"][0]["total_cases"] = data["organizations"][0]["processes"][0].pop("arrivals")

    _config_rejected(tmp_path, r"processes\[0\]: unknown key 'total_cases'; allowed: arrivals, ", prosimos_name)


def test_carry_links_is_no_key_of_a_process_entry(tmp_path):
    # carry is set on the message entry whose links it passes on
    def on_the_process(data):
        data["organizations"][0]["processes"][0]["carry_links"] = True

    _config_rejected(tmp_path, r"processes\[0\]: unknown key 'carry_links'; allowed: arrivals, ", on_the_process)


def test_case_attribute_is_no_key_of_a_reference(tmp_path):
    def prosimos_name(data):
        reference = data["organizations"][0]["processes"][0]["references"][0]
        reference["case_attribute"] = reference.pop("object_attribute")

    _config_rejected(tmp_path, r"processes\[0\].references\[0\]: unknown key 'case_attribute'; allowed: catalogue, ",
                     prosimos_name)


def test_case_attributes_is_no_section_of_a_processs_settings(tmp_path):
    def prosimos_name(settings):
        settings["case_attributes"] = settings.pop("object_attributes")

    _settings_rejected(tmp_path, r"sales.json: top level: unknown key 'case_attributes'; allowed: ", prosimos_name)


@pytest.mark.parametrize("section, where", [
    (publish(fan_out={"case_attribute": "items"}), r"messages.publish\[0\].fan_out"),
    (consume(collect={"case_attribute": "items"}), r"messages.consume\[0\].collect"),
    (consume(collect={"min": {"case_attribute": "items"}}), r"messages.consume\[0\].collect.min"),
    (consume(condition=[[{"attribute": "order_id", "comparison": "=", "case_attribute": "order_id"}]]),
     r"messages.consume\[0\].condition\[0\]\[0\]"),
])
def test_case_attribute_is_no_key_of_the_messages_section(bpmn, section, where):
    _messages_rejected(section, bpmn, rf"{where}: unknown key 'case_attribute'; allowed: ")


def test_case_id_is_no_reserved_name_of_a_message(bpmn):
    _messages_rejected(publish(attributes=["case_id", "city"]), bpmn,
                       r"messages.publish\[0\]: 'case_id' not declared as an object, global or event attribute")


# duplicate ids

def test_a_pool_of_100000_passes_the_catalogue_check_well_under_a_second():
    assets = "testing_scripts/assets/order_management"
    specs = [ProcessSpec(name, f"{assets}/{name.lower()}.bpmn", f"{assets}/{name.lower()}.json")
             for name in ("Sales", "Warehouse", "Packaging")]

    started = time.perf_counter()
    checked_catalogue(one_organization(specs, {"customer": CataloguePool(100_000, ())}))

    assert time.perf_counter() - started < 0.5
