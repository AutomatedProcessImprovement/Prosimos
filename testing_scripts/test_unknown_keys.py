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
from prosimos.messaging_parser import parse_messages
from prosimos.orchestration.catalogue import CataloguePool, _checked_catalogue
from prosimos.orchestration.config import ProcessSpec, SimulationConfig
from prosimos.orchestration.prosimos_engine import load_process
from testing_scripts.test_catalogue import _config_with
from testing_scripts.test_messaging_parser import bpmn, consume, publish  # noqa: F401 (bpmn is a fixture)

START = pytz.utc.localize(datetime(2024, 1, 1, 9))


def _config_rejected(tmp_path, message, change):
    path = _config_with(tmp_path)
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        SimulationConfig.from_json(path)


# simulation.json

def test_the_top_level(tmp_path):
    def misspelled(data):
        data["sead"] = data.pop("seed")

    _config_rejected(tmp_path, r"simulation.json: top level: unknown key 'sead'; allowed: consumer_groups, processes, "
                               r"seed, start_time$", misspelled)


def test_a_process_entry(tmp_path):
    def misspelled(data):
        data["processes"][2]["object_typ"] = data["processes"][2].pop("object_type")

    _config_rejected(tmp_path, r"processes\[2\]: unknown key 'object_typ'; allowed: arrivals, bpmn_path, "
                               r"carry_links, catalogues, json_path, name, object_type, ocel_attributes, qualifier, "
                               r"qualifier_by_activity, references$",
                     misspelled)


def test_an_object_of_a_listed_catalogue(tmp_path):
    def misspelled(data):
        customers = data["processes"][0]["catalogues"]["customer"]
        customers[2]["attribute"] = {}

    _config_rejected(tmp_path, r"processes\[0\].catalogues.customer\[2\]: unknown key 'attribute'; allowed: attributes, id$",
                     misspelled)


def test_a_pool(tmp_path):
    def misspelled(data):
        data["processes"][0]["catalogues"]["customer"] = {"count": 10, "atributes": []}

    _config_rejected(tmp_path, r"processes\[0\].catalogues.customer: unknown key 'atributes'; allowed: attributes, count$",
                     misspelled)


def test_a_growing_catalogue(tmp_path):
    # a misspelled count would otherwise make the pool a growing catalogue
    def misspelled(data):
        data["processes"][0]["catalogues"]["customer"] = {"cont": 10, "attributes": []}

    _config_rejected(tmp_path, r"processes\[0\].catalogues.customer: unknown key 'cont'; allowed: attributes, count$",
                     misspelled)


def test_a_reference(tmp_path):
    def misspelled(data):
        reference = data["processes"][0]["references"][0]
        reference["link_event"] = reference.pop("link_events")
        reference["revers"] = reference.pop("reverse")

    _config_rejected(tmp_path, r"processes\[0\].references\[0\]: unknown keys 'link_event', 'revers'; allowed: "
                               r"catalogue, count, link_events, o2o, object_attribute, pick, reverse, tag_along, with$",
                     misspelled)


def test_a_pick_object(tmp_path):
    def misspelled(data):
        data["processes"][0]["references"][0]["pick"] = {"weight": {"c1": 1}}

    _config_rejected(tmp_path, r"processes\[0\].references\[0\].pick: unknown key 'weight'; allowed: new_share, weights$",
                     misspelled)


def test_catalogues_and_references_are_keys_of_a_process_not_of_the_top_level(tmp_path):
    def at_the_top_level(data):
        data["references"] = data["processes"][0].pop("references")

    _config_rejected(tmp_path, r"simulation.json: top level: unknown key 'references'; allowed: ", at_the_top_level)


def test_a_reference_names_no_process_its_entry_is_its_process(tmp_path):
    def naming_its_process(data):
        data["processes"][0]["references"][0]["process"] = "Sales"

    _config_rejected(tmp_path, r"processes\[0\].references\[0\]: unknown key 'process'; allowed: ", naming_its_process)


# the messages section

def _messages_rejected(section, bpmn_path, message):
    with pytest.raises(InvalidSimScenarioException, match=message):
        parse_messages(section, bpmn_path, {"items", "city"})


def test_a_publish_entry(bpmn):
    _messages_rejected(publish(atributes=["object_id"]), bpmn,
                       r"messages.publish\[0\]: unknown key 'atributes'; allowed: attributes, count, event_id, o2o, "
                       r"qualifier, type$")


def test_a_consume_entry(bpmn):
    _messages_rejected(consume(colect="all"), bpmn,
                       r"messages.consume\[0\]: unknown key 'colect'; allowed: aggregate, append, capacity, collect, "
                       r"condition, copy, event_id, o2o, qualifier, same, type$")


def test_a_count_object(bpmn):
    _messages_rejected(publish(count={"object_atribute": "items"}), bpmn,
                       r"messages.publish\[0\].count: unknown key 'object_atribute'; allowed: as, list, object_attribute$")


def test_a_collect_object(bpmn):
    _messages_rejected(consume(collect={"vale": 3}), bpmn,
                       r"messages.consume\[0\].collect: unknown key 'vale'; allowed: max, min, object_attribute$")


def test_a_capacity_object(bpmn):
    _messages_rejected(consume(capacity={"atribute": "seats"}), bpmn,
                       r"messages.consume\[0\].capacity: unknown key 'atribute'; allowed: attribute, max, min$")


def test_count_on_a_consume_entry_keeps_its_own_reason(bpmn):
    _messages_rejected(consume(count=2), bpmn, r"count is only for publish entries")


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
        data["processes"][0]["total_cases"] = data["processes"][0].pop("arrivals")

    _config_rejected(tmp_path, r"processes\[0\]: unknown key 'total_cases'; allowed: arrivals, ", prosimos_name)


def test_case_attribute_is_no_key_of_a_reference(tmp_path):
    def prosimos_name(data):
        reference = data["processes"][0]["references"][0]
        reference["case_attribute"] = reference.pop("object_attribute")

    _config_rejected(tmp_path, r"processes\[0\].references\[0\]: unknown key 'case_attribute'; allowed: catalogue, ",
                     prosimos_name)


def test_case_attributes_is_no_section_of_a_processs_settings(tmp_path):
    def prosimos_name(settings):
        settings["case_attributes"] = settings.pop("object_attributes")

    _settings_rejected(tmp_path, r"sales.json: top level: unknown key 'case_attributes'; allowed: ", prosimos_name)


@pytest.mark.parametrize("section, where", [
    (publish(count={"case_attribute": "items"}), r"messages.publish\[0\].count"),
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
    specs = [ProcessSpec(name, f"{assets}/{name.lower()}.bpmn", f"{assets}/{name.lower()}.json",
                         catalogues={"customer": CataloguePool(100_000, ())} if name == "Sales" else {})
             for name in ("Sales", "Warehouse", "Packaging")]

    started = time.perf_counter()
    _checked_catalogue(specs)

    assert time.perf_counter() - started < 0.5
