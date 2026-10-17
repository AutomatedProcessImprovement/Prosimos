"""
Unknown keys (docs/orchestrator.md, docs/messaging.md): a key the configuration readers don't know is a load error,
naming the key, where it is and the allowed keys, since ignored, a misspelled key would change the run without any
message. One test per place, each with a misspelled key.
"""
import json
import time

import pytest

from prosimos.exceptions import InvalidSimScenarioException
from prosimos.messaging_parser import parse_messages
from prosimos.orchestrator import CataloguePool, ProcessSpec, SimulationConfig, _checked_catalogue
from testing_scripts.test_catalogue import CONFIG, _config_with
from testing_scripts.test_messaging_parser import bpmn, consume, publish  # noqa: F401 (bpmn is a fixture)


def _config_rejected(tmp_path, message, change):
    with open(CONFIG) as file:
        data = json.load(file)
    change(data)
    with pytest.raises(ValueError, match=message):
        SimulationConfig.from_json(_config_with(tmp_path, **data))


# simulation.json

def test_the_top_level(tmp_path):
    def misspelled(data):
        data["catalog"] = data.pop("catalogue")

    _config_rejected(tmp_path, r"simulation.json: top level: unknown key 'catalog'; allowed: catalogue, "
                               r"consumer_groups, processes, references, seed, start_time$", misspelled)


def test_a_process_entry(tmp_path):
    def misspelled(data):
        data["processes"][1]["object_typ"] = data["processes"][1].pop("object_type")

    _config_rejected(tmp_path, r"processes\[1\]: unknown key 'object_typ'; allowed: bpmn_path, carry_links, "
                               r"json_path, name, object_type, qualifier, qualifier_by_activity, total_cases$",
                     misspelled)


def test_an_object_of_a_listed_catalogue(tmp_path):
    def misspelled(data):
        data["catalogue"]["customers"][2]["attribute"] = data["catalogue"]["customers"][2].pop("attributes")

    _config_rejected(tmp_path, r"catalogue.customers\[2\]: unknown key 'attribute'; allowed: attributes, id$",
                     misspelled)


def test_a_pool(tmp_path):
    def misspelled(data):
        data["catalogue"]["customers"] = {"count": 10, "atributes": []}

    _config_rejected(tmp_path, r"catalogue.customers: unknown key 'atributes'; allowed: attributes, count$", misspelled)


def test_a_growing_catalogue(tmp_path):
    # a misspelled count would otherwise make the pool a growing catalogue
    def misspelled(data):
        data["catalogue"]["customers"] = {"cont": 10, "attributes": []}

    _config_rejected(tmp_path, r"catalogue.customers: unknown key 'cont'; allowed: attributes, count$", misspelled)


def test_a_reference(tmp_path):
    def misspelled(data):
        data["references"][0]["link_event"] = data["references"][0].pop("link_events")
        data["references"][0]["revers"] = data["references"][0].pop("reverse")

    _config_rejected(tmp_path, r"references\[0\]: unknown keys 'link_event', 'revers'; allowed: case_attribute, "
                               r"catalogue, link_events, o2o, pick, process, reverse$", misspelled)


def test_a_pick_object(tmp_path):
    def misspelled(data):
        data["references"][0]["pick"] = {"weight": {"c1": 1}}

    _config_rejected(tmp_path, r"references\[0\].pick: unknown key 'weight'; allowed: new_share, weights$", misspelled)


# the messages section

def _messages_rejected(section, bpmn_path, message):
    with pytest.raises(InvalidSimScenarioException, match=message):
        parse_messages(section, bpmn_path, {"items", "city"})


def test_a_publish_entry(bpmn):
    _messages_rejected(publish(atributes=["case_id"]), bpmn,
                       r"messages.publish\[0\]: unknown key 'atributes'; allowed: attributes, count, event_id, o2o, "
                       r"qualifier, type$")


def test_a_consume_entry(bpmn):
    _messages_rejected(consume(colect="all"), bpmn,
                       r"messages.consume\[0\]: unknown key 'colect'; allowed: aggregate, append, capacity, collect, "
                       r"condition, copy, event_id, o2o, qualifier, type$")


def test_a_count_object(bpmn):
    _messages_rejected(publish(count={"case_atribute": "items"}), bpmn,
                       r"messages.publish\[0\].count: unknown key 'case_atribute'; allowed: as, case_attribute, list, "
                       r"value$")


def test_a_collect_object(bpmn):
    _messages_rejected(consume(collect={"vale": 3}), bpmn,
                       r"messages.consume\[0\].collect: unknown key 'vale'; allowed: case_attribute, max, min, value$")


def test_a_capacity_object(bpmn):
    _messages_rejected(consume(capacity={"atribute": "seats"}), bpmn,
                       r"messages.consume\[0\].capacity: unknown key 'atribute'; allowed: attribute, value$")


def test_count_on_a_consume_entry_keeps_its_own_reason(bpmn):
    _messages_rejected(consume(count={"value": 2}), bpmn, r"count is only for publish entries")


# duplicate ids

def test_a_pool_of_100000_passes_the_catalogue_check_well_under_a_second():
    assets = "testing_scripts/assets/order_management"
    specs = [ProcessSpec(name, f"{assets}/{name.lower()}.bpmn", f"{assets}/{name.lower()}.json")
             for name in ("Sales", "Customer", "Warehouse", "Packaging")]

    started = time.perf_counter()
    _checked_catalogue(specs, [spec.name for spec in specs], {"customers": CataloguePool(100_000, ())}, ())

    assert time.perf_counter() - started < 0.5
