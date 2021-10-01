import json
import xml.etree.ElementTree as ET

from bpdfr_simulation_engine.control_flow_manager import BPMNGraph, ElementInfo, BPMN
from bpdfr_simulation_engine.resource_calendar import RCalendar, convert_time_unit_from_to, convertion_table, to_seconds
from bpdfr_simulation_engine.resource_profile import ResourceProfile
from bpdfr_simulation_engine.probability_distributions import *

bpmn_schema_url = 'http://www.omg.org/spec/BPMN/20100524/MODEL'
bpmn_element_ns = {'xmlns': bpmn_schema_url}


def parse_calendar_from_json(res_calendar_path):
    with open(res_calendar_path) as json_file:
        json_data = json.load(json_file)
        resources_map = dict()
        calendars_map = dict()
        for r_id in json_data:
            calendar_id = "%s_timetable" % r_id
            resources_map[r_id] = ResourceProfile(r_id, r_id, calendar_id, 1.0)
            r_calendar = RCalendar(calendar_id)
            for c_item in json_data[r_id]:
                r_calendar.add_calendar_item(c_item['from'], c_item['to'], c_item['beginTime'], c_item['endTime'])
            r_calendar.compute_cumulative_durations()
            calendars_map[r_calendar.calendar_id] = r_calendar
        return resources_map, calendars_map


def parse_simulation_parameters(arrival_dist, gateway_prob, task_res_dist):
    element_distribution = dict()

    with open(task_res_dist) as json_file:
        task_resource_distribution = json.load(json_file)

    with open(arrival_dist) as json_file:
        element_distribution['arrivalTime'] = json.load(json_file)

    with open(gateway_prob) as json_file:
        data = json.load(json_file)

        for gateway_id in data:
            probability_list = list()
            out_arc = list()
            for flow_arc in data[gateway_id]:
                out_arc.append(flow_arc)
                probability_list.append(data[gateway_id][flow_arc])
            element_distribution[gateway_id] = Choice(out_arc, probability_list)

    return element_distribution, task_resource_distribution


def parse_simulation_model(bpmn_path):
    tree = ET.parse(bpmn_path)
    root = tree.getroot()

    to_extract = {'xmlns:task': BPMN.TASK,
                  'xmlns:startEvent': BPMN.START_EVENT,
                  'xmlns:endEvent': BPMN.END_EVENT,
                  'xmlns:exclusiveGateway': BPMN.EXCLUSIVE_GATEWAY,
                  'xmlns:parallelGateway': BPMN.PARALLEL_GATEWAY,
                  'xmlns:inclusiveGateway': BPMN.INCLUSIVE_GATEWAY}

    bpmn_graph = BPMNGraph()
    for process in root.findall('xmlns:process', bpmn_element_ns):
        for xmlns_key in to_extract:
            for bpmn_element in process.findall(xmlns_key, bpmn_element_ns):
                name = bpmn_element.attrib["name"] \
                    if "name" in bpmn_element.attrib and len(bpmn_element.attrib["name"]) > 0 \
                    else bpmn_element.attrib["id"]
                bpmn_graph.add_bpmn_element(bpmn_element.attrib["id"],
                                            ElementInfo(to_extract[xmlns_key], bpmn_element.attrib["id"], name))
        for flow_arc in process.findall('xmlns:sequenceFlow', bpmn_element_ns):
            bpmn_graph.add_flow_arc(flow_arc.attrib["id"], flow_arc.attrib["sourceRef"], flow_arc.attrib["targetRef"])
    bpmn_graph.encode_or_join_predecesors()
    return bpmn_graph
