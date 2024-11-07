import datetime
import json
import xml.etree.ElementTree as ET

from dateutil import parser
from numpy import exp, log, sqrt
from pix_framework.discovery.resource_calendar_and_performance.crisp.resource_calendar import RCalendar
from pix_framework.statistics.distribution import DurationDistribution

from prosimos.fuzzy_engine.fuzzy_calendar import FuzzyModel, WeeklyFuzzyCalendar
from prosimos.batch_processing_parser import BatchProcessingParser
from prosimos.case_attributes import AllCaseAttributes, CaseAttribute
from prosimos.control_flow_manager import BPMN, EVENT_TYPE, BPMNGraph, ElementInfo
from prosimos.histogram_distribution import HistogramDistribution
from prosimos.multitasking.multitasking_struct import MultiTaskDS
from prosimos.prioritisation import AllPriorityRules
from prosimos.prioritisation_parser import PrioritisationParser
from prosimos.probability_distributions import Choice
from prosimos.resource_profile import PoolInfo, ResourceProfile
from prosimos.branch_condition_parser import BranchConditionParser
from prosimos.branch_condition_rules import AllBranchConditionRules
from prosimos.event_attributes import AllEventAttributes
from prosimos.event_attributes_parser import EventAttributesParser
from prosimos.gateway_condition_choice import GatewayConditionChoice
from prosimos.global_attributes_parser import GlobalAttributesParser
from prosimos.global_attributes import AllGlobalAttributes
from prosimos.all_attributes import AllAttributes
from prosimos.warning_logger import warning_logger

bpmn_schema_url = "http://www.omg.org/spec/BPMN/20100524/MODEL"
simod_ns = {"qbp": "http://www.qbp-simulator.com/Schema201212"}
bpmn_element_ns = {"xmlns": bpmn_schema_url}

EVENT_DISTRIBUTION_SECTION = "event_distribution"
BATCH_PROCESSING_SECTION = "batch_processing"
CASE_ATTRIBUTES_SECTION = "case_attributes"
PRIORITISATION_RULES_SECTION = "prioritisation_rules"
ARRIVAL_TIME_CALENDAR = "arrival_time_calendar"
RESOURCE_CALENDARS = "resource_calendars"
BRANCH_RULES = "branch_rules"
EVENT_ATTRIBUTES = "event_attributes"
GLOBAL_ATTRIBUTES = "global_attributes"
GATEWAY_EXECUTION_LIMIT = "gateway_execution_limit"
MULTITASKING_SECTION = "multitask"

DEFAULT_GATEWAY_EXECUTION_LIMIT = 1000

granule_units = {"SECONDS": 1 / 60, "MINUTES": 1, "HOURS": 60}
int_week_days = {"MONDAY": 0, "TUESDAY": 1, "WEDNESDAY": 2, "THURSDAY": 3, "FRIDAY": 4, "SATURDAY": 5, "SUNDAY": 6}


def parse_json_sim_parameters(json_path):
    with open(json_path) as json_file:
        json_data = json.load(json_file)
        model_type = json_data["model_type"] if "model_type" in json_data else "CRSIP"

        resources_map, res_pool = parse_resource_profiles(json_data["resource_profiles"])
        # calendars_map = parse_resource_calendars(json_data[RESOURCE_CALENDARS])

        calendars_map = (
            parse_fuzzy_calendar(json_data)
            if model_type == "FUZZY"
            else parse_resource_calendars(json_data[RESOURCE_CALENDARS])
        )

        task_resource_distribution = parse_task_resource_distributions(
            json_data["task_resource_distribution"], res_pool
        )

        branch_rules = (
            BranchConditionParser(json_data[BRANCH_RULES]).parse()
            if BRANCH_RULES in json_data
            else AllBranchConditionRules([])
        )

        element_distribution = parse_arrival_branching_probabilities(
            json_data["arrival_time_distribution"],
            json_data["gateway_branching_probabilities"]
        )

        gateway_conditions = parse_gateway_conditions(json_data["gateway_branching_probabilities"], 
                                                      branch_rules)

        arrival_calendar = parse_arrival_calendar(json_data)
        event_distibution = (
            parse_event_distribution(json_data[EVENT_DISTRIBUTION_SECTION])
            if EVENT_DISTRIBUTION_SECTION in json_data
            else dict()
        )
        batch_processing = (
            BatchProcessingParser(json_data[BATCH_PROCESSING_SECTION]).parse()
            if BATCH_PROCESSING_SECTION in json_data
            else dict()
        )
        case_attributes = (
            parse_case_attr(json_data[CASE_ATTRIBUTES_SECTION])
            if CASE_ATTRIBUTES_SECTION in json_data
            else AllCaseAttributes([])
        )
        event_attributes = (
            EventAttributesParser(json_data[EVENT_ATTRIBUTES]).parse()
            if EVENT_ATTRIBUTES in json_data
            else AllEventAttributes({})
        )

        global_attributes = (
            GlobalAttributesParser(json_data[GLOBAL_ATTRIBUTES]).parse()
            if GLOBAL_ATTRIBUTES in json_data
            else AllGlobalAttributes({})
        )

        all_attributes = AllAttributes(global_attributes, case_attributes, event_attributes)

        prioritisation_rules = (
            PrioritisationParser(json_data[PRIORITISATION_RULES_SECTION]).parse()
            if PRIORITISATION_RULES_SECTION in json_data
            else AllPriorityRules([])
        )

        gateway_execution_limit = json_data[GATEWAY_EXECUTION_LIMIT] \
            if GATEWAY_EXECUTION_LIMIT in json_data \
            else DEFAULT_GATEWAY_EXECUTION_LIMIT

        multitasking_info = parse_multitasking_model(json_data[MULTITASKING_SECTION], task_resource_distribution) \
            if MULTITASKING_SECTION in json_data \
            else None

        return (
            resources_map,
            calendars_map,
            element_distribution,
            task_resource_distribution,
            arrival_calendar,
            event_distibution,
            batch_processing,
            prioritisation_rules,
            branch_rules,
            gateway_conditions,
            all_attributes,
            gateway_execution_limit,
            model_type,
            multitasking_info
        )


def parse_multitasking_model(json_data, task_resource_distribution):
    if json_data["type"] == "local":
        return _parse_local_multitasking(json_data, task_resource_distribution)
    elif json_data["type"] == "global":
        return _parse_global_multitasking(json_data, task_resource_distribution)
    return None


def _parse_global_multitasking(json_data, task_res_distr):
    multi_info = MultiTaskDS(json_data["type"])
    for res_info in json_data["values"]:
        r_id = res_info["resource_id"]
        multi_info.update_expected_workload(r_id, res_info["r_workload"])
        multi_info.register_resource(r_id)
        for mt_info in res_info["multitask_info"]:
            multi_info.register_multitasks(r_id, mt_info["parallel_tasks"], mt_info["probability"])
    # multi_info.init_relative_workload(task_res_distr)
    return multi_info


def _parse_local_multitasking(json_data, task_res_distr):
    multi_info = MultiTaskDS(
        json_data["type"],
        60
    )
    # multi_info = MultiTaskDS(
    #     json_data["type"],
    #     json_data["granule_size"]["value"] * granule_units[(json_data["granule_size"]["time_unit"]).upper()]
    # )
    for res_info in json_data["values"]:
        r_id = res_info["resource_id"]
        multi_info.register_resource(r_id)
        multi_info.update_expected_workload(r_id, res_info["r_workload"])
        for wd_info in res_info["weekly_probability"]:
            for gr_info in wd_info:
                time_periods = convert_to_fuzzy_time_periods(gr_info)
                for p_info in time_periods:
                    for mt_info in gr_info["multitask_info"]:
                        multi_info.register_local_multitasks(r_id,
                                                             int_week_days[p_info["weekDay"]],
                                                             parse_datetime(p_info["beginTime"], False),
                                                             parse_datetime(p_info["endTime"], False),
                                                             mt_info["parallel_tasks"],
                                                             mt_info["probability"])
    # multi_info.init_relative_workload(task_res_distr)
    return multi_info


def parse_fuzzy_calendar(json_data):
    granule_size = json_data["granule_size"]["value"] * granule_units[(json_data["granule_size"]["time_unit"]).upper()]
    fuzzy_calendars = dict()
    resource_calendars = json_data["resource_calendars"]
    for r_info in resource_calendars:
        fuzzy_model = FuzzyModel(r_info["id"])
        for prob_type in ["time_periods", "workload_ratio"]:
            f_calendar = WeeklyFuzzyCalendar(granule_size)
            avail_probabilities = r_info[prob_type]
            for i_info in avail_probabilities:
                fuzzy_intervals = convert_to_fuzzy_time_periods(i_info)
                for p_info in fuzzy_intervals:
                    f_calendar.add_weekday_intervals(
                        int_week_days[p_info["weekDay"]],
                        parse_datetime(p_info["beginTime"], False),
                        parse_datetime(p_info["endTime"], False),
                        float(p_info["probability"]),
                    )
            f_calendar.index_consecutive_boundaries()
            fuzzy_model.update_model(prob_type, f_calendar)
        fuzzy_calendars[r_info["id"]] = fuzzy_model
    return fuzzy_calendars


def convert_to_fuzzy_time_periods(time_period):
    from_day = int_week_days[time_period["from"]]
    to_day = int_week_days[time_period["to"]]

    time_periods = []

    for day in range(from_day, to_day + 1):
        week_day = list(int_week_days.keys())[list(int_week_days.values()).index(day)]
        time_period = {
            "weekDay": week_day,
            "beginTime": time_period["beginTime"],
            "endTime": time_period["endTime"],
            "probability": time_period["probability"] if "probability" in time_period else 0.0,
        }
        time_periods.append(time_period)

    return time_periods


# def parse_pool_info(json_data, resources_map):
#     for pool_id in json_data:
#         pool_name = json_data[pool_id]["name"]
#         for res_info in json_data[pool_id]["resource_list"]:
#             r_id = res_info["id"]
#             resources_map[r_id].pool_info = PoolInfo(pool_id, pool_name)
#             resources_map[r_id].resource_name = res_info["name"]
#             resources_map[r_id].cost_per_hour = float(res_info["cost_per_hour"])
#             resources_map[r_id].resource_amount = int(res_info["amount"])


def parse_arrival_calendar(json_data):
    arrival_calendar = None
    if ARRIVAL_TIME_CALENDAR in json_data:
        arrival_calendar = RCalendar("arrival_time_calendar")
        for c_item in json_data[ARRIVAL_TIME_CALENDAR]:
            arrival_calendar.add_calendar_item(c_item["from"], c_item["to"], c_item["beginTime"], c_item["endTime"])
    return arrival_calendar


def parse_resource_profiles(json_data):
    resources_map = dict()
    resource_pool = dict()
    for pool_entry in json_data:
        for r_info in pool_entry["resource_list"]:
            r_id = r_info["id"]
            r_count = int(r_info["amount"])
            resource_pool[r_id] = list()
            for i in range(0, r_count):
                r_i = "%s_%d" % (r_id, i) if r_count > 1 else r_id
                name = "%s_%d" % (r_info["name"], i) if r_count > 1 else r_info["name"]
                resource_pool[r_id].append(r_i)
                resources_map[r_i] = ResourceProfile(r_i, name, r_info["calendar"], float(r_info["cost_per_hour"]))
                resources_map[r_i].resource_amount = 1
                resources_map[r_i].pool_info = PoolInfo(pool_entry["id"], pool_entry["name"])
    return resources_map, resource_pool


def parse_resource_calendars(json_data):
    calendars_info = dict()
    for c_info in json_data:
        r_calendar = RCalendar(c_info["id"])
        for c_item in c_info["time_periods"]:
            r_calendar.add_calendar_item(c_item["from"], c_item["to"], c_item["beginTime"], c_item["endTime"])
        r_calendar.compute_cumulative_durations()
        calendars_info[r_calendar.calendar_id] = r_calendar
    return calendars_info


def parse_task_resource_distributions(json_data, res_pool):
    task_resource_distribution = dict()
    for perf_info in json_data:
        t_id = perf_info["task_id"]
        if t_id not in task_resource_distribution:
            task_resource_distribution[t_id] = dict()
        for r_info in perf_info["resources"]:
            for r_id in res_pool[r_info["resource_id"]]:
                task_resource_distribution[t_id][r_id] = DurationDistribution.from_dict(r_info)

    return task_resource_distribution


def parse_event_distribution(event_json_data):
    """
    Parse "event_distribution" section of json data
    """
    event_distibution = dict()

    for event_info in event_json_data:
        e_id = event_info["event_id"]

        if e_id not in event_distibution:
            event_distibution[e_id] = DurationDistribution.from_dict(event_info)

    return event_distibution


def parse_case_attr(json_data) -> AllCaseAttributes:
    case_attributes = []
    for curr_case_attr in json_data:
        case_attr = CaseAttribute(curr_case_attr["name"], curr_case_attr["type"], curr_case_attr["values"])
        case_attributes.append(case_attr)

    return AllCaseAttributes(case_attributes)


# def parse_calendar_from_json(json_data):
#     resources_map = dict()
#     calendars_map = dict()
#     for r_id in json_data:
#         calendar_id = "%s_timetable" % r_id
#         resources_map[r_id] = ResourceProfile(r_id, r_id, calendar_id, 1.0)
#         r_calendar = RCalendar(calendar_id)
#         for c_item in json_data[r_id]:
#             r_calendar.add_calendar_item(c_item['from'], c_item['to'], c_item['beginTime'], c_item['endTime'])
#         r_calendar.compute_cumulative_durations()
#         calendars_map[r_calendar.calendar_id] = r_calendar
#     return resources_map, calendars_map

def parse_gateway_conditions(gateway_json, branch_rules):
    gateway_conditions = dict()

    for g_info in gateway_json:
        g_id = g_info["gateway_id"]
        gateway_rules = list()
        out_arc = list()
        missing_conditions = list()

        for prob_info in g_info["probabilities"]:
            if "condition_id" in prob_info:
                out_arc.append(prob_info["path_id"])
                curr_branch_rules = branch_rules.get_branch_condition_by_id(prob_info["condition_id"])
                gateway_rules.append(curr_branch_rules)
            else:
                missing_conditions.append(prob_info["path_id"])

        if len(prob_info) > 0:
            gateway_conditions[g_id] = GatewayConditionChoice(out_arc, gateway_rules)

        if len(missing_conditions) > 0 and len(gateway_rules) != len(g_info["probabilities"]):
            warning_logger.add_warning(f"Gateway {g_id} is using conditions, but some are missing. Flows without conditions: {', '.join(missing_conditions)}")

    return gateway_conditions


def parse_arrival_branching_probabilities(arrival_json, gateway_json):
    element_distribution = dict()

    dist_name = arrival_json["distribution_name"]
    if dist_name == "histogram_sampling":
        # Custom distribution: we expect a list of inter-arrival interval values (floats),
        # prosimos will take randomly a value from this list each time it needs a new
        # observation, so the output will follow (if the sample is big enough) the same
        # "unknown distribution" than the specified data.
        element_distribution["arrivalTime"] = HistogramDistribution.from_dict(arrival_json)
    else:
        # handling all other types apart from "histogram_sampling"
        # since end-user provides user-friendly parameters,
        # we transform them to ones suitable for being used with Scipy library
        element_distribution["arrivalTime"] = DurationDistribution.from_dict(arrival_json)

    for g_info in gateway_json:
        g_id = g_info["gateway_id"]
        probability_list = list()
        out_arc = list()

        total_probability = sum([float(prob_info.get("value", 0)) for prob_info in g_info["probabilities"]])
        missing_probability = 1 - total_probability
        missing_values_count = sum(1 for prob_info in g_info["probabilities"] if "value" not in prob_info)
        value_to_assign = missing_probability / missing_values_count if missing_values_count else 0
        auto_assigned = {}

        for prob_info in g_info["probabilities"]:
            out_arc.append(prob_info["path_id"])
            if "value" not in prob_info:
                prob_info["value"] = value_to_assign
                auto_assigned[prob_info["path_id"]] = value_to_assign

            probability_list.append(float(prob_info["value"]))

        if auto_assigned:
            auto_assigned_flows = ", ".join([f"{key}: {value}" for key, value in auto_assigned.items()])
            warning_logger.add_warning(f"Gateway {g_id} has auto-assigned probabilities for flows: {auto_assigned_flows}")

        element_distribution[g_id] = Choice(out_arc, probability_list)
    return element_distribution


def parse_simulation_model(bpmn_path):
    tree = ET.parse(bpmn_path)
    root = tree.getroot()

    to_extract = {
        "xmlns:task": BPMN.TASK,
        "xmlns:startEvent": BPMN.START_EVENT,
        "xmlns:endEvent": BPMN.END_EVENT,
        "xmlns:exclusiveGateway": BPMN.EXCLUSIVE_GATEWAY,
        "xmlns:parallelGateway": BPMN.PARALLEL_GATEWAY,
        "xmlns:inclusiveGateway": BPMN.INCLUSIVE_GATEWAY,
        "xmlns:eventBasedGateway": BPMN.EVENT_BASED_GATEWAY,
        "xmlns:intermediateCatchEvent": BPMN.INTERMEDIATE_EVENT,
    }

    bpmn_graph = BPMNGraph()
    elements_map = dict()
    for process in root.findall("xmlns:process", bpmn_element_ns):
        for xmlns_key in to_extract:
            for bpmn_element in process.findall(xmlns_key, bpmn_element_ns):
                name = (
                    bpmn_element.attrib["name"]
                    if "name" in bpmn_element.attrib and len(bpmn_element.attrib["name"]) > 0
                    else bpmn_element.attrib["id"]
                )
                elem_general_type: BPMN = to_extract[xmlns_key]

                event_type = (
                    _get_event_type_from_element(name, bpmn_element) if BPMN.is_event(elem_general_type) else None
                )
                e_info = ElementInfo(elem_general_type, bpmn_element.attrib["id"], name, event_type)

                bpmn_graph.add_bpmn_element(bpmn_element.attrib["id"], e_info)
                elements_map[e_info.id] = {"in": 0, "out": 0, "info": e_info}

        # Counting incoming/outgoing flow arcs to handle cases of multiple in/out arcs simultaneously
        pending_flow_arcs = list()
        for flow_arc in process.findall("xmlns:sequenceFlow", bpmn_element_ns):
            # Fixing the case in which a task may have multiple incoming/outgoing flow-arcs
            pending_flow_arcs.append(flow_arc)
            if flow_arc.attrib["sourceRef"] in elements_map:
                elements_map[flow_arc.attrib["sourceRef"]]["out"] += 1
            if flow_arc.attrib["targetRef"] in elements_map:
                elements_map[flow_arc.attrib["targetRef"]]["in"] += 1
            # bpmn_graph.add_flow_arc(flow_arc.attrib["id"], flow_arc.attrib["sourceRef"], flow_arc.attrib["targetRef"])

        # Adding fake gateways for tasks with multiple incoming/outgoing flow arcs
        join_gateways = dict()
        split_gateways = dict()
        for t_id in elements_map:
            e_info = elements_map[t_id]["info"]
            if e_info.type is BPMN.TASK:
                if elements_map[t_id]["in"] > 1:
                    _add_fake_gateway(
                        bpmn_graph,
                        "xor_join_%s" % t_id,
                        BPMN.EXCLUSIVE_GATEWAY,
                        t_id,
                        join_gateways,
                    )
                if elements_map[t_id]["out"] > 1:
                    _add_fake_gateway(
                        bpmn_graph,
                        "and_split_%s" % t_id,
                        BPMN.PARALLEL_GATEWAY,
                        t_id,
                        split_gateways,
                        False,
                    )
            elif e_info.type is BPMN.END_EVENT:
                if elements_map[t_id]["in"] > 1:
                    _add_fake_gateway(
                        bpmn_graph,
                        "or_join_%s" % t_id,
                        BPMN.INCLUSIVE_GATEWAY,
                        t_id,
                        join_gateways,
                    )
            elif e_info.is_gateway():
                if elements_map[t_id]["in"] > 1 and elements_map[t_id]["out"] > 1:
                    _add_fake_gateway(bpmn_graph, "join_%s" % t_id, e_info.type, t_id, join_gateways)

        for flow_arc in pending_flow_arcs:
            source_id = flow_arc.attrib["sourceRef"]
            target_id = flow_arc.attrib["targetRef"]
            if source_id in split_gateways:
                source_id = split_gateways[source_id]
            if target_id in join_gateways:
                target_id = join_gateways[target_id]
            bpmn_graph.add_flow_arc(flow_arc.attrib["id"], source_id, target_id)

    bpmn_graph.encode_or_join_predecesors()
    bpmn_graph.validate_model()
    return bpmn_graph


def _get_event_type_from_element(name: str, bpmn_element):
    # children = bpmn_element.getchildren()
    children = list(bpmn_element)

    for child in children:
        if "EventDefinition" in child.tag:
            # tag example: '{http://www.omg.org/spec/BPMN/20100524/MODEL}timerEventDefinition'
            type_name = child.tag.split("}")[1]
            switcher = {
                "timerEventDefinition": EVENT_TYPE.TIMER,
                "messageEventDefinition": EVENT_TYPE.MESSAGE,
                "linkEventDefinition": EVENT_TYPE.LINK,
                "signalEventDefinition": EVENT_TYPE.SIGNAL,
                "terminateEventDefinition": EVENT_TYPE.TERMINATE,
            }

            event_type = switcher.get(type_name, EVENT_TYPE.UNDEFINED)
            if event_type == EVENT_TYPE.UNDEFINED:
                print(f"WARNING: {name} event has an undefined event type")

            return event_type


def _add_fake_gateway(bpmn_graph, g_id, g_type, t_id, e_map, in_front=True):
    bpmn_graph.add_bpmn_element(g_id, ElementInfo(g_type, g_id, g_id, None))
    if in_front:
        bpmn_graph.add_flow_arc("%s_%s" % (g_id, t_id), g_id, t_id)
    else:
        bpmn_graph.add_flow_arc("%s_%s" % (t_id, g_id), t_id, g_id)
    e_map[t_id] = g_id


def parse_qbp_simulation_process(qbp_bpmn_path, out_file):
    tree = ET.parse(qbp_bpmn_path)
    root = tree.getroot()
    simod_root = root.find("qbp:processSimulationInfo", simod_ns)
    if simod_root is None:
        print("PARSING ABORTED: Input BPMN model is not a simulation model, i.e., simulation parameters are missing.")
        return

    # 1. Extracting gateway branching probabilities
    gateways_branching = dict()
    reverse_map = dict()
    for process in root.findall("xmlns:process", bpmn_element_ns):
        for xmlns_key in ["xmlns:exclusiveGateway", "xmlns:inclusiveGateway"]:
            for bpmn_element in process.findall(xmlns_key, bpmn_element_ns):
                if bpmn_element.attrib["gatewayDirection"] == "Diverging":
                    gateways_branching[bpmn_element.attrib["id"]] = dict()
                    for out_flow in bpmn_element.findall("xmlns:outgoing", bpmn_element_ns):
                        arc_id = out_flow.text.strip()
                        gateways_branching[bpmn_element.attrib["id"]][arc_id] = 0
                        reverse_map[arc_id] = bpmn_element.attrib["id"]
    for flow_prob in simod_root.find("qbp:sequenceFlows", simod_ns).findall("qbp:sequenceFlow", simod_ns):
        flow_id = flow_prob.attrib["elementId"]
        gateways_branching[reverse_map[flow_id]][flow_id] = flow_prob.attrib["executionProbability"]

    # 2. Extracting Resource Calendars
    resource_pools = dict()
    calendars_map = dict()
    bpmn_calendars = simod_root.find("qbp:timetables", simod_ns)
    arrival_calendar_id = None

    for calendar_info in bpmn_calendars:
        calendar_id = calendar_info.attrib["id"]
        if calendar_id not in calendars_map:
            calendars_map[calendar_id] = list()

        time_tables = calendar_info.find("qbp:rules", simod_ns).findall("qbp:rule", simod_ns)
        if "ARRIVAL_CALENDAR" in calendar_id or (arrival_calendar_id is None and "DEFAULT_TIMETABLE" in calendar_id):
            arrival_calendar_id = calendar_id
        for time_table in time_tables:
            calendars_map[calendar_id].append(
                {
                    "from": time_table.attrib["fromWeekDay"],
                    "to": time_table.attrib["toWeekDay"],
                    "beginTime": format_date(time_table.attrib["fromTime"]),
                    "endTime": format_date(time_table.attrib["toTime"]),
                }
            )

    # 3. Extracting Arrival time distribution
    arrival_time_dist = extract_dist_params(simod_root.find("qbp:arrivalRateDistribution", simod_ns))

    # 4. Extracting task-resource duration distributions
    bpmn_resources = simod_root.find("qbp:resources", simod_ns)
    simod_elements = simod_root.find("qbp:elements", simod_ns)
    pools_json = dict()

    resource_calendars = dict()
    for resource in bpmn_resources:
        pools_json[resource.attrib["id"]] = {
            "name": resource.attrib["name"],
            "resource_list": list(),
        }
        resource_pools[resource.attrib["id"]] = list()
        calendar_id = resource.attrib["timetableId"]
        for i in range(1, int(resource.attrib["totalAmount"]) + 1):
            nr_id = "%s_%d" % (resource.attrib["id"], i)
            pools_json[resource.attrib["id"]]["resource_list"].append(
                {
                    "id": nr_id,
                    "name": "%s_%d" % (resource.attrib["name"], i),
                    "cost_per_hour": resource.attrib["costPerHour"],
                    "amount": 1,
                }
            )
            resource_pools[resource.attrib["id"]].append(nr_id)
            resource_calendars[nr_id] = calendars_map[calendar_id]

    task_resource_dist = dict()
    for e_inf in simod_elements:
        task_id = e_inf.attrib["elementId"]
        rpool_id = e_inf.find("qbp:resourceIds", simod_ns).find("qbp:resourceId", simod_ns).text
        dist_info = e_inf.find("qbp:durationDistribution", simod_ns)

        t_dist = extract_dist_params(dist_info)
        if task_id not in task_resource_dist:
            task_resource_dist[task_id] = dict()
        for rp_id in resource_pools[rpool_id]:
            task_resource_dist[task_id][rp_id] = t_dist

    # 5.Saving all in a single JSON file

    to_save = {
        "resource_profiles": pools_json,
        "arrival_time_distribution": arrival_time_dist,
        "arrival_time_calendar": calendars_map[arrival_calendar_id],
        "gateway_branching_probabilities": gateways_branching,
        "task_resource_distribution": task_resource_dist,
        "resource_calendars": resource_calendars,
    }
    with open(out_file, "w") as file_writter:
        json.dump(to_save, file_writter)


def format_date(date_str):
    date_splt = date_str.split("+")
    if len(date_splt) == 2 and date_splt[1] == "00:00":
        return date_splt[0]
    return date_str


def extract_dist_params(dist_info):
    # time_unit = dist_info.find("qbp:timeUnit", simod_ns).text
    # The time_tables produced by bimp always have the parameters in seconds, although it shouws other time units in
    # the XML file.
    dist_params = {
        "mean": float(dist_info.attrib["mean"]),
        "arg1": float(dist_info.attrib["arg1"]),
        "arg2": float(dist_info.attrib["arg2"]),
    }
    dist_name = dist_info.attrib["type"].upper()
    if dist_name == "EXPONENTIAL":
        # input: loc = 0, scale = mean
        return {
            "distribution_name": "expon",
            "distribution_params": [0, dist_params["arg1"]],
        }
    if dist_name == "NORMAL":
        # input: loc = mean, scale = standard deviation
        return {
            "distribution_name": "norm",
            "distribution_params": [dist_params["mean"], dist_params["arg1"]],
        }
    if dist_name == "FIXED":
        return {
            "distribution_name": "fix",
            "distribution_params": [dist_params["mean"], 0, 1],
        }
    if dist_name == "UNIFORM":
        # input: loc = from, scale = to - from
        return {
            "distribution_name": "uniform",
            "distribution_params": [
                dist_params["arg1"],
                dist_params["arg2"] - dist_params["arg1"],
            ],
        }
    if dist_name == "GAMMA":
        # input: shape, loc=0, scale
        mean, variance = dist_params["mean"], dist_params["arg1"]
        return {
            "distribution_name": "gamma",
            "distribution_params": [pow(mean, 2) / variance, 0, variance / mean],
        }
    if dist_name == "TRIANGULAR":
        # input: c = mode, loc = min, scale = max - min
        return {
            "distribution_name": "triang",
            "distribution_params": [
                dist_params["mean"],
                dist_params["arg1"],
                dist_params["arg2"] - dist_params["arg1"],
            ],
        }
    if dist_name == "LOGNORMAL":
        mean_2 = dist_params["mean"] ** 2
        variance = dist_params["arg1"]
        phi = sqrt([variance + mean_2])[0]
        mu = log(mean_2 / phi)
        sigma = sqrt([log(phi**2 / mean_2)])[0]

        # input: s = sigma = standard deviation, loc = 0, scale = exp(mu)
        return {
            "distribution_name": "lognorm",
            "distribution_params": [sigma, 0, exp(mu)],
        }
    return None


def parse_datetime(time, has_date):
    time_formats = (
        ["%H:%M:%S.%f", "%H:%M", "%I:%M%p", "%H:%M:%S", "%I:%M:%S%p"]
        if not has_date
        else [
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%b %d %Y %I:%M%p",
            "%b %d %Y at %I:%M%p",
            "%B %d, %Y, %H:%M:%S",
            "%a,%d/%m/%y,%I:%M%p",
            "%a, %d %B, %Y",
            "%Y-%m-%dT%H:%M:%SZ",
        ]
    )
    try:
        return parser.parse(time)
    except:
        for time_format in time_formats:
            try:
                return datetime.datetime.strptime(time, time_format)
            except ValueError:
                pass
    raise ValueError
