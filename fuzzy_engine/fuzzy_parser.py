import json
import logging

from bpdfr_simulation_engine.resource_calendar import parse_datetime, RCalendar
from bpdfr_simulation_engine.simulation_properties_parser \
    import parse_resource_profiles, parse_task_resource_distributions, parse_arrival_branching_probabilities, \
    parse_arrival_calendar, parse_resource_calendars
from fuzzy_engine.fuzzy_calendar import WeeklyFuzzyCalendar, FuzzyModel

granule_units = {'SECONDS': 1 / 60, 'MINUTES': 1, 'HOURS': 60}
int_week_days = {'MONDAY': 0, 'TUESDAY': 1, 'WEDNESDAY': 2, 'THURSDAY': 3, 'FRIDAY': 4, 'SATURDAY': 5, 'SUNDAY': 6}


def parse_json_sim_parameters(json_path):
    with open(json_path) as json_file:
        json_data = json.load(json_file)
        model_type = json_data["model_type"]

        resources_map = parse_resource_profiles(json_data["resource_profiles"])
        calendars_map = parse_fuzzy_calendar(json_data) if model_type == "FUZZY"  \
            else parse_resource_calendars(json_data["resource_calendars"])

        task_resource_distribution = parse_task_resource_distributions(json_data["task_resource_distribution"])

        element_distribution = parse_arrival_branching_probabilities(json_data["arrival_time_distribution"],
                                                                     json_data["gateway_branching_probabilities"])
        arrival_calendar = parse_arrival_calendar(json_data)

        return resources_map, calendars_map, element_distribution, \
            task_resource_distribution, arrival_calendar, model_type


def parse_fuzzy_calendar(json_data):
    granule_size = json_data['granule_size']['value'] * granule_units[(json_data['granule_size']['time_unit']).upper()]
    fuzzy_calendars = dict()
    resource_calendars = json_data['resource_calendars']
    for r_info in resource_calendars:
        fuzzy_model = FuzzyModel(r_info['id'])
        for prob_type in ['time_periods', 'workload_ratio']:
            f_calendar = WeeklyFuzzyCalendar(granule_size)
            avail_probabilities = r_info[prob_type]
            for i_info in avail_probabilities:
                fuzzy_intervals = convert_to_fuzzy_time_periods(i_info)
                for p_info in fuzzy_intervals:
                    f_calendar.add_weekday_intervals(int_week_days[p_info['weekDay']],
                                                     parse_datetime(p_info['beginTime'], False),
                                                     parse_datetime(p_info['endTime'], False),
                                                     float(p_info['probability']))
            f_calendar.index_consecutive_boundaries()
            fuzzy_model.update_model(prob_type, f_calendar)
        fuzzy_calendars[r_info['id']] = fuzzy_model
    return fuzzy_calendars

def convert_to_fuzzy_time_periods(time_period):
    from_day = int_week_days[time_period['from']]
    to_day = int_week_days[time_period['to']]

    time_periods = []

    for day in range(from_day, to_day + 1):
        week_day = list(int_week_days.keys())[list(int_week_days.values()).index(day)]
        time_period = {
            'weekDay': week_day,
            'beginTime': time_period['beginTime'],
            'endTime': time_period['endTime'],
            'probability': time_period['probability']
        }
        time_periods.append(time_period)

    return time_periods
