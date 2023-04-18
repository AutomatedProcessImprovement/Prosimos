import json

from bpdfr_simulation_engine.resource_calendar import parse_datetime
from bpdfr_simulation_engine.simulation_properties_parser \
    import parse_resource_profiles, parse_task_resource_distributions, parse_arrival_branching_probabilities, \
    parse_arrival_calendar, parse_resource_calendars
from fuzzy_engine.fuzzy_calendar import WeeklyFuzzyCalendar, FuzzyModel

granule_units = {'SECONDS': 1 / 60, 'MINUTES': 1, 'HOURS': 60}
int_week_days = {'MONDAY': 0, 'TUESDAY': 1, 'WEDNESDAY': 2, 'THURSDAY': 3, 'FRIDAY': 4, 'SATURDAY': 5, 'SUNDAY': 6}


def parse_json_sim_parameters(json_path, is_fuzzy):
    with open(json_path) as json_file:
        json_data = json.load(json_file)

        resources_map = parse_resource_profiles(json_data["resource_profiles"])
        calendars_map = parse_fuzzy_calendar(json_data) if is_fuzzy \
            else parse_resource_calendars(json_data["resource_calendars"])

        task_resource_distribution = parse_task_resource_distributions(json_data["task_resource_distribution"])

        element_distribution = parse_arrival_branching_probabilities(json_data["arrival_time_distribution"],
                                                                     json_data["gateway_branching_probabilities"])
        arrival_calendar = parse_arrival_calendar(json_data)

        return resources_map, calendars_map, element_distribution, task_resource_distribution, arrival_calendar


def parse_fuzzy_calendar(json_data):
    granule_size = json_data['granule_size']['value'] * granule_units[json_data['granule_size']['time_unit']]
    fuzzy_calendars = dict()
    availability_calendars = json_data['resource_calendars']
    for r_info in availability_calendars:
        fuzzy_model = FuzzyModel(r_info['id'])
        for prob_type in ['availability_probabilities', 'workload_ratio']:
            f_calendar = WeeklyFuzzyCalendar(granule_size)
            avail_probabilities = r_info[prob_type]
            for i_info in avail_probabilities:
                for p_info in i_info['fuzzy_intervals']:
                    f_calendar.add_weekday_intervals(int_week_days[i_info['week_day']],
                                                     parse_datetime(p_info['begin_time'], False),
                                                     parse_datetime(p_info['end_time'], False),
                                                     float(p_info['probability']))
            f_calendar.index_consecutive_boundaries()
            fuzzy_model.update_model(prob_type, f_calendar)
        fuzzy_calendars[r_info['id']] = fuzzy_model

    return fuzzy_calendars
