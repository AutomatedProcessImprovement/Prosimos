import json
import os

from pm4py.objects.log.importer.xes import importer as xes_importer

from bpdfr_simulation_engine.resource_calendar import CalendarFactory


def main():
    log_name = 'purchasing_example'
    out_file_path = "%s.json" % log_name
    log_traces = xes_importer.apply('./../input_files/xes_files/PurchasingExample.xes')

    calendar_factory = CalendarFactory(15)

    for trace in log_traces:
        for event in trace:
            calendar_factory.check_date_time(event['org:resource'], event['concept:name'], event['time:timestamp'])

    calendar_candidates = calendar_factory.build_weekly_calendars(0.1, 0.7, 0.4)
    json_calendar = dict()
    for resource_id in calendar_candidates:
        if calendar_candidates[resource_id] is not None:
            json_calendar[resource_id] = calendar_candidates[resource_id].to_json()
    with open(out_file_path, 'w') as file_writter:
        json.dump(json_calendar, file_writter)

    os._exit(0)


if __name__ == "__main__":
    main()