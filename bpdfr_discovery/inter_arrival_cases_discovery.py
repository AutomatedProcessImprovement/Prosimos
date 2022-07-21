import enum
from datetime import timedelta

import pandas as pd
import os
# from prophet import Prophet
# import matplotlib.pyplot as plt
# from neuralprophet import NeuralProphet

from bpdfr_discovery.emd_metric import discretize_to_minutes, discretize_to_hour

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from bpdfr_discovery.log_parser import transform_xes_to_csv
from testing_scripts.bpm_2022_testing_files import experiment_logs, process_files


class DTimeInfo:
    def __init__(self):
        self.year_freq = dict()
        self.month_freq = dict()
        self.week_freq = dict()
        self.weekday_freq = dict()
        self.day_freq = dict()
        self.hour_freq = dict()

    def check_datetime(self, date_time):
        _update_freq_dict(self.year_freq, date_time.year, 1)
        _update_freq_dict(self.week_freq, date_time.weekofyear, 1)
        _update_freq_dict(self.month_freq, date_time.month, 1)
        _update_freq_dict(self.weekday_freq, date_time.weekday(), 1)
        _update_freq_dict(self.day_freq, date_time.day, 1)
        _update_freq_dict(self.hour_freq, date_time.hour, 1)

    def print_freq_stats(self):
        print("Yearly Freq -----------------")
        print_dictionary(self.year_freq)
        print("Monthly Freq -----------------")
        print_dictionary(self.month_freq)
        print("Weekly Freq -----------------")
        print_dictionary(self.week_freq)
        print("WeekDay Freq -----------------")
        print_dictionary(self.weekday_freq)
        print("Day Freq -----------------")
        print_dictionary(self.day_freq)
        print("Hour Freq -----------------")
        print_dictionary(self.hour_freq)


def _update_freq_dict(in_dict, key, increase_by):
    if key not in in_dict:
        in_dict[key] = 0
    in_dict[key] += increase_by


def print_dictionary(in_dict):
    keys = sorted(in_dict.keys())
    for key in keys:
        print("%s: %s" % (str(key), str(in_dict[key])))


class GranuleSize(enum.Enum):
    Hour = 'H'
    Day = 'D'
    Month = 'M'
    Year = 'Y'
    Week = 'W'


def group_events_by_intervals(timestamps: list, g_size: GranuleSize):
    timestamps.sort()
    event_intervals = dict()
    max_length = 0
    for tstamp in timestamps:
        key = get_key(tstamp, g_size)
        if key not in event_intervals:
            event_intervals[key] = []
        event_intervals[key].append(tstamp)
        max_length = max(max_length, len(event_intervals[key]))
    return event_intervals, max_length


def build_histogram(event_intervals: dict, max_lenght: int, g_size: GranuleSize):
    discretize = discretize_func(g_size)
    discretized_intervals = dict()
    for interval_start in event_intervals:
        discretized_intervals[interval_start] = [0] * max_lenght * len(event_intervals[interval_start])
        for timestamp in event_intervals[interval_start]:
            discretized_intervals[interval_start].append(discretize((timestamp - interval_start).total_seconds()))


def discretize_func(g_size: GranuleSize):
    if g_size is GranuleSize.Hour:
        return discretize_to_minutes
    elif g_size is GranuleSize.Day:
        return discretize_to_hour


def get_key(timestamp, g_size: GranuleSize):
    res_dt = pd.to_datetime('1700-01-01 00:00:00.000')

    if g_size is GranuleSize.Year:
        return res_dt.replace(year=timestamp.year)
    if g_size is GranuleSize.Month:
        return res_dt.replace(year=timestamp.year, month=timestamp.month)
    if g_size is GranuleSize.Week:
        n_dt = timestamp.date()
        return res_dt.replace(year=n_dt.year, month=n_dt.month, day=n_dt.day) - timestamp.dayofweek * timedelta(days=1)
    if g_size is GranuleSize.Day:
        return res_dt.replace(year=timestamp.year, month=timestamp.month, day=timestamp.day)
    if g_size is GranuleSize.Hour:
        return res_dt.replace(year=timestamp.year, month=timestamp.month, day=timestamp.day, hour=timestamp.hour)
    return timestamp


def parse_datetime(date_str, time_stats):
    date_t = pd.to_datetime(date_str, utc=True).tz_localize(None)
    time_stats.check_datetime(date_t)
    return date_t
    # if g_size is None:
    #     return date_t
    # elif g_size is GranuleSize.Hour:
    #     return date_t.replace(minute=0, second=0, microsecond=0)
    # else:
    #     date_t = date_t.date()
    #     if g_size is GranuleSize.Month:
    #         return date_t.replace(day=1)
    #     if g_size is GranuleSize.Year:
    #         return date_t.replace(month=1, day=1)
    #     return date_t


def discover_inter_arrival(log_cases, g_size: GranuleSize):
    time_stats = DTimeInfo()
    max_value = 0

    arrival_times = dict()
    start_events = []
    for trace in log_cases:
        first_date = parse_datetime(sorted(trace.event_list, key=lambda evt: evt.started_at)[0].started_at, time_stats)
        _update_freq_dict(arrival_times, first_date, 1)
        start_events.append(first_date)

        max_value = max(max_value, arrival_times[first_date])
    # time_stats.print_freq_stats()
    event_intervals, max_size = group_events_by_intervals(start_events, g_size)
    build_histogram(event_intervals, max_size, g_size)

    # prophet_forecasting(arrival_times, max_value)


# def prophet_forecasting(arrival_times, max_value):
#     # arrival_times.sort()
#     ds = arrival_times.keys()
#     y = arrival_times.values()
#
#     dataset = pd.DataFrame({'ds': ds, 'y': y})
#     f_dataset = iqr_filter_outliers(dataset)
#
#     f_dataset['cap'] = max_value
#     f_dataset['floor'] = 0
#
#     pr_estimator = Prophet(growth='logistic')
#     pr_estimator.fit(f_dataset)
#
#     future = pr_estimator.make_future_dataframe(periods=365, freq='D')
#     future['cap'] = max_value
#     future['floor'] = 0
#     forecast = pr_estimator.predict(future)
#     pr_estimator.plot(forecast)
#
#     est_val = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
#     # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     #     print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(580))
#
#     # pr_estimator.plot_components(forecast)
#     plt.show()
#     # neural_prophet_forecast(dataset)

# def neural_prophet_forecast(dataframe):
#     n_prophet = NeuralProphet()
#     n_prophet.fit(dataframe)
#     future = n_prophet.make_future_dataframe(df=dataframe, periods=365)
#     forecast = n_prophet.predict(df=future)
#     fig_forecast = n_prophet.plot(forecast)
#     plt.show()


def dataframe_from_csv(log_path, extended_out=False):
    event_log = pd.read_csv(log_path)
    event_log['start_time'] = pd.to_datetime(event_log['start_time'], utc=True)
    event_log['end_time'] = pd.to_datetime(event_log['end_time'], utc=True)
    event_log.sort_values(by=['case_id', 'end_time'], inplace=True, ascending=[True, True])

    # act_freq = event_log['activity'].value_counts()
    # res_freq = event_log['resource'].value_counts()


def iqr_filter_outliers(data_frame):
    q1 = data_frame['y'].quantile(0.25)
    q3 = data_frame['y'].quantile(0.75)
    iqr = q3 - q1
    upper_limit = q3 + 1.5 * iqr
    lower_limit = q1 - 1.5 * iqr
    filt_df = data_frame.copy()
    filt_df.loc[(filt_df['y'] < lower_limit) & (filt_df['y'] > upper_limit), 'y'] = None
    return filt_df
