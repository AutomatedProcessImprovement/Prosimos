import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

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


def discover_inter_arrival(log_cases):
    time_stats = DTimeInfo()
    max_value = 0

    arrival_times = list()

    for trace in log_cases:
        first_date = pd.to_datetime(sorted(trace.event_list,
                                           key=lambda evt: evt.started_at)[0].started_at, utc=True).tz_localize(None)
        arrival_times.append(first_date)
        time_stats.check_datetime(first_date)
    time_stats.print_freq_stats()


    arrival_times.sort()
    ds = []
    y = []

    for i in range(0, len(arrival_times) - 1):
        ds.append(arrival_times[i])
        y.append((arrival_times[i + 1] - arrival_times[i]).total_seconds())
        max_value = max(max_value, y[i])

    dataset = pd.DataFrame({'ds': ds, 'y': y})
    f_dataset = iqr_filter_outliers(dataset)

    # dataset['cap'] = max_value
    # dataset['floor'] = 0


    pr_estimator = Prophet()
    pr_estimator.fit(dataset)

    future = pr_estimator.make_future_dataframe(periods=8760, freq='h')
    # future['cap'] = max_value
    # future['floor'] = 0
    forecast = pr_estimator.predict(future)
    pr_estimator.plot(forecast)
    plt.show()
    est_val = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(580))

    # pr_estimator.plot_components(forecast)
    pr_estimator = Prophet()
    pr_estimator.fit(f_dataset)
    forecast = pr_estimator.predict(future)
    pr_estimator.plot(forecast)
    plt.show()


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




