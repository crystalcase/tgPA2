import pandas as pd
from datetime import datetime
import numpy as np
from sklearn import preprocessing


def get_min_max(data):
    return data['Absatz'].min(), data['Absatz'].max()


def parse_date(date_str):
    return datetime.strptime(date_str, "%d-%m-%Y %H:%M:%S")


def sum_data(raw_data):
    raw_data.index.name = 'id'

    transformed_data = None

    last_entry = ""
    day_sum = 0

    for index, data in raw_data.iterrows():
        day = data['Kalendertag']
        value = data['Absatz']
        if day == last_entry:
            day_sum += value
        else:
            new_line = pd.DataFrame(
                data={
                    'Kalendertag': [last_entry],
                    'Absatz': [day_sum]
                }
            )
            if transformed_data is None:
                transformed_data = new_line
            else:
                transformed_data = transformed_data.append(new_line, ignore_index=True)
            day_sum = value
        last_entry = day

    transformed_data['Kalendertag'].replace('', np.nan, inplace=True)
    transformed_data.dropna(subset=['Kalendertag'], inplace=True)
    return transformed_data


def fill_dates(data):
    data.set_index('Kalendertag', inplace=True)
    date_range = pd.date_range('01-01-2014', '28-05-2020')
    new_data = data.reindex(date_range, fill_value=0)
    new_data.reset_index(level=0, inplace=True)
    new_data['Kalendertag'] = pd.to_datetime(new_data['index'])
    return new_data


def split_features(data_file):
    new_data = pd.DataFrame()
    for index, data in data_file.iterrows():
        new_line = pd.DataFrame(
            data={
                "Kalendertag": [data.Kalendertag.day],
                "Monat": [data.Kalendertag.month],
                "Jahr": [data.Kalendertag.year],
                "Kalenderwoche": [data.Kalendertag.weekofyear],
                "Wochentag": [data.Kalendertag.dayofweek],
                "Absatz": [data.Absatz]

            }
        )
        new_data = new_data.append(new_line)

    return new_data


def normalize_sells(data):
    sell_values = data['Absatz']
    scaler = preprocessing.MinMaxScaler()
    data['Absatz'] = scaler.fit_transform(np.array(sell_values).reshape(-1, 1))
    return data


def get_tb_dir():
    return "C:/dev/TensorflowNetworks/logs/"
