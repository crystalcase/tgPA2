import pandas as pd
from datetime import datetime

def get_min_max(data):
    return data['SumAbsatz'].min(), data['SumAbsatz'].max()

def date_parser(date_str):
    return datetime.strptime(date_str, "%d-%m-%Y")


def read_data():
    return pd.read_csv('C:/dev/BierData/raw_data.csv',
                            header=0,
                            parse_dates=['Kalendertag'],
                            date_parser=date_parser,
                            )

def get_summed_data():
    data_file = read_data()
    data_file.index.name = 'id'

    transformed_data = None

    last_entry = ""
    day_sum = 0

    for index, data in data_file.iterrows():
        day = data['Kalendertag']
        value = data['Absatz']
        if day == last_entry:
            day_sum += value
        else:
            new_line = pd.DataFrame(
                data={
                    'Kalendertag': [last_entry],
                    'SumAbsatz': [day_sum]
                }
            )
            if transformed_data is None:
                transformed_data = new_line
            else:
                transformed_data = transformed_data.append(new_line, ignore_index=False)
            day_sum = value
        last_entry = day



    return transformed_data