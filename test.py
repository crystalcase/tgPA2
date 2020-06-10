import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def date_parser(date_str):
    return datetime.strptime(date_str, "%d-%m-%Y")

data_file = pd.read_csv('C:/dev/BierData/raw_data.csv',
                        header=0,
                        parse_dates=['Kalendertag'],
                        date_parser=date_parser,
                        )
data_file.index.name = 'id'

print("file", data_file.iterrows())

transformed_data = pd.DataFrame()
last_entry = ""
day_sum = 0

for index, data in data_file.iterrows():
    day = data['Kalendertag']
    value = data['Absatz']
    print('value', value)
    if day == last_entry:
        day_sum += value
    else:
        new_line = pd.DataFrame(
            data={
                'Kalendertag': [last_entry],
                'SumAbsatz': [day_sum]
            }
        )
        transformed_data = transformed_data.append(new_line)
        day_sum = value
    last_entry = day

