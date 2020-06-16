from utils import *



def read_raw_data():
    return pd.read_csv('./data/raw_data.csv',
                       header=0,
                       parse_dates=['Kalendertag'])

def read_data(data_name):
    file_path = 'C:/dev/BierData/' + data_name + '.csv'
    return pd.read_csv(file_path)

def create_data():
    raw_data = read_raw_data()
    summed_data = sum_data(raw_data)
    summed_data['Absatz'] = summed_data['Absatz'].astype(np.float32)
    summed_data.to_csv(r'C:/dev/BierData/summed_data.csv', index=False, header=True)
    print("summed_data created")

    split_data = split_features(summed_data)
    split_data.to_csv(r'C:/dev/BierData/split_data.csv', index=False, header=True)
    print("split_data created")

    normalized_data = normalize_sells(split_data)
    normalized_data.to_csv(r'C:/dev/BierData/split_normalized_data.csv', index=False, header=True)
    print("split_normalized_data created")

    filled_data = fill_dates(summed_data)
    filled_data.to_csv(r'C:/dev/BierData/filled_data.csv', index=False, header=True)
    print("filled_data created")

    split_data = split_features(filled_data)
    split_data.to_csv(r'C:/dev/BierData/filled_split_data.csv', index=False, header=True)
    print("filled_split_data created")

    normalized_data = normalize_sells(split_data)
    normalized_data.to_csv(r'C:/dev/BierData/filled_split_normalized_data.csv', index=False, header=True)
    print("filled_split_normalized_data created")


create_data()