import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import config as cfg


def create_sequence_wise_data(train, mean_temp, time_steps):
    x_train = []
    y_train = []
    for value in range(len(train) - time_steps):
        temps = train.iloc[value:(value + time_steps)].values
        x_train.append(temps)        
        y_train.append(mean_temp.iloc[value + time_steps])
    return np.array(x_train), np.array(y_train)

def generate_formatted_data(data_dir, csv_name, train_data_ratio, time_steps):
    # read csv file 
    df = pd.read_csv(os.path.join(data_dir, csv_name))
    # renaming the coloumn names
    df.rename(columns = {'date' : 'DATE',
                        'temperature_2m_max (°C)': 'max_temp', 
                        'temperature_2m_min (°C)': 'min_temp', 
                        'temperature_2m_mean (°C)': 'mean_temp',
                        'rain_sum (mm)' : 'rain'}, 
                        inplace=True)
    #using only mean temp for weather prediction for now
    df_temp = df[['mean_temp']]
    #indesxing time column with mean temp
    df_temp.index = pd.to_datetime(df[['time']].stack(), format='%m%d%y', errors='ignore')

    # diving train and test data
    train_size = int(len(df_temp) * train_data_ratio)
    test_size = len(df_temp) - train_size
    train, test = df_temp.iloc[0:train_size], df_temp.iloc[train_size:len(df_temp)]
    print(f'Train Data len -> {len(train)}')
    print(f'Test Data len -> {len(test)}')

    # Normalize the data
    scaler = MinMaxScaler()
    scaler = scaler.fit(train[['mean_temp']])
    train['mean_temp'] = scaler.transform(train[['mean_temp']])
    test['mean_temp'] = scaler.transform(test[['mean_temp']])

    # create time series data 
    x_train, y_train = create_sequence_wise_data(train, train.mean_temp, time_steps)
    x_test, y_test = create_sequence_wise_data(test, test.mean_temp, time_steps)

    print(f'Train Data Shape -> {x_train.shape}')
    print(f'Prediction Data Shape -> {y_train.shape}')

    return scaler, x_train, y_train, x_test, y_test


if __name__ == '__main__':
    scaler, x_train, y_train, x_test, y_test = generate_formatted_data(cfg.DATA_DIR, cfg.CSV_NAME, cfg.TRAIN_DATA_RATIO, cfg.TIME_STEPS)
    # print(scaler.value)