import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import model as md
import config as cfg

def get_scaler(data_dir, csv_name):
    df = pd.read_csv(os.path.join(data_dir, csv_name))
    df.rename(columns = {'temperature_2m_mean (°C)': 'mean_temp'}, inplace=True)
    scaler = MinMaxScaler()
    return scaler.fit(df[['mean_temp']])

def get_prediction(past_records):
    model_path = os.path.join(cfg.MODEL_DIR, f'it_{cfg.ITERATION}_lstm.h5')
    model = md.LSTM_model(cfg.TIME_STEPS, cfg.LSTM_LAYER)
    model.load_weights(model_path)

    scaler = get_scaler(cfg.DATA_DIR, cfg.CSV_NAME)
    y = scaler.transform(past_records)
    print(y)
    y = np.array([past_records])

    y = model.predict(y)
    
    return scaler.inverse_transform(y)



if __name__ == '__main__':
    records = [[28], [28.4], [27.9], [29.2], [27],
    [22], [22.4], [27.9], [29.2], [27]]
    print(get_prediction(records))
    
