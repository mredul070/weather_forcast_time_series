import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

import config as cfg

def get_scaler(data_dir, csv_name):
    df = pd.read_csv(os.path.join(data_dir, csv_name))
    df.rename(columns = {'temperature_2m_mean (°C)': 'mean_temp'}, inplace=True)
    scaler = MinMaxScaler()
    return scaler.fit(df[['mean_temp']])

def get_prediction(past_records):
    # define model 
    model_path = os.path.join(cfg.MODEL_DIR, f'it_{cfg.ITERATION}_lstm.h5')
    # load model into memory
    model = load_model(model_path)
    # define normalizer
    scaler = get_scaler(cfg.DATA_DIR, cfg.CSV_NAME)
    # normalize
    y = scaler.transform(past_records)
    # get model predictiom
    y = model.predict([y])  
    # denormalize
    return scaler.inverse_transform(y)


if __name__ == '__main__':
    records = [[32], [28.4], [27.9], [29.2], [32]]
    print(get_prediction(records))
    
