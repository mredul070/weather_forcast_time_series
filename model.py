import config as cfg
from tensorflow import keras
from tensorflow.keras import backend, models, layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout

def LSTM_model(time_steps, lstm_layer):
    #initiate the model
    model = keras.Sequential()
    # LSTM Layer
    model.add(keras.layers.LSTM(lstm_layer, input_shape=(time_steps, 1)))
    model.add(Dropout(0.2))
    # Output Later
    model.add(keras.layers.Dense(1))
    # Compile the model
    model.compile(loss='mean_squared_error',
                optimizer=keras.optimizers.Adam(0.001))
    model.summary()

    return model


