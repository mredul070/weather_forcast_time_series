import matplotlib.pyplot as plt

import model as md
import config as cfg
from data_preprocess import generate_formatted_data


def plot_history(history):
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('logs/training_logs.png')


if __name__ == '__main__':
    # generate train Test Data
    x_train, y_train, x_test, y_test = generate_formatted_data(cfg.DATA_DIR, cfg.CSV_NAME, cfg.TRAIN_DATA_RATIO, cfg.TIME_STEPS)
    # create the model
    model = md.LSTM_model(cfg.TIME_STEPS, cfg.LSTM_LAYER)
    # Train the model
    history = model.fit(x_train, 
                        y_train, 
                        epochs=cfg.EPOCHS, 
                        batch_size=cfg.BATCH_SIZE, 
                        validation_data = (x_test, y_test),
                        verbose=1, 
                        shuffle=False)

    # save model
    itr = str(cfg.ITERATION)
    model.save(f'models/it_{itr}_lstm.h5')
    # save logs
    plot_history(history)

    # evaluate model performance
    results = model.evaluate(x_test, y_test)
    print("Model Prerformance Report")
    print(f'{model.metrics_names} -> {results}')