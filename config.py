import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'dataset'   # dataset path
MODEL_DIR = 'models'   # models save dir
CSV_NAME = 'archive.csv'   # csv file name
TIME_STEPS = 5          # how previous records to predict future data
TRAIN_DATA_RATIO = 0.8   # how much data will be used for training
LSTM_LAYER = 16    # Number lstm layer in the base model
ITERATION = 4       # iteration name
EPOCHS = 5          # How to many epochs the model will train
BATCH_SIZE = 32     # Batch size of training

