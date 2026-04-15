# src/config.py

# File paths
DATA_PATH = "data/IoT_Intrusion.csv"
MODEL_DIR = "models/"
REPORTS_DIR = "reports/"

# Data preprocessing settings
RARE_LABEL_THRESHOLD = 100

# Autoencoder settings
INPUT_DIM = 46 # Number of features after dropping label
ENCODING_DIM = 20
AE_EPOCHS = 50
AE_BATCH_SIZE = 256
AE_LEARNING_RATE = 1e-3

# XGBoost settings
XGB_N_ESTIMATORS = 200
XGB_LEARNING_RATE = 0.1