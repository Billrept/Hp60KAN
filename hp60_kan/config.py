"""Configuration settings for Hp60 prediction."""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'hpodata_2000-2025.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models', 'hp60_kan')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler_params.npz')

# Features
FEATURE_COLUMNS = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                   'doy_sin', 'doy_cos', 'ap60',
                   'ap60_hour_sin', 'ap60_hour_cos',
                   'ap60_month_sin', 'ap60_month_cos',
                   'ap60_log']
TARGET_COLUMN = 'Hp60'

# Data split
DATA_USE_RATIO = 0.9
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

# Model architecture
HIDDEN_DIMS = [64, 64, 64]

# KAN parameters
GRID_SIZE = 5
SPLINE_ORDER = 3
GRID_EPS = 0.02

# Training
EPOCHS = 60
BATCH_SIZE = 256
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5

# Regularization
USE_ENTROPY_REG = True
REG_ACTIVATION = 0.01
REG_ENTROPY = 0.01
UPDATE_GRID_EVERY = 0
GRID_UPDATE_SAMPLE_SIZE = 2000

# Sample weighting
USE_SAMPLE_WEIGHTS = True
WEIGHT_POWER = 2.0

# Output
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
TRAINING_HISTORY_PLOT = os.path.join(OUTPUT_DIR, 'training_history.png')
PREDICTIONS_PLOT = os.path.join(OUTPUT_DIR, 'predictions.png')
