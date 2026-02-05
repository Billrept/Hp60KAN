"""Configuration settings for Hp60 prediction."""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'hpodata_2000-2025.csv')
NEW_DATA = os.path.join(BASE_DIR, 'hpodata_2026_1-2.csv')
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

# Training (tuned parameters from trial 22)
EPOCHS = 60
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Regularization
USE_ENTROPY_REG = True
REG_ACTIVATION = 0.01
REG_ENTROPY = 0.01
UPDATE_GRID_EVERY = 0
GRID_UPDATE_SAMPLE_SIZE = 2000

# Sample weighting
USE_SAMPLE_WEIGHTS = True
WEIGHT_POWER = 2.5  # Tuned from hyperparameter search (was 3.0)

# Target transformation (to capture high activity trends)
USE_LOG_TRANSFORM = False  # Disabled - log compresses high values too much

# Asymmetric loss (penalize under-prediction of high values more)
USE_ASYMMETRIC_LOSS = True
ASYMMETRIC_ALPHA = 6.0  # Tuned: lower penalty works better (was 8.0)
HIGH_VALUE_THRESHOLD = 0.35  # Tuned: focus on truly extreme events (was 0.3)

# Stratified sampling
USE_STRATIFIED_SAMPLING = True
HIGH_ACTIVITY_RATIO = 0.4  # Tuned: 40% high-activity samples per batch (was 0.5)
HIGH_ACTIVITY_PERCENTILE = 70  # Top 30% considered "high activity"

# Output
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
TRAINING_HISTORY_PLOT = os.path.join(OUTPUT_DIR, 'training_history.png')
PREDICTIONS_PLOT = os.path.join(OUTPUT_DIR, 'predictions.png')
