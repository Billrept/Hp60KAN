"""Hyperparameter tuning configurations for high-activity capture."""

# =============================================================================
# TUNING PARAMETERS - Parameters to search over
# =============================================================================
# These are the new parameters for capturing high magnetic activity

# Full search space (81 combinations, ~7 hours at 5 min/trial)
TUNING_PARAMS = {
    # Asymmetric loss parameters
    'asymmetric_alpha': [6.0, 10.0, 14.0],        # Penalty multiplier for under-prediction
    'high_value_threshold': [0.25, 0.30, 0.35],   # Normalized threshold for "high" values
    
    # Stratified sampling parameters  
    'high_activity_ratio': [0.4, 0.5, 0.6],       # Ratio of high-activity samples per batch
    
    # Sample weighting
    'weight_power': [2.5, 3.0, 3.5],              # Exponential weight for high values
}

# Smaller search space for faster tuning (16 combinations, ~80 min)
TUNING_PARAMS_SMALL = {
    'asymmetric_alpha': [8.0, 12.0],
    'high_value_threshold': [0.25, 0.35],
    'high_activity_ratio': [0.4, 0.6],
    'weight_power': [2.5, 3.5],
}

# Quick test (4 combinations, ~20 min)
TUNING_PARAMS_QUICK = {
    'asymmetric_alpha': [8.0, 12.0],
    'high_value_threshold': [0.3],
    'high_activity_ratio': [0.5],
    'weight_power': [3.0, 3.5],
}

# =============================================================================
# FIXED PARAMETERS - Parameters kept constant during tuning
# =============================================================================

FIXED_PARAMS = {
    # Training parameters
    'epochs': 40,                    # Reduced for faster tuning
    'batch_size': 256,
    'learning_rate': 1e-4,           # Fixed based on previous tuning
    'weight_decay': 1e-5,
    
    # Model architecture (fixed based on previous tuning)
    'hidden_dims': [64, 64, 64],
    
    # Regularization
    'reg_activation': 0.01,
    'reg_entropy': 0.01,
    'use_entropy_reg': True,
    
    # Enable all new features
    'use_sample_weights': True,
    'use_asymmetric_loss': True,
    'use_stratified_sampling': True,
    'use_log_transform': False,      # Disabled - compresses high values too much
    
    # Grid updates
    'update_grid_every': 0,
    'grid_update_sample_size': 2000,
    
    # Fixed stratified sampling percentile
    'high_activity_percentile': 70,
}

# =============================================================================
# OPTIMIZATION SETTINGS
# =============================================================================

# Primary metric to optimize (lower is better)
OPTIMIZATION_METRIC = 'val_high_mae'  # Focus on high-activity MAE

# Secondary metrics to track
TRACK_METRICS = ['val_mae', 'val_mse', 'val_high_mae', 'val_high_mse']

# Whether to use small search space (faster) or full search space
USE_SMALL_SEARCH = False  # Set to True for quick testing

# Number of epochs for final training with best params
FINAL_TRAINING_EPOCHS = 60
