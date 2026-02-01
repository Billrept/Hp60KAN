"""Hyperparameter tuning configurations."""

# Define the search space (parameters to tune)
# REMOVE grid_size since train_model_pipeline doesn't accept it
TUNING_PARAMS = {
    'learning_rate': [1e-5, 5e-5, 1e-4],
    'weight_power': [1.5, 2.0, 3.0],
    'hidden_dims': [[64, 64, 64], [64, 64], [128, 64, 32]],
}

# Fixed parameters
FIXED_PARAMS = {
    'epochs': 40,
    'batch_size': 256,
    'weight_decay': 1e-5,
    'reg_activation': 0.01,
    'reg_entropy': 0.01,
    'use_entropy_reg': True,
    'use_sample_weights': True,
    'update_grid_every': 0,
    'grid_update_sample_size': 2000,
}

OPTIMIZATION_METRIC = 'val_mae'
