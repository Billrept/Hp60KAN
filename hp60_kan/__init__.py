"""
HP60 Prediction using MLX-KAN

A modular package for predicting Hp60 geomagnetic index values.

Modules:
    - config: Configuration settings
    - data_loader: Data loading and preprocessing
    - model: KAN model definition
    - train: Training pipeline
    - predict: Prediction utilities
    - visualize: Visualization utilities

Quick Start:
    # Train a model
    from hp60_kan.train import train_model_pipeline
    model, normalizer, history = train_model_pipeline()
    
    # Make predictions
    from hp60_kan.predict import load_predictor
    predictor = load_predictor()
    hp60 = predictor.predict(hour=12, month=6, day_of_year=172, ap60=50)
    
    # Visualize data
    from hp60_kan.visualize import create_visualizer
    viz = create_visualizer()
    viz.plot_daily_pattern()
"""

from hp60_kan.data_loader import DataLoader, DataNormalizer, split_data
from hp60_kan.model import Hp60KANModel, create_model
from hp60_kan.predict import Hp60Predictor, load_predictor, quick_predict
from hp60_kan.visualize import Hp60Visualizer, create_visualizer
from hp60_kan import config

__version__ = "1.0.0"
__all__ = [
    # Data
    'DataLoader',
    'DataNormalizer', 
    'split_data',
    # Model
    'Hp60KANModel',
    'create_model',
    # Prediction
    'Hp60Predictor',
    'load_predictor',
    'quick_predict',
    # Visualization
    'Hp60Visualizer',
    'create_visualizer',
    # Config
    'config'
]
