"""
HP60 Prediction using MLX-KAN

A modular package for predicting Hp60 geomagnetic index values.

Modules:
    - config: Configuration settings
    - data_loader: Data loading and preprocessing
    - model: KAN model definition (MLX-based, macOS only)
    - train: Training pipeline (MLX-based, macOS only)
    - predict: Prediction utilities (MLX-based, macOS only)
    - predict_numpy: Windows-compatible prediction using NumPy
    - visualize: Visualization utilities

Quick Start:
    # For Windows: Use NumPy-based inference
    from hp60_kan.predict_numpy import Hp60PredictorNumpy
    predictor = Hp60PredictorNumpy()
    predictor.load()
    hp60 = predictor.predict(ap60=50, dt=datetime(2024, 5, 15, 12, 0))
    
    # For macOS: Use MLX-based training and prediction
    from hp60_kan.predict import load_predictor
    predictor = load_predictor()
    hp60 = predictor.predict(hour=12, month=6, day_of_year=172, ap60=50)
"""

# Always available
from hp60_kan.data_loader import DataLoader, DataNormalizer, split_data
from hp60_kan import config

# MLX-dependent imports (lazy load for macOS only)
def __getattr__(name):
    if name in ['Hp60KANModel', 'create_model']:
        from hp60_kan.model import Hp60KANModel, create_model
        return {'Hp60KANModel': Hp60KANModel, 'create_model': create_model}[name]
    elif name in ['Hp60Predictor', 'load_predictor', 'quick_predict']:
        from hp60_kan.predict import Hp60Predictor, load_predictor, quick_predict
        return {'Hp60Predictor': Hp60Predictor, 'load_predictor': load_predictor, 'quick_predict': quick_predict}[name]
    elif name in ['Hp60Visualizer', 'create_visualizer']:
        from hp60_kan.visualize import Hp60Visualizer, create_visualizer
        return {'Hp60Visualizer': Hp60Visualizer, 'create_visualizer': create_visualizer}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = "1.0.0"
__all__ = [
    # Data
    'DataLoader',
    'DataNormalizer', 
    'split_data',
    # Model (MLX-based, macOS only)
    'Hp60KANModel',
    'create_model',
    # Prediction (MLX-based, macOS only)
    'Hp60Predictor',
    'load_predictor',
    'quick_predict',
    # Windows-compatible NumPy prediction
    'Hp60PredictorNumpy',
    # Visualization (MLX-based, macOS only)
    'Hp60Visualizer',
    'create_visualizer',
    # Config
    'config'
]