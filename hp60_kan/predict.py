"""Prediction utilities for Hp60 KAN model."""
import sys
import os
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(parent_dir, 'mlx-kan'))

import numpy as np
import mlx.core as mx
from typing import List, Dict
from datetime import datetime

from hp60_kan.data_loader import DataNormalizer
from hp60_kan.model import Hp60KANModel
from hp60_kan import config


class Hp60Predictor:
    """Predictor class for Hp60 values."""
    
    def __init__(self, model_dir: str = None, scaler_path: str = None):
        self.model_dir = model_dir or config.MODEL_DIR
        self.scaler_path = scaler_path or config.SCALER_PATH
        self.model = None
        self.normalizer = None
        self.is_loaded = False
        
    def load(self) -> bool:
        try:
            config_path = os.path.join(self.model_dir, 'config.json')
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            
            self.model = Hp60KANModel(
                input_dim=model_config['input_dim'],
                hidden_dims=model_config['hidden_dims'],
                output_dim=model_config['output_dim'],
                grid_size=model_config.get('grid_size'),
                spline_order=model_config.get('spline_order'),
                grid_eps=model_config.get('grid_eps')
            )
            self.model.load(self.model_dir)
            
            self.normalizer = DataNormalizer()
            self.normalizer.load(self.scaler_path)
            
            self.is_loaded = True
            print("Predictor loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading predictor: {e}")
            return False
    
    def _create_features(self, hour: float, month: int, day_of_year: int, 
                         ap60: float) -> np.ndarray:
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        doy_sin = np.sin(2 * np.pi * day_of_year / 365)
        doy_cos = np.cos(2 * np.pi * day_of_year / 365)
        
        ap60_norm = ap60 / 100.0
        ap60_hour_sin = ap60_norm * hour_sin
        ap60_hour_cos = ap60_norm * hour_cos
        ap60_month_sin = ap60_norm * month_sin
        ap60_month_cos = ap60_norm * month_cos
        ap60_log = np.log1p(ap60)
        
        return np.array([[hour_sin, hour_cos, month_sin, month_cos, 
                          doy_sin, doy_cos, ap60,
                          ap60_hour_sin, ap60_hour_cos,
                          ap60_month_sin, ap60_month_cos,
                          ap60_log]])
    
    def predict(self, hour: float, month: int, day_of_year: int, 
                ap60: float) -> float:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        features = self._create_features(hour, month, day_of_year, ap60)
        features_norm = self.normalizer.transform_X(features)
        features_mx = mx.array(features_norm.astype(np.float32))
        prediction_norm = self.model(features_mx)
        prediction = self.normalizer.inverse_transform_y(np.array(prediction_norm))
        
        return float(prediction[0, 0])
    
    def predict_from_datetime(self, dt: datetime, ap60: float) -> float:
        hour = dt.hour + dt.minute / 60
        month = dt.month
        day_of_year = dt.timetuple().tm_yday
        return self.predict(hour, month, day_of_year, ap60)
    
    def predict_batch(self, data: List[Dict]) -> List[float]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        all_features = []
        for d in data:
            features = self._create_features(
                d['hour'], d['month'], d['day_of_year'], d['ap60']
            )
            all_features.append(features[0])
        
        features = np.array(all_features)
        features_norm = self.normalizer.transform_X(features)
        features_mx = mx.array(features_norm.astype(np.float32))
        predictions_norm = self.model(features_mx)
        predictions = self.normalizer.inverse_transform_y(np.array(predictions_norm))
        
        return predictions.flatten().tolist()
    
    def get_model_info(self) -> Dict:
        if not self.is_loaded:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "model_dir": self.model_dir,
            "scaler_path": self.scaler_path,
            "architecture": self.model.layers_hidden,
            "parameters": self.model.count_parameters(),
            "feature_columns": config.FEATURE_COLUMNS
        }


def load_predictor(model_dir: str = None, scaler_path: str = None) -> Hp60Predictor:
    predictor = Hp60Predictor(model_dir, scaler_path)
    predictor.load()
    return predictor


def quick_predict(hour: float, month: int, day_of_year: int, ap60: float,
                  predictor: Hp60Predictor = None) -> float:
    if predictor is None:
        predictor = load_predictor()
    return predictor.predict(hour, month, day_of_year, ap60)


if __name__ == "__main__":
    print("Testing Hp60 Predictor...")
    print("=" * 50)
    
    predictor = load_predictor()
    
    pred = predictor.predict(hour=12, month=6, day_of_year=172, ap60=50)
    print(f"\nSingle prediction: Hp60 = {pred:.2f}")
    
    dt = datetime(2024, 7, 15, 14, 30)
    pred = predictor.predict_from_datetime(dt, ap60=60)
    print(f"Datetime prediction ({dt}): Hp60 = {pred:.2f}")
    
    batch_data = [
        {'hour': 0, 'month': 1, 'day_of_year': 1, 'ap60': 10},
        {'hour': 12, 'month': 6, 'day_of_year': 172, 'ap60': 50},
        {'hour': 18, 'month': 12, 'day_of_year': 350, 'ap60': 100},
    ]
    preds = predictor.predict_batch(batch_data)
    print(f"\nBatch predictions: {preds}")
    
    print(f"\nModel info: {predictor.get_model_info()}")
