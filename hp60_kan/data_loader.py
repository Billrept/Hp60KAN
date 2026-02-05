"""Data loading and preprocessing for Hp60 prediction."""
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict


class DataLoader:
    """Load and preprocess HPO data."""
    
    def __init__(self, filepath: str, use_ratio: float = 0.9):
        self.filepath = filepath
        self.use_ratio = use_ratio
        self.df = None
        self.df_clean = None
        self.feature_columns = None
        self.target_column = None
        
    def load(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.filepath)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        return self.df
    
    def create_cyclical_features(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")
            
        df_clean = self.df.dropna().copy()
        
        n_use = int(len(df_clean) * self.use_ratio)
        df_clean = df_clean.iloc[:n_use].copy()
        print(f"Using {self.use_ratio*100:.0f}% of data: {n_use} samples")
        
        # Cyclical time features
        df_clean['hour_sin'] = np.sin(2 * np.pi * df_clean['hh.h'] / 24)
        df_clean['hour_cos'] = np.cos(2 * np.pi * df_clean['hh.h'] / 24)
        df_clean['month_sin'] = np.sin(2 * np.pi * df_clean['MM'] / 12)
        df_clean['month_cos'] = np.cos(2 * np.pi * df_clean['MM'] / 12)
        
        df_clean['day_of_year'] = pd.to_datetime(df_clean[['YYYY', 'MM', 'DD']].rename(
            columns={'YYYY': 'year', 'MM': 'month', 'DD': 'day'}
        )).dt.dayofyear
        df_clean['doy_sin'] = np.sin(2 * np.pi * df_clean['day_of_year'] / 365)
        df_clean['doy_cos'] = np.cos(2 * np.pi * df_clean['day_of_year'] / 365)
        
        # Interaction features
        ap60_norm = df_clean['ap60'] / 100.0
        df_clean['ap60_hour_sin'] = ap60_norm * df_clean['hour_sin']
        df_clean['ap60_hour_cos'] = ap60_norm * df_clean['hour_cos']
        df_clean['ap60_month_sin'] = ap60_norm * df_clean['month_sin']
        df_clean['ap60_month_cos'] = ap60_norm * df_clean['month_cos']
        
        # Log transform
        df_clean['ap60_log'] = np.log1p(df_clean['ap60'])
        
        self.df_clean = df_clean
        print(f"Cleaned dataset shape: {df_clean.shape}")
        return df_clean
    
    def prepare_features(self, feature_columns: List[str], 
                         target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        if self.df_clean is None:
            raise ValueError("Features not created. Call create_cyclical_features() first.")
            
        self.feature_columns = feature_columns
        self.target_column = target_column
        
        X = self.df_clean[feature_columns].values
        y = self.df_clean[target_column].values
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        return X, y
    
    def get_raw_dataframe(self) -> pd.DataFrame:
        return self.df_clean


class DataNormalizer:
    """Data normalization with optional log transform for target."""
    
    def __init__(self, use_log_transform: bool = False):
        self.X_min = None
        self.X_max = None
        self.y_min = None
        self.y_max = None
        self.y_log_min = None
        self.y_log_max = None
        self.use_log_transform = use_log_transform
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_min = X.min(axis=0)
        self.X_max = X.max(axis=0)
        self.y_min = y.min()
        self.y_max = y.max()
        
        if self.use_log_transform:
            y_log = np.log1p(y)
            self.y_log_min = y_log.min()
            self.y_log_max = y_log.max()
            print(f"Log transform enabled: y_log range [{self.y_log_min:.4f}, {self.y_log_max:.4f}]")
        
        self.is_fitted = True
        
    def transform_X(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        return 2 * (X - self.X_min) / (self.X_max - self.X_min + 1e-8) - 1
    
    def transform_y(self, y: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        if self.use_log_transform:
            y_log = np.log1p(y)
            return (y_log - self.y_log_min) / (self.y_log_max - self.y_log_min + 1e-8)
        else:
            return (y - self.y_min) / (self.y_max - self.y_min + 1e-8)
    
    def inverse_transform_y(self, y_norm: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        if self.use_log_transform:
            y_log = y_norm * (self.y_log_max - self.y_log_min) + self.y_log_min
            return np.expm1(y_log)
        else:
            return y_norm * (self.y_max - self.y_min) + self.y_min
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.fit(X, y)
        X_norm = self.transform_X(X)
        y_norm = self.transform_y(y)
        
        print(f"X normalized range: [{X_norm.min():.4f}, {X_norm.max():.4f}]")
        print(f"y normalized range: [{y_norm.min():.4f}, {y_norm.max():.4f}]")
        
        return X_norm, y_norm
    
    def get_params(self) -> Dict:
        params = {
            'X_min': self.X_min,
            'X_max': self.X_max,
            'y_min': self.y_min,
            'y_max': self.y_max,
            'use_log_transform': self.use_log_transform
        }
        if self.use_log_transform:
            params['y_log_min'] = self.y_log_min
            params['y_log_max'] = self.y_log_max
        return params
    
    def set_params(self, params: Dict):
        self.X_min = params['X_min']
        self.X_max = params['X_max']
        self.y_min = float(params['y_min'])
        self.y_max = float(params['y_max'])
        self.use_log_transform = bool(params.get('use_log_transform', False))
        if self.use_log_transform:
            self.y_log_min = float(params['y_log_min'])
            self.y_log_max = float(params['y_log_max'])
        self.is_fitted = True
    
    def save(self, filepath: str):
        np.savez(filepath, **self.get_params())
        print(f"Scaler parameters saved to {filepath}")
        
    def load(self, filepath: str):
        params = np.load(filepath, allow_pickle=True)
        self.set_params(params)
        print(f"Scaler parameters loaded from {filepath}")


class StratifiedBatchSampler:
    """Stratified batch sampler that ensures high activity samples in each batch."""
    
    def __init__(self, y: np.ndarray, high_activity_ratio: float = 0.3, 
                 high_activity_percentile: float = 75):
        """
        Args:
            y: Target values (normalized)
            high_activity_ratio: Ratio of high activity samples per batch
            high_activity_percentile: Percentile threshold for high activity
        """
        self.high_activity_ratio = high_activity_ratio
        threshold = np.percentile(y, high_activity_percentile)
        
        self.high_indices = np.where(y.flatten() >= threshold)[0]
        self.low_indices = np.where(y.flatten() < threshold)[0]
        
        print(f"Stratified sampling: {len(self.high_indices)} high activity samples "
              f"(threshold >= {threshold:.4f}), {len(self.low_indices)} regular samples")
    
    def iterate(self, batch_size: int, X: np.ndarray, y: np.ndarray, 
                weights: np.ndarray = None):
        """Yield stratified batches."""
        n_high = max(1, int(batch_size * self.high_activity_ratio))
        n_low = batch_size - n_high
        
        # Shuffle indices
        high_idx_shuffled = np.random.permutation(self.high_indices)
        low_idx_shuffled = np.random.permutation(self.low_indices)
        
        # Calculate number of complete batches
        n_high_batches = len(high_idx_shuffled) // n_high
        n_low_batches = len(low_idx_shuffled) // n_low
        n_batches = min(n_high_batches, n_low_batches)
        
        for i in range(n_batches):
            high_batch_idx = high_idx_shuffled[i * n_high:(i + 1) * n_high]
            low_batch_idx = low_idx_shuffled[i * n_low:(i + 1) * n_low]
            batch_idx = np.concatenate([high_batch_idx, low_batch_idx])
            np.random.shuffle(batch_idx)
            
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            
            if weights is not None:
                w_batch = weights[batch_idx]
                yield X_batch, y_batch, w_batch
            else:
                yield X_batch, y_batch


def split_data(X: np.ndarray, y: np.ndarray, 
               train_ratio: float = 0.8, 
               val_ratio: float = 0.1) -> Tuple:
    """Split data chronologically into train, validation, and test sets."""
    n_samples = len(X)
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
