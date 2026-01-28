"""Training for Hp60 prediction using MLX-KAN."""
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(parent_dir, 'mlx-kan'))

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Dict, Tuple
from trainer.sft import batch_iterate

from hp60_kan.data_loader import DataLoader, DataNormalizer, split_data
from hp60_kan.model import Hp60KANModel, create_model
from hp60_kan import config


class Trainer:
    """Trainer class for KAN model."""
    
    def __init__(self, model: Hp60KANModel, learning_rate: float = 5e-5, 
                 weight_decay: float = 1e-5, use_entropy_reg: bool = False,
                 reg_activation: float = 0.01, reg_entropy: float = 0.01,
                 use_sample_weights: bool = False, weight_power: float = 2.0):
        self.model = model
        self.use_entropy_reg = use_entropy_reg
        self.reg_activation = reg_activation
        self.reg_entropy = reg_entropy
        self.use_sample_weights = use_sample_weights
        self.weight_power = weight_power
        
        self.optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        
        if use_sample_weights:
            if use_entropy_reg:
                self.loss_and_grad_fn = nn.value_and_grad(
                    model.model, 
                    lambda m, X, y, w: self._weighted_loss_with_regularization(m, X, y, w)
                )
            else:
                self.loss_and_grad_fn = nn.value_and_grad(
                    model.model, 
                    lambda m, X, y, w: self._weighted_mse_loss(m, X, y, w)
                )
            print(f"Using sample weights with power={weight_power}")
        else:
            if use_entropy_reg:
                self.loss_and_grad_fn = nn.value_and_grad(
                    model.model, 
                    lambda m, X, y: self._loss_with_regularization(m, X, y)
                )
            else:
                self.loss_and_grad_fn = nn.value_and_grad(model.model, self._mse_loss)
        
        if use_entropy_reg:
            print(f"Using entropy regularization: activation={reg_activation}, entropy={reg_entropy}")
        
        self.history = {'train_loss': [], 'val_mse': [], 'val_mae': [], 'reg_loss': []}
        
    @staticmethod
    def _mse_loss(model, X, y):
        predictions = model(X)
        return mx.mean((predictions - y) ** 2)
    
    @staticmethod
    def _weighted_mse_loss(model, X, y, weights):
        predictions = model(X)
        squared_errors = (predictions - y) ** 2
        weighted_errors = squared_errors * weights
        return mx.mean(weighted_errors)
    
    def _loss_with_regularization(self, model, X, y):
        predictions = model(X)
        mse_loss = mx.mean((predictions - y) ** 2)
        reg_loss = model.regularization_loss(
            regularize_activation=self.reg_activation,
            regularize_entropy=self.reg_entropy
        )
        return mse_loss + reg_loss
    
    def _weighted_loss_with_regularization(self, model, X, y, weights):
        predictions = model(X)
        squared_errors = (predictions - y) ** 2
        weighted_errors = squared_errors * weights
        mse_loss = mx.mean(weighted_errors)
        reg_loss = model.regularization_loss(
            regularize_activation=self.reg_activation,
            regularize_entropy=self.reg_entropy
        )
        return mse_loss + reg_loss
    
    def _batch_iterate_with_weights(self, batch_size, X, y, weights):
        n = X.shape[0]
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        for s in range(0, n, batch_size):
            batch_idx = indices[s:s + batch_size]
            X_batch = mx.array(np.array(X)[batch_idx])
            y_batch = mx.array(np.array(y)[batch_idx])
            w_batch = mx.array(np.array(weights)[batch_idx])
            yield X_batch, y_batch, w_batch
    
    def _train_epoch(self, X_train: mx.array, y_train: mx.array, 
                     batch_size: int, weights_train: mx.array = None) -> float:
        total_loss = 0
        num_batches = 0
        
        if self.use_sample_weights and weights_train is not None:
            for X_batch, y_batch, w_batch in self._batch_iterate_with_weights(batch_size, X_train, y_train, weights_train):
                loss, grads = self.loss_and_grad_fn(self.model.model, X_batch, y_batch, w_batch)
                self.optimizer.update(self.model.model, grads)
                mx.eval(self.model.model.parameters(), self.optimizer.state)
                total_loss += loss.item()
                num_batches += 1
        else:
            for X_batch, y_batch in batch_iterate(batch_size, X_train, y_train):
                loss, grads = self.loss_and_grad_fn(self.model.model, X_batch, y_batch)
                self.optimizer.update(self.model.model, grads)
                mx.eval(self.model.model.parameters(), self.optimizer.state)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _evaluate(self, X: mx.array, y: mx.array, batch_size: int = 1024) -> Tuple[float, float]:
        total_mse = 0
        total_mae = 0
        num_batches = 0
        
        n_samples = X.shape[0]
        for s in range(0, n_samples, batch_size):
            X_batch = X[s:s + batch_size]
            y_batch = y[s:s + batch_size]
            
            predictions = self.model(X_batch)
            mse = mx.mean((predictions - y_batch) ** 2)
            mae = mx.mean(mx.abs(predictions - y_batch))
            
            total_mse += mse.item()
            total_mae += mae.item()
            num_batches += 1
        
        return total_mse / num_batches, total_mae / num_batches
    
    def update_grid(self, X_sample: mx.array, margin: float = 0.01):
        print("    Updating grid based on data distribution...")
        for layer in self.model.model.layers:
            layer.update_grid(X_sample, margin=margin)
        print("    Grid updated successfully.")
    
    def train(self, X_train: mx.array, y_train: mx.array,
              X_val: mx.array, y_val: mx.array,
              epochs: int, batch_size: int,
              update_grid_every: int = 0,
              grid_update_sample_size: int = 2000,
              weights_train: mx.array = None) -> Dict:
        best_val_mse = float('inf')
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Batch size: {batch_size}")
        if update_grid_every > 0:
            print(f"Grid update: every {update_grid_every} epochs")
        if self.use_entropy_reg:
            print(f"Entropy regularization: enabled")
        if self.use_sample_weights:
            print(f"Sample weighting: enabled (power={self.weight_power})")
        print("-" * 60)
        
        for epoch in range(epochs):
            if update_grid_every > 0 and (epoch + 1) % update_grid_every == 0:
                n_samples = min(grid_update_sample_size, X_train.shape[0])
                indices = np.random.permutation(X_train.shape[0])[:n_samples]
                X_sample = mx.array(np.array(X_train)[indices])
                self.update_grid(X_sample)
            
            train_loss = self._train_epoch(X_train, y_train, batch_size, weights_train)
            val_mse, val_mae = self._evaluate(X_val, y_val)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_mse'].append(val_mse)
            self.history['val_mae'].append(val_mae)
            
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_marker = " *"
            else:
                best_marker = ""
            
            print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.6f} | "
                  f"Val MSE: {val_mse:.6f} | Val MAE: {val_mae:.6f}{best_marker}")
        
        print("-" * 60)
        print(f"Best validation MSE: {best_val_mse:.6f}")
        
        return self.history
    
    def get_history(self) -> Dict:
        return self.history


def train_model_pipeline(data_path: str = None, model_save_dir: str = None,
                         epochs: int = None, batch_size: int = None,
                         learning_rate: float = None, weight_decay: float = None,
                         hidden_dims: list = None, use_ratio: float = None,
                         use_entropy_reg: bool = None, reg_activation: float = None,
                         reg_entropy: float = None, update_grid_every: int = None,
                         grid_update_sample_size: int = None,
                         use_sample_weights: bool = None, weight_power: float = None):
    """Complete training pipeline."""
    data_path = data_path or config.DATA_PATH
    model_save_dir = model_save_dir or config.MODEL_DIR
    scaler_save_path = config.SCALER_PATH
    epochs = epochs or config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE
    learning_rate = learning_rate or config.LEARNING_RATE
    weight_decay = weight_decay or config.WEIGHT_DECAY
    hidden_dims = hidden_dims or config.HIDDEN_DIMS
    use_ratio = use_ratio or config.DATA_USE_RATIO
    
    use_entropy_reg = use_entropy_reg if use_entropy_reg is not None else config.USE_ENTROPY_REG
    reg_activation = reg_activation if reg_activation is not None else config.REG_ACTIVATION
    reg_entropy = reg_entropy if reg_entropy is not None else config.REG_ENTROPY
    update_grid_every = update_grid_every if update_grid_every is not None else config.UPDATE_GRID_EVERY
    grid_update_sample_size = grid_update_sample_size or config.GRID_UPDATE_SAMPLE_SIZE
    
    use_sample_weights = use_sample_weights if use_sample_weights is not None else config.USE_SAMPLE_WEIGHTS
    weight_power = weight_power if weight_power is not None else config.WEIGHT_POWER
    
    print("=" * 60)
    print("Hp60 KAN Model Training")
    print("=" * 60)
    
    print("\n[1/6] Loading data...")
    loader = DataLoader(data_path, use_ratio=use_ratio)
    loader.load()
    
    print("\n[2/6] Creating cyclical features...")
    loader.create_cyclical_features()
    
    print("\n[3/6] Preparing features...")
    X, y = loader.prepare_features(config.FEATURE_COLUMNS, config.TARGET_COLUMN)
    
    print("\n[4/6] Normalizing data...")
    normalizer = DataNormalizer()
    X_norm, y_norm = normalizer.fit_transform(X, y)
    
    print("\n[5/6] Splitting data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(
        X_norm, y_norm, config.TRAIN_RATIO, config.VAL_RATIO
    )
    
    weights_train = None
    if use_sample_weights:
        n_total = len(y)
        train_end = int(n_total * config.TRAIN_RATIO)
        y_train_original = y[:train_end]
        
        max_hp60 = y_train_original.max()
        weights = 1.0 + (y_train_original / max_hp60) ** weight_power
        weights = weights / weights.mean()
        
        print(f"Sample weights computed: min={weights.min():.2f}, max={weights.max():.2f}, mean={weights.mean():.2f}")
        weights_train = mx.array(weights.astype(np.float32)).reshape(-1, 1)
    
    X_train_mx = mx.array(X_train.astype(np.float32))
    y_train_mx = mx.array(y_train.astype(np.float32)).reshape(-1, 1)
    X_val_mx = mx.array(X_val.astype(np.float32))
    y_val_mx = mx.array(y_val.astype(np.float32)).reshape(-1, 1)
    X_test_mx = mx.array(X_test.astype(np.float32))
    y_test_mx = mx.array(y_test.astype(np.float32)).reshape(-1, 1)
    
    print("\n[6/6] Creating and training model...")
    input_dim = len(config.FEATURE_COLUMNS)
    model = create_model(input_dim, hidden_dims)
    
    trainer = Trainer(
        model, learning_rate=learning_rate, weight_decay=weight_decay,
        use_entropy_reg=use_entropy_reg, reg_activation=reg_activation,
        reg_entropy=reg_entropy, use_sample_weights=use_sample_weights,
        weight_power=weight_power
    )
    
    history = trainer.train(
        X_train_mx, y_train_mx, X_val_mx, y_val_mx, 
        epochs=epochs, batch_size=batch_size,
        update_grid_every=update_grid_every,
        grid_update_sample_size=grid_update_sample_size,
        weights_train=weights_train
    )
    
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)
    test_mse, test_mae = trainer._evaluate(X_test_mx, y_test_mx)
    
    predictions = model(X_test_mx)
    ss_res = mx.sum((y_test_mx - predictions) ** 2)
    ss_tot = mx.sum((y_test_mx - mx.mean(y_test_mx)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    y_pred_denorm = normalizer.inverse_transform_y(np.array(predictions))
    y_test_denorm = normalizer.inverse_transform_y(np.array(y_test_mx))
    
    test_mae_original = np.mean(np.abs(y_pred_denorm - y_test_denorm))
    test_rmse_original = np.sqrt(np.mean((y_pred_denorm - y_test_denorm) ** 2))
    
    print(f"\nNormalized metrics:")
    print(f"  MSE: {test_mse:.6f}")
    print(f"  MAE: {test_mae:.6f}")
    print(f"  R²:  {r2.item():.6f}")
    print(f"\nOriginal scale metrics:")
    print(f"  MAE:  {test_mae_original:.4f} Hp60 units")
    print(f"  RMSE: {test_rmse_original:.4f} Hp60 units")
    
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    model.save(model_save_dir)
    normalizer.save(scaler_save_path)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return model, normalizer, history


if __name__ == "__main__":
    model, normalizer, history = train_model_pipeline()
