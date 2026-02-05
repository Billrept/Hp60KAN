"""Training for Hp60 prediction using MLX-KAN with improvements for high activity capture."""
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

from hp60_kan.data_loader import DataLoader, DataNormalizer, StratifiedBatchSampler, split_data
from hp60_kan.model import Hp60KANModel, create_model
from hp60_kan import config


class Trainer:
    """Trainer class for KAN model with asymmetric loss and stratified sampling."""
    
    def __init__(self, model: Hp60KANModel, learning_rate: float = 5e-5, 
                 weight_decay: float = 1e-5, use_entropy_reg: bool = False,
                 reg_activation: float = 0.01, reg_entropy: float = 0.01,
                 use_sample_weights: bool = False, weight_power: float = 2.0,
                 use_asymmetric_loss: bool = False, asymmetric_alpha: float = 2.0,
                 high_value_threshold: float = 0.4):
        self.model = model
        self.use_entropy_reg = use_entropy_reg
        self.reg_activation = reg_activation
        self.reg_entropy = reg_entropy
        self.use_sample_weights = use_sample_weights
        self.weight_power = weight_power
        self.use_asymmetric_loss = use_asymmetric_loss
        self.asymmetric_alpha = asymmetric_alpha
        self.high_value_threshold = high_value_threshold
        
        self.optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        
        # Select the appropriate loss function
        self._setup_loss_function()
        
        if use_asymmetric_loss:
            print(f"Using asymmetric loss: alpha={asymmetric_alpha}, threshold={high_value_threshold}")
        if use_sample_weights:
            print(f"Using sample weights with power={weight_power}")
        if use_entropy_reg:
            print(f"Using entropy regularization: activation={reg_activation}, entropy={reg_entropy}")
        
        self.history = {'train_loss': [], 'val_mse': [], 'val_mae': [], 
                        'val_high_mae': [], 'val_high_mse': [], 'reg_loss': []}
    
    def _setup_loss_function(self):
        """Setup the appropriate loss function based on configuration."""
        if self.use_sample_weights:
            if self.use_asymmetric_loss:
                if self.use_entropy_reg:
                    self.loss_and_grad_fn = nn.value_and_grad(
                        self.model.model,
                        lambda m, X, y, w: self._weighted_asymmetric_loss_with_reg(m, X, y, w)
                    )
                else:
                    self.loss_and_grad_fn = nn.value_and_grad(
                        self.model.model,
                        lambda m, X, y, w: self._weighted_asymmetric_loss(m, X, y, w)
                    )
            else:
                if self.use_entropy_reg:
                    self.loss_and_grad_fn = nn.value_and_grad(
                        self.model.model, 
                        lambda m, X, y, w: self._weighted_loss_with_regularization(m, X, y, w)
                    )
                else:
                    self.loss_and_grad_fn = nn.value_and_grad(
                        self.model.model, 
                        lambda m, X, y, w: self._weighted_mse_loss(m, X, y, w)
                    )
        else:
            if self.use_asymmetric_loss:
                if self.use_entropy_reg:
                    self.loss_and_grad_fn = nn.value_and_grad(
                        self.model.model,
                        lambda m, X, y: self._asymmetric_loss_with_reg(m, X, y)
                    )
                else:
                    self.loss_and_grad_fn = nn.value_and_grad(
                        self.model.model,
                        lambda m, X, y: self._asymmetric_loss(m, X, y)
                    )
            else:
                if self.use_entropy_reg:
                    self.loss_and_grad_fn = nn.value_and_grad(
                        self.model.model, 
                        lambda m, X, y: self._loss_with_regularization(m, X, y)
                    )
                else:
                    self.loss_and_grad_fn = nn.value_and_grad(self.model.model, self._mse_loss)
    
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
    
    def _asymmetric_loss(self, model, X, y):
        """
        Progressive asymmetric loss that penalizes under-prediction more severely
        as the target value increases. This forces the model to capture high activity.
        
        For high y values (y > threshold):
        - Under-prediction penalty = alpha * (1 + y)^2
        - This makes extreme events VERY costly to under-predict
        """
        predictions = model(X)
        errors = predictions - y  # positive = over-prediction, negative = under-prediction
        squared_errors = errors ** 2
        
        # Progressive penalty: penalty increases quadratically with y value
        # For y=0.3 (threshold): penalty = alpha * 1.69
        # For y=0.5: penalty = alpha * 2.25
        # For y=0.8: penalty = alpha * 3.24
        # For y=1.0: penalty = alpha * 4.0
        progressive_alpha = self.asymmetric_alpha * (1.0 + y) ** 2
        
        is_high = y > self.high_value_threshold
        is_under_predicted = errors < 0
        
        # Apply progressive penalty only for under-predictions of high values
        alpha_mask = mx.where(
            is_high & is_under_predicted,
            progressive_alpha,
            mx.array(1.0)
        )
        
        asymmetric_errors = squared_errors * alpha_mask
        return mx.mean(asymmetric_errors)
    
    def _weighted_asymmetric_loss(self, model, X, y, weights):
        """Weighted version of progressive asymmetric loss."""
        predictions = model(X)
        errors = predictions - y
        squared_errors = errors ** 2
        
        # Progressive penalty
        progressive_alpha = self.asymmetric_alpha * (1.0 + y) ** 2
        
        is_high = y > self.high_value_threshold
        is_under_predicted = errors < 0
        
        alpha_mask = mx.where(
            is_high & is_under_predicted,
            progressive_alpha,
            mx.array(1.0)
        )
        
        asymmetric_errors = squared_errors * alpha_mask * weights
        return mx.mean(asymmetric_errors)
    
    def _asymmetric_loss_with_reg(self, model, X, y):
        """Asymmetric loss with entropy regularization."""
        loss = self._asymmetric_loss(model, X, y)
        reg_loss = model.regularization_loss(
            regularize_activation=self.reg_activation,
            regularize_entropy=self.reg_entropy
        )
        return loss + reg_loss
    
    def _weighted_asymmetric_loss_with_reg(self, model, X, y, weights):
        """Weighted progressive asymmetric loss with entropy regularization."""
        predictions = model(X)
        errors = predictions - y
        squared_errors = errors ** 2
        
        # Progressive penalty
        progressive_alpha = self.asymmetric_alpha * (1.0 + y) ** 2
        
        is_high = y > self.high_value_threshold
        is_under_predicted = errors < 0
        
        alpha_mask = mx.where(
            is_high & is_under_predicted,
            progressive_alpha,
            mx.array(1.0)
        )
        
        asymmetric_errors = squared_errors * alpha_mask * weights
        mse_loss = mx.mean(asymmetric_errors)
        
        reg_loss = model.regularization_loss(
            regularize_activation=self.reg_activation,
            regularize_entropy=self.reg_entropy
        )
        return mse_loss + reg_loss
    
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
                     batch_size: int, weights_train: mx.array = None,
                     stratified_sampler: StratifiedBatchSampler = None) -> float:
        total_loss = 0
        num_batches = 0
        
        if stratified_sampler is not None:
            # Use stratified sampling
            X_np = np.array(X_train)
            y_np = np.array(y_train)
            w_np = np.array(weights_train) if weights_train is not None else None
            
            for batch_data in stratified_sampler.iterate(batch_size, X_np, y_np, w_np):
                if self.use_sample_weights and weights_train is not None:
                    X_batch, y_batch, w_batch = batch_data
                    X_batch = mx.array(X_batch)
                    y_batch = mx.array(y_batch)
                    w_batch = mx.array(w_batch)
                    loss, grads = self.loss_and_grad_fn(self.model.model, X_batch, y_batch, w_batch)
                else:
                    X_batch, y_batch = batch_data
                    X_batch = mx.array(X_batch)
                    y_batch = mx.array(y_batch)
                    loss, grads = self.loss_and_grad_fn(self.model.model, X_batch, y_batch)
                
                self.optimizer.update(self.model.model, grads)
                mx.eval(self.model.model.parameters(), self.optimizer.state)
                total_loss += loss.item()
                num_batches += 1
        elif self.use_sample_weights and weights_train is not None:
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
    
    def _evaluate(self, X: mx.array, y: mx.array, batch_size: int = 1024,
                  high_threshold: float = 0.4) -> Tuple[float, float, float, float]:
        """Evaluate with both overall and high-activity specific metrics."""
        total_mse = 0
        total_mae = 0
        high_mse_sum = 0
        high_mae_sum = 0
        high_count = 0
        num_batches = 0
        
        n_samples = X.shape[0]
        for s in range(0, n_samples, batch_size):
            X_batch = X[s:s + batch_size]
            y_batch = y[s:s + batch_size]
            
            predictions = self.model(X_batch)
            
            # Overall metrics
            mse = mx.mean((predictions - y_batch) ** 2)
            mae = mx.mean(mx.abs(predictions - y_batch))
            
            total_mse += mse.item()
            total_mae += mae.item()
            num_batches += 1
            
            # High activity metrics
            y_np = np.array(y_batch).flatten()
            pred_np = np.array(predictions).flatten()
            high_mask = y_np > high_threshold
            
            if np.any(high_mask):
                high_errors = pred_np[high_mask] - y_np[high_mask]
                high_mse_sum += np.mean(high_errors ** 2) * np.sum(high_mask)
                high_mae_sum += np.mean(np.abs(high_errors)) * np.sum(high_mask)
                high_count += np.sum(high_mask)
        
        avg_mse = total_mse / num_batches
        avg_mae = total_mae / num_batches
        
        # High activity averages
        if high_count > 0:
            high_mse = high_mse_sum / high_count
            high_mae = high_mae_sum / high_count
        else:
            high_mse = 0.0
            high_mae = 0.0
        
        return avg_mse, avg_mae, high_mse, high_mae
    
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
              weights_train: mx.array = None,
              stratified_sampler: StratifiedBatchSampler = None) -> Dict:
        best_val_mse = float('inf')
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Batch size: {batch_size}")
        if update_grid_every > 0:
            print(f"Grid update: every {update_grid_every} epochs")
        if stratified_sampler is not None:
            print(f"Stratified sampling: enabled")
        print("-" * 80)
        print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val MSE':>10} | {'Val MAE':>10} | "
              f"{'High MSE':>10} | {'High MAE':>10}")
        print("-" * 80)
        
        for epoch in range(epochs):
            if update_grid_every > 0 and (epoch + 1) % update_grid_every == 0:
                n_samples = min(grid_update_sample_size, X_train.shape[0])
                indices = np.random.permutation(X_train.shape[0])[:n_samples]
                X_sample = mx.array(np.array(X_train)[indices])
                self.update_grid(X_sample)
            
            train_loss = self._train_epoch(X_train, y_train, batch_size, weights_train, stratified_sampler)
            val_mse, val_mae, val_high_mse, val_high_mae = self._evaluate(X_val, y_val)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_mse'].append(val_mse)
            self.history['val_mae'].append(val_mae)
            self.history['val_high_mse'].append(val_high_mse)
            self.history['val_high_mae'].append(val_high_mae)
            
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_marker = " *"
            else:
                best_marker = ""
            
            print(f"{epoch+1:>6}/{epochs} | {train_loss:>12.6f} | {val_mse:>10.6f} | {val_mae:>10.6f} | "
                  f"{val_high_mse:>10.6f} | {val_high_mae:>10.6f}{best_marker}")
        
        print("-" * 80)
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
                         use_sample_weights: bool = None, weight_power: float = None,
                         use_log_transform: bool = None,
                         use_asymmetric_loss: bool = None, asymmetric_alpha: float = None,
                         high_value_threshold: float = None,
                         use_stratified_sampling: bool = None,
                         high_activity_ratio: float = None,
                         high_activity_percentile: float = None):
    """Complete training pipeline with improvements for high activity capture."""
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
    
    # New parameters for high activity capture
    use_log_transform = use_log_transform if use_log_transform is not None else config.USE_LOG_TRANSFORM
    use_asymmetric_loss = use_asymmetric_loss if use_asymmetric_loss is not None else config.USE_ASYMMETRIC_LOSS
    asymmetric_alpha = asymmetric_alpha if asymmetric_alpha is not None else config.ASYMMETRIC_ALPHA
    high_value_threshold = high_value_threshold if high_value_threshold is not None else config.HIGH_VALUE_THRESHOLD
    use_stratified_sampling = use_stratified_sampling if use_stratified_sampling is not None else config.USE_STRATIFIED_SAMPLING
    high_activity_ratio = high_activity_ratio if high_activity_ratio is not None else config.HIGH_ACTIVITY_RATIO
    high_activity_percentile = high_activity_percentile if high_activity_percentile is not None else config.HIGH_ACTIVITY_PERCENTILE
    
    print("=" * 80)
    print("Hp60 KAN Model Training (Enhanced for High Activity Capture)")
    print("=" * 80)
    
    print("\n[1/7] Loading data...")
    loader = DataLoader(data_path, use_ratio=use_ratio)
    loader.load()
    
    print("\n[2/7] Creating cyclical features...")
    loader.create_cyclical_features()
    
    print("\n[3/7] Preparing features...")
    X, y = loader.prepare_features(config.FEATURE_COLUMNS, config.TARGET_COLUMN)
    
    print("\n[4/7] Normalizing data...")
    normalizer = DataNormalizer(use_log_transform=use_log_transform)
    X_norm, y_norm = normalizer.fit_transform(X, y)
    
    print("\n[5/7] Splitting data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(
        X_norm, y_norm, config.TRAIN_RATIO, config.VAL_RATIO
    )
    
    # Create stratified sampler if enabled
    stratified_sampler = None
    if use_stratified_sampling:
        print("\n[5.5/7] Setting up stratified sampling...")
        stratified_sampler = StratifiedBatchSampler(
            y_train, 
            high_activity_ratio=high_activity_ratio,
            high_activity_percentile=high_activity_percentile
        )
    
    # Compute sample weights
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
    
    print("\n[6/7] Creating and training model...")
    input_dim = len(config.FEATURE_COLUMNS)
    model = create_model(input_dim, hidden_dims)
    
    trainer = Trainer(
        model, learning_rate=learning_rate, weight_decay=weight_decay,
        use_entropy_reg=use_entropy_reg, reg_activation=reg_activation,
        reg_entropy=reg_entropy, use_sample_weights=use_sample_weights,
        weight_power=weight_power, use_asymmetric_loss=use_asymmetric_loss,
        asymmetric_alpha=asymmetric_alpha, high_value_threshold=high_value_threshold
    )
    
    history = trainer.train(
        X_train_mx, y_train_mx, X_val_mx, y_val_mx, 
        epochs=epochs, batch_size=batch_size,
        update_grid_every=update_grid_every,
        grid_update_sample_size=grid_update_sample_size,
        weights_train=weights_train,
        stratified_sampler=stratified_sampler
    )
    
    print("\n" + "=" * 80)
    print("EVALUATION ON TEST SET")
    print("=" * 80)
    test_mse, test_mae, test_high_mse, test_high_mae = trainer._evaluate(X_test_mx, y_test_mx)
    
    predictions = model(X_test_mx)
    ss_res = mx.sum((y_test_mx - predictions) ** 2)
    ss_tot = mx.sum((y_test_mx - mx.mean(y_test_mx)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    y_pred_denorm = normalizer.inverse_transform_y(np.array(predictions))
    y_test_denorm = normalizer.inverse_transform_y(np.array(y_test_mx))
    
    test_mae_original = np.mean(np.abs(y_pred_denorm - y_test_denorm))
    test_rmse_original = np.sqrt(np.mean((y_pred_denorm - y_test_denorm) ** 2))
    
    # Calculate high activity metrics in original scale
    # Find samples where original Hp60 > 4
    y_test_orig_full = y[int(len(y) * (config.TRAIN_RATIO + config.VAL_RATIO)):]
    high_mask = y_test_orig_full > 4.0
    if np.any(high_mask):
        high_mae_orig = np.mean(np.abs(y_pred_denorm.flatten()[high_mask] - y_test_denorm.flatten()[high_mask]))
        high_rmse_orig = np.sqrt(np.mean((y_pred_denorm.flatten()[high_mask] - y_test_denorm.flatten()[high_mask]) ** 2))
        high_count = np.sum(high_mask)
    else:
        high_mae_orig = 0.0
        high_rmse_orig = 0.0
        high_count = 0
    
    print(f"\nNormalized metrics:")
    print(f"  Overall MSE: {test_mse:.6f}")
    print(f"  Overall MAE: {test_mae:.6f}")
    print(f"  High-Activity MSE: {test_high_mse:.6f}")
    print(f"  High-Activity MAE: {test_high_mae:.6f}")
    print(f"  R²:  {r2.item():.6f}")
    print(f"\nOriginal scale metrics:")
    print(f"  Overall MAE:  {test_mae_original:.4f} Hp60 units")
    print(f"  Overall RMSE: {test_rmse_original:.4f} Hp60 units")
    print(f"\nHigh Activity (Hp60 > 4) metrics ({high_count} samples):")
    print(f"  High-Activity MAE:  {high_mae_orig:.4f} Hp60 units")
    print(f"  High-Activity RMSE: {high_rmse_orig:.4f} Hp60 units")
    
    print("\n[7/7] Saving model...")
    model.save(model_save_dir)
    normalizer.save(scaler_save_path)
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    
    return model, normalizer, history


if __name__ == "__main__":
    model, normalizer, history = train_model_pipeline()
