# Hp60 Geomagnetic Index Prediction with KAN

A machine learning project using **Kolmogorov-Arnold Networks (KAN)** to predict the Hp60 geomagnetic index from ap60 and temporal features. Built with Apple's MLX framework for efficient computation on Apple Silicon.

## Highlights

- **R² = 0.993** on held-out test data
- **Captures extreme geomagnetic storms** (Hp60 > 7) with MAE < 0.1
- **Progressive asymmetric loss** to prioritize high-activity prediction
- **Stratified sampling** ensures rare storm events are well-represented in training

## Model Performance

### Overall Metrics (Reserved 10% Data)

| Metric | Value |
|--------|-------|
| MAE | 0.095 Hp60 units |
| RMSE | 0.118 Hp60 units |
| R² | 0.993 |
| Pearson Correlation | 0.996 |

### Performance by Geomagnetic Activity Level

| Activity Level | ap60 Range | MAE | Samples |
|----------------|------------|-----|---------|
| Quiet | 0-10 | 0.122 | 14,616 |
| Low | 10-30 | 0.044 | 6,172 |
| Moderate | 30-50 | 0.054 | 1,329 |
| Active | 50-100 | 0.063 | 496 |
| Storm | 100-200 | 0.099 | 123 |
| Severe | 200+ | 0.131 | 57 |

### Performance by Hp60 Level

| Hp60 Range | MAE | Bias | Interpretation |
|------------|-----|------|----------------|
| 0-1 | 0.149 | +0.107 | Slight over-prediction |
| 1-2 | 0.097 | -0.066 | Slight under-prediction |
| 2-3 | 0.074 | -0.040 | Good |
| 3-4 | 0.049 | +0.040 | Excellent |
| 4-5 | 0.054 | +0.033 | Excellent |
| 5-7 | 0.069 | +0.064 | Good (safely over-predicts) |
| 7-9 | 0.087 | +0.071 | Good (safely over-predicts) |

The model has a **positive bias for extreme events (Hp60 > 5)**, meaning it slightly over-predicts storms. This is the desired behavior for a geomagnetic warning system where missing a storm is more dangerous than a false alarm.

## The Challenge: Capturing High Magnetic Activity

### Problem Statement

The Hp60 distribution is **severely imbalanced**:
- 38% of samples have Hp60 in range 0-1
- 27% in range 1-2
- Only 4% in range 4-5
- Less than 1% above Hp60 = 6

Standard MSE loss causes models to regress toward the mean, systematically **under-predicting rare extreme events**. Initial models predicted Hp60 = 6.7 for actual values of 10.6 during severe storms.

### Solution: Multi-Pronged Approach

We implemented several techniques to force the model to capture high-activity trends:

#### 1. Progressive Asymmetric Loss

Instead of standard MSE, we use an asymmetric loss that **penalizes under-prediction of high values quadratically**:

```python
# For high y values, penalty increases with target magnitude
progressive_alpha = alpha * (1 + y)^2

# Under-prediction of high values gets penalized more
loss = MSE * progressive_alpha  # when y > threshold and prediction < y
```

Configuration (after tuning):
- `ASYMMETRIC_ALPHA = 6.0` - Base penalty multiplier
- `HIGH_VALUE_THRESHOLD = 0.35` - Normalized threshold for "high" values

#### 2. Stratified Batch Sampling

Every training batch contains a **guaranteed proportion of high-activity samples**:

```python
# 40% of each batch must be high-activity samples (tuned from 50%)
HIGH_ACTIVITY_RATIO = 0.4
HIGH_ACTIVITY_PERCENTILE = 70  # Top 30% of Hp60 values
```

This ensures the model sees extreme events in every batch, not just occasionally.

#### 3. Sample Weighting

Higher Hp60 values receive exponentially higher sample weights:

```python
weight = 1 + (Hp60 / max_Hp60)^2.5  # tuned from 3.0
```

With `WEIGHT_POWER = 2.5`, a sample with Hp60 = 9 has ~2x the weight of a quiet sample.

#### 4. Log Transform Option

Optional log transform of the target variable to compress the range:
- Currently disabled (`USE_LOG_TRANSFORM = False`) as it compressed high values too much

### Results: Before vs After

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Severe storm MAE | 0.743 | 0.131 | **82% better** |
| Storm MAE | 0.294 | 0.099 | **66% better** |
| Hp60 5-7 MAE | 0.188 | 0.069 | **63% better** |
| Hp60 7-9 MAE | 0.358 | 0.087 | **76% better** |
| Hp60 7-9 bias | -0.11 (under) | +0.07 (over) | **Now safe!** |

Example - May 2024 Severe Storm (ap60 = 456):
- **Before**: Actual 10.66 → Predicted 6.71 (missed by 4 units!)
- **After**: Actual 10.66 → Predicted 11.46 (slight over-prediction, safe!)

## Project Structure

```
hp60_kan/
├── config.py          # Configuration and hyperparameters
├── data_loader.py     # Data loading, normalization, stratified sampling
├── model.py           # KAN model wrapper
├── train.py           # Training with asymmetric loss
├── predict.py         # Hp60Predictor class for inference
├── visualize.py       # Visualization utilities
├── evaluate.py        # Evaluation on held-out data
└── __init__.py

main.py                # CLI entry point
hpodata_2000-2025.csv  # Geomagnetic data (2000-2025, ~228K samples)
saved_models/          # Trained model files
outputs/               # Training plots and analysis
Hyperparameter Tuning/ # Optuna hyperparameter search results
```

## Installation

### Requirements

- Python 3.10+
- Apple Silicon Mac (for MLX acceleration)

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install mlx numpy pandas matplotlib

# Install mlx-kan library
pip install -e ./mlx-kan
```

## Usage

### Training

```bash
source venv/bin/activate
python -m hp60_kan.train
```

This will:
1. Load and preprocess the Hp60 dataset (90% for training/validation)
2. Create 12 engineered features including interaction terms
3. Set up stratified sampling for high-activity events
4. Train with progressive asymmetric loss
5. Save the model to `saved_models/hp60_kan/`

Training output shows both overall and high-activity metrics:
```
Epoch |   Train Loss |    Val MSE |    Val MAE |   High MSE |   High MAE
   60 |     0.091727 |   0.000171 |   0.011131 |   0.000036 |   0.004177
```

### Evaluation

```bash
python -m hp60_kan.evaluate
```

Evaluates on the reserved 10% of data (2023-2025) and generates:
- Performance breakdown by hour, month, season
- Performance by geomagnetic activity level
- Performance by Hp60 level with bias analysis
- Top 20 worst predictions
- Error analysis plots

### Prediction (Python API)

```python
from datetime import datetime
from hp60_kan import Hp60Predictor

# Load trained model
predictor = Hp60Predictor()

# Predict Hp60 from ap60 and datetime
hp60 = predictor.predict(ap60=45, dt=datetime(2024, 5, 15, 12, 0))
print(f"Predicted Hp60: {hp60:.2f}")

# Predict for storm conditions
hp60_storm = predictor.predict(ap60=300, dt=datetime(2024, 5, 11, 2, 0))
print(f"Storm Hp60: {hp60_storm:.2f}")  # Will predict ~9-10
```

## Features

The model uses 12 engineered features:

| Feature | Description |
|---------|-------------|
| `hour_sin`, `hour_cos` | Cyclical encoding of hour (0-24) |
| `month_sin`, `month_cos` | Cyclical encoding of month (1-12) |
| `doy_sin`, `doy_cos` | Cyclical encoding of day of year |
| `ap60` | Raw ap60 index |
| `ap60_hour_sin`, `ap60_hour_cos` | Interaction: ap60 × hour pattern |
| `ap60_month_sin`, `ap60_month_cos` | Interaction: ap60 × seasonal pattern |
| `ap60_log` | Log transform of ap60 for extreme values |

## Model Architecture

```
KAN Model: [12, 64, 64, 64, 1]
- Input: 12 features
- Hidden layers: 3 × 64 neurons with B-spline activations
- Output: 1 (Hp60 prediction)
- Total parameters: 81,409
- Grid size: 5, Spline order: 3
```

## Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Epochs | 60 | Training iterations |
| Batch size | 256 | Samples per batch |
| Learning rate | 1e-4 | AdamW learning rate |
| Weight decay | 1e-5 | L2 regularization |
| Entropy regularization | 0.01 | KAN-specific regularization |
| **Sample weight power** | 3.0 | Exponential weighting for high values |
| **Asymmetric alpha** | 8.0 | Under-prediction penalty multiplier |
| **High value threshold** | 0.3 | Threshold for asymmetric loss |
| **Stratified ratio** | 0.5 | 50% high-activity samples per batch |
| **High activity percentile** | 70 | Top 30% considered "high activity" |

## Hyperparameter Tuning

Hyperparameter optimization was performed using Optuna with 27 trials. Key findings:
- Hidden dimensions [64, 64, 64] performed best
- Learning rate 1e-4 optimal
- Higher weight power (3.0) better for extreme events

Results are saved in `Hyperparameter Tuning/` directory.

## Data

The model is trained on `hpodata_2000-2025.csv`:
- **Period**: 2000-2025 (25 years of data)
- **Samples**: ~228,000 hourly observations
- **Split**: 90% training (2000-2023), 10% reserved test (2023-2025)
- **Features**: DateTime, Hp60, ap60, and derived indices

## Key Files

| File | Description |
|------|-------------|
| `config.py` | All hyperparameters and paths |
| `data_loader.py` | `DataLoader`, `DataNormalizer`, `StratifiedBatchSampler` |
| `train.py` | `Trainer` class with progressive asymmetric loss |
| `evaluate.py` | Comprehensive evaluation with breakdown analysis |
| `predict.py` | `Hp60Predictor` for inference |

## References

- **KAN Paper**: [Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756)
- **MLX-KAN**: Implementation of KAN for Apple MLX framework
- **Hp60 Index**: High-resolution geomagnetic activity index (Hpo indices)

## License

MIT License
