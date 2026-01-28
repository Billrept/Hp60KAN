# Hp60 Geomagnetic Index Prediction with KAN

A machine learning project using **Kolmogorov-Arnold Networks (KAN)** to predict the Hp60 geomagnetic index from solar wind and temporal features. Built with Apple's MLX framework for efficient computation on Apple Silicon.


### Key Features

- **KAN Architecture**: Uses B-splines instead of fixed activation functions for learnable, interpretable transformations
- **12 Engineered Features**: Cyclical time encoding + interaction features for extreme event prediction
- **Sample Weighting**: Emphasizes rare extreme events during training

## Model Performance

| Metric | Value |
|--------|-------|
| MAE | 0.56 Hp60 units |
| RMSE | 0.73 Hp60 units |
| R² | 0.72 |

## Project Structure

```
hp60_kan/
├── config.py          # Configuration settings
├── data_loader.py     # Data loading and feature engineering
├── model.py           # KAN model wrapper
├── train.py           # Training logic with KAN optimizations
├── predict.py         # Hp60Predictor class for inference
├── visualize.py       # Visualization utilities
├── evaluate.py        # Evaluation on held-out data
└── __init__.py

main.py                # CLI entry point
hpodata_2000-2025.csv  # Geomagnetic data (2000-2025)
saved_models/          # Trained model files
outputs/               # Training plots and analysis
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
python main.py train
```

This will:
1. Load and preprocess the Hp60 dataset (90% of data)
2. Create 12 engineered features including interaction terms
3. Train a KAN model with architecture [12, 64, 64, 64, 1]
4. Save the model to `saved_models/hp60_kan/`

### Evaluation

```bash
python main.py evaluate
```

Evaluates the model on the reserved 10% of data and generates error analysis plots.

### Prediction (Python API)

```python
from hp60_kan import Hp60Predictor

# Load trained model
predictor = Hp60Predictor()

# Predict Hp60 from ap60 and datetime
hp60 = predictor.predict(ap60=45, dt=datetime(2024, 5, 15, 12, 0))
print(f"Predicted Hp60: {hp60:.2f}")
```

## Features

The model uses 12 features:

| Feature | Description |
|---------|-------------|
| `hour_sin`, `hour_cos` | Cyclical encoding of hour (0-24) |
| `month_sin`, `month_cos` | Cyclical encoding of month (1-12) |
| `doy_sin`, `doy_cos` | Cyclical encoding of day of year |
| `ap60` | Normalized ap60 index |
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

| Parameter | Value |
|-----------|-------|
| Epochs | 60 |
| Batch size | 256 |
| Learning rate | 5e-5 |
| Weight decay | 1e-5 |
| Entropy regularization | 0.01 |
| Sample weight power | 2.0 |
