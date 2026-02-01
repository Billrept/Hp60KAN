"""Hyperparameter tuning script."""
import sys
import os

# ========================================
# PATH SETUP - UPDATED FOR NEW LOCATION
# ========================================
# Get the project root directory (Hp60KAN/)
script_dir = os.path.dirname(os.path.abspath(__file__))  # tuning/
parent_dir = os.path.dirname(script_dir)                 # Hyperparameter tuning/
project_root = os.path.dirname(parent_dir)               # Hp60KAN/

# Add project root to Python path
sys.path.insert(0, project_root)
# ========================================

import itertools
import json
from datetime import datetime
import pandas as pd

# These imports now work because project_root is in sys.path
from hp60_kan.train import train_model_pipeline
from tuning.tune_config import TUNING_PARAMS, FIXED_PARAMS, OPTIMIZATION_METRIC


def generate_combinations(param_grid):
    """Generate all combinations from parameter grid."""
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = []
    
    for combo in itertools.product(*values):
        param_dict = dict(zip(keys, combo))
        combinations.append(param_dict)
    
    return combinations


def train_with_params(params, trial_num, total_trials):
    """Train model with specific hyperparameters."""
    print("\n" + "=" * 70)
    print(f"TRIAL {trial_num}/{total_trials}")
    print("=" * 70)
    print("Testing parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print("=" * 70 + "\n")
    
    try:
        all_params = {**FIXED_PARAMS, **params}
        
        model, normalizer, history = train_model_pipeline(**all_params)
        
        final_val_mae = history['val_mae'][-1]
        final_val_mse = history['val_mse'][-1]
        
        result = {
            'trial': trial_num,
            'val_mae': final_val_mae,
            'val_mse': final_val_mse,
            'params': params,
            'status': 'success',
            'history': history
        }
        
        print(f"\n✓ Trial {trial_num} completed: MAE={final_val_mae:.4f}")
        return result
        
    except Exception as e:
        print(f"\n✗ Trial {trial_num} failed: {e}")
        return {
            'trial': trial_num,
            'val_mae': float('inf'),
            'val_mse': float('inf'),
            'params': params,
            'status': 'failed',
            'error': str(e)
        }


def run_grid_search():
    """Run grid search hyperparameter tuning."""
    print("\n" + "=" * 70)
    print("STARTING HYPERPARAMETER TUNING")
    print("=" * 70)
    
    combinations = generate_combinations(TUNING_PARAMS)
    total_trials = len(combinations)
    
    print(f"\nTotal combinations to try: {total_trials}")
    print(f"Estimated time: ~{total_trials * 10} minutes (assuming 10 min per trial)\n")
    
    input("Press Enter to start tuning...")
    
    results = []
    start_time = datetime.now()
    
    for i, params in enumerate(combinations, 1):
        result = train_with_params(params, i, total_trials)
        results.append(result)
        save_results(results, intermediate=True)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    print("\n" + "=" * 70)
    print("TUNING COMPLETE!")
    print("=" * 70)
    print(f"Total time: {duration:.1f} minutes")
    
    save_results(results, intermediate=False)
    print_summary(results)
    
    return results


def save_results(results, intermediate=False):
    """Save results to JSON and CSV."""
    # Create results directory inside tuning folder
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if intermediate:
        json_path = os.path.join(results_dir, 'tuning_results_intermediate.json')
        csv_path = os.path.join(results_dir, 'tuning_results_intermediate.csv')
    else:
        json_path = os.path.join(results_dir, f'tuning_results_{timestamp}.json')
        csv_path = os.path.join(results_dir, f'tuning_results_{timestamp}.csv')
    
    # Save to JSON
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save to CSV
    df_results = []
    for r in results:
        row = {
            'trial': r['trial'],
            'val_mae': r['val_mae'],
            'val_mse': r['val_mse'],
            'status': r['status'],
        }
        for key, value in r['params'].items():
            row[f'param_{key}'] = str(value)
        df_results.append(row)
    
    df = pd.DataFrame(df_results)
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")


def print_summary(results):
    """Print summary of tuning results."""
    successful = [r for r in results if r['status'] == 'success']
    
    if not successful:
        print("\n⚠️ No successful trials!")
        return
    
    successful.sort(key=lambda x: x['val_mae'])
    
    print("\n" + "=" * 70)
    print("TOP 5 BEST CONFIGURATIONS")
    print("=" * 70)
    
    for i, result in enumerate(successful[:5], 1):
        print(f"\n#{i} - MAE: {result['val_mae']:.4f}, MSE: {result['val_mse']:.4f}")
        print("Parameters:")
        for key, value in result['params'].items():
            print(f"  {key}: {value}")
    
    best = successful[0]
    print("\n" + "=" * 70)
    print("🏆 BEST CONFIGURATION")
    print("=" * 70)
    print(f"Validation MAE: {best['val_mae']:.4f}")
    print(f"Validation MSE: {best['val_mse']:.4f}")
    print("\nBest parameters:")
    for key, value in best['params'].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    results = run_grid_search()
