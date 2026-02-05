"""Hyperparameter tuning script for high-activity capture optimization."""
import sys
import os

# ========================================
# PATH SETUP
# ========================================
script_dir = os.path.dirname(os.path.abspath(__file__))  # tuning/
parent_dir = os.path.dirname(script_dir)                 # Hyperparameter tuning/
project_root = os.path.dirname(parent_dir)               # project root

sys.path.insert(0, project_root)
sys.path.insert(0, parent_dir)
# ========================================

import itertools
import json
from datetime import datetime
import pandas as pd
import numpy as np

from hp60_kan.train import train_model_pipeline
from tuning.tune_config import (
    TUNING_PARAMS, TUNING_PARAMS_SMALL, TUNING_PARAMS_QUICK, FIXED_PARAMS, 
    OPTIMIZATION_METRIC, TRACK_METRICS, USE_SMALL_SEARCH,
    FINAL_TRAINING_EPOCHS
)


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
        
        # Extract final metrics
        final_val_mae = history['val_mae'][-1]
        final_val_mse = history['val_mse'][-1]
        final_val_high_mae = history.get('val_high_mae', [float('inf')])[-1]
        final_val_high_mse = history.get('val_high_mse', [float('inf')])[-1]
        
        # Calculate a combined score (weighted average of overall and high-activity MAE)
        # Prioritize high-activity performance
        combined_score = 0.3 * final_val_mae + 0.7 * final_val_high_mae
        
        result = {
            'trial': trial_num,
            'val_mae': final_val_mae,
            'val_mse': final_val_mse,
            'val_high_mae': final_val_high_mae,
            'val_high_mse': final_val_high_mse,
            'combined_score': combined_score,
            'params': params,
            'status': 'success',
            'history': {k: v for k, v in history.items() if k in TRACK_METRICS}
        }
        
        print(f"\n✓ Trial {trial_num} completed:")
        print(f"  Overall MAE: {final_val_mae:.6f}")
        print(f"  High-Activity MAE: {final_val_high_mae:.6f}")
        print(f"  Combined Score: {combined_score:.6f}")
        return result
        
    except Exception as e:
        import traceback
        print(f"\n✗ Trial {trial_num} failed: {e}")
        traceback.print_exc()
        return {
            'trial': trial_num,
            'val_mae': float('inf'),
            'val_mse': float('inf'),
            'val_high_mae': float('inf'),
            'val_high_mse': float('inf'),
            'combined_score': float('inf'),
            'params': params,
            'status': 'failed',
            'error': str(e)
        }


def run_grid_search(search_mode=None, auto_start=False):
    """Run grid search hyperparameter tuning."""
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING FOR HIGH-ACTIVITY CAPTURE")
    print("=" * 70)
    
    # Select search space based on mode
    if search_mode == 'quick':
        search_params = TUNING_PARAMS_QUICK
        mode_name = 'QUICK'
    elif search_mode == 'small':
        search_params = TUNING_PARAMS_SMALL
        mode_name = 'SMALL'
    else:
        search_params = TUNING_PARAMS
        mode_name = 'FULL'
    
    combinations = generate_combinations(search_params)
    total_trials = len(combinations)
    
    print(f"\nSearch space: {mode_name}")
    print(f"Total combinations to try: {total_trials}")
    print(f"Optimization metric: {OPTIMIZATION_METRIC}")
    print(f"Estimated time: ~{total_trials * 5} minutes (assuming 5 min per trial)\n")
    
    print("Parameters being tuned:")
    for key, values in search_params.items():
        print(f"  {key}: {values}")
    
    print("\nFixed parameters:")
    for key, value in FIXED_PARAMS.items():
        print(f"  {key}: {value}")
    
    if not auto_start:
        try:
            response = input("\nPress Enter to start tuning (or 'q' to quit)... ")
            if response.lower() == 'q':
                print("Tuning cancelled.")
                return []
        except EOFError:
            # Running in non-interactive mode, auto-start
            print("\nNon-interactive mode detected. Starting automatically...")
    else:
        print("\nAuto-start enabled. Beginning tuning...")
    
    results = []
    start_time = datetime.now()
    
    for i, params in enumerate(combinations, 1):
        result = train_with_params(params, i, total_trials)
        results.append(result)
        save_results(results, intermediate=True)
        
        # Print progress
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        avg_time = elapsed / i
        remaining = avg_time * (total_trials - i)
        print(f"\n[Progress] {i}/{total_trials} trials | "
              f"Elapsed: {elapsed:.1f}min | Remaining: ~{remaining:.1f}min")
    
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
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if intermediate:
        json_path = os.path.join(results_dir, 'tuning_results_intermediate.json')
        csv_path = os.path.join(results_dir, 'tuning_results_intermediate.csv')
    else:
        json_path = os.path.join(results_dir, f'tuning_results_{timestamp}.json')
        csv_path = os.path.join(results_dir, f'tuning_results_{timestamp}.csv')
    
    # Prepare results for JSON (remove non-serializable items)
    json_results = []
    for r in results:
        jr = {k: v for k, v in r.items() if k != 'history'}
        jr['history_final'] = {k: v[-1] if v else None for k, v in r.get('history', {}).items()}
        json_results.append(jr)
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Save to CSV
    df_results = []
    for r in results:
        row = {
            'trial': r['trial'],
            'val_mae': r['val_mae'],
            'val_mse': r['val_mse'],
            'val_high_mae': r['val_high_mae'],
            'val_high_mse': r['val_high_mse'],
            'combined_score': r['combined_score'],
            'status': r['status'],
        }
        for key, value in r['params'].items():
            row[f'param_{key}'] = str(value)
        df_results.append(row)
    
    df = pd.DataFrame(df_results)
    df = df.sort_values('combined_score')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")


def print_summary(results):
    """Print summary of tuning results."""
    successful = [r for r in results if r['status'] == 'success']
    
    if not successful:
        print("\n⚠️ No successful trials!")
        return
    
    # Sort by combined score (prioritizes high-activity performance)
    successful.sort(key=lambda x: x['combined_score'])
    
    print("\n" + "=" * 70)
    print("TOP 5 BEST CONFIGURATIONS (by Combined Score)")
    print("=" * 70)
    
    for i, result in enumerate(successful[:5], 1):
        print(f"\n#{i}")
        print(f"  Combined Score: {result['combined_score']:.6f}")
        print(f"  Overall MAE: {result['val_mae']:.6f}")
        print(f"  High-Activity MAE: {result['val_high_mae']:.6f}")
        print("  Parameters:")
        for key, value in result['params'].items():
            print(f"    {key}: {value}")
    
    # Also show best by high-activity MAE only
    by_high_mae = sorted(successful, key=lambda x: x['val_high_mae'])
    
    print("\n" + "=" * 70)
    print("TOP 3 BY HIGH-ACTIVITY MAE ONLY")
    print("=" * 70)
    
    for i, result in enumerate(by_high_mae[:3], 1):
        print(f"\n#{i} - High-Activity MAE: {result['val_high_mae']:.6f} "
              f"(Overall MAE: {result['val_mae']:.6f})")
        print("  Parameters:")
        for key, value in result['params'].items():
            print(f"    {key}: {value}")
    
    best = successful[0]
    print("\n" + "=" * 70)
    print("🏆 BEST CONFIGURATION")
    print("=" * 70)
    print(f"Combined Score: {best['combined_score']:.6f}")
    print(f"Overall MAE: {best['val_mae']:.6f}")
    print(f"Overall MSE: {best['val_mse']:.6f}")
    print(f"High-Activity MAE: {best['val_high_mae']:.6f}")
    print(f"High-Activity MSE: {best['val_high_mse']:.6f}")
    print("\nBest parameters:")
    for key, value in best['params'].items():
        print(f"  {key}: {value}")
    
    # Generate config update suggestion
    print("\n" + "=" * 70)
    print("📋 SUGGESTED CONFIG UPDATE")
    print("=" * 70)
    print("Add these to hp60_kan/config.py:\n")
    for key, value in best['params'].items():
        config_key = key.upper()
        if isinstance(value, float):
            print(f"{config_key} = {value}")
        else:
            print(f"{config_key} = {value}")


def run_quick_test():
    """Run a quick test with just 2 trials to verify setup."""
    print("\n" + "=" * 70)
    print("QUICK TEST MODE (2 trials)")
    print("=" * 70)
    
    test_params = [
        {'asymmetric_alpha': 8.0, 'high_value_threshold': 0.3, 
         'high_activity_ratio': 0.5, 'high_activity_percentile': 70, 'weight_power': 3.0},
        {'asymmetric_alpha': 12.0, 'high_value_threshold': 0.25, 
         'high_activity_ratio': 0.6, 'high_activity_percentile': 65, 'weight_power': 3.5},
    ]
    
    # Use fewer epochs for quick test
    quick_fixed = {**FIXED_PARAMS, 'epochs': 20}
    
    results = []
    for i, params in enumerate(test_params, 1):
        all_params = {**quick_fixed, **params}
        result = train_with_params(params, i, len(test_params))
        results.append(result)
    
    print_summary(results)
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for Hp60 KAN')
    parser.add_argument('--quick', action='store_true', 
                        help='Run quick search (4 combinations)')
    parser.add_argument('--small', action='store_true',
                        help='Use small search space (16 combinations)')
    parser.add_argument('--full', action='store_true',
                        help='Use full search space (81 combinations)')
    parser.add_argument('--auto', '-y', action='store_true',
                        help='Auto-start without confirmation')
    args = parser.parse_args()
    
    if args.quick:
        mode = 'quick'
    elif args.small:
        mode = 'small'
    else:
        mode = 'full'
    
    results = run_grid_search(search_mode=mode, auto_start=args.auto)
