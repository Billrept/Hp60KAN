#!/usr/bin/env python3
"""Main entry point for hp60_kan package."""
import sys
import os
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(parent_dir, 'mlx-kan'))


def main():
    parser = argparse.ArgumentParser(description='Hp60 KAN Prediction System')
    parser.add_argument('command', nargs='?', default='all',
                        choices=['train', 'predict', 'visualize', 'evaluate', 'all'],
                        help='Command to run (default: all)')
    args = parser.parse_args()
    
    if args.command in ['train', 'all']:
        print("\n" + "=" * 60)
        print("TRAINING MODEL")
        print("=" * 60)
        from hp60_kan.train import train_model_pipeline
        model, normalizer, history = train_model_pipeline()

        from hp60_kan.visualize import Hp60Visualizer
        from hp60_kan import config
        viz = Hp60Visualizer()
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        viz.plot_training_history(history, save_path=config.TRAINING_HISTORY_PLOT)

    if args.command in ['predict', 'all']:
        print("\n" + "=" * 60)
        print("TESTING PREDICTIONS")
        print("=" * 60)
        from hp60_kan.predict import load_predictor
        from datetime import datetime
        
        predictor = load_predictor()
        
        examples = [
            (12, 2, 46, 30, "Feb 15, 12:00 PM, ap60=30"),
            (20, 7, 200, 80, "Jul 19, 8:00 PM, ap60=80"),
            (0, 10, 280, 10, "Oct 7, 12:00 AM, ap60=10"),
            (6, 3, 90, 40, "Mar 31, 6:00 AM, ap60=40"),
        ]

        print("\nExample Predictions:")
        print("-" * 50)
        for hour, month, doy, ap60, desc in examples:
            pred = predictor.predict(hour, month, doy, ap60)
            print(f"{desc} -> Hp60 = {pred:.2f}")

        dt = datetime.now()
        pred = predictor.predict_from_datetime(dt, ap60=15.0)
        print(f"\nCurrent time ({dt.strftime('%Y-%m-%d %H:%M')}), ap60=27.0 -> Hp60 = {pred:.2f}")
    
    if args.command in ['visualize', 'all']:
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)
        from hp60_kan.visualize import create_visualizer
        from hp60_kan import config
        
        viz = create_visualizer(load_model=False, load_data=True)
        
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        viz.plot_hp60_distribution(
            save_path=os.path.join(config.OUTPUT_DIR, 'hp60_distribution.png'))
        viz.plot_daily_pattern(
            save_path=os.path.join(config.OUTPUT_DIR, 'daily_pattern.png'))
        viz.plot_monthly_pattern(
            save_path=os.path.join(config.OUTPUT_DIR, 'monthly_pattern.png'))
        viz.plot_geomagnetic_activity(
            start_date='2024-01-01', end_date='2024-01-31',
            save_path=os.path.join(config.OUTPUT_DIR, 'geomagnetic_jan2024.png'))
        
        stats = viz.get_statistics()
        print(f"\nData Statistics:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        print(f"  Hp60 - Mean: {stats['hp60']['mean']:.2f}, Std: {stats['hp60']['std']:.2f}")
    
    if args.command == 'evaluate':
        print("\n" + "=" * 60)
        print("EVALUATING ON RESERVED DATA")
        print("=" * 60)
        from hp60_kan.evaluate import evaluate_model
        evaluate_model()
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
