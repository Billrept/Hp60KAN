"""
Evaluation script for analyzing model predictions on reserved data.
Identifies which time windows the model predicts well vs poorly.
"""
import sys
import os

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(parent_dir, 'mlx-kan'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

from hp60_kan.data_loader import DataLoader, DataNormalizer
from hp60_kan.predict import load_predictor
from hp60_kan import config


class ModelEvaluator:
    """Evaluates model performance on reserved data across different time windows."""
    
    def __init__(self, predictor=None):
        """Initialize evaluator with a predictor."""
        self.predictor = predictor or load_predictor()
        self.reserved_data = None
        self.predictions = None
        self.errors = None
        
    def load_reserved_data(self, data_path: str = None, use_ratio: float = None) -> pd.DataFrame:
        """
        Load the reserved 10% of data that was not used for training.
        
        Args:
            data_path: Path to CSV data
            use_ratio: Fraction that was used for training (default 0.9)
            
        Returns:
            DataFrame with reserved data
        """
        data_path = data_path or config.NEW_DATA
        use_ratio = use_ratio or config.DATA_USE_RATIO
        
        # Load full data
        loader = DataLoader(data_path, use_ratio=1.0)  # Load 100%
        loader.load()
        loader.create_cyclical_features()
        
        full_df = loader.df
        
        # Create datetime column if not exists
        if 'datetime' not in full_df.columns:
            full_df['datetime'] = pd.to_datetime(
                full_df[['YYYY', 'MM', 'DD']].rename(
                    columns={'YYYY': 'year', 'MM': 'month', 'DD': 'day'}
                )
            ) + pd.to_timedelta(full_df['hh.h'], unit='h')
        
        # Create doy column if not exists
        if 'doy' not in full_df.columns:
            full_df['doy'] = full_df['datetime'].dt.dayofyear
        
        # Get the reserved portion (last 10%)
        n_total = len(full_df)
        n_used = int(n_total * use_ratio)
        
        self.reserved_data = full_df.iloc[n_used:].copy()
        
        print(f"Reserved data: {len(self.reserved_data)} samples")
        print(f"Date range: {self.reserved_data['datetime'].min()} to {self.reserved_data['datetime'].max()}")
        
        return self.reserved_data
    
    def predict_reserved(self) -> pd.DataFrame:
        """
        Make predictions on reserved data.
        
        Returns:
            DataFrame with predictions and errors
        """
        if self.reserved_data is None:
            self.load_reserved_data()
        
        df = self.reserved_data.copy()
        
        # Make predictions
        predictions = []
        for _, row in df.iterrows():
            pred = self.predictor.predict(
                hour=row['hh.h'],
                month=row['MM'],
                day_of_year=row['doy'],
                ap60=row['ap60']
            )
            predictions.append(pred)
        
        df['predicted'] = predictions
        df['actual'] = df['Hp60']
        df['error'] = df['predicted'] - df['actual']
        df['abs_error'] = np.abs(df['error'])
        df['squared_error'] = df['error'] ** 2
        
        self.predictions = df
        
        # Overall metrics
        mae = df['abs_error'].mean()
        rmse = np.sqrt(df['squared_error'].mean())
        r2 = 1 - (df['squared_error'].sum() / ((df['actual'] - df['actual'].mean()) ** 2).sum())
        
        print(f"\n=== Overall Performance on Reserved Data ===")
        print(f"MAE:  {mae:.4f} Hp60 units")
        print(f"RMSE: {rmse:.4f} Hp60 units")
        print(f"R²:   {r2:.4f}")
        
        return df
    
    def analyze_by_hour(self) -> pd.DataFrame:
        """Analyze performance by hour of day."""
        if self.predictions is None:
            self.predict_reserved()
            
        hourly = self.predictions.groupby('hh.h').agg({
            'abs_error': ['mean', 'std', 'count'],
            'squared_error': 'mean',
            'actual': 'mean'
        }).round(4)
        
        hourly.columns = ['MAE', 'MAE_std', 'count', 'MSE', 'actual_mean']
        hourly['RMSE'] = np.sqrt(hourly['MSE'])
        hourly = hourly.sort_values('MAE', ascending=False)
        
        print("\n=== Performance by Hour of Day ===")
        print("(Sorted by MAE, worst first)")
        print(hourly[['MAE', 'RMSE', 'actual_mean', 'count']].head(10))
        
        return hourly
    
    def analyze_by_month(self) -> pd.DataFrame:
        """Analyze performance by month."""
        if self.predictions is None:
            self.predict_reserved()
            
        monthly = self.predictions.groupby('MM').agg({
            'abs_error': ['mean', 'std', 'count'],
            'squared_error': 'mean',
            'actual': 'mean'
        }).round(4)
        
        monthly.columns = ['MAE', 'MAE_std', 'count', 'MSE', 'actual_mean']
        monthly['RMSE'] = np.sqrt(monthly['MSE'])
        monthly = monthly.sort_values('MAE', ascending=False)
        
        month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                       7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        monthly.index = monthly.index.map(month_names)
        
        print("\n=== Performance by Month ===")
        print("(Sorted by MAE, worst first)")
        print(monthly[['MAE', 'RMSE', 'actual_mean', 'count']])
        
        return monthly
    
    def analyze_by_ap60_level(self) -> pd.DataFrame:
        """Analyze performance by ap60 activity level."""
        if self.predictions is None:
            self.predict_reserved()
            
        # Create ap60 bins
        bins = [0, 10, 30, 50, 100, 200, 500]
        labels = ['Quiet (0-10)', 'Low (10-30)', 'Moderate (30-50)', 
                  'Active (50-100)', 'Storm (100-200)', 'Severe (200+)']
        
        df = self.predictions.copy()
        df['ap60_level'] = pd.cut(df['ap60'], bins=bins, labels=labels, include_lowest=True)
        
        by_level = df.groupby('ap60_level', observed=True).agg({
            'abs_error': ['mean', 'std', 'count'],
            'squared_error': 'mean',
            'actual': 'mean'
        }).round(4)
        
        by_level.columns = ['MAE', 'MAE_std', 'count', 'MSE', 'actual_mean']
        by_level['RMSE'] = np.sqrt(by_level['MSE'])
        by_level = by_level.sort_values('MAE', ascending=False)
        
        print("\n=== Performance by Geomagnetic Activity Level ===")
        print("(Sorted by MAE, worst first)")
        print(by_level[['MAE', 'RMSE', 'actual_mean', 'count']])
        
        return by_level
    
    def analyze_by_hp60_level(self) -> pd.DataFrame:
        """Analyze performance by actual Hp60 level."""
        if self.predictions is None:
            self.predict_reserved()
            
        # Create Hp60 bins
        bins = [0, 1, 2, 3, 4, 5, 7, 9]
        labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-7', '7-9']
        
        df = self.predictions.copy()
        df['hp60_level'] = pd.cut(df['actual'], bins=bins, labels=labels, include_lowest=True)
        
        by_level = df.groupby('hp60_level', observed=True).agg({
            'abs_error': ['mean', 'std', 'count'],
            'squared_error': 'mean',
            'error': 'mean'  # Bias: positive = overpredict
        }).round(4)
        
        by_level.columns = ['MAE', 'MAE_std', 'count', 'MSE', 'bias']
        by_level['RMSE'] = np.sqrt(by_level['MSE'])
        
        print("\n=== Performance by Actual Hp60 Level ===")
        print("(Shows where model struggles with high/low values)")
        print(by_level[['MAE', 'RMSE', 'bias', 'count']])
        
        return by_level
    
    def analyze_by_season(self) -> pd.DataFrame:
        """Analyze performance by season."""
        if self.predictions is None:
            self.predict_reserved()
            
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Fall'
        
        df = self.predictions.copy()
        df['season'] = df['MM'].apply(get_season)
        
        by_season = df.groupby('season').agg({
            'abs_error': ['mean', 'std', 'count'],
            'squared_error': 'mean',
            'actual': 'mean'
        }).round(4)
        
        by_season.columns = ['MAE', 'MAE_std', 'count', 'MSE', 'actual_mean']
        by_season['RMSE'] = np.sqrt(by_season['MSE'])
        by_season = by_season.sort_values('MAE', ascending=False)
        
        print("\n=== Performance by Season ===")
        print("(Sorted by MAE, worst first)")
        print(by_season[['MAE', 'RMSE', 'actual_mean', 'count']])
        
        return by_season
    
    def find_worst_predictions(self, n: int = 20) -> pd.DataFrame:
        """Find the worst predictions."""
        if self.predictions is None:
            self.predict_reserved()
            
        worst = self.predictions.nlargest(n, 'abs_error')[
            ['datetime', 'hh.h', 'MM', 'ap60', 'actual', 'predicted', 'error']
        ]
        
        print(f"\n=== Top {n} Worst Predictions ===")
        print(worst.to_string())
        
        return worst
    
    def plot_error_analysis(self, save_dir: str = None):
        """Generate plots for error analysis."""
        if self.predictions is None:
            self.predict_reserved()
        
        save_dir = save_dir or config.OUTPUT_DIR
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Error by hour
        hourly = self.predictions.groupby('hh.h')['abs_error'].mean()
        axes[0, 0].bar(hourly.index, hourly.values, color='steelblue')
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('MAE (Hp60 units)')
        axes[0, 0].set_title('Prediction Error by Hour')
        axes[0, 0].axhline(y=hourly.mean(), color='red', linestyle='--', label=f'Mean: {hourly.mean():.3f}')
        axes[0, 0].legend()
        
        # 2. Error by month
        monthly = self.predictions.groupby('MM')['abs_error'].mean()
        axes[0, 1].bar(monthly.index, monthly.values, color='coral')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('MAE (Hp60 units)')
        axes[0, 1].set_title('Prediction Error by Month')
        axes[0, 1].set_xticks(range(1, 13))
        axes[0, 1].axhline(y=monthly.mean(), color='red', linestyle='--', label=f'Mean: {monthly.mean():.3f}')
        axes[0, 1].legend()
        
        # 3. Predicted vs Actual scatter
        sample = self.predictions.sample(min(5000, len(self.predictions)))
        axes[1, 0].scatter(sample['actual'], sample['predicted'], alpha=0.3, s=5)
        axes[1, 0].plot([0, 9], [0, 9], 'r--', label='Perfect prediction')
        axes[1, 0].set_xlabel('Actual Hp60')
        axes[1, 0].set_ylabel('Predicted Hp60')
        axes[1, 0].set_title('Predicted vs Actual')
        axes[1, 0].legend()
        
        # 4. Error distribution by ap60 level
        bins = [0, 10, 30, 50, 100, 200, 500]
        labels = ['0-10', '10-30', '30-50', '50-100', '100-200', '200+']
        df = self.predictions.copy()
        df['ap60_bin'] = pd.cut(df['ap60'], bins=bins, labels=labels, include_lowest=True)
        ap60_mae = df.groupby('ap60_bin', observed=True)['abs_error'].mean()
        axes[1, 1].bar(range(len(ap60_mae)), ap60_mae.values, color='green')
        axes[1, 1].set_xticks(range(len(ap60_mae)))
        axes[1, 1].set_xticklabels(ap60_mae.index, rotation=45)
        axes[1, 1].set_xlabel('ap60 Level')
        axes[1, 1].set_ylabel('MAE (Hp60 units)')
        axes[1, 1].set_title('Prediction Error by Geomagnetic Activity')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'error_analysis.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"\nError analysis plot saved to {save_path}")
    
    def full_analysis(self, save_plots: bool = True) -> Dict:
        """Run complete analysis."""
        print("=" * 60)
        print("EVALUATING MODEL ON RESERVED 10% DATA")
        print("=" * 60)
        
        self.load_reserved_data()
        self.predict_reserved()
        
        results = {
            'by_hour': self.analyze_by_hour(),
            'by_month': self.analyze_by_month(),
            'by_season': self.analyze_by_season(),
            'by_ap60': self.analyze_by_ap60_level(),
            'by_hp60': self.analyze_by_hp60_level(),
            'worst': self.find_worst_predictions()
        }
        
        if save_plots:
            self.plot_error_analysis()
        
        # Summary of problem areas
        print("\n" + "=" * 60)
        print("SUMMARY: Problem Areas to Fix")
        print("=" * 60)
        
        # Worst hours
        worst_hours = results['by_hour'].head(3)
        print(f"\nWorst hours: {list(worst_hours.index)}")
        
        # Worst ap60 levels
        print(f"\nStruggles most with: High geomagnetic activity (storm conditions)")
        
        # Bias by hp60 level
        hp60_bias = results['by_hp60']
        print(f"\nBias pattern:")
        for level in hp60_bias.index:
            bias = hp60_bias.loc[level, 'bias']
            if abs(bias) > 0.3:
                direction = "overpredicts" if bias > 0 else "underpredicts"
                print(f"  - Hp60 {level}: Model {direction} by {abs(bias):.2f}")
        
        return results


def evaluate_model():
    """Main function to evaluate model on reserved data."""
    evaluator = ModelEvaluator()
    results = evaluator.full_analysis()
    return results


if __name__ == "__main__":
    evaluate_model()
