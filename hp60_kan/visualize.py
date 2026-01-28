"""Visualization utilities for Hp60 prediction."""
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(parent_dir, 'mlx-kan'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Tuple
from datetime import datetime, timedelta

from hp60_kan.predict import Hp60Predictor, load_predictor
from hp60_kan.data_loader import DataLoader
from hp60_kan import config


class Hp60Visualizer:
    """Visualization class for Hp60 data and predictions."""
    
    def __init__(self, predictor: Hp60Predictor = None):
        self.predictor = predictor
        self.df = None
        
    def load_predictor(self, model_path: str = None, scaler_path: str = None):
        self.predictor = load_predictor(model_path, scaler_path)
        
    def load_data(self, data_path: str = None, use_ratio: float = 0.9) -> pd.DataFrame:
        data_path = data_path or config.DATA_PATH
        loader = DataLoader(data_path, use_ratio=use_ratio)
        loader.load()
        loader.create_cyclical_features()
        self.df = loader.get_raw_dataframe()
        
        self.df['datetime'] = pd.to_datetime(
            self.df[['YYYY', 'MM', 'DD']].rename(
                columns={'YYYY': 'year', 'MM': 'month', 'DD': 'day'}
            )
        ) + pd.to_timedelta(self.df['hh.h'], unit='h')
        
        return self.df
    
    def plot_training_history(self, history: Dict, 
                               save_path: str = None,
                               figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
        axes[0].plot(history['val_mse'], label='Val MSE', color='orange')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(history['val_mae'], label='Val MAE', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Validation MAE')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        
        return fig
    
    def plot_predictions_vs_actual(self, n_samples: int = 200,
                                    save_path: str = None,
                                    figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        if self.predictor is None or not self.predictor.is_loaded:
            raise ValueError("Predictor not loaded. Call load_predictor() first.")
        
        sample_df = self.df.tail(n_samples).copy()
        
        predictions = []
        for _, row in sample_df.iterrows():
            pred = self.predictor.predict(
                hour=row['hh.h'],
                month=int(row['MM']),
                day_of_year=int(row['day_of_year']),
                ap60=row['ap60']
            )
            predictions.append(pred)
        
        sample_df['predicted'] = predictions
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(sample_df['datetime'], sample_df['Hp60'], 
                label='Actual', alpha=0.8, linewidth=1.5)
        ax.plot(sample_df['datetime'], sample_df['predicted'], 
                label='Predicted', alpha=0.8, linewidth=1.5)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Hp60')
        ax.set_title('Hp60: Actual vs Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Predictions plot saved to {save_path}")
        
        return fig
    
    def plot_hp60_distribution(self, save_path: str = None,
                                figsize: Tuple[int, int] = (10, 5)) -> plt.Figure:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        axes[0].hist(self.df['Hp60'], bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Hp60')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Hp60 Distribution')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(self.df['ap60'], bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1].set_xlabel('ap60')
        axes[1].set_ylabel('Count')
        axes[1].set_title('ap60 Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Distribution plot saved to {save_path}")
        
        return fig
    
    def plot_daily_pattern(self, save_path: str = None,
                            figsize: Tuple[int, int] = (10, 5)) -> plt.Figure:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        hourly_stats = self.df.groupby('hh.h')['Hp60'].agg(['mean', 'std']).reset_index()
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.fill_between(hourly_stats['hh.h'], 
                        hourly_stats['mean'] - hourly_stats['std'],
                        hourly_stats['mean'] + hourly_stats['std'],
                        alpha=0.3)
        ax.plot(hourly_stats['hh.h'], hourly_stats['mean'], 'o-', linewidth=2)
        
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Hp60')
        ax.set_title('Average Hp60 by Hour of Day')
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Daily pattern saved to {save_path}")
        
        return fig
    
    def plot_monthly_pattern(self, save_path: str = None,
                              figsize: Tuple[int, int] = (10, 5)) -> plt.Figure:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        monthly_stats = self.df.groupby('MM')['Hp60'].agg(['mean', 'std']).reset_index()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(monthly_stats['MM'], monthly_stats['mean'], 
               yerr=monthly_stats['std'], capsize=3, alpha=0.7)
        
        ax.set_xlabel('Month')
        ax.set_ylabel('Hp60')
        ax.set_title('Average Hp60 by Month')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(months)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Monthly pattern saved to {save_path}")
        
        return fig
    
    def plot_geomagnetic_activity(self, start_date: str = None, end_date: str = None,
                                   save_path: str = None,
                                   figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        df_plot = self.df.copy()
        
        if start_date:
            df_plot = df_plot[df_plot['datetime'] >= start_date]
        if end_date:
            df_plot = df_plot[df_plot['datetime'] <= end_date]
        
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        axes[0].plot(df_plot['datetime'], df_plot['Hp60'], linewidth=0.8)
        axes[0].set_ylabel('Hp60')
        axes[0].set_title('Geomagnetic Activity')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(df_plot['datetime'], df_plot['ap60'], linewidth=0.8, color='orange')
        axes[1].set_ylabel('ap60')
        axes[1].set_xlabel('Date')
        axes[1].grid(True, alpha=0.3)
        
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Activity plot saved to {save_path}")
        
        return fig
    
    def get_statistics(self) -> Dict:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return {
            'total_samples': len(self.df),
            'date_range': {
                'start': self.df['datetime'].min().strftime('%Y-%m-%d'),
                'end': self.df['datetime'].max().strftime('%Y-%m-%d')
            },
            'hp60': {
                'mean': float(self.df['Hp60'].mean()),
                'std': float(self.df['Hp60'].std()),
                'min': float(self.df['Hp60'].min()),
                'max': float(self.df['Hp60'].max())
            },
            'ap60': {
                'mean': float(self.df['ap60'].mean()),
                'std': float(self.df['ap60'].std()),
                'min': float(self.df['ap60'].min()),
                'max': float(self.df['ap60'].max())
            }
        }


def create_visualizer(load_model: bool = True, load_data: bool = True,
                       model_path: str = None, data_path: str = None) -> Hp60Visualizer:
    viz = Hp60Visualizer()
    
    if load_model:
        viz.load_predictor(model_path)
    
    if load_data:
        viz.load_data(data_path)
    
    return viz
