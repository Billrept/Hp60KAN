"""KAN model for Hp60 prediction."""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mlx-kan'))

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from kan.kan import KAN
from kan.args import ModelArgs
from typing import List

from hp60_kan import config as hp60_config


class Hp60KANModel:
    """Wrapper for KAN model."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int = 1,
                 grid_size: int = None, spline_order: int = None, grid_eps: float = None):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.layers_hidden = [input_dim] + hidden_dims + [output_dim]
        
        self.grid_size = grid_size or getattr(hp60_config, 'GRID_SIZE', 5)
        self.spline_order = spline_order or getattr(hp60_config, 'SPLINE_ORDER', 3)
        self.grid_eps = grid_eps or getattr(hp60_config, 'GRID_EPS', 0.02)
        
        self.model = None
        self._build_model()
        
    def _build_model(self):
        print(f"Model architecture: {self.layers_hidden}")
        
        model_args = ModelArgs()
        model_args.layers_hidden = self.layers_hidden
        model_args.in_features = self.input_dim
        model_args.out_features = self.output_dim
        model_args.hidden_dim = self.hidden_dims[0]
        model_args.num_layers = len(self.layers_hidden) - 1
        model_args.num_classes = self.output_dim
        model_args.grid_size = self.grid_size
        model_args.spline_order = self.spline_order
        model_args.scale_noise = 0.1
        model_args.scale_base = 1.0
        model_args.scale_spline = 1.0
        model_args.hidden_act = nn.SiLU
        model_args.grid_eps = self.grid_eps
        model_args.grid_range = [-1, 1]
        
        self.model = KAN(layers_hidden=self.layers_hidden, args=model_args)
        
        self.model.num_layers = len(self.model.layers)
        for i, layer in enumerate(self.model.layers):
            setattr(self.model, f'layer_{i}', layer)
        
        print(f"Total parameters: {self.count_parameters():,}")
        print(f"Number of layers: {self.model.num_layers}")
        print(f"Grid size: {self.grid_size}, Spline order: {self.spline_order}")
    
    def count_parameters(self) -> int:
        total = 0
        for layer in self.model.layers:
            total += layer.base_weight.size
            total += layer.spline_weight.size
            if layer.bias:
                total += layer.bias_param.size
        return total
    
    def __call__(self, X: mx.array) -> mx.array:
        return self.model(X)
    
    def parameters(self):
        return self.model.parameters()
    
    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(save_dir, "model.safetensors")
        flattened_tree = tree_flatten(self.model.trainable_parameters())
        mx.save_safetensors(model_path, dict(flattened_tree))
        print(f"Model saved to {model_path}")
        
        config_path = os.path.join(save_dir, "config.json")
        config = self.get_config()
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Config saved to {config_path}")
        
    def load(self, save_dir: str):
        model_path = os.path.join(save_dir, "model.safetensors")
        weights = tree_unflatten(list(mx.load(model_path).items()))
        self.model.update(weights)
        mx.eval(self.model.parameters())
        print(f"Model loaded from {model_path}")
    
    def get_config(self) -> dict:
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'grid_size': self.grid_size,
            'spline_order': self.spline_order,
            'grid_eps': self.grid_eps
        }


def create_model(input_dim: int, hidden_dims: List[int], output_dim: int = 1,
                 grid_size: int = None, spline_order: int = None) -> Hp60KANModel:
    return Hp60KANModel(input_dim, hidden_dims, output_dim, 
                        grid_size=grid_size, spline_order=spline_order)
