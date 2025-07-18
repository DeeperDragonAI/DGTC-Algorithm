import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from .compression import WaveletCompressor

class DGTCOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, compression_ratio: float = 0.1):
        defaults = dict(lr=lr, compression_ratio=compression_ratio)
        super().__init__(params, defaults)
        self.compressor = WaveletCompressor(compression_ratio=compression_ratio)
        self.gradient_buffer = {}
        self.step_counter = 0
        self.compression_stats = {
            'total_gradients': 0,
            'compressed_size': 0,
            'original_size': 0
        }
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                if p in self.gradient_buffer:
                    comp_grad, metadata = self.gradient_buffer[p]
                    full_grad = self.compressor.decompress(metadata)
                    p.add_(full_grad.to(p.device), alpha=-group['lr'])
                    self._update_stats(comp_grad, full_grad)
                
                self.gradient_buffer[p] = self.compressor.compress(p.grad)
                p.grad = None
        
        self.step_counter += 1
        return loss
    
    def _update_stats(self, comp_grad, full_grad):
        self.compression_stats['compressed_size'] += comp_grad.element_size() * comp_grad.nelement()
        self.compression_stats['original_size'] += full_grad.element_size() * full_grad.nelement()
        self.compression_stats['total_gradients'] += 1
    
    def get_compression_ratio(self) -> float:
        if self.compression_stats['original_size'] == 0:
            return 0.0
        return 1 - (self.compression_stats['compressed_size'] / self.compression_stats['original_size'])
    
    def get_performance_stats(self) -> Dict:
        return {
            'compression_ratio': self.get_compression_ratio(),
            'memory_savings': self.compression_stats['original_size'] - self.compression_stats['compressed_size'],
            'total_gradients': self.compression_stats['total_gradients'],
            'steps': self.step_counter
        }