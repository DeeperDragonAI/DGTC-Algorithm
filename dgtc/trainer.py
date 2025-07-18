import torch
import torch.nn as nn
import time
import numpy as np
import contextlib
from typing import Dict
from .optimizer import DGTCOptimizer
from .sharding import TAPSModel

class DGTCTrainer:
    def __init__(self, model: nn.Module, device: torch.device, 
                 train_loader, lr: float = 1e-3, compression_ratio: float = 0.1,
                 min_parameters: int = 10**6):
        self.device = device
        self.model = TAPSModel(model, device, min_parameters)
        self.optimizer = DGTCOptimizer(
            self.model.parameters(),
            lr=lr,
            compression_ratio=compression_ratio
        )
        self.train_loader = train_loader
        self.epoch = 0
        self.step = 0
        self.loss_history = []
        self.start_time = time.time()
    
    def train(self, epochs: int = 10):
        for epoch in range(epochs):
            self.epoch = epoch
            self.model.train()
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.step += 1
                data, target = data.to(self.device), target.to(self.device)
                
                with self._async_pipeline():
                    output = self.model(data)
                    loss = self._compute_loss(output, target)
                    loss.backward()
                    self.optimizer.step()
                
                self.loss_history.append(loss.item())
                
                if self.step % 100 == 0:
                    self._report_progress()
            
            if epoch % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
    
    def _async_pipeline(self):
        return contextlib.nullcontext()
    
    def _compute_loss(self, output, target):
        return nn.CrossEntropyLoss()(output, target)
    
    def _report_progress(self):
        stats = self.optimizer.get_performance_stats()
        elapsed = time.time() - self.start_time
        steps_per_sec = 100 / elapsed
        
        print(f"Step {self.step} | "
              f"Loss: {self.loss_history[-1]:.4f} | "
              f"Compression: {stats['compression_ratio']*100:.1f}% | "
              f"Speed: {steps_per_sec:.1f} steps/sec")
        
        self.start_time = time.time()
    
    def save_checkpoint(self, filename: str):
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'compression_stats': self.optimizer.get_performance_stats()
        }
        torch.save(checkpoint, filename)