import torch.nn as nn
import torch
from typing import List

class TAPSModel(nn.Module):
    def __init__(self, model: nn.Module, device: torch.device, min_parameters: int = 10**6):
        super().__init__()
        self.device = device
        self.min_parameters = min_parameters
        self.slices = self._partition_model(model)
        
        for i, slice in enumerate(self.slices):
            slice.to(self.device)
            slice.device = self.device
    
    def _partition_model(self, model: nn.Module) -> List[nn.Module]:
        modules = list(model.named_children())
        slices = []
        current_slice = []
        current_params = 0
        
        for name, module in modules:
            params = sum(p.numel() for p in module.parameters())
            
            if current_params > 0 and current_params + params > self.min_parameters:
                slices.append(nn.Sequential(*current_slice))
                current_slice = []
                current_params = 0
            
            current_slice.append(module)
            current_params += params
        
        if current_slice:
            slices.append(nn.Sequential(*current_slice))
        
        return nn.ModuleList(slices)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, slice in enumerate(self.slices):
            if i > 0:
                x = x.detach()
            x = slice(x)
            if i < len(self.slices) - 1:
                x = self._async_transfer(x, self.slices[i+1].device)
        return x
    
    def _async_transfer(self, tensor: torch.Tensor, target_device: torch.device) -> torch.Tensor:
        return tensor.to(target_device, non_blocking=True)