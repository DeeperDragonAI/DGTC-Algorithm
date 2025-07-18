import torch
import numpy as np
import math
from typing import Dict, Tuple

class WaveletCompressor:
    """基于小波变换的梯度压缩器"""
    
    def __init__(self, wavelet: str = 'db8', compression_ratio: float = 0.1):
        self.wavelet = wavelet
        self.compression_ratio = max(0.01, min(0.2, compression_ratio))
        self._init_wavelet_filters()
    
    def _init_wavelet_filters(self):
        """初始化小波滤波器(伪实现)"""
        self.low_pass = np.ones(8) / np.sqrt(2)
        self.high_pass = np.ones(8) / np.sqrt(2)
    
    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        original_shape = tensor.shape
        flat_tensor = tensor.view(-1).cpu().numpy()
        
        # 伪小波变换 - 实际项目应使用pywavelets
        coeffs = self._dwt(flat_tensor)
        threshold = np.percentile(np.abs(coeffs), 100 * (1 - self.compression_ratio))
        compressed = np.where(np.abs(coeffs) > threshold, coeffs, 0)
        
        nonzero_indices = np.nonzero(compressed)[0]
        nonzero_values = compressed[nonzero_indices]
        
        metadata = {
            'original_shape': original_shape,
            'threshold': threshold,
            'indices': nonzero_indices,
            'values': nonzero_values
        }
        
        return torch.tensor(nonzero_values), metadata
    
    def decompress(self, metadata: Dict) -> torch.Tensor:
        size = math.prod(metadata['original_shape'])
        coeffs = np.zeros(size, dtype=np.float32)
        coeffs[metadata['indices']] = metadata['values']
        reconstructed = self._idwt(coeffs)
        return torch.tensor(reconstructed).view(metadata['original_shape'])
    
    def _dwt(self, data: np.ndarray) -> np.ndarray:
        return np.fft.rfft(data)
    
    def _idwt(self, coeffs: np.ndarray) -> np.ndarray:
        return np.fft.irfft(coeffs)