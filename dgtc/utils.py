import os
import torch

class MemoryOptimizer:
    @staticmethod
    def enable_mixed_precision():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('medium')
    
    @staticmethod
    def configure_pytorch_memory(device_id: int = 0, max_split_size: int = 128):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{max_split_size}"
        torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()
    
    @staticmethod
    def optimize_model_memory(model: nn.Module):
        for name, param in model.named_parameters():
            if 'embedding' in name or 'bias' in name:
                param.requires_grad = False