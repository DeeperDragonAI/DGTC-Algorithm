import torch
import torchvision
from dgtc import DGTCTrainer, MemoryOptimizer

# 初始化配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MemoryOptimizer.configure_pytorch_memory()
MemoryOptimizer.enable_mixed_precision()

# 准备模型和数据
model = torchvision.models.resnet50()
train_loader = ...  # 您的数据加载器

# 创建训练器
trainer = DGTCTrainer(
    model=model,
    device=device,
    train_loader=train_loader,
    lr=1e-4,
    compression_ratio=0.05
)

# 开始训练
trainer.train(epochs=50)