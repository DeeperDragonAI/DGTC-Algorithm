import unittest
import torch
import torch.nn as nn
from dgtc.optimizer import DGTCOptimizer

class TestDGTCOptimizer(unittest.TestCase):
    def setUp(self):
        self.model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        self.optimizer = DGTCOptimizer(
            self.model.parameters(), 
            lr=0.01,
            compression_ratio=0.1
        )
        self.loss_fn = nn.CrossEntropyLoss()
        
    def test_optimizer_step(self):
        """测试优化器是否能正常更新参数"""
        # 保存初始参数
        initial_params = [p.clone() for p in self.model.parameters()]
        
        # 模拟训练步骤
        inputs = torch.randn(5, 10)
        targets = torch.randint(0, 2, (5,))
        
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()
        
        # 检查参数是否更新
        for initial, current in zip(initial_params, self.model.parameters()):
            self.assertFalse(torch.equal(initial, current))
            
    def test_compression_stats(self):
        """测试压缩统计信息是否准确"""
        # 执行多次训练步骤
        for _ in range(10):
            inputs = torch.randn(5, 10)
            targets = torch.randint(0, 2, (5,))
            
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
        
        # 获取统计信息
        stats = self.optimizer.get_performance_stats()
        
        # 验证统计信息
        self.assertGreater(stats['total_gradients'], 0)
        self.assertGreater(stats['compression_ratio'], 0.05)
        self.assertGreater(stats['memory_savings'], 0)

if __name__ == '__main__':
    unittest.main()