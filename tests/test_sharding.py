import unittest
import torch
import torch.nn as nn
from dgtc.sharding import TAPSModel

class TestTAPSModel(unittest.TestCase):
    def setUp(self):
        # 创建一个测试模型
        self.model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Linear(30, 2)
        )
        
    def test_sharding(self):
        """测试模型分片是否正常工作"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sharded_model = TAPSModel(self.model, device, min_parameters=100)
        
        # 验证分片数量
        self.assertGreater(len(sharded_model.slices), 1)
        
        # 验证前向传播
        inputs = torch.randn(5, 10)
        outputs = sharded_model(inputs)
        
        # 验证输出形状
        self.assertEqual(outputs.shape, (5, 2))
        
        # 验证设备
        for slice in sharded_model.slices:
            self.assertEqual(next(slice.parameters()).device, device)
            
    def test_output_consistency(self):
        """测试分片模型与原模型输出是否一致"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sharded_model = TAPSModel(self.model, device)
        
        inputs = torch.randn(5, 10)
        
        # 原始模型输出
        original_output = self.model(inputs)
        
        # 分片模型输出
        sharded_output = sharded_model(inputs)
        
        # 验证输出是否接近
        diff = (original_output - sharded_output).abs().max()
        self.assertLess(diff, 1e-5)

if __name__ == '__main__':
    unittest.main()