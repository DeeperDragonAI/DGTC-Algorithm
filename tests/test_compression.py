import unittest
import torch
from dgtc.compression import WaveletCompressor

class TestWaveletCompressor(unittest.TestCase):
    def setUp(self):
        self.compressor = WaveletCompressor(compression_ratio=0.1)
        self.test_tensor = torch.randn(100, 100)  # 10,000个元素
        
    def test_compression_decompression(self):
        """测试压缩和解压缩是否一致"""
        compressed, metadata = self.compressor.compress(self.test_tensor)
        decompressed = self.compressor.decompress(metadata)
        
        # 验证形状相同
        self.assertEqual(self.test_tensor.shape, decompressed.shape)
        
        # 验证值接近（由于有损压缩，不能完全相等）
        diff = (self.test_tensor - decompressed).abs().mean()
        self.assertLess(diff, 0.1)  # 平均差异应小于0.1
        
    def test_compression_ratio(self):
        """测试压缩率是否在预期范围内"""
        compressed, _ = self.compressor.compress(self.test_tensor)
        original_size = self.test_tensor.numel()
        compressed_size = compressed.numel()
        
        # 计算压缩率
        ratio = compressed_size / original_size
        self.assertAlmostEqual(ratio, 0.1, delta=0.05)  # 压缩率应在0.1±0.05范围内

if __name__ == '__main__':
    unittest.main()