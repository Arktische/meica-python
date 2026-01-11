import unittest
import torch
from unittest.mock import MagicMock
from trainer.loss_accumulator import LossAccumulator

class TestLossAccumulator(unittest.TestCase):
    def test_finalize_zero_count(self):
        # Mock accelerator
        accelerator = MagicMock()
        accelerator.device = torch.device("cpu")
        # Mock gather to return input tensor (simulating gather on 1 process)
        accelerator.gather.side_effect = lambda x: x 
        
        loss_acc = LossAccumulator(accelerator)
        # Manually set internal state to 0
        loss_acc.loss_sum = torch.tensor(0.0)
        loss_acc.sample_count = torch.tensor(0)
        
        # Should return 0.0 and log warning
        result = loss_acc.finalize()
        self.assertEqual(result.item(), 0.0)

    def test_finalize_normal(self):
        accelerator = MagicMock()
        accelerator.device = torch.device("cpu")
        accelerator.gather.side_effect = lambda x: x
        
        loss_acc = LossAccumulator(accelerator)
        loss_acc.loss_sum = torch.tensor(10.0)
        loss_acc.sample_count = torch.tensor(5)
        
        result = loss_acc.finalize()
        self.assertEqual(result.item(), 2.0)

if __name__ == '__main__':
    unittest.main()
