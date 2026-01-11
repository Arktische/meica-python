import unittest
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch
from trainer.trainer import Trainer
from types import SimpleNamespace

class TestCheckpoint(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.project_config = SimpleNamespace(
            project_dir=self.test_dir,
            automatic_checkpoint_naming=False,
            total_limit=3  # Adjusted for first test
        )
        self.trainer = Trainer()
        self.trainer.project_configuration = self.project_config
        self.trainer.checkpoint_dir = os.path.join(self.test_dir, "checkpoints")
        self.trainer.__progress__ = SimpleNamespace(global_step=0, epoch=0, step=0)
        self.trainer.save_state = MagicMock()
        
        # Initialize time map for mocking
        self.time_map = {}

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _mock_ctime(self, path):
        basename = os.path.basename(path)
        return self.time_map.get(basename, 0.0)

    @patch('os.path.getctime')
    def test_checkpoint_sorting(self, mock_getctime):
        mock_getctime.side_effect = self._mock_ctime
        base_dir = self.trainer.checkpoint_dir
        os.makedirs(base_dir)

        # 1. Valid step 100, time 100
        os.makedirs(os.path.join(base_dir, "ckpt_100_epoch_1_step_100"))
        self.time_map["ckpt_100_epoch_1_step_100"] = 100.0
        
        # 2. Valid step 200, time 200
        os.makedirs(os.path.join(base_dir, "ckpt_200_epoch_2_step_200"))
        self.time_map["ckpt_200_epoch_2_step_200"] = 200.0
        
        # 3. Invalid name, time 50 (Oldest invalid)
        os.makedirs(os.path.join(base_dir, "invalid_old"))
        self.time_map["invalid_old"] = 50.0
        
        # 4. Invalid name, time 300 (Newest invalid)
        os.makedirs(os.path.join(base_dir, "invalid_new"))
        self.time_map["invalid_new"] = 300.0

        # Trigger save new checkpoint (Step 300)
        # We need to set time for the new checkpoint too, but it's created inside the function.
        # The function calls getctime on it AFTER creation? No, it only calls getctime on EXISTING dirs.
        # The new one is created AFTER cleanup.
        # Wait, check code:
        # entries = listdir...
        # sort...
        # while len >= limit: delete...
        # os.makedirs(output_dir) -> Create NEW one.
        # So sorting only affects EXISTING directories.
        
        self.trainer.__progress__.global_step = 300
        self.trainer.__progress__.epoch = 3
        self.trainer.__progress__.step = 300
        
        # Current Existing: 4. Limit: 3.
        # Should delete 4 - 3 + 1 (for new one) = 2?
        # Logic: while len(entries) >= limit: pop(0).
        # entries count: 4. limit: 3.
        # pop(0) -> count 3.
        # loop ends.
        # Then create new one -> Total 4.
        # Wait, usually limit includes the new one.
        # Code: while len(entries) >= limit.
        # If limit is 3, and we have 4, we pop 1, leaving 3.
        # Then we create 1 -> Total 4.
        # So we actually have limit + 1.
        # To strictly maintain limit, we should pop until len < limit?
        # Standard implementation: while len >= limit: pop. Leaves limit-1? No, leaves limit-1 if we pop one more.
        # If len == limit, we pop 1. Leaves limit-1. +1 = limit. Correct.
        # Here len=4, limit=3.
        # 1. 4 >= 3. pop(0). entries=3.
        # 2. 3 >= 3. pop(0). entries=2.
        # 3. 2 < 3. Stop.
        # Create new -> Total 3.
        # So 2 items will be removed.
        
        # Keys:
        # invalid_old: (-1, 50)
        # invalid_new: (-1, 300)
        # ckpt_100: (100, 100)
        # ckpt_200: (200, 200)
        
        # Sorted: invalid_old, invalid_new, ckpt_100, ckpt_200
        # Removed 2: invalid_old, invalid_new.
        
        self.trainer.__save_checkpoint__()
        
        remaining = os.listdir(base_dir)
        self.assertIn("checkpoint_300_epoch_3_step_300", remaining)
        self.assertIn("ckpt_200_epoch_2_step_200", remaining)
        self.assertIn("ckpt_100_epoch_1_step_100", remaining)
        self.assertNotIn("invalid_old", remaining)
        self.assertNotIn("invalid_new", remaining)

    @patch('os.path.getctime')
    def test_checkpoint_sorting_invalid_retention(self, mock_getctime):
        mock_getctime.side_effect = self._mock_ctime
        base_dir = self.trainer.checkpoint_dir
        os.makedirs(base_dir)
        
        # inv_1: time 100
        os.makedirs(os.path.join(base_dir, "inv_1"))
        self.time_map["inv_1"] = 100.0
        
        # inv_2: time 200
        os.makedirs(os.path.join(base_dir, "inv_2"))
        self.time_map["inv_2"] = 200.0
        
        # inv_3: time 300
        os.makedirs(os.path.join(base_dir, "inv_3"))
        self.time_map["inv_3"] = 300.0
        
        self.trainer.project_configuration.total_limit = 2
        # Existing: 3. Limit: 2.
        # 1. 3 >= 2. Pop inv_1. (Len 2)
        # 2. 2 >= 2. Pop inv_2. (Len 1)
        # Create new. Total 2.
        
        self.trainer.__progress__.global_step = 400
        self.trainer.__save_checkpoint__()
        
        remaining = os.listdir(base_dir)
        self.assertIn("inv_3", remaining)
        self.assertIn("checkpoint_400_epoch_0_step_0", remaining)
        self.assertNotIn("inv_1", remaining)
        self.assertNotIn("inv_2", remaining)

if __name__ == '__main__':
    unittest.main()
