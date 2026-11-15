from __future__ import annotations

import sys
import unittest
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARGUS_0501_ROOT = ROOT / "local_runtime" / "detection_baselines" / "argus" / "0501"
if str(ARGUS_0501_ROOT) not in sys.path:
    sys.path.insert(0, str(ARGUS_0501_ROOT))

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from loaders.load_utils import edge_tv_split
else:
    torch = None
    edge_tv_split = None

@unittest.skipUnless(HAS_TORCH, "requires torch")
class Argus0501ValidationTests(unittest.TestCase):
    def test_edge_tv_split_keeps_one_validation_edge_for_small_slices(self) -> None:
        train_mask, val_mask = edge_tv_split(torch.tensor([[0, 1], [1, 2]]))

        self.assertEqual(int(train_mask.sum().item()), 1)
        self.assertEqual(int(val_mask.sum().item()), 1)

    def test_edge_tv_split_keeps_singleton_slice_in_train(self) -> None:
        train_mask, val_mask = edge_tv_split(torch.tensor([[0], [1]]))

        self.assertEqual(train_mask.tolist(), [True])
        self.assertEqual(val_mask.tolist(), [False])

if __name__ == "__main__":
    unittest.main()
