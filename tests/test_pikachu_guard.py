from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

EG1_ROOT = Path(__file__).resolve().parents[1] / "EG1"
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(EG1_ROOT) not in sys.path:
    sys.path.insert(0, str(EG1_ROOT))

from pikachu_run import _preflight_report

class PikachuGuardTests(unittest.TestCase):
    def test_preflight_marks_large_input_unsupported(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "input.csv"
            output_dir = root / "out"
            output_dir.mkdir(parents=True, exist_ok=True)
            with input_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["timestamp", "src_computer", "dst_computer", "label", "campaign", "snapshot"],
                )
                writer.writeheader()
                for snapshot in range(4):
                    for node in range(100):
                        writer.writerow(
                            {
                                "timestamp": snapshot,
                                "src_computer": str(node),
                                "dst_computer": str((node + 1) % 100),
                                "label": 0,
                                "campaign": "",
                                "snapshot": snapshot,
                            }
                        )

            report = _preflight_report(
                dataset="synthetic",
                input_path=input_path,
                output_dir=output_dir,
                dimensions=128,
                max_estimated_gib=0.0001,
            )

            self.assertFalse(report["supported"])
            self.assertEqual(report["reason"], "estimated_working_set_exceeds_threshold")
            self.assertTrue((output_dir / "pikachu_preflight.json").exists())

if __name__ == "__main__":
    unittest.main()
