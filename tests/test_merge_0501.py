from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

EG1_ROOT = Path(__file__).resolve().parents[1] / "EG1"
if str(EG1_ROOT) not in sys.path:
    sys.path.insert(0, str(EG1_ROOT))

from merge_0501 import build_sorted_combined_events, iter_sorted_events

def make_raw_row(ts: float, src: str, dst: str, label: int) -> list[str]:
    row = ["-"] * 23
    row[0] = f"{ts:.6f}"
    row[2] = src
    row[4] = dst
    row[-1] = str(label)
    return row

class Merge0501Tests(unittest.TestCase):
    def test_build_sorted_combined_events_uses_clean_background_and_shifted_segments(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "raw_data" / "0501"
            root.mkdir(parents=True, exist_ok=True)

            samples = {
                "label_oilrig.log": [
                    make_raw_row(1000, "bg_a", "bg_b", 0),
                    make_raw_row(1005, "bg_c", "bg_d", 0),
                    make_raw_row(1300, "oil_1", "oil_2", 1),
                    make_raw_row(1305, "oil_n", "oil_m", 0),
                    make_raw_row(1310, "oil_3", "oil_4", 1),
                ],
                "label_sandworm.log": [
                    make_raw_row(1000, "bg_a", "bg_b", 0),
                    make_raw_row(1005, "bg_c", "bg_d", 0),
                    make_raw_row(2300, "snd_1", "snd_2", 1),
                    make_raw_row(2305, "snd_n", "snd_m", 0),
                    make_raw_row(2310, "snd_3", "snd_4", 1),
                ],
                "label_wizard_spider.log": [
                    make_raw_row(1000, "bg_a", "bg_b", 0),
                    make_raw_row(1005, "bg_c", "bg_d", 0),
                    make_raw_row(3300, "wiz_1", "wiz_2", 1),
                    make_raw_row(3305, "wiz_n", "wiz_m", 0),
                    make_raw_row(3310, "wiz_3", "wiz_4", 1),
                ],
            }

            for name, rows in samples.items():
                with (root / name).open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle, delimiter="\t")
                    writer.writerows(rows)

            metadata = build_sorted_combined_events(
                raw_root=root.parent,
                work_root=Path(tmp_dir) / "work",
                day_seconds=100,
                insertion_starts={"oilrig": 10, "sandworm": 40, "wizard_spider": 70},
            )
            rows = list(iter_sorted_events(Path(str(metadata["sorted_path"]))))

        self.assertEqual(
            rows,
            [
                (1000.0, "bg_a", "bg_b", 0, ""),
                (1005.0, "bg_c", "bg_d", 0, ""),
                (1010.0, "oil_1", "oil_2", 1, "oilrig"),
                (1015.0, "oil_n", "oil_m", 0, ""),
                (1020.0, "oil_3", "oil_4", 1, "oilrig"),
                (1040.0, "snd_1", "snd_2", 1, "sandworm"),
                (1045.0, "snd_n", "snd_m", 0, ""),
                (1050.0, "snd_3", "snd_4", 1, "sandworm"),
                (1070.0, "wiz_1", "wiz_2", 1, "wizard_spider"),
                (1075.0, "wiz_n", "wiz_m", 0, ""),
                (1080.0, "wiz_3", "wiz_4", 1, "wizard_spider"),
            ],
        )

if __name__ == "__main__":
    unittest.main()
