from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
EG1_ROOT = ROOT / "EG1"
PREPROC_ROOT = ROOT / "local_runtime" / "pre_processing"
if str(EG1_ROOT) not in sys.path:
    sys.path.insert(0, str(EG1_ROOT))
if str(PREPROC_ROOT) not in sys.path:
    sys.path.insert(0, str(PREPROC_ROOT))

from campaign_grouping import assign_temporal_entity_campaigns
from pikachu_preprocess import preprocess_lanl

class TemporalCampaignGroupingTests(unittest.TestCase):
    def test_assign_temporal_entity_campaigns_avoids_merging_shared_target_only(self) -> None:
        events = [
            {"timestamp": 0, "src_computer": "src_a", "dst_computer": "target"},
            {"timestamp": 5, "src_computer": "src_b", "dst_computer": "target"},
            {"timestamp": 8, "src_computer": "src_a", "dst_computer": "target"},
            {"timestamp": 400, "src_computer": "src_a", "dst_computer": "target"},
        ]

        campaigns, _ = assign_temporal_entity_campaigns(
            events,
            dataset="optc",
            entity_keys=("src_computer",),
            time_window_seconds=300,
        )

        self.assertEqual(campaigns[0], campaigns[2])
        self.assertNotEqual(campaigns[0], campaigns[1])
        self.assertNotEqual(campaigns[2], campaigns[3])

    def test_eg1_lanl_preprocess_uses_first_redteam_row_and_assigns_campaigns(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            raw_root = Path(tmp_dir) / "raw_data" / "LANL"
            raw_root.mkdir(parents=True, exist_ok=True)

            auth_rows = [
                "100,U1@DOM1,U9@DOM1,C1,C10,NTLM,Network,Auth,Success\n",
                "200,U1@DOM1,U9@DOM1,C1,C11,NTLM,Network,Auth,Success\n",
                "4000,U2@DOM1,U9@DOM1,C2,C20,NTLM,Network,Auth,Success\n",
                "4010,U2@DOM1,U9@DOM1,C2,C21,NTLM,Network,Auth,Success\n",
            ]
            (raw_root / "auth.txt").write_text("".join(auth_rows), encoding="utf-8")

            redteam_rows = [
                "100,U1@DOM1,C1,C10\n",
                "200,U1@DOM1,C1,C11\n",
                "4000,U2@DOM1,C2,C20\n",
            ]
            (raw_root / "redteam.txt").write_text("".join(redteam_rows), encoding="utf-8")

            frame = preprocess_lanl(raw_root.parent, snapshot_seconds=60, neg_ratio=20, seed=42)

        positives = frame[frame["label"] == 1].sort_values(by=["timestamp", "dst_computer"]).reset_index(drop=True)
        self.assertEqual(len(positives), 3)
        self.assertEqual(int(positives.iloc[0]["timestamp"]), 100)
        self.assertEqual(
            positives["campaign"].tolist(),
            ["lanl_campaign_001", "lanl_campaign_001", "lanl_campaign_002"],
        )

if __name__ == "__main__":
    unittest.main()
