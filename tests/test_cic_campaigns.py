from __future__ import annotations

import csv
import importlib
import os
import pickle
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EG1_ROOT = ROOT / "EG1"
if str(EG1_ROOT) not in sys.path:
    sys.path.insert(0, str(EG1_ROOT))

from pikachu_preprocess import preprocess_cic

def _write_cic_day(path: Path, rows: list[dict[str, str]]) -> None:
    frame = pd.DataFrame(rows, columns=["Timestamp", "Src IP", "Dst IP", "Label"])
    frame.to_csv(path, index=False)

class CicCampaignTests(unittest.TestCase):
    def test_eg1_cic_preprocess_uses_raw_labels_as_campaigns(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            raw_root = Path(tmp_dir) / "raw_data" / "CIC-new"
            raw_root.mkdir(parents=True, exist_ok=True)

            samples = {
                "monday.csv": [{"Timestamp": "2017-07-03 08:00:00.000000", "Src IP": "10.0.0.1", "Dst IP": "10.0.0.2", "Label": "BENIGN"}],
                "tuesday.csv": [{"Timestamp": "2017-07-04 08:01:00.000000", "Src IP": "10.0.0.3", "Dst IP": "10.0.0.4", "Label": "FTP-Patator"}],
                "wednesday.csv": [{"Timestamp": "2017-07-05 08:02:00.000000", "Src IP": "10.0.0.5", "Dst IP": "10.0.0.6", "Label": "DoS Hulk"}],
                "thursday.csv": [{"Timestamp": "2017-07-06 08:03:00.000000", "Src IP": "10.0.0.7", "Dst IP": "10.0.0.8", "Label": "Web Attack - SQL Injection"}],
                "friday.csv": [{"Timestamp": "2017-07-07 08:04:00.000000", "Src IP": "10.0.0.9", "Dst IP": "10.0.0.10", "Label": "Botnet"}],
            }
            for name, rows in samples.items():
                _write_cic_day(raw_root / name, rows)

            frame = preprocess_cic(raw_root.parent, snapshot_seconds=60)

        positive_campaigns = set(frame.loc[frame["label"] == 1, "campaign"].tolist())
        self.assertEqual(
            positive_campaigns,
            {"FTP-Patator", "DoS Hulk", "Web Attack - SQL Injection", "Botnet"},
        )
        self.assertEqual(frame.loc[frame["label"] == 0, "campaign"].tolist(), [""])

    def test_argus_cic_loader_keeps_campaign_specific_edge_samples(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir)
            cic_root = data_root / "cic"
            cic_root.mkdir(parents=True, exist_ok=True)

            with (cic_root / "0.txt").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow([0, 0, 1, 10, 20, 1, 1, "Botnet"])
                writer.writerow([0, 0, 1, 10, 20, 1, 1, "Portscan"])
                writer.writerow([0, 0, 1, 10, 20, 1, 0, ""])
                writer.writerow([150, 0, 1, 10, 20, 1, 0, ""])

            with (cic_root / "nmap.pkl").open("wb") as handle:
                pickle.dump(["host_a", "host_b"], handle, protocol=pickle.HIGHEST_PROTOCOL)

            previous_root = os.environ.get("ARGUS_DATA_ROOT")
            os.environ["ARGUS_DATA_ROOT"] = str(data_root)
            sys.path.insert(0, str(ROOT / "local_runtime" / "detection_baselines" / "argus" / "CIC"))
            try:
                sys.modules.pop("loaders.load_cic", None)
                sys.modules.pop("loaders", None)
                loader = importlib.import_module("loaders.load_cic")
                data = loader.load_partial_lanl(start=0, end=200, delta=100, is_test=True)
            finally:
                sys.path.pop(0)
                if previous_root is None:
                    os.environ.pop("ARGUS_DATA_ROOT", None)
                else:
                    os.environ["ARGUS_DATA_ROOT"] = previous_root

        self.assertEqual(data.campaigns[0], ["Botnet", "Portscan", ""])
        self.assertEqual(data.eis[0].shape[1], 3)

if __name__ == "__main__":
    unittest.main()
