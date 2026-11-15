from __future__ import annotations

import csv
import importlib
import os
import pickle
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]

def _write_csv(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

def _load_module(package_root: Path, module_name: str, env_key: str, env_value: str):
    previous = os.environ.get(env_key)
    os.environ[env_key] = env_value
    sys.path.insert(0, str(package_root))
    try:
        for name in list(sys.modules):
            if name == "loaders" or name.startswith("loaders."):
                sys.modules.pop(name, None)
        module = importlib.import_module(module_name)
    finally:
        sys.path.pop(0)
        if previous is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = previous
    return module

class GraphCampaignTests(unittest.TestCase):
    def test_argus_lanl_preprocess_keeps_first_redteam_event_and_writes_campaign(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            raw_root = Path(tmp_dir) / "raw"
            out_root = Path(tmp_dir) / "out"
            raw_root.mkdir(parents=True, exist_ok=True)
            out_root.mkdir(parents=True, exist_ok=True)

            (raw_root / "auth.txt").write_text(
                "".join(
                    [
                        "150900,U1@DOM1,U9@DOM1,C1,C10,NTLM,Network,Auth,Success\n",
                        "150920,U1@DOM1,U9@DOM1,C1,C11,NTLM,Network,Auth,Success\n",
                        "151500,U2@DOM1,U9@DOM1,C2,C20,NTLM,Network,Auth,Success\n",
                    ]
                ),
                encoding="utf-8",
            )
            (raw_root / "redteam.txt").write_text(
                "".join(
                    [
                        "150900,U1@DOM1,C1,C10\n",
                        "150920,U1@DOM1,C1,C11\n",
                        "151500,U2@DOM1,C2,C20\n",
                    ]
                ),
                encoding="utf-8",
            )
            (raw_root / "flows.txt").write_text("", encoding="utf-8")

            cmd = [
                sys.executable,
                str(ROOT / "local_runtime" / "pre_processing" / "argus" / "process_lanl_argus.py"),
                "-i",
                str(raw_root) + os.sep,
                "-o",
                str(out_root) + os.sep,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            with (out_root / "150000.txt").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))

        self.assertEqual(rows[0][9], "1")
        self.assertEqual(rows[0][10], "lanl_campaign_001")
        self.assertEqual(rows[1][10], "lanl_campaign_001")
        self.assertEqual(rows[2][10], "lanl_campaign_002")

    def test_argus_lanl_loader_preserves_campaign_specific_edges(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir)
            lanl_root = data_root / "lanl"
            lanl_root.mkdir(parents=True, exist_ok=True)
            _write_csv(
                lanl_root / "0.txt",
                [
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, "lanl_campaign_001"],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, "lanl_campaign_002"],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, ""],
                    [150, 0, 1, 0, 0, 0, 0, 0, 0, 0, ""],
                ],
            )
            with (lanl_root / "nmap.pkl").open("wb") as handle:
                pickle.dump(["host_a", "host_b"], handle, protocol=pickle.HIGHEST_PROTOCOL)
            with (lanl_root / "umap.pkl").open("wb") as handle:
                pickle.dump(["U1"], handle, protocol=pickle.HIGHEST_PROTOCOL)

            loader = _load_module(
                ROOT / "local_runtime" / "detection_baselines" / "argus" / "LANL_OpTC",
                "loaders.load_lanl",
                "ARGUS_DATA_ROOT",
                str(data_root),
            )
            data = loader.load_partial_lanl(start=0, end=200, delta=100, is_test=True, use_flows=False)

        self.assertEqual(data.campaigns[0], ["lanl_campaign_001", "lanl_campaign_002", ""])
        self.assertEqual(data.eis[0].shape[1], 3)

    def test_argus_lanl_loader_flushes_final_partial_window(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir)
            lanl_root = data_root / "lanl"
            lanl_root.mkdir(parents=True, exist_ok=True)
            _write_csv(
                lanl_root / "0.txt",
                [
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, "lanl_campaign_001"],
                    [50, 0, 1, 0, 0, 0, 0, 0, 0, 0, ""],
                ],
            )
            with (lanl_root / "nmap.pkl").open("wb") as handle:
                pickle.dump(["host_a", "host_b"], handle, protocol=pickle.HIGHEST_PROTOCOL)
            with (lanl_root / "umap.pkl").open("wb") as handle:
                pickle.dump(["U1"], handle, protocol=pickle.HIGHEST_PROTOCOL)

            loader = _load_module(
                ROOT / "local_runtime" / "detection_baselines" / "argus" / "LANL_OpTC",
                "loaders.load_lanl",
                "ARGUS_DATA_ROOT",
                str(data_root),
            )
            data = loader.load_partial_lanl(start=0, end=200, delta=100, is_test=True, use_flows=False)

        self.assertEqual(data.T, 1)
        self.assertEqual(data.campaigns[0], ["lanl_campaign_001", ""])
        self.assertEqual(data.eis[0].shape[1], 2)

    def test_euler_optc_loader_preserves_campaign_specific_edges(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir)
            optc_root = data_root / "optc"
            optc_root.mkdir(parents=True, exist_ok=True)
            _write_csv(
                optc_root / "0.txt",
                [
                    [0, 0, 1, 10, 20, 1, 1, 22, 1, "optc_campaign_001"],
                    [0, 0, 1, 10, 20, 1, 1, 22, 1, "optc_campaign_002"],
                    [0, 0, 1, 10, 20, 1, 1, 22, 0, ""],
                    [150, 0, 1, 10, 20, 1, 1, 22, 0, ""],
                ],
            )
            with (optc_root / "nmap.pkl").open("wb") as handle:
                pickle.dump(["host_a", "host_b"], handle, protocol=pickle.HIGHEST_PROTOCOL)

            loader = _load_module(
                ROOT / "local_runtime" / "detection_baselines" / "euler" / "LANL_OpTC",
                "loaders.load_optc",
                "EULER_DATA_ROOT",
                str(data_root),
            )
            data = loader.load_partial_lanl(start=0, end=200, delta=100, is_test=True)

        self.assertEqual(data.campaigns[0], ["optc_campaign_001", "optc_campaign_002", ""])
        self.assertEqual(data.eis[0].shape[1], 3)

    def test_vgrnn_optc_loader_preserves_campaign_specific_edges(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir)
            optc_root = data_root / "optc"
            optc_root.mkdir(parents=True, exist_ok=True)
            _write_csv(
                optc_root / "0.txt",
                [
                    [0, 0, 1, 10, 20, 1, 1, 22, 1, "optc_campaign_001"],
                    [0, 0, 1, 10, 20, 1, 1, 22, 1, "optc_campaign_002"],
                    [0, 0, 1, 10, 20, 1, 1, 22, 0, ""],
                    [150, 0, 1, 10, 20, 1, 1, 22, 0, ""],
                ],
            )
            with (optc_root / "nmap.pkl").open("wb") as handle:
                pickle.dump(["host_a", "host_b"], handle, protocol=pickle.HIGHEST_PROTOCOL)

            loader = _load_module(
                ROOT / "local_runtime" / "detection_baselines" / "vgrnn" / "LANL_OpTC",
                "loaders.load_optc",
                "VGRNN_DATA_ROOT",
                str(data_root),
            )
            data = loader.load_partial_lanl(start=0, end=200, delta=100, is_test=True)

        self.assertEqual(data.campaigns[0], ["optc_campaign_001", "optc_campaign_002", ""])
        self.assertEqual(data.eis[0].shape[1], 3)

if __name__ == "__main__":
    unittest.main()
