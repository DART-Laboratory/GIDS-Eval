
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from eg3_compat import load_eg3_compat

load_eg3_compat()

THIS_DIR = Path(__file__).resolve().parent

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EG3 PIKACHU runner")
    p.add_argument("--dataset", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--detector-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--metrics-csv", required=True)
    p.add_argument("--calibration", required=True, choices=["transductive", "inductive"])
    p.add_argument("--val-fraction", type=float, default=0.5)
    p.add_argument("--dimensions", default="100")
    p.add_argument("--alpha", default="0.01")
    p.add_argument("--support", default="10")
    p.add_argument("--gpu", default="-1")
    p.add_argument("--walklen", default="500")
    p.add_argument("--numwalk", default="1")
    p.add_argument("--epoch", default="50")
    p.add_argument("--iter", default="10")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    with metadata_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    output_dir = Path(args.output_dir)
    runtime_dir = output_dir / "_pikachu_runtime"
    (runtime_dir / "weights").mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "joblib-tmp").mkdir(parents=True, exist_ok=True)

    detector_root = Path(args.detector_root)

    eg3_detect = THIS_DIR / "eg3_anomaly_detection.py"

    command = [
        sys.executable,
        str(THIS_DIR / "eg3_compat.py"),
        str(eg3_detect),
        "--dataset", args.dataset,
        "--input", str(metadata["input_path"]),
        "--trainwin", str(metadata["trainwin"]),
        "--dimensions", str(args.dimensions),
        "--alpha", str(args.alpha),
        "--support", str(args.support),
        "--walklen", str(args.walklen),
        "--numwalk", str(args.numwalk),
        "--epoch", str(args.epoch),
        "--iter", str(args.iter),
        "--output", str(output_dir),
        "--metrics_csv", str(args.metrics_csv),
        "--gpu", str(args.gpu),
        "--calibration", args.calibration,
        "--val_fraction", str(args.val_fraction),
        "--detector_root", str(detector_root),
    ]

    env = dict(os.environ)
    env.setdefault("EG1_PIKACHU_POOL_BACKEND", "thread")
    env.setdefault("EG1_PIKACHU_MAX_WORKERS", str(min(os.cpu_count() or 1, 4)))
    env.setdefault("JOBLIB_TEMP_FOLDER", str(runtime_dir / "joblib-tmp"))

    completed = subprocess.run(command, cwd=str(runtime_dir), env=env, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

if __name__ == "__main__":
    main()
