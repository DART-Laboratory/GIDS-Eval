from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

sys.path.insert(0, str(REPO_ROOT / "src"))

from gids_eval.process_monitor import maybe_profile_phase

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Anomal-E from EG5 metadata")
    parser.add_argument("--dataset", choices=["cic", "0501"], required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--detector-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics-csv", required=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    with Path(args.metadata).open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    input_path = Path(str(metadata["input_path"]))
    split_index = int(metadata["split_index"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    detector_root = Path(args.detector_root)
    run_script = detector_root / ("CIC" if args.dataset == "cic" else "0501") / "run.py"
    command = [
        sys.executable,
        str(run_script),
        "--dataset",
        args.dataset,
        "--file",
        str(input_path),
        "--split",
        str(split_index),
    ]

    env = dict(os.environ)
    env["ANOMAL_RESULTS_ROOT"] = str(output_dir)
    env["ANOMAL_MODELS_ROOT"] = str(models_dir)
    env["GIDS_PREDICTIONS_PATH"] = str(output_dir / "predictions.csv")

    with maybe_profile_phase("detection"):
        completed = subprocess.run(command, cwd=str(detector_root), env=env, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    metrics_path = output_dir / "metrics.csv"
    if not metrics_path.exists():
        raise SystemExit(f"Anomal-E did not produce metrics at {metrics_path}")

if __name__ == "__main__":
    main()
