from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from gids_eval.status_codes import UNSUPPORTED_EXIT_CODE
from runtime_shims import load_eg1_sitecustomize

load_eg1_sitecustomize()

THIS_DIR = Path(__file__).resolve().parent
FLOAT32_BYTES = 4
DEFAULT_MAX_ESTIMATED_GIB = 16.0
WORKING_SET_MULTIPLIER = 4.0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PIKACHU with SC1-derived metadata")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--detector-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--dimensions", default="100")
    parser.add_argument("--alpha", default="0.01")
    parser.add_argument("--support", default="10")
    parser.add_argument("--gpu", default="-1")
    parser.add_argument("--walklen", default="500")
    parser.add_argument("--numwalk", default="1")
    parser.add_argument("--epoch", default="50")
    parser.add_argument("--iter", default="10")
    return parser.parse_args()

def _estimate_input_shape(input_path: Path) -> tuple[int, int]:
    snapshots: set[str] = set()
    nodes: set[str] = set()
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"snapshot", "src_computer", "dst_computer"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            missing = sorted(required.difference(set(reader.fieldnames or [])))
            raise SystemExit(f"PIKACHU input is missing required columns: {missing}")
        for row in reader:
            snapshots.add(str(row["snapshot"]))
            nodes.add(str(row["src_computer"]))
            nodes.add(str(row["dst_computer"]))
    return len(snapshots), len(nodes)

def _preflight_report(
    *,
    dataset: str,
    input_path: Path,
    output_dir: Path,
    dimensions: int,
    max_estimated_gib: float,
) -> dict[str, object]:
    snapshot_count, node_count = _estimate_input_shape(input_path)
    tensor_bytes = snapshot_count * node_count * dimensions * FLOAT32_BYTES
    tensor_gib = tensor_bytes / (1024.0 ** 3)
    estimated_working_set_gib = tensor_gib * WORKING_SET_MULTIPLIER
    supported = estimated_working_set_gib <= max_estimated_gib
    report = {
        "dataset": dataset,
        "input_path": str(input_path),
        "snapshot_count": snapshot_count,
        "node_count": node_count,
        "dimensions": dimensions,
        "estimated_tensor_gib": round(tensor_gib, 3),
        "estimated_working_set_gib": round(estimated_working_set_gib, 3),
        "working_set_multiplier": WORKING_SET_MULTIPLIER,
        "max_estimated_gib": max_estimated_gib,
        "supported": supported,
        "reason": (
            "estimated_working_set_exceeds_threshold"
            if not supported
            else "within_threshold"
        ),
    }
    report_path = output_dir / "pikachu_preflight.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report

def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    output_dir = Path(args.output_dir)
    runtime_dir = output_dir / "_pikachu_runtime"
    (runtime_dir / "weights").mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "joblib-tmp").mkdir(parents=True, exist_ok=True)

    detector_root = Path(args.detector_root)
    main_py = detector_root / "main.py"
    input_path = Path(str(metadata["input_path"]))
    max_estimated_gib = float(os.environ.get("EG1_PIKACHU_MAX_ESTIMATED_GIB", DEFAULT_MAX_ESTIMATED_GIB))
    preflight = _preflight_report(
        dataset=args.dataset,
        input_path=input_path,
        output_dir=output_dir,
        dimensions=int(args.dimensions),
        max_estimated_gib=max_estimated_gib,
    )
    if not bool(preflight["supported"]):
        print(
            (
                "PIKACHU unsupported for this EG1 input: "
                f"estimated_working_set_gib={preflight['estimated_working_set_gib']} "
                f"> max_estimated_gib={preflight['max_estimated_gib']}"
            ),
            file=sys.stderr,
        )
        raise SystemExit(UNSUPPORTED_EXIT_CODE)

    command = [
        sys.executable,
        str(THIS_DIR / "legacy_entrypoint.py"),
        str(main_py),
        "--dataset",
        args.dataset,
        "--input",
        str(input_path),
        "--trainwin",
        str(metadata["trainwin"]),
        "--dimensions",
        str(args.dimensions),
        "--alpha",
        str(args.alpha),
        "--support",
        str(args.support),
        "--walklen",
        str(args.walklen),
        "--numwalk",
        str(args.numwalk),
        "--epoch",
        str(args.epoch),
        "--iter",
        str(args.iter),
        "--output",
        str(output_dir),
        "--metrics_csv",
        str(args.metrics_csv),
        "--gpu",
        str(args.gpu),
    ]

    env = dict(os.environ)
    env.setdefault("EG1_PIKACHU_POOL_BACKEND", "thread")
    env.setdefault("EG1_PIKACHU_MAX_WORKERS", str(min(os.cpu_count() or 1, 4)))
    env.setdefault("JOBLIB_TEMP_FOLDER", str(runtime_dir / "joblib-tmp"))
    completed = subprocess.run(
        command,
        cwd=str(runtime_dir),
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

if __name__ == "__main__":
    main()
