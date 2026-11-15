from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from gids_eval.io_utils import ensure_dir

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gids_eval.module_tasks")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke-stage", help="Run a lightweight synthetic stage")
    smoke.add_argument("--stage", required=True)
    smoke.add_argument("--output-dir", required=True)
    smoke.add_argument("--input-payload")
    smoke.add_argument("--staged-manifests")
    smoke.add_argument("--delta", type=float, default=1.0)
    smoke.add_argument("--tag", default="")
    smoke.add_argument("--write-metrics", action="store_true")
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "smoke-stage":
        run_smoke_stage(args)
        return

    raise SystemExit(f"Unsupported command: {args.command}")

def run_smoke_stage(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir)
    input_payload = _load_json(args.input_payload) if args.input_payload else {}
    datasets = _resolve_datasets(input_payload, args.staged_manifests)
    history = list(input_payload.get("history", []))

    step = {
        "stage": args.stage,
        "delta": args.delta,
        "tag": args.tag,
    }
    history.append(step)

    score = round(float(input_payload.get("score", 0.0)) + float(args.delta), 6)
    payload = {
        "stage": args.stage,
        "score": score,
        "datasets": datasets,
        "history": history,
        "step_count": len(history),
        "source_payload": str(args.input_payload or ""),
    }

    _write_json(output_dir / "payload.json", payload)
    (output_dir / "summary.txt").write_text(
        f"stage={args.stage}\nscore={score}\ntag={args.tag}\nsteps={len(history)}\n",
        encoding="utf-8",
    )

    if args.write_metrics:
        _write_metrics(output_dir / "metrics.csv", payload)

def _resolve_datasets(input_payload: dict[str, Any], staged_manifests_path: str | None) -> list[str]:
    datasets = input_payload.get("datasets")
    if isinstance(datasets, list) and datasets:
        return [str(item) for item in datasets]

    if staged_manifests_path:
        manifests = _load_json(staged_manifests_path)
        if isinstance(manifests, list):
            result = [str(item.get("dataset_id", "unknown")) for item in manifests]
            return result or ["synthetic"]
    return ["synthetic"]

def _write_metrics(path: Path, payload: dict[str, Any]) -> None:
    score = float(payload["score"])
    threshold = round(0.05 * int(payload["step_count"]), 4)
    train_time = round(0.5 * int(payload["step_count"]), 4)
    test_time = round(0.1 * int(payload["step_count"]), 4)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "auc",
                "ap",
                "threshold",
                "tp",
                "fp",
                "fn",
                "tn",
                "tpr",
                "fpr",
                "fnr",
                "tnr",
                "precision",
                "recall",
                "f1",
                "train_time",
                "test_time",
            ],
        )
        writer.writeheader()
        for index, dataset in enumerate(payload["datasets"], start=1):
            auc = min(0.999, round(0.55 + (score / 20.0), 6))
            ap = min(0.999, round(0.5 + (score / 25.0), 6))
            tp = 10 + index + int(score)
            fp = max(0, 3 - index)
            fn = max(0, 2 - index)
            tn = 50 + index
            precision = round(tp / max(1, tp + fp), 6)
            recall = round(tp / max(1, tp + fn), 6)
            f1 = round((2 * precision * recall) / max(1e-9, precision + recall), 6)
            writer.writerow(
                {
                    "dataset": dataset,
                    "auc": auc,
                    "ap": ap,
                    "threshold": threshold,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "tpr": recall,
                    "fpr": round(fp / max(1, fp + tn), 6),
                    "fnr": round(fn / max(1, fn + tp), 6),
                    "tnr": round(tn / max(1, tn + fp), 6),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "train_time": train_time,
                    "test_time": test_time,
                }
            )

def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)

def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

if __name__ == "__main__":
    main()
