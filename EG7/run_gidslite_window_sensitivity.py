from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from gids_eval.paths import get_paths

_PATHS = get_paths()
from typing import Any

REPO_ROOT = _PATHS.repo_root
SRC_ROOT = REPO_ROOT / "src"
CONDA_ROOT = _PATHS.conda_root
WORKER_PYTHON_CANDIDATES = (
    CONDA_ROOT / "envs" / "sc1-pikachu" / "bin" / "python",
    CONDA_ROOT / "envs" / "sc1-torch-base" / "bin" / "python",
    Path(sys.executable),
)
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gids_eval.io_utils import dump_json, dump_yaml, ensure_dir, load_yaml, write_csv_rows
from gids_eval.process_monitor import run_command_logged

EG7_WORKER = REPO_ROOT / "EG7" / "eg7_worker.py"
ARTIFACT_PATH = REPO_ROOT / "gids-lite" / "detector.yaml"
DATA_ROOT = _PATHS.sc_data_root("eg7") / "gidslite_window_sensitivity"
CONFIG_ROOT = DATA_ROOT / "configs"
EXPERIMENT_ROOT = DATA_ROOT / "experiments"
REPORT_ROOT = DATA_ROOT / "reports"
RAW_RESULTS_PATH = REPORT_ROOT / "raw_results.csv"
SUMMARY_BY_DATASET_PATH = REPORT_ROOT / "summary_by_dataset.csv"
PAPER_TABLE_PATH = REPORT_ROOT / "paper_table.csv"
CONCLUSION_PATH = REPORT_ROOT / "conclusion.md"
RUN_SUMMARY_PATH = REPORT_ROOT / "run_summary.json"

WINDOW_GRID = {
    "0501": (75, 150, 300, 450),
    "cic": (600, 900, 1800, 3000),
    "lanl": (1800, 3600, 5400, 7200),
    "optc": (360, 1800, 3600, 9000),
}
DATASET_ORDER = ("0501", "lanl", "cic", "optc")
DATASET_LABELS = {
    "0501": "Campus",
    "lanl": "LANL",
    "cic": "CI-2017",
    "optc": "OpTC",
}
COMMON_CONFIG = {
    "schema_version": 1,
    "kind": "eg7_job",
    "val_fraction": 0.33,
    "neg_ratio": 20,
    "preprocess_seed": 1337,
    "one_hot_dim": 64,
    "projection_dim": 16,
    "mcc_grid_size": 201,
    "code_version": "eg7_gidslite_window_sensitivity_v1",
}
RAW_FIELDS = [
    "dataset",
    "dataset_label",
    "window_seconds",
    "window_label",
    "experiment_id",
    "featurization",
    "graph_encoder",
    "temporal_encoder",
    "loss",
    "status",
    "ap",
    "cdr",
    "precision",
    "fpr",
    "threshold",
    "campaign_count",
    "detected_campaigns",
    "cdr_meaningful",
    "train_time",
    "test_time",
    "total_wall_time_sec",
    "run_peak_rss_gib",
    "metrics_path",
    "predictions_path",
    "config_path",
    "stdout_path",
    "stderr_path",
]
SUMMARY_FIELDS = [
    "dataset",
    "dataset_label",
    "ap_min",
    "ap_max",
    "ap_range_display",
    "ap_swing",
    "ap_swing_percent",
    "cdr_min",
    "cdr_max",
    "cdr_range",
    "cdr_range_pp",
    "best_window_seconds",
    "best_window_label",
    "best_ap",
]

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_gidslite_window_sensitivity")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset list")
    parser.add_argument("--windows", default=None, help="Comma-separated integer window sizes")
    parser.add_argument("--force", action="store_true", help="Re-run even when outputs already exist")
    parser.add_argument("--max-jobs", type=int, default=None, help="Stop after N attempted jobs")
    return parser

def parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]

def format_window_label(seconds: int) -> str:
    if seconds % 3600 == 0:
        return f"{seconds / 3600:g}h"
    if seconds % 60 == 0:
        return f"{seconds / 60:g}m"
    return f"{seconds}s"

def worker_python_bin() -> str:
    for candidate in WORKER_PYTHON_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return sys.executable

def artifact_specs() -> dict[str, dict[str, Any]]:
    payload = load_yaml(ARTIFACT_PATH)
    jobs = payload.get("jobs") or []
    specs: dict[str, dict[str, Any]] = {}
    for row in jobs:
        dataset = str(row["dataset"])
        specs[dataset] = {
            "featurization": str(row["featurization"]),
            "graph_encoder": str(row["graph_encoder"]),
            "temporal_encoder": str(row["temporal_encoder"]),
            "loss": str(row["loss"]),
        }
    missing = [dataset for dataset in DATASET_ORDER if dataset not in specs]
    if missing:
        raise ValueError(f"gids-lite is missing dataset specs: {missing}")
    return specs

def ensure_dirs() -> None:
    for path in (DATA_ROOT, CONFIG_ROOT, EXPERIMENT_ROOT, REPORT_ROOT):
        ensure_dir(path)

def selected_jobs(*, datasets: list[str], windows: list[int]) -> list[tuple[str, int]]:
    chosen_datasets = datasets or list(DATASET_ORDER)
    selected: list[tuple[str, int]] = []
    allowed_windows = set(windows)
    for dataset in DATASET_ORDER:
        if dataset not in chosen_datasets:
            continue
        for window_seconds in WINDOW_GRID[dataset]:
            if allowed_windows and window_seconds not in allowed_windows:
                continue
            selected.append((dataset, window_seconds))
    return selected

def build_config(dataset: str, window_seconds: int, spec: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    window_label = format_window_label(window_seconds)
    experiment_id = f"eg7_gidslite_window_{dataset}_{window_label.replace('.', 'p')}"
    workspace_root = EXPERIMENT_ROOT / dataset / window_label / "eg7"
    metadata_root = workspace_root / "metadata"
    output_root = workspace_root / "output"
    result_root = output_root / "results"
    config = {
        **COMMON_CONFIG,
        "name": f"GIDS-Lite window sensitivity {dataset} {window_label}",
        "experiment_id": experiment_id,
        "dataset": dataset,
        "window_seconds": int(window_seconds),
        "featurization": spec["featurization"],
        "graph_encoder": spec["graph_encoder"],
        "temporal_encoder": spec["temporal_encoder"],
        "loss": spec["loss"],
        "workspace_root": str(workspace_root),
        "metadata_root": str(metadata_root),
        "output_root": str(output_root),
        "result_root": str(result_root),
        "paths": {
            "repo_root": str(REPO_ROOT),
            "data_root": str(DATA_ROOT),
            "cache_root": str(DATA_ROOT / "cache"),
            "workspace_root": str(workspace_root),
            "metadata_root": str(metadata_root),
            "output_root": str(output_root),
            "result_root": str(result_root),
        },
    }
    config_path = CONFIG_ROOT / f"{dataset}_{window_seconds}.yaml"
    return config, config_path

def read_single_csv_row(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            return dict(row)
    raise ValueError(f"No rows found in {path}")

def safe_float(value: Any) -> float:
    if value in ("", None):
        return 0.0
    return float(value)

def _gib_from_kb(value: Any) -> float:
    return round(float(value or 0.0) / (1024.0 * 1024.0), 6)

def existing_result(dataset: str, window_seconds: int, spec: dict[str, Any]) -> dict[str, Any] | None:
    config, config_path = build_config(dataset, window_seconds, spec)
    metadata_root = Path(config["metadata_root"])
    metrics_path = metadata_root / "baseline_metrics.csv"
    stdout_path = metadata_root / "outer.stdout.log"
    stderr_path = metadata_root / "outer.stderr.log"
    summary_path = metadata_root / "run_summary.json"
    if not metrics_path.exists() or not summary_path.exists():
        return None
    summary = load_json(summary_path)
    if str(summary.get("status", "")) != "success":
        return None
    metrics_row = read_single_csv_row(metrics_path)
    return {
        "dataset": dataset,
        "dataset_label": DATASET_LABELS[dataset],
        "window_seconds": int(window_seconds),
        "window_label": format_window_label(window_seconds),
        "experiment_id": config["experiment_id"],
        "featurization": spec["featurization"],
        "graph_encoder": spec["graph_encoder"],
        "temporal_encoder": spec["temporal_encoder"],
        "loss": spec["loss"],
        "status": "success",
        "ap": safe_float(metrics_row.get("ap")),
        "cdr": safe_float(metrics_row.get("cdr")),
        "precision": safe_float(metrics_row.get("precision")),
        "fpr": safe_float(metrics_row.get("fpr")),
        "threshold": safe_float(metrics_row.get("threshold")),
        "campaign_count": int(float(metrics_row.get("campaign_count") or 0)),
        "detected_campaigns": int(float(metrics_row.get("detected_campaigns") or 0)),
        "cdr_meaningful": str(metrics_row.get("cdr_meaningful", "")),
        "train_time": safe_float(metrics_row.get("train_time")),
        "test_time": safe_float(metrics_row.get("test_time")),
        "total_wall_time_sec": safe_float(summary.get("total_wall_time_sec")),
        "run_peak_rss_gib": safe_float(summary.get("run_peak_rss_gib")),
        "metrics_path": str(metrics_path),
        "predictions_path": str(Path(config["result_root"]) / "predictions.csv"),
        "config_path": str(config_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }

def load_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))

def run_job(dataset: str, window_seconds: int, spec: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    if not force:
        cached = existing_result(dataset, window_seconds, spec)
        if cached is not None:
            return cached
    config, config_path = build_config(dataset, window_seconds, spec)
    dump_yaml(config_path, config)
    metadata_root = Path(config["metadata_root"])
    ensure_dir(metadata_root)
    stdout_path = metadata_root / "outer.stdout.log"
    stderr_path = metadata_root / "outer.stderr.log"
    metrics_path = metadata_root / "baseline_metrics.csv"
    predictions_path = Path(config["result_root"]) / "predictions.csv"
    run_summary_path = metadata_root / "run_summary.json"
    env = dict(os.environ)
    pythonpath = [str(REPO_ROOT / "EG7"), str(REPO_ROOT / "src")]
    current_pythonpath = env.get("PYTHONPATH", "").strip()
    if current_pythonpath:
        pythonpath.append(current_pythonpath)
    env["PYTHONPATH"] = ":".join(pythonpath)
    env["GIDS_PHASE_PROFILE_PATH"] = str(metadata_root / "resource_profile.json")
    command = [worker_python_bin(), str(EG7_WORKER), "--config", str(config_path)]
    usage = run_command_logged(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        poll_interval=0.5,
        monitor_resources=True,
    )
    status = "success" if usage.exit_code == 0 and metrics_path.exists() else "failed"
    dump_json(
        run_summary_path,
        {
            "status": status,
            "dataset": dataset,
            "dataset_label": DATASET_LABELS[dataset],
            "window_seconds": int(window_seconds),
            "window_label": format_window_label(window_seconds),
            "experiment_id": config["experiment_id"],
            "command": command,
            "exit_code": int(usage.exit_code),
            "total_wall_time_sec": round(float(usage.wall_time_sec), 6),
            "run_peak_rss_kb": int(usage.peak_rss_kb),
            "run_peak_rss_gib": _gib_from_kb(usage.peak_rss_kb),
            "metrics_path": str(metrics_path),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        },
    )
    if status != "success":
        raise RuntimeError(f"{dataset} {window_seconds} failed with exit code {usage.exit_code}")
    metrics_row = read_single_csv_row(metrics_path)
    return {
        "dataset": dataset,
        "dataset_label": DATASET_LABELS[dataset],
        "window_seconds": int(window_seconds),
        "window_label": format_window_label(window_seconds),
        "experiment_id": config["experiment_id"],
        "featurization": spec["featurization"],
        "graph_encoder": spec["graph_encoder"],
        "temporal_encoder": spec["temporal_encoder"],
        "loss": spec["loss"],
        "status": status,
        "ap": safe_float(metrics_row.get("ap")),
        "cdr": safe_float(metrics_row.get("cdr")),
        "precision": safe_float(metrics_row.get("precision")),
        "fpr": safe_float(metrics_row.get("fpr")),
        "threshold": safe_float(metrics_row.get("threshold")),
        "campaign_count": int(float(metrics_row.get("campaign_count") or 0)),
        "detected_campaigns": int(float(metrics_row.get("detected_campaigns") or 0)),
        "cdr_meaningful": str(metrics_row.get("cdr_meaningful", "")),
        "train_time": safe_float(metrics_row.get("train_time")),
        "test_time": safe_float(metrics_row.get("test_time")),
        "total_wall_time_sec": round(float(usage.wall_time_sec), 6),
        "run_peak_rss_gib": _gib_from_kb(usage.peak_rss_kb),
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "config_path": str(config_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }

def best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    default_window = artifact_default_window(str(rows[0]["dataset"]))
    return max(
        rows,
        key=lambda row: (
            float(row["ap"]),
            float(row["cdr"]),
            -abs(int(row["window_seconds"]) - default_window),
            -int(row["window_seconds"]),
        ),
    )

def artifact_default_window(dataset: str) -> int:
    jobs = load_yaml(ARTIFACT_PATH).get("jobs") or []
    for job in jobs:
        if str(job.get("dataset")) == dataset:
            return int(job["window_seconds"])
    raise KeyError(dataset)

def summarize_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {dataset: [] for dataset in DATASET_ORDER}
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    summary_rows: list[dict[str, Any]] = []
    paper_rows: list[dict[str, Any]] = []
    largest_ap_swing = ("", -1.0)
    largest_cdr_range = ("", -1.0)
    for dataset in DATASET_ORDER:
        current = sorted(grouped[dataset], key=lambda row: int(row["window_seconds"]))
        if not current:
            continue
        ap_values = [float(row["ap"]) for row in current]
        cdr_values = [float(row["cdr"]) for row in current]
        min_ap = min(ap_values)
        max_ap = max(ap_values)
        min_cdr = min(cdr_values)
        max_cdr = max(cdr_values)
        ap_swing = 0.0 if math.isclose(max_ap, 0.0) else (max_ap - min_ap) / max_ap
        cdr_range = max_cdr - min_cdr
        chosen = best_row(current)
        summary_row = {
            "dataset": dataset,
            "dataset_label": DATASET_LABELS[dataset],
            "ap_min": round(min_ap, 6),
            "ap_max": round(max_ap, 6),
            "ap_range_display": f"{min_ap:.3f}--{max_ap:.3f}",
            "ap_swing": round(ap_swing, 6),
            "ap_swing_percent": round(ap_swing * 100.0, 1),
            "cdr_min": round(min_cdr, 6),
            "cdr_max": round(max_cdr, 6),
            "cdr_range": round(cdr_range, 6),
            "cdr_range_pp": round(cdr_range * 100.0, 1),
            "best_window_seconds": int(chosen["window_seconds"]),
            "best_window_label": str(chosen["window_label"]),
            "best_ap": round(float(chosen["ap"]), 6),
        }
        summary_rows.append(summary_row)
        paper_rows.append(
            {
                "dataset": DATASET_LABELS[dataset],
                "ap_range": summary_row["ap_range_display"],
                "ap_swing_percent": f"{summary_row['ap_swing_percent']:.1f}\\%",
                "cdr_range_pp": f"{summary_row['cdr_range_pp']:.1f}",
                "best_window": summary_row["best_window_label"],
            }
        )
        if ap_swing > largest_ap_swing[1]:
            largest_ap_swing = (DATASET_LABELS[dataset], ap_swing)
        if cdr_range > largest_cdr_range[1]:
            largest_cdr_range = (DATASET_LABELS[dataset], cdr_range)
    conclusion = {
        "largest_ap_swing_dataset": largest_ap_swing[0],
        "largest_ap_swing_percent": round(largest_ap_swing[1] * 100.0, 1),
        "largest_cdr_range_dataset": largest_cdr_range[0],
        "largest_cdr_range_pp": round(largest_cdr_range[1] * 100.0, 1),
    }
    return summary_rows, paper_rows, conclusion

def write_reports(rows: list[dict[str, Any]]) -> None:
    summary_rows, paper_rows, conclusion = summarize_rows(rows)
    write_csv_rows(RAW_RESULTS_PATH, rows, RAW_FIELDS)
    write_csv_rows(SUMMARY_BY_DATASET_PATH, summary_rows, SUMMARY_FIELDS)
    write_csv_rows(
        PAPER_TABLE_PATH,
        paper_rows,
        ["dataset", "ap_range", "ap_swing_percent", "cdr_range_pp", "best_window"],
    )
    dump_json(
        RUN_SUMMARY_PATH,
        {
            "rows": rows,
            "summary_by_dataset": summary_rows,
            "paper_table": paper_rows,
            "conclusion": conclusion,
        },
    )
    lines = [
        "# GIDS-Lite Window Sensitivity",
        "",
        f"- largest_ap_swing: `{conclusion['largest_ap_swing_percent']:.1f}%` on `{conclusion['largest_ap_swing_dataset']}`",
        f"- largest_cdr_range: `{conclusion['largest_cdr_range_pp']:.1f}` percentage points on `{conclusion['largest_cdr_range_dataset']}`",
        "",
        "## Table Rows",
        "",
    ]
    for row in paper_rows:
        lines.append(
            f"- {row['dataset']}: AP {row['ap_range']}, swing {row['ap_swing_percent']}, "
            f"CDR range {row['cdr_range_pp']}, best window {row['best_window']}"
        )
    CONCLUSION_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> int:
    args = build_parser().parse_args()
    ensure_dirs()
    specs = artifact_specs()
    datasets = parse_csv_list(args.datasets)
    windows = [int(item) for item in parse_csv_list(args.windows)]
    jobs = selected_jobs(datasets=datasets, windows=windows)
    rows: list[dict[str, Any]] = []
    attempted = 0
    for dataset, window_seconds in jobs:
        rows.append(run_job(dataset, window_seconds, specs[dataset], force=bool(args.force)))
        attempted += 1
        if args.max_jobs is not None and attempted >= args.max_jobs:
            break
    if not rows:
        existing_rows = []
        if RAW_RESULTS_PATH.exists():
            with RAW_RESULTS_PATH.open("r", encoding="utf-8", newline="") as handle:
                existing_rows = list(csv.DictReader(handle))
        rows = existing_rows
    rows = sorted(rows, key=lambda row: (DATASET_ORDER.index(str(row["dataset"])), int(row["window_seconds"])))
    write_reports(rows)
    print(RAW_RESULTS_PATH)
    print(SUMMARY_BY_DATASET_PATH)
    print(PAPER_TABLE_PATH)
    print(CONCLUSION_PATH)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
