
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from gids_eval.io_utils import ensure_dir
from gids_eval.paths import get_paths
from gids_eval.process_monitor import run_command_logged

_PATHS = get_paths()
REPO_ROOT = _PATHS.repo_root
EG7_CODE_ROOT = REPO_ROOT / "EG7"
DATA_ROOT = _PATHS.sc_data_root("eg7")
GENERATED_CONFIG_ROOT = DATA_ROOT / "generated-configs"
ORCHESTRATOR_ROOT = DATA_ROOT / "orchestrator"
REPORT_ROOT = DATA_ROOT / "reports"
EXPERIMENT_ROOT = DATA_ROOT / "experiments"
LOG_ROOT = DATA_ROOT / "logs"
CACHE_ROOT = DATA_ROOT / "cache"
RUN_LABEL = "eg7"

PROGRESS_JSON_PATH = ORCHESTRATOR_ROOT / "progress.json"
PROGRESS_MD_PATH = ORCHESTRATOR_ROOT / "progress.md"

DATASET_WINDOW_SECONDS: dict[str, int] = {
    "0501": 300,
    "cic": 3600,
    "optc": 3600,
    "lanl": 3600,
}
DATASETS = list(DATASET_WINDOW_SECONDS)

FEATURIZATION_OPTIONS = (
    "one_hot",
    "flow_features",
    "structural",
    "one_hot_structural",
)
GRAPH_ENCODERS = (
    "none",
    "linear",
    "gcn",
    "graphsage",
    "mpnn",
)
TEMPORAL_ENCODERS = (
    "none",
    "gru",
    "lstm",
    "gc_lstm",
)
LOSS_OPTIONS = (
    "reconstruction",
    "contrastive",
    "link_prediction",
)

VAL_FRACTION = 0.33
NEG_RATIO = 20
PREPROCESS_SEED = 1337
ONE_HOT_DIM = 64
PROJECTION_DIM = 16
MCC_GRID_SIZE = 201
CODE_VERSION = "eg7_reference_v1"

HEATMAP_METRICS = ("ap", "cdr_auc", "mcc")

GRAPH_ENCODER_SHORT = {
    "none": "none",
    "linear": "lin",
    "gcn": "gcn",
    "graphsage": "sage",
    "mpnn": "mpnn",
}
TEMPORAL_ENCODER_SHORT = {
    "none": "none",
    "gru": "gru",
    "lstm": "lstm",
    "gc_lstm": "gclstm",
}

@dataclass(frozen=True)
class EG7Job:
    dataset: str
    featurization: str
    graph_encoder: str
    temporal_encoder: str
    loss: str

    @property
    def experiment_id(self) -> str:
        return (
            f"eg7_{self.dataset}_{self.featurization}_"
            f"{self.graph_encoder}_{self.temporal_encoder}_{self.loss}"
        )

    @property
    def config_path(self) -> Path:
        return GENERATED_CONFIG_ROOT / f"{self.experiment_id}.yaml"

    @property
    def workspace_root(self) -> Path:
        return EXPERIMENT_ROOT / self.experiment_id / RUN_LABEL

    @property
    def metadata_root(self) -> Path:
        return self.workspace_root / "metadata"

    @property
    def result_root(self) -> Path:
        return self.workspace_root / "output" / "results"

    @property
    def output_root(self) -> Path:
        return self.workspace_root / "output"

    @property
    def baseline_metrics_path(self) -> Path:
        return self.metadata_root / "baseline_metrics.csv"

    @property
    def cost_metrics_path(self) -> Path:
        return self.metadata_root / "cost_metrics.csv"

    @property
    def run_summary_path(self) -> Path:
        return self.metadata_root / "run_summary.json"

def ensure_eg7_dirs() -> None:
    for path in (
        GENERATED_CONFIG_ROOT,
        ORCHESTRATOR_ROOT,
        REPORT_ROOT,
        EXPERIMENT_ROOT,
        LOG_ROOT,
        CACHE_ROOT,
    ):
        ensure_dir(path)

def stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def iter_jobs(
    *,
    datasets: Iterable[str] | None = None,
    featurizations: Iterable[str] | None = None,
    graph_encoders: Iterable[str] | None = None,
    temporal_encoders: Iterable[str] | None = None,
    losses: Iterable[str] | None = None,
) -> list[EG7Job]:
    selected_datasets = list(datasets or DATASETS)
    selected_featurizations = list(featurizations or FEATURIZATION_OPTIONS)
    selected_graph_encoders = list(graph_encoders or GRAPH_ENCODERS)
    selected_temporal_encoders = list(temporal_encoders or TEMPORAL_ENCODERS)
    selected_losses = list(losses or LOSS_OPTIONS)

    jobs: list[EG7Job] = []
    for dataset in selected_datasets:
        if dataset not in DATASET_WINDOW_SECONDS:
            raise ValueError(f"Unknown dataset: {dataset}")
        for featurization in selected_featurizations:
            if featurization not in FEATURIZATION_OPTIONS:
                raise ValueError(f"Unknown featurization: {featurization}")
            for graph_encoder in selected_graph_encoders:
                if graph_encoder not in GRAPH_ENCODERS:
                    raise ValueError(f"Unknown graph encoder: {graph_encoder}")
                for temporal_encoder in selected_temporal_encoders:
                    if temporal_encoder not in TEMPORAL_ENCODERS:
                        raise ValueError(f"Unknown temporal encoder: {temporal_encoder}")
                    for loss in selected_losses:
                        if loss not in LOSS_OPTIONS:
                            raise ValueError(f"Unknown loss: {loss}")
                        jobs.append(
                            EG7Job(
                                dataset=dataset,
                                featurization=featurization,
                                graph_encoder=graph_encoder,
                                temporal_encoder=temporal_encoder,
                                loss=loss,
                            )
                        )
    return jobs

def same_job_set(left: Iterable[EG7Job], right: Iterable[EG7Job]) -> bool:
    return {job.experiment_id for job in left} == {job.experiment_id for job in right}

def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload

def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]

def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

def generate_config(job: EG7Job) -> Path:
    ensure_eg7_dirs()
    payload = {
        "schema_version": 1,
        "kind": "eg7_job",
        "name": f"EG7 {job.dataset} {job.featurization} {job.graph_encoder} {job.temporal_encoder} {job.loss}",
        "experiment_id": job.experiment_id,
        "dataset": job.dataset,
        "window_seconds": DATASET_WINDOW_SECONDS[job.dataset],
        "featurization": job.featurization,
        "graph_encoder": job.graph_encoder,
        "temporal_encoder": job.temporal_encoder,
        "loss": job.loss,
        "val_fraction": VAL_FRACTION,
        "neg_ratio": NEG_RATIO,
        "preprocess_seed": PREPROCESS_SEED,
        "one_hot_dim": ONE_HOT_DIM,
        "projection_dim": PROJECTION_DIM,
        "mcc_grid_size": MCC_GRID_SIZE,
        "code_version": CODE_VERSION,
        "workspace_root": str(job.workspace_root),
        "metadata_root": str(job.metadata_root),
        "output_root": str(job.output_root),
        "result_root": str(job.result_root),
        "paths": {
            "repo_root": str(REPO_ROOT),
            "data_root": str(DATA_ROOT),
            "cache_root": str(CACHE_ROOT),
            "workspace_root": str(job.workspace_root),
            "metadata_root": str(job.metadata_root),
            "output_root": str(job.output_root),
            "result_root": str(job.result_root),
        },
    }
    dump_yaml(job.config_path, payload)
    return job.config_path

def run_job(job: EG7Job, *, dry_run: bool = False, force: bool = False) -> str:
    config_path = generate_config(job)
    if not force and job_status(job) == "success":
        return "skipped"

    ensure_dir(job.metadata_root / "logs")
    stdout_log = job.metadata_root / "logs" / "worker.stdout.log"
    stderr_log = job.metadata_root / "logs" / "worker.stderr.log"
    process_json = job.metadata_root / "logs" / "worker_process.json"
    env = dict(os.environ)
    pythonpath_parts = [str(EG7_CODE_ROOT), str(REPO_ROOT / "src")]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    env["GIDS_PHASE_PROFILE_PATH"] = str(job.metadata_root / "resource_profile.json")

    command = [sys.executable, str(EG7_CODE_ROOT / "eg7_worker.py"), "--config", str(config_path)]
    if dry_run:
        command.append("--dry-run")

    usage = run_command_logged(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        stdout_path=stdout_log,
        stderr_path=stderr_log,
        monitor_resources=True,
    )
    process_json.write_text(
        json.dumps(
            {
                "exit_code": usage.exit_code,
                "wall_time_sec": usage.wall_time_sec,
                "peak_rss_kb": usage.peak_rss_kb,
                "mean_rss_kb": usage.mean_rss_kb,
                "peak_cpu_percent": usage.peak_cpu_percent,
                "mean_cpu_percent": usage.mean_cpu_percent,
                "peak_process_count": usage.peak_process_count,
                "sample_count": usage.sample_count,
                "cpu_sample_count": usage.cpu_sample_count,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return "success" if usage.exit_code == 0 else "failed"

def job_status(job: EG7Job) -> str:
    baseline_rows = read_csv(job.baseline_metrics_path)
    cost_rows = read_csv(job.cost_metrics_path)
    if baseline_rows and cost_rows:
        phases = {str(row.get("phase", "")).strip(): str(row.get("status", "")).strip() for row in cost_rows}
        if phases and all(status in {"success", "cache_hit"} for status in phases.values()):
            return "success"
    process_path = job.metadata_root / "logs" / "worker_process.json"
    if process_path.exists():
        try:
            process_payload = json.loads(process_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            process_payload = {}
        if int(process_payload.get("exit_code", 0) or 0) != 0:
            return "failed"
    if job.run_summary_path.exists():
        try:
            summary = json.loads(job.run_summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "failed"
        if str(summary.get("status", "")).strip() == "failed":
            return "failed"
        if str(summary.get("status", "")).strip() == "dry_run":
            return "pending"
    if job.metadata_root.exists():
        return "partial"
    return "pending"

STATUS_FIELDNAMES = [
    "dataset",
    "featurization",
    "graph_encoder",
    "temporal_encoder",
    "loss",
    "experiment_id",
    "status",
    "config_path",
]

def build_status_rows(jobs: list[EG7Job]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": job.dataset,
            "featurization": job.featurization,
            "graph_encoder": job.graph_encoder,
            "temporal_encoder": job.temporal_encoder,
            "loss": job.loss,
            "experiment_id": job.experiment_id,
            "status": job_status(job),
            "config_path": str(job.config_path),
        }
        for job in jobs
    ]

def write_matrix_report(jobs: list[EG7Job], path: Path | None = None) -> Path:
    path = path or (ORCHESTRATOR_ROOT / "job_matrix.csv")
    rows = [
        {
            "dataset": job.dataset,
            "featurization": job.featurization,
            "graph_encoder": job.graph_encoder,
            "temporal_encoder": job.temporal_encoder,
            "loss": job.loss,
            "experiment_id": job.experiment_id,
            "config_path": str(job.config_path),
        }
        for job in jobs
    ]
    write_csv(
        path,
        rows,
        [
            "dataset",
            "featurization",
            "graph_encoder",
            "temporal_encoder",
            "loss",
            "experiment_id",
            "config_path",
        ],
    )
    return path

def write_status_report(jobs: list[EG7Job], path: Path | None = None) -> Path:
    path = path or (ORCHESTRATOR_ROOT / "job_status.csv")
    write_csv(path, build_status_rows(jobs), STATUS_FIELDNAMES)
    return path

def write_progress_report(
    jobs: list[EG7Job],
    *,
    json_path: Path | None = None,
    md_path: Path | None = None,
    attempted_jobs: int | None = None,
    failed_jobs: int | None = None,
) -> tuple[Path, Path]:
    rows = build_status_rows(jobs)
    counts = {
        status: sum(1 for row in rows if row["status"] == status)
        for status in ("pending", "partial", "failed", "success")
    }
    total = len(rows)
    completed = counts["success"] + counts["failed"]
    completion_ratio = completed / total if total else 0.0

    per_dataset = []
    for dataset in DATASETS:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        if not dataset_rows:
            continue
        per_dataset.append(
            {
                "dataset": dataset,
                "total": len(dataset_rows),
                "success": sum(1 for row in dataset_rows if row["status"] == "success"),
                "failed": sum(1 for row in dataset_rows if row["status"] == "failed"),
                "partial": sum(1 for row in dataset_rows if row["status"] == "partial"),
                "pending": sum(1 for row in dataset_rows if row["status"] == "pending"),
            }
        )

    payload = {
        "total_jobs": total,
        "completed_jobs": completed,
        "success_jobs": counts["success"],
        "failed_jobs": counts["failed"],
        "partial_jobs": counts["partial"],
        "pending_jobs": counts["pending"],
        "completion_ratio": completion_ratio,
        "attempted_jobs": attempted_jobs,
        "run_failures": failed_jobs,
        "per_dataset": per_dataset,
    }

    json_path = json_path or PROGRESS_JSON_PATH
    md_path = md_path or PROGRESS_MD_PATH
    ensure_dir(json_path.parent)
    ensure_dir(md_path.parent)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# EG7 Progress",
        "",
        f"- total_jobs: {total}",
        f"- completed: {completed}",
        f"- success: {counts['success']}",
        f"- failed: {counts['failed']}",
        f"- partial: {counts['partial']}",
        f"- pending: {counts['pending']}",
        f"- completion_ratio: {completion_ratio:.1%}",
    ]
    if attempted_jobs is not None:
        lines.append(f"- attempted_this_run: {attempted_jobs}")
    if failed_jobs is not None:
        lines.append(f"- failures_this_run: {failed_jobs}")
    lines.extend(["", "## Per Dataset"])
    for row in per_dataset:
        lines.append(
            f"- {row['dataset']}: {row['success']}/{row['total']} success, "
            f"failed={row['failed']}, partial={row['partial']}, pending={row['pending']}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path

RAW_FIELDNAMES = [
    "dataset",
    "featurization",
    "graph_encoder",
    "temporal_encoder",
    "loss",
    "experiment_id",
    "ap",
    "auc",
    "cdr",
    "cdr_auc",
    "mcc",
    "threshold",
    "precision",
    "recall",
    "f1",
    "tpr",
    "fpr",
    "campaign_count",
    "detected_campaigns",
    "cdr_meaningful",
    "train_time",
    "test_time",
    "preprocess_wall_time_sec",
    "representation_wall_time_sec",
    "detection_wall_time_sec",
    "total_wall_time_sec",
    "preprocess_peak_rss_kb",
    "representation_peak_rss_kb",
    "detection_peak_rss_kb",
    "run_peak_rss_kb",
    "run_mean_rss_kb",
    "run_mean_cpu_percent",
    "metrics_path",
]

def _to_float(value: Any) -> float:
    if value in ("", None):
        return 0.0
    return float(value)

def _optional_float(value: Any) -> float | str:
    if value in ("", None):
        return ""
    return float(value)

def summarize_jobs(jobs: list[EG7Job], output_root: Path | None = None) -> tuple[Path, Path]:
    output_root = output_root or REPORT_ROOT
    ensure_dir(output_root)

    raw_rows: list[dict[str, Any]] = []
    for job in jobs:
        if job_status(job) != "success":
            continue
        metric_rows = read_csv(job.baseline_metrics_path)
        if not metric_rows:
            continue
        cost_rows = read_csv(job.cost_metrics_path)
        cost_by_phase = {
            str(row.get("phase", "")).strip(): row
            for row in cost_rows
            if str(row.get("phase", "")).strip()
        }
        total_wall = sum(_to_float(row.get("wall_time_sec")) for row in cost_rows)
        peak_values = [
            _optional_float(row.get("peak_rss_kb"))
            for row in cost_rows
            if _optional_float(row.get("peak_rss_kb")) != ""
        ]
        mean_rss_rows = [
            _optional_float(row.get("mean_rss_kb"))
            for row in cost_rows
            if _optional_float(row.get("mean_rss_kb")) != ""
        ]
        mean_cpu_rows = [
            _optional_float(row.get("mean_cpu_percent"))
            for row in cost_rows
            if _optional_float(row.get("mean_cpu_percent")) != ""
        ]
        for row in metric_rows:
            raw_rows.append(
                {
                    "dataset": job.dataset,
                    "featurization": job.featurization,
                    "graph_encoder": job.graph_encoder,
                    "temporal_encoder": job.temporal_encoder,
                    "loss": job.loss,
                    "experiment_id": job.experiment_id,
                    "ap": _to_float(row.get("ap")),
                    "auc": _to_float(row.get("auc")),
                    "cdr": _to_float(row.get("cdr")),
                    "cdr_auc": _to_float(row.get("cdr_auc")),
                    "mcc": _to_float(row.get("mcc")),
                    "threshold": _to_float(row.get("threshold")),
                    "precision": _to_float(row.get("precision")),
                    "recall": _to_float(row.get("recall")),
                    "f1": _to_float(row.get("f1")),
                    "tpr": _to_float(row.get("tpr")),
                    "fpr": _to_float(row.get("fpr")),
                    "campaign_count": int(float(row.get("campaign_count", 0) or 0)),
                    "detected_campaigns": int(float(row.get("detected_campaigns", 0) or 0)),
                    "cdr_meaningful": str(row.get("cdr_meaningful", "")),
                    "train_time": _to_float(row.get("train_time")),
                    "test_time": _to_float(row.get("test_time")),
                    "preprocess_wall_time_sec": _to_float(cost_by_phase.get("preprocess", {}).get("wall_time_sec")),
                    "representation_wall_time_sec": _to_float(cost_by_phase.get("representation", {}).get("wall_time_sec")),
                    "detection_wall_time_sec": _to_float(cost_by_phase.get("detection", {}).get("wall_time_sec")),
                    "total_wall_time_sec": total_wall,
                    "preprocess_peak_rss_kb": _optional_float(cost_by_phase.get("preprocess", {}).get("peak_rss_kb")),
                    "representation_peak_rss_kb": _optional_float(cost_by_phase.get("representation", {}).get("peak_rss_kb")),
                    "detection_peak_rss_kb": _optional_float(cost_by_phase.get("detection", {}).get("peak_rss_kb")),
                    "run_peak_rss_kb": max(peak_values) if peak_values else "",
                    "run_mean_rss_kb": statistics.mean(float(v) for v in mean_rss_rows) if mean_rss_rows else "",
                    "run_mean_cpu_percent": statistics.mean(float(v) for v in mean_cpu_rows) if mean_cpu_rows else "",
                    "metrics_path": str(job.baseline_metrics_path),
                }
            )

    raw_path = output_root / "raw_results.csv"
    write_csv(raw_path, raw_rows, RAW_FIELDNAMES)

    option_rows = _build_option_summary(raw_rows)
    option_path = output_root / "summary_by_option.csv"
    write_csv(
        option_path,
        option_rows,
        [
            "dataset",
            "dimension",
            "option",
            "mean_ap",
            "mean_cdr_auc",
            "mean_mcc",
            "mean_total_wall_time_sec",
            "mean_run_peak_rss_kb",
        ],
    )

    leaderboard_rows = _build_leaderboard_rows(raw_rows)
    leaderboard_path = output_root / "leaderboard.csv"
    write_csv(
        leaderboard_path,
        leaderboard_rows,
        [
            "dataset",
            "rank",
            "experiment_id",
            "featurization",
            "graph_encoder",
            "temporal_encoder",
            "loss",
            "ap",
            "cdr_auc",
            "mcc",
            "total_wall_time_sec",
            "run_peak_rss_kb",
        ],
    )

    for dataset in DATASETS:
        dataset_rows = [row for row in raw_rows if row["dataset"] == dataset]
        for metric in HEATMAP_METRICS:
            write_heatmap_svg(
                dataset_rows,
                metric_key=metric,
                title=f"EG7 {dataset.upper()} {metric.upper()} Heatmap",
                output_path=output_root / f"plot_{dataset}_{metric}_heatmap.svg",
            )

    write_conclusion_md(raw_rows, output_root / "conclusion.md")
    return raw_path, option_path

def _build_option_summary(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        for dimension in ("featurization", "graph_encoder", "temporal_encoder", "loss"):
            key = (str(row["dataset"]), dimension, str(row[dimension]))
            grouped.setdefault(key, []).append(row)

    rows: list[dict[str, Any]] = []
    for (dataset, dimension, option), items in sorted(grouped.items()):
        peak_rss_values = [float(item["run_peak_rss_kb"]) for item in items if item["run_peak_rss_kb"] != ""]
        rows.append(
            {
                "dataset": dataset,
                "dimension": dimension,
                "option": option,
                "mean_ap": statistics.mean(float(item["ap"]) for item in items),
                "mean_cdr_auc": statistics.mean(float(item["cdr_auc"]) for item in items),
                "mean_mcc": statistics.mean(float(item["mcc"]) for item in items),
                "mean_total_wall_time_sec": statistics.mean(float(item["total_wall_time_sec"]) for item in items),
                "mean_run_peak_rss_kb": statistics.mean(peak_rss_values) if peak_rss_values else "",
            }
        )
    return rows

def _build_leaderboard_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in raw_rows:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)
    for dataset, items in sorted(by_dataset.items()):
        ranked = sorted(
            items,
            key=lambda row: (
                float(row["ap"]),
                float(row["cdr_auc"]),
                float(row["mcc"]),
                -float(row["total_wall_time_sec"]),
            ),
            reverse=True,
        )
        for index, row in enumerate(ranked, start=1):
            rows.append(
                {
                    "dataset": dataset,
                    "rank": index,
                    "experiment_id": row["experiment_id"],
                    "featurization": row["featurization"],
                    "graph_encoder": row["graph_encoder"],
                    "temporal_encoder": row["temporal_encoder"],
                    "loss": row["loss"],
                    "ap": row["ap"],
                    "cdr_auc": row["cdr_auc"],
                    "mcc": row["mcc"],
                    "total_wall_time_sec": row["total_wall_time_sec"],
                    "run_peak_rss_kb": row["run_peak_rss_kb"],
                }
            )
    return rows

def write_heatmap_svg(
    raw_rows: list[dict[str, Any]],
    *,
    metric_key: str,
    title: str,
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    if not raw_rows:
        output_path.write_text(_empty_svg(title, "No successful jobs yet."), encoding="utf-8")
        return

    row_labels = [
        f"{feat}|{loss}"
        for feat in FEATURIZATION_OPTIONS
        for loss in LOSS_OPTIONS
    ]
    col_labels = [
        f"{GRAPH_ENCODER_SHORT[graph]}|{TEMPORAL_ENCODER_SHORT[temp]}"
        for graph in GRAPH_ENCODERS
        for temp in TEMPORAL_ENCODERS
    ]
    lookup = {
        (
            str(row["featurization"]),
            str(row["loss"]),
            str(row["graph_encoder"]),
            str(row["temporal_encoder"]),
        ): float(row[metric_key])
        for row in raw_rows
    }
    width = 220 + 78 * len(col_labels)
    height = 110 + 30 * len(row_labels)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:monospace;font-size:11px}.title{font-size:16px;font-weight:bold}.cell{font-size:10px}</style>',
        f'<text class="title" x="20" y="24">{title}</text>',
    ]
    for index, label in enumerate(col_labels):
        x = 210 + index * 78
        parts.append(f'<text x="{x}" y="54" transform="rotate(-45 {x},{54})">{label}</text>')
    for row_index, row_label in enumerate(row_labels):
        y = 78 + row_index * 30
        parts.append(f'<text x="20" y="{y + 18}">{row_label}</text>')
        feat, loss = row_label.split("|", 1)
        for col_index, col_label in enumerate(col_labels):
            graph_short, temp_short = col_label.split("|", 1)
            graph = next(name for name, short in GRAPH_ENCODER_SHORT.items() if short == graph_short)
            temporal = next(name for name, short in TEMPORAL_ENCODER_SHORT.items() if short == temp_short)
            value = lookup.get((feat, loss, graph, temporal))
            if value is None:
                normalized = 0.0
            elif metric_key == "mcc":
                normalized = max(0.0, min(1.0, (float(value) + 1.0) / 2.0))
            else:
                normalized = max(0.0, min(1.0, float(value)))
            shade = 255 - int(normalized * 155)
            fill = f"rgb({shade},255,{shade})"
            x = 180 + col_index * 78
            parts.append(f'<rect x="{x}" y="{y}" width="70" height="24" fill="{fill}" stroke="#999"/>')
            label = "n/a" if value is None else f"{value:.3f}"
            parts.append(f'<text class="cell" x="{x + 10}" y="{y + 16}">{label}</text>')
    parts.append("</svg>")
    output_path.write_text("".join(parts), encoding="utf-8")

def write_conclusion_md(raw_rows: list[dict[str, Any]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    if not raw_rows:
        output_path.write_text("# EG7 Conclusion\n\nNo successful EG7 jobs yet.\n", encoding="utf-8")
        return

    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in raw_rows:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)

    lines = [
        "# EG7 Conclusion",
        "",
        "EG7 fixes graph construction and detection, then ablates featurization,",
        "graph encoder, temporal encoder, and scoring loss.",
        "",
        "## Dataset Winners",
        "",
        "| Dataset | Best AP | Best CDR-AUC | Best MCC |",
        "|---------|---------|--------------|----------|",
    ]
    for dataset, rows in sorted(by_dataset.items()):
        best_ap = max(rows, key=lambda row: float(row["ap"]))
        best_cdr = max(rows, key=lambda row: float(row["cdr_auc"]))
        best_mcc = max(rows, key=lambda row: float(row["mcc"]))
        lines.append(
            "| "
            f"{dataset} | "
            f"{best_ap['experiment_id']} ({float(best_ap['ap']):.3f}) | "
            f"{best_cdr['experiment_id']} ({float(best_cdr['cdr_auc']):.3f}) | "
            f"{best_mcc['experiment_id']} ({float(best_mcc['mcc']):.3f}) |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _empty_svg(title: str, message: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="160" viewBox="0 0 800 160">'
        '<style>text{font-family:monospace;font-size:14px}.title{font-size:18px;font-weight:bold}</style>'
        f'<text class="title" x="20" y="30">{title}</text>'
        f'<text x="20" y="70">{message}</text>'
        "</svg>"
    )

def parse_csv_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]
