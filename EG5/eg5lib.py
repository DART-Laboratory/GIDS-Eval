from __future__ import annotations

import csv
import json
import os
import pickle
import statistics
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from gids_eval.paths import get_paths

_PATHS = get_paths()
REPO_ROOT = _PATHS.repo_root
EG5_CODE_ROOT = REPO_ROOT / "EG5"
EG1_DATA_ROOT = _PATHS.sc_data_root("eg1")
DATA_ROOT = _PATHS.sc_data_root("eg5")
GENERATED_CONFIG_ROOT = DATA_ROOT / "generated-configs"
ORCHESTRATOR_ROOT = DATA_ROOT / "orchestrator"
REPORT_ROOT = DATA_ROOT / "reports"
RUN_LABEL = "eg5"
DEFAULT_SEEDS = (0,)
SCALES = (25, 50, 75, 100)
ANOMAL_DATASETS = ("cic", "0501")
PROGRESS_JSON_PATH = ORCHESTRATOR_ROOT / "progress.json"
PROGRESS_MD_PATH = ORCHESTRATOR_ROOT / "progress.md"
ANCHOR_INPUT_STATS_PATH = ORCHESTRATOR_ROOT / "anchor_input_stats.json"
EG1_JOB_MATRIX_PATH = EG1_DATA_ROOT / "orchestrator" / "job_matrix.csv"
EG1_JOB_STATUS_PATH = EG1_DATA_ROOT / "orchestrator" / "job_status.csv"
EG1_RAW_RESULTS_PATH = EG1_DATA_ROOT / "reports" / "raw_results.csv"
CONDA_ROOT = _PATHS.conda_root
ANOMAL_ENV = CONDA_ROOT / "envs/sc1-anomal" / "bin/python"

DATASET_ALIASES = {
    "0501": "0501",
    "cic": "cic",
    "lanl": "lanl",
    "optc": "optc",
}

EG5_DEFAULT_WINDOW_SECONDS = {
    "argus": {
        "lanl": 3600,
        "optc": 360,
        "cic": 900,
        "0501": 150,
    },
    "euler": {
        "lanl": 1800,
        "optc": 9000,
        "cic": 3000,
        "0501": 150,
    },
    "vgrnn": {
        "lanl": 5400,
        "optc": 1800,
        "cic": 600,
        "0501": 150,
    },
    "pikachu": {
        "lanl": 3600,
        "optc": 3600,
        "cic": 3000,
        "0501": 300,
    },
}

@dataclass(frozen=True)
class EG5Job:
    job_kind: str
    detector: str
    dataset: str
    scale_percent: int
    seed: int
    window_seconds: int | None = None
    window_label: str | None = None
    source_experiment_id: str = ""
    source_config_path: str = ""

    @property
    def is_anchor(self) -> bool:
        return self.job_kind == "eg1" and int(self.scale_percent) == 100

    @property
    def experiment_id(self) -> str:
        if self.job_kind == "anomal":
            return f"eg5_anomal_{self.dataset}_p{self.scale_percent}_s{self.seed:02d}"
        return f"eg5_{self.detector}_{self.dataset}_{self.window_label}_p{self.scale_percent}_s{self.seed:02d}"

    @property
    def config_path(self) -> Path:
        return GENERATED_CONFIG_ROOT / f"{self.experiment_id}.yaml"

    @property
    def workspace_root(self) -> Path:
        return DATA_ROOT / "experiments" / self.experiment_id / RUN_LABEL

    @property
    def metadata_root(self) -> Path:
        return self.workspace_root / "metadata"

    @property
    def baseline_metrics_path(self) -> Path:
        return self.metadata_root / "baseline_metrics.csv"

    @property
    def cost_metrics_path(self) -> Path:
        return self.metadata_root / "cost_metrics.csv"

    @property
    def run_summary_path(self) -> Path:
        return self.metadata_root / "run_summary.json"

    @property
    def scale_metadata_path(self) -> Path:
        if self.job_kind == "anomal":
            return self.scaled_input_root / "metadata.json"
        if self.detector == "pikachu":
            return self.scaled_input_root / "metadata.json"
        return self.scaled_input_root / "eg5_scale_metadata.json"

    @property
    def scaled_input_root(self) -> Path:
        base = self.workspace_root / "output" / "modules" / "featurization" / "output" / "data_root"
        if self.job_kind == "anomal":
            return base / "anomal" / self.dataset
        if self.detector == "pikachu":
            return base / "pikachu_sc1" / self.dataset
        if self.dataset == "0501":
            return base / self.detector / "0501" / "0501"
        return base / self.detector / self.dataset

    @property
    def source_featurization_root(self) -> Path:
        if self.job_kind != "eg1":
            raise ValueError("Anomal-E jobs do not have EG1 source featurization roots")
        base = EG1_DATA_ROOT / "experiments" / self.source_experiment_id / "eg1" / "output" / "modules" / "featurization" / "output" / "data_root"
        if self.detector == "pikachu":
            return base / "pikachu_sc1" / self.dataset
        if self.dataset == "0501":
            return base / self.detector / "0501" / "0501"
        return base / self.detector / self.dataset

    @property
    def source_eg1_results_row(self) -> dict[str, Any] | None:
        return _eg1_results_by_experiment().get(self.source_experiment_id)

def ensure_eg5_dirs() -> None:
    for path in (GENERATED_CONFIG_ROOT, ORCHESTRATOR_ROOT, REPORT_ROOT):
        path.mkdir(parents=True, exist_ok=True)

def detector_list() -> list[str]:
    return ["argus", "euler", "vgrnn", "pikachu", "anomal"]

def dataset_list() -> list[str]:
    return ["0501", "cic", "lanl", "optc"]

def scale_list() -> list[int]:
    return list(SCALES)

def anomal_python_bin() -> str:
    return str(ANOMAL_ENV) if ANOMAL_ENV.exists() else sys.executable

def canonical_dataset(value: str) -> str:
    lowered = value.strip().lower()
    return DATASET_ALIASES.get(lowered, lowered)

def parse_csv_list(value: str | None, *, kind: str) -> list[Any] | None:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if kind == "dataset":
        items = [canonical_dataset(item) for item in items]
    elif kind == "scale":
        items = [int(item) for item in items]
    return list(dict.fromkeys(items))

def parse_seed_list(value: str | None) -> list[int]:
    if not value:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in value.split(",") if item.strip()]

def iter_jobs(
    detectors: Iterable[str] | None = None,
    datasets: Iterable[str] | None = None,
    scales: Iterable[int] | None = None,
    seeds: Iterable[int] | None = None,
) -> list[EG5Job]:
    selected_detectors = list(detectors or detector_list())
    selected_datasets = list(datasets or dataset_list())
    selected_scales = list(scales or SCALES)
    selected_seeds = list(seeds or DEFAULT_SEEDS)

    jobs: list[EG5Job] = []
    for row in _eg1_matrix_rows():
        detector = str(row["detector"])
        dataset = str(row["dataset"])
        if detector not in selected_detectors or dataset not in selected_datasets:
            continue
        default_window_seconds = EG5_DEFAULT_WINDOW_SECONDS.get(detector, {}).get(dataset)
        if default_window_seconds is None:
            continue
        if int(row["window_seconds"]) != int(default_window_seconds):
            continue
        for scale in selected_scales:
            jobs.append(
                EG5Job(
                    job_kind="eg1",
                    detector=detector,
                    dataset=dataset,
                    scale_percent=int(scale),
                    seed=int(row["seed"]),
                    window_seconds=int(row["window_seconds"]),
                    window_label=str(row["window_label"]),
                    source_experiment_id=str(row["experiment_id"]),
                    source_config_path=str(row["config_path"]),
                )
            )

    if "anomal" in selected_detectors:
        for dataset in ANOMAL_DATASETS:
            if dataset not in selected_datasets:
                continue
            for scale in selected_scales:
                for seed in selected_seeds:
                    jobs.append(
                        EG5Job(
                            job_kind="anomal",
                            detector="anomal",
                            dataset=dataset,
                            scale_percent=int(scale),
                            seed=int(seed),
                            window_seconds=None,
                            window_label="na",
                        )
                    )
    return jobs

def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in YAML file: {path}")
    return payload

def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]

def generate_config(job: EG5Job) -> Path:
    ensure_eg5_dirs()
    if job.is_anchor:
        return job.config_path
    if job.job_kind == "anomal":
        payload = _build_anomal_config(job)
        dump_yaml(job.config_path, payload)
        return job.config_path

    payload = load_yaml(Path(job.source_config_path))
    generated = deepcopy(payload)
    generated["name"] = f"EG5 {job.detector.upper()} {job.dataset} {job.window_label} {job.scale_percent}% seed {job.seed}"
    generated["experiment_id"] = job.experiment_id
    generated["run_label"] = RUN_LABEL
    generated.setdefault("runtime", {})
    generated["runtime"]["data_root"] = str(DATA_ROOT)
    featurization = generated["modules"]["featurization"]
    featurization["runner"] = "shell_command"
    featurization["workdir"] = "${repo_root}"
    featurization["env"] = {}

    if job.detector == "pikachu":
        featurization["command"] = [
            "${python_bin}",
            "${repo_root}/EG5/scale_eg1_input.py",
            "--kind",
            "pikachu",
            "--detector",
            job.detector,
            "--dataset",
            job.dataset,
            "--scale-percent",
            str(job.scale_percent),
            "--source-metadata",
            str(job.source_featurization_root / "metadata.json"),
            "--output-root",
            str(job.scaled_input_root),
            "--source-experiment-id",
            job.source_experiment_id,
        ]
    else:
        featurization["command"] = [
            "${python_bin}",
            "${repo_root}/EG5/scale_eg1_input.py",
            "--kind",
            "graph",
            "--detector",
            job.detector,
            "--dataset",
            job.dataset,
            "--scale-percent",
            str(job.scale_percent),
            "--source-root",
            str(job.source_featurization_root),
            "--output-root",
            str(job.scaled_input_root),
            "--source-experiment-id",
            job.source_experiment_id,
        ]

    dump_yaml(job.config_path, generated)
    return job.config_path

def _build_anomal_config(job: EG5Job) -> dict[str, Any]:
    manifests = (
        [str(REPO_ROOT / "configs/data/cic.yaml")]
        if job.dataset == "cic"
        else [
            str(REPO_ROOT / "configs/data/0501_oilrig.yaml"),
            str(REPO_ROOT / "configs/data/0501_sandworm.yaml"),
            str(REPO_ROOT / "configs/data/0501_wizard_spider.yaml"),
        ]
    )
    results_rel = f"results/{job.dataset}"
    data_rel = f"anomal/{job.dataset}"
    return {
        "schema_version": 2,
        "kind": "experiment",
        "name": f"EG5 ANOMAL {job.dataset} {job.scale_percent}% seed {job.seed}",
        "experiment_id": job.experiment_id,
        "run_label": RUN_LABEL,
        "seed": job.seed,
        "python_bin": anomal_python_bin(),
        "stage_mode": "symlink",
        "runtime": {
            "repo_root": str(REPO_ROOT),
            "data_root": str(DATA_ROOT),
            "runtime_root": str(REPO_ROOT / "local_runtime"),
        },
        "manifests": manifests,
        "modules": {
            "graph_construction": {
                "runner": "noop",
                "inputs": {"data_root/raw_data": "${input_root}/raw_data"},
                "artifacts": {"data_root": "data_root"},
            },
            "featurization": {
                "runner": "shell_command",
                "inputs": {"data_root/raw_data": "${graph_construction_artifact_data_root}/raw_data"},
                "workdir": "${repo_root}",
                "env": {"EG5_ANOMAL_FEATURIZATION_SCHEMA": "v2"},
                "command": [
                    "${python_bin}",
                    "${repo_root}/EG5/build_anomal_scaled_input.py",
                    "--dataset",
                    job.dataset,
                    "--raw-root",
                    "${module_output_dir}/data_root/raw_data",
                    "--output-dir",
                    f"${{module_output_dir}}/data_root/{data_rel}",
                    "--scale-percent",
                    str(job.scale_percent),
                ],
                "artifacts": {"data_root": "data_root"},
            },
            "graph_encoder": {
                "runner": "noop",
                "inputs": {"data_root": "${featurization_artifact_data_root}"},
                "artifacts": {"data_root": "data_root"},
            },
            "temporal_encoder": {
                "runner": "noop",
                "inputs": {"data_root": "${graph_encoder_artifact_data_root}"},
                "artifacts": {"data_root": "data_root"},
            },
            "loss": {
                "runner": "noop",
                "inputs": {"data_root": "${temporal_encoder_artifact_data_root}"},
                "artifacts": {"data_root": "data_root"},
            },
            "detection": {
                "runner": "command_group",
                "inputs": {"data_root": "${loss_artifact_data_root}"},
                "artifacts": {"data_root": "data_root", "results_root": "results"},
                "jobs": [
                    {
                        "id": job.dataset,
                        "dataset": job.dataset,
                        "workdir": "${repo_root}",
                        "command": [
                            "${python_bin}",
                            "${repo_root}/EG5/anomal_run.py",
                            "--dataset",
                            job.dataset,
                            "--metadata",
                            f"${{module_output_dir}}/data_root/{data_rel}/metadata.json",
                            "--detector-root",
                            "${runtime_root}/detection_baselines/anomal",
                            "--output-dir",
                            f"${{module_output_dir}}/{results_rel}",
                            "--metrics-csv",
                            f"${{module_output_dir}}/{results_rel}/metrics.csv",
                        ],
                        "metrics_relpath": f"{results_rel}/metrics.csv",
                    }
                ],
            },
        },
    }

def run_job(job: EG5Job, *, dry_run: bool = False, force: bool = False) -> str:
    if job.is_anchor:
        return "skipped"
    config_path = generate_config(job)
    if not force:
        status = job_status(job)
        if status in {"success"}:
            return "skipped"
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH", "")
    parts = [str(EG5_CODE_ROOT.resolve()), str((REPO_ROOT / "src").resolve())]
    if pythonpath:
        parts.append(pythonpath)
    env["PYTHONPATH"] = ":".join(parts)
    command = [sys.executable, "-m", "gids_eval", "run"]
    if dry_run:
        command.append("--dry-run")
    command.append(str(config_path))
    completed = subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=False)
    if completed.returncode != 0:
        return "failed"
    return job_status(job)

def job_status(job: EG5Job) -> str:
    if job.is_anchor:
        return "success" if _eg1_status_by_experiment().get(job.source_experiment_id, "") == "success" else "failed"
    cost_rows = read_csv(job.cost_metrics_path)
    baseline_rows = read_csv(job.baseline_metrics_path)
    if cost_rows and baseline_rows:
        detection_rows = [row for row in cost_rows if row.get("phase") == "detection"]
        if detection_rows and all(row.get("status") in {"success", "cache_hit"} for row in detection_rows):
            return "success"
    if job.run_summary_path.exists():
        try:
            with job.run_summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
        except json.JSONDecodeError:
            return "failed"
        records = summary.get("records", [])
        if any(record.get("status") == "failed" for record in records):
            return "failed"
    stage_records = _stage_status_records(job)
    if any(record.get("status") == "failed" for record in stage_records):
        return "failed"
    if any(any(subjob.get("status") == "failed" for subjob in record.get("subjobs", [])) for record in stage_records):
        return "failed"
    if job.metadata_root.exists():
        if not _job_process_active(job):
            return "failed"
        return "partial"
    return "pending"

def _stage_status_records(job: EG5Job) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for stage in ("graph_construction", "featurization", "graph_encoder", "temporal_encoder", "loss", "detection"):
        path = job.metadata_root / f"{stage}.json"
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records

def _active_process_snapshot() -> str:
    completed = subprocess.run(
        ["ps", "-ef"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout if completed.returncode == 0 else ""

def _job_process_active(job: EG5Job) -> bool:
    snapshot = _active_process_snapshot()
    if not snapshot:
        return False
    needles = (
        job.experiment_id,
        str(job.config_path),
        str(job.workspace_root),
    )
    return any(needle in snapshot for needle in needles)

def build_status_rows(jobs: list[EG5Job]) -> list[dict[str, Any]]:
    rows = []
    for job in jobs:
        rows.append(
            {
                "detector": job.detector,
                "dataset": job.dataset,
                "window_seconds": job.window_seconds or "",
                "window_label": job.window_label or "",
                "scale_percent": job.scale_percent,
                "seed": job.seed,
                "job_kind": job.job_kind,
                "experiment_id": job.experiment_id,
                "source_experiment_id": job.source_experiment_id,
                "status": job_status(job),
                "config_path": str(job.config_path) if not job.is_anchor else "",
                "baseline_metrics_path": str(job.baseline_metrics_path) if not job.is_anchor else str(job.source_eg1_results_row.get("metrics_path", "") if job.source_eg1_results_row else ""),
                "cost_metrics_path": str(job.cost_metrics_path) if not job.is_anchor else "",
                "run_summary_path": str(job.run_summary_path) if not job.is_anchor else "",
            }
        )
    return rows

def write_matrix_report(jobs: list[EG5Job], path: Path | None = None) -> Path:
    rows = []
    for job in jobs:
        rows.append(
            {
                "detector": job.detector,
                "dataset": job.dataset,
                "window_seconds": job.window_seconds or "",
                "window_label": job.window_label or "",
                "scale_percent": job.scale_percent,
                "seed": job.seed,
                "job_kind": job.job_kind,
                "experiment_id": job.experiment_id,
                "source_experiment_id": job.source_experiment_id,
                "config_path": str(job.config_path) if not job.is_anchor else "",
            }
        )
    path = path or (ORCHESTRATOR_ROOT / "job_matrix.csv")
    write_csv(
        path,
        rows,
        [
            "detector",
            "dataset",
            "window_seconds",
            "window_label",
            "scale_percent",
            "seed",
            "job_kind",
            "experiment_id",
            "source_experiment_id",
            "config_path",
        ],
    )
    return path

def write_status_report(jobs: list[EG5Job], path: Path | None = None) -> Path:
    rows = build_status_rows(jobs)
    path = path or (ORCHESTRATOR_ROOT / "job_status.csv")
    write_csv(
        path,
        rows,
        [
            "detector",
            "dataset",
            "window_seconds",
            "window_label",
            "scale_percent",
            "seed",
            "job_kind",
            "experiment_id",
            "source_experiment_id",
            "status",
            "config_path",
            "baseline_metrics_path",
            "cost_metrics_path",
            "run_summary_path",
        ],
    )
    return path

def write_progress_report(
    jobs: list[EG5Job],
    *,
    attempted_jobs: int | None = None,
    failed_jobs: int | None = None,
) -> tuple[Path, Path]:
    rows = build_status_rows(jobs)
    total = len(rows)
    counts = {status: sum(1 for row in rows if row["status"] == status) for status in ("pending", "partial", "failed", "success")}
    completed = counts["success"] + counts["failed"]
    payload = {
        "total_jobs": total,
        "completed_jobs": completed,
        "remaining_jobs": total - completed,
        "success_jobs": counts["success"],
        "failed_jobs": counts["failed"],
        "partial_jobs": counts["partial"],
        "pending_jobs": counts["pending"],
        "completion_ratio": (completed / total) if total else 0.0,
        "attempted_jobs": attempted_jobs,
        "run_failures": failed_jobs,
    }
    PROGRESS_JSON_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# EG5 Progress",
        "",
        f"- total_jobs: {total}",
        f"- completed_jobs: {completed}",
        f"- remaining_jobs: {total - completed}",
        f"- success_jobs: {counts['success']}",
        f"- failed_jobs: {counts['failed']}",
        f"- partial_jobs: {counts['partial']}",
        f"- pending_jobs: {counts['pending']}",
        f"- completion_ratio: {(completed / total) if total else 0.0:.1%}",
    ]
    if attempted_jobs is not None:
        lines.append(f"- attempted_jobs_in_current_run: {attempted_jobs}")
    if failed_jobs is not None:
        lines.append(f"- failures_in_current_run: {failed_jobs}")
    PROGRESS_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return PROGRESS_JSON_PATH, PROGRESS_MD_PATH

def summarize_jobs(jobs: list[EG5Job]) -> tuple[Path, Path]:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, Any]] = []
    anchor_stats = _anchor_input_stats(jobs)
    for job in jobs:
        if job.is_anchor:
            source_row = job.source_eg1_results_row
            if not source_row:
                continue
            scale_stats = anchor_stats.get(job.source_experiment_id, {})
            raw_rows.append(_build_anchor_result_row(job, source_row, scale_stats))
            continue
        if job_status(job) != "success":
            continue
        cost_rows = read_csv(job.cost_metrics_path)
        phase_rows = {str(row.get("phase", "")): row for row in cost_rows if row.get("phase")}
        detection_row = phase_rows.get("detection", {})
        featurization_row = phase_rows.get("featurization", {})
        total_wall_time_sec = sum(_to_float(row.get("wall_time_sec")) for row in cost_rows)
        run_peak_rss_kb = max((_to_float(row.get("peak_rss_kb")) for row in cost_rows), default=0.0)
        scale_stats = _read_scale_metadata(job.scale_metadata_path)
        for metrics_row in read_csv(job.baseline_metrics_path):
            raw_rows.append(
                {
                    "source_kind": job.job_kind,
                    "detector": job.detector,
                    "dataset": job.dataset,
                    "window_seconds": job.window_seconds or "",
                    "window_label": job.window_label or "",
                    "scale_percent": job.scale_percent,
                    "seed": job.seed,
                    "experiment_id": job.experiment_id,
                    "source_experiment_id": job.source_experiment_id,
                    "status": "success",
                    "auc": _to_float(metrics_row.get("auc")),
                    "ap": _to_float(metrics_row.get("ap")),
                    "cdr": _optional_float(metrics_row.get("cdr")),
                    "campaign_count": _optional_float(metrics_row.get("campaign_count")),
                    "detected_campaigns": _optional_float(metrics_row.get("detected_campaigns")),
                    "cdr_meaningful": _to_bool(metrics_row.get("cdr_meaningful")),
                    "train_time": _optional_float(metrics_row.get("train_time")),
                    "test_time": _optional_float(metrics_row.get("test_time")),
                    "featurization_wall_time_sec": _optional_float(featurization_row.get("wall_time_sec")),
                    "detection_wall_time_sec": _optional_float(detection_row.get("wall_time_sec")),
                    "total_wall_time_sec": total_wall_time_sec,
                    "run_peak_rss_gib": _optional_kb_to_gib(run_peak_rss_kb),
                    "input_rows": _optional_float(scale_stats.get("row_count")),
                    "positive_rows": _optional_float(scale_stats.get("positive_rows")),
                    "node_count": _optional_float(scale_stats.get("node_count")),
                    "input_unit_count": _optional_float(
                        scale_stats.get("snapshot_count", scale_stats.get("file_count"))
                    ),
                    "kept_pre_units": _optional_float(
                        scale_stats.get("kept_pre_snapshot_count", scale_stats.get("kept_pre_file_count", scale_stats.get("kept_pre_rows")))
                    ),
                    "kept_post_units": _optional_float(
                        scale_stats.get("kept_post_snapshot_count", scale_stats.get("kept_post_file_count", scale_stats.get("kept_post_rows")))
                    ),
                    "metrics_path": str(job.baseline_metrics_path),
                    "scale_metadata_path": str(job.scale_metadata_path),
                }
            )

    raw_path = REPORT_ROOT / "raw_results.csv"
    write_csv(raw_path, raw_rows, _raw_fieldnames())

    summary_rows = []
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for row in raw_rows:
        grouped.setdefault(
            (str(row["detector"]), str(row["dataset"]), str(row["window_label"]), int(float(row["scale_percent"]))),
            [],
        ).append(row)
    for (detector, dataset, window_label, scale_percent), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "detector": detector,
                "dataset": dataset,
                "window_label": window_label,
                "scale_percent": scale_percent,
                "seed_count": len(rows),
                "mean_total_wall_time_sec": _aggregate_mean(rows, "total_wall_time_sec"),
                "mean_run_peak_rss_gib": _aggregate_mean(rows, "run_peak_rss_gib"),
                "mean_input_rows": _aggregate_mean(rows, "input_rows"),
                "mean_node_count": _aggregate_mean(rows, "node_count"),
            }
        )
    summary_path = REPORT_ROOT / "summary_by_scale.csv"
    write_csv(summary_path, summary_rows, list(summary_rows[0].keys()) if summary_rows else _summary_fieldnames())

    fit_rows = []
    family_grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        family_grouped.setdefault((str(row["detector"]), str(row["dataset"]), str(row["window_label"])), []).append(row)
    for (detector, dataset, window_label), rows in sorted(family_grouped.items()):
        fit_rows.append(_fit_family(detector, dataset, window_label, rows))
    fit_path = REPORT_ROOT / "fit_summary.csv"
    write_csv(fit_path, fit_rows, list(fit_rows[0].keys()) if fit_rows else _fit_fieldnames())
    _write_conclusion(fit_rows, REPORT_ROOT / "conclusion.md")
    return raw_path, summary_path

def _build_anchor_result_row(job: EG5Job, source_row: dict[str, Any], scale_stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_kind": "eg1_anchor",
        "detector": job.detector,
        "dataset": job.dataset,
        "window_seconds": job.window_seconds or "",
        "window_label": job.window_label or "",
        "scale_percent": 100,
        "seed": job.seed,
        "experiment_id": job.experiment_id,
        "source_experiment_id": job.source_experiment_id,
        "status": "success",
        "auc": _to_float(source_row.get("auc")),
        "ap": _to_float(source_row.get("ap")),
        "cdr": _optional_float(source_row.get("cdr")),
        "campaign_count": _optional_float(source_row.get("campaign_count")),
        "detected_campaigns": _optional_float(source_row.get("detected_campaigns")),
        "cdr_meaningful": _to_bool(source_row.get("cdr_meaningful")),
        "train_time": _optional_float(source_row.get("train_time")),
        "test_time": _optional_float(source_row.get("test_time")),
        "featurization_wall_time_sec": _optional_float(source_row.get("featurization_wall_time_sec")),
        "detection_wall_time_sec": _optional_float(source_row.get("detection_wall_time_sec")),
        "total_wall_time_sec": _optional_float(source_row.get("total_wall_time_sec")),
        "run_peak_rss_gib": _optional_float(source_row.get("run_peak_rss_gib")),
        "input_rows": _optional_float(scale_stats.get("row_count")),
        "positive_rows": _optional_float(scale_stats.get("positive_rows")),
        "node_count": _optional_float(scale_stats.get("node_count")),
        "input_unit_count": _optional_float(scale_stats.get("input_unit_count")),
        "kept_pre_units": _optional_float(scale_stats.get("kept_pre_units")),
        "kept_post_units": _optional_float(scale_stats.get("kept_post_units")),
        "metrics_path": str(source_row.get("metrics_path", "")),
        "scale_metadata_path": "",
    }

def _anchor_input_stats(jobs: list[EG5Job] | None = None) -> dict[str, dict[str, Any]]:
    expected_anchor_ids = None
    if jobs is not None:
        expected_anchor_ids = {
            job.source_experiment_id
            for job in jobs
            if job.is_anchor and job.source_experiment_id
        }
    if ANCHOR_INPUT_STATS_PATH.exists():
        with ANCHOR_INPUT_STATS_PATH.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        if expected_anchor_ids is None or expected_anchor_ids.issubset(set(cached.keys())):
            return cached
    tasks: list[tuple[str, str, str]] = []
    for row in _eg1_matrix_rows():
        experiment_id = str(row["experiment_id"])
        if expected_anchor_ids is not None and experiment_id not in expected_anchor_ids:
            continue
        tasks.append((experiment_id, str(row["detector"]), str(row["dataset"])))

    payload: dict[str, dict[str, Any]] = {}
    max_workers = min(len(tasks), max(1, min(4, os.cpu_count() or 1)))
    if max_workers <= 1:
        for task in tasks:
            experiment_id, stats = _compute_anchor_input_stats_task(task)
            payload[experiment_id] = stats
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for experiment_id, stats in executor.map(_compute_anchor_input_stats_task, tasks):
                payload[experiment_id] = stats
    ANCHOR_INPUT_STATS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload

def _compute_anchor_input_stats_task(task: tuple[str, str, str]) -> tuple[str, dict[str, Any]]:
    experiment_id, detector, dataset = task
    source_root = _source_featurization_root(experiment_id, detector, dataset)
    if detector == "pikachu":
        return experiment_id, _inspect_pikachu_input(source_root / "metadata.json")
    return experiment_id, _inspect_graph_input(source_root, dataset)

def _inspect_pikachu_input(metadata_path: Path) -> dict[str, Any]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    input_unit_count = int(payload.get("snapshot_count", 0))
    kept_pre = int(payload.get("trainwin", payload.get("first_attack_snapshot", 0)))
    return {
        "row_count": int(payload.get("row_count", 0)),
        "positive_rows": int(payload.get("positive_rows", 0)),
        "node_count": int(payload.get("node_count", 0)),
        "input_unit_count": input_unit_count,
        "kept_pre_units": kept_pre,
        "kept_post_units": max(0, input_unit_count - kept_pre),
    }

def _inspect_graph_input(root: Path, dataset: str) -> dict[str, Any]:
    numeric_paths = sorted(path for path in root.glob("*.txt") if path.stem.isdigit())
    row_count = 0
    positive_rows = 0
    label_index = {"lanl": 9, "optc": 8, "cic": 6, "0501": 3}[dataset]
    for path in numeric_paths:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                stripped = raw_line.rstrip("\n")
                if not stripped:
                    continue
                parts = stripped.split(",")
                if len(parts) <= label_index:
                    continue
                row_count += 1
                try:
                    if int(float(parts[label_index].strip() or "0")) == 1:
                        positive_rows += 1
                except Exception:
                    pass
    node_count = 0
    nmap_path = root / "nmap.pkl"
    if nmap_path.exists():
        with nmap_path.open("rb") as handle:
            try:
                node_count = len(pickle.load(handle))
            except Exception:
                node_count = 0
    first_positive_ms = 0
    if dataset == "0501":
        meta_path = root / "eg1_0501_metadata.json"
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            first_positive_ms = int(payload.get("first_positive_ms", 0))
    else:
        found_first_positive = False
        for path in numeric_paths:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for raw_line in handle:
                    parts = raw_line.rstrip("\n").split(",")
                    if len(parts) <= label_index:
                        continue
                    try:
                        if int(float(parts[label_index].strip() or "0")) == 1:
                            first_positive_ms = int(float(parts[0]))
                            found_first_positive = True
                            break
                    except Exception:
                        continue
                if found_first_positive:
                    break
    file_delta = 0
    if len(numeric_paths) >= 2:
        file_delta = int(numeric_paths[1].stem) - int(numeric_paths[0].stem)
    kept_pre = 0 if file_delta <= 0 else int(first_positive_ms // file_delta)
    input_units = len(numeric_paths)
    return {
        "row_count": row_count,
        "positive_rows": positive_rows,
        "node_count": node_count,
        "input_unit_count": input_units,
        "kept_pre_units": kept_pre,
        "kept_post_units": max(0, input_units - kept_pre),
    }

def _source_featurization_root(experiment_id: str, detector: str, dataset: str) -> Path:
    base = EG1_DATA_ROOT / "experiments" / experiment_id / "eg1" / "output" / "modules" / "featurization" / "output" / "data_root"
    if detector == "pikachu":
        return base / "pikachu_sc1" / dataset
    if dataset == "0501":
        return base / detector / "0501" / "0501"
    return base / detector / dataset

def _read_scale_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def _fit_family(detector: str, dataset: str, window_label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: float(row["scale_percent"]))
    xs = [float(row["scale_percent"]) for row in ordered]
    time_ys = [float(row["total_wall_time_sec"]) for row in ordered if row.get("total_wall_time_sec") not in ("", None)]
    mem_ys = [float(row["run_peak_rss_gib"]) for row in ordered if row.get("run_peak_rss_gib") not in ("", None)]
    return {
        "detector": detector,
        "dataset": dataset,
        "window_label": window_label,
        "point_count": len(ordered),
        "time_slope": _fit_linear(xs, [float(row["total_wall_time_sec"]) for row in ordered])[0] if len(ordered) >= 2 else "",
        "time_intercept": _fit_linear(xs, [float(row["total_wall_time_sec"]) for row in ordered])[1] if len(ordered) >= 2 else "",
        "time_r2": _fit_linear(xs, [float(row["total_wall_time_sec"]) for row in ordered])[2] if len(ordered) >= 2 else "",
        "mem_slope": _fit_linear(xs, [float(row["run_peak_rss_gib"]) for row in ordered])[0] if len(ordered) >= 2 else "",
        "mem_intercept": _fit_linear(xs, [float(row["run_peak_rss_gib"]) for row in ordered])[1] if len(ordered) >= 2 else "",
        "mem_r2": _fit_linear(xs, [float(row["run_peak_rss_gib"]) for row in ordered])[2] if len(ordered) >= 2 else "",
        "max_scale_percent": max(xs) if xs else "",
        "max_time_sec": max(time_ys) if time_ys else "",
        "max_peak_rss_gib": max(mem_ys) if mem_ys else "",
    }

def _fit_linear(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    n = len(xs)
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    if ss_xx == 0:
        return 0.0, mean_y, 0.0
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x
    predictions = [slope * x + intercept for x in xs]
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - y_hat) ** 2 for y, y_hat in zip(ys, predictions))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot else 1.0
    return slope, intercept, r2

def _write_conclusion(fit_rows: list[dict[str, Any]], path: Path) -> None:
    lines = ["# EG5 Conclusion", ""]
    if not fit_rows:
        lines.append("No completed EG5 results yet.")
    else:
        for row in fit_rows:
            lines.append(
                f"- {row['detector']} {row['dataset']} {row['window_label']}: "
                f"time_r2={row['time_r2']}, mem_r2={row['mem_r2']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _aggregate_mean(rows: list[dict[str, Any]], key: str) -> float | str:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return statistics.mean(values) if values else ""

def _to_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)

def _optional_float(value: Any) -> float | str:
    if value in (None, ""):
        return ""
    return float(value)

def _optional_kb_to_gib(value: Any) -> float | str:
    if value in (None, ""):
        return ""
    return float(value) / (1024.0 * 1024.0)

def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return str(value).strip().lower() == "true"

def _raw_fieldnames() -> list[str]:
    return [
        "source_kind",
        "detector",
        "dataset",
        "window_seconds",
        "window_label",
        "scale_percent",
        "seed",
        "experiment_id",
        "source_experiment_id",
        "status",
        "auc",
        "ap",
        "cdr",
        "campaign_count",
        "detected_campaigns",
        "cdr_meaningful",
        "train_time",
        "test_time",
        "featurization_wall_time_sec",
        "detection_wall_time_sec",
        "total_wall_time_sec",
        "run_peak_rss_gib",
        "input_rows",
        "positive_rows",
        "node_count",
        "input_unit_count",
        "kept_pre_units",
        "kept_post_units",
        "metrics_path",
        "scale_metadata_path",
    ]

def _summary_fieldnames() -> list[str]:
    return ["detector", "dataset", "window_label", "scale_percent", "seed_count", "mean_total_wall_time_sec", "mean_run_peak_rss_gib", "mean_input_rows", "mean_node_count"]

def _fit_fieldnames() -> list[str]:
    return ["detector", "dataset", "window_label", "point_count", "time_slope", "time_intercept", "time_r2", "mem_slope", "mem_intercept", "mem_r2", "max_scale_percent", "max_time_sec", "max_peak_rss_gib"]

_EG1_MATRIX_CACHE: list[dict[str, Any]] | None = None
_EG1_STATUS_CACHE: dict[str, str] | None = None
_EG1_RESULTS_CACHE: dict[str, dict[str, Any]] | None = None

def _eg1_matrix_rows() -> list[dict[str, Any]]:
    global _EG1_MATRIX_CACHE
    if _EG1_MATRIX_CACHE is None:
        _EG1_MATRIX_CACHE = read_csv(EG1_JOB_MATRIX_PATH)
    return _EG1_MATRIX_CACHE

def _eg1_status_by_experiment() -> dict[str, str]:
    global _EG1_STATUS_CACHE
    if _EG1_STATUS_CACHE is None:
        _EG1_STATUS_CACHE = {str(row["experiment_id"]): str(row["status"]) for row in read_csv(EG1_JOB_STATUS_PATH)}
    return _EG1_STATUS_CACHE

def _eg1_results_by_experiment() -> dict[str, dict[str, Any]]:
    global _EG1_RESULTS_CACHE
    if _EG1_RESULTS_CACHE is None:
        _EG1_RESULTS_CACHE = {str(row["experiment_id"]): row for row in read_csv(EG1_RAW_RESULTS_PATH)}
    return _EG1_RESULTS_CACHE
