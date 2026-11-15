from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from gids_eval.paths import get_paths

_PATHS = get_paths()
REPO_ROOT = _PATHS.repo_root
EG4_CODE_ROOT = REPO_ROOT / "EG4"
DATA_ROOT = _PATHS.sc_data_root("eg4")
GENERATED_CONFIG_ROOT = DATA_ROOT / "generated-configs"
ORCHESTRATOR_ROOT = DATA_ROOT / "orchestrator"
REPORT_ROOT = DATA_ROOT / "reports"
RUN_LABEL = "eg4"
DETECTOR = "argus"
DEFAULT_SEEDS = (0,)
CONDA_ROOT = _PATHS.conda_root
DETECTOR_ENV_PREFIX = CONDA_ROOT / "envs/sc1-argus"
BASE_CONFIG = REPO_ROOT / "configs/experiments/argus_pipeline.yaml"
PROGRESS_JSON_PATH = ORCHESTRATOR_ROOT / "progress.json"
PROGRESS_MD_PATH = ORCHESTRATOR_ROOT / "progress.md"

DATASET_ALIASES = {
    "0501": "0501",
    "wizard": "0501",
    "wizard_spider": "0501",
    "oilrig": "0501",
    "sandworm": "0501",
    "cic": "cic",
    "lanl": "lanl",
    "optc": "optc",
}

FLOW_MODE_ALIASES = {
    "with_flows": "with_flows",
    "flow": "with_flows",
    "flows": "with_flows",
    "on": "with_flows",
    "with": "with_flows",
    "without_flows": "without_flows",
    "noflow": "without_flows",
    "no_flows": "without_flows",
    "off": "without_flows",
    "without": "without_flows",
}

DATASETS = ("lanl", "optc", "cic", "0501")
FLOW_MODES = ("without_flows", "with_flows")

MANIFESTS = {
    "lanl": REPO_ROOT / "configs/data/lanl.yaml",
    "optc": REPO_ROOT / "configs/data/optc.yaml",
    "cic": REPO_ROOT / "configs/data/cic.yaml",
    "0501": (
        REPO_ROOT / "configs/data/0501_oilrig.yaml",
        REPO_ROOT / "configs/data/0501_sandworm.yaml",
        REPO_ROOT / "configs/data/0501_wizard_spider.yaml",
    ),
}

JOB_IDS = {
    "lanl": "lanl",
    "optc": "optc",
    "cic": "cic",
    "0501": "oilrig",
}

PREPROCESS_SPECS = {
    "lanl": {
        "workdir": "${runtime_root}/pre_processing/argus",
        "command": [
            "${python_bin}",
            "process_lanl_argus.py",
            "-i",
            "${module_output_dir}/data_root/raw_data/LANL",
            "-o",
            "${module_output_dir}/data_root/argus/lanl",
        ],
    },
    "optc": {
        "workdir": "${runtime_root}/pre_processing/argus",
        "command": [
            "${python_bin}",
            "process_optc_argus.py",
            "-i",
            "${module_output_dir}/data_root/raw_data/OpTC",
            "-o",
            "${module_output_dir}/data_root/argus/optc",
        ],
    },
    "cic": {
        "workdir": "${runtime_root}/pre_processing/argus",
        "command": [
            "${python_bin}",
            "process_CIC_argus.py",
            "-i",
            "${module_output_dir}/data_root/raw_data/CIC-new",
            "-o",
            "${module_output_dir}/data_root/argus/cic",
        ],
    },
    "0501": {
        "workdir": "${repo_root}",
        "command": [
            "${python_bin}",
            "${repo_root}/EG4/build_0501_unified_argus.py",
            "--raw-root",
            "${module_output_dir}/data_root/raw_data",
            "--output-root",
            "${module_output_dir}/data_root/argus/0501/0501",
        ],
    },
}

FEATURE_SCHEMAS = {
    "lanl": ("duration", "packets", "bytes"),
    "optc": ("pid", "ppid", "dest_port"),
    "cic": ("duration", "bytes", "packets"),
    "0501": ("duration", "bytes", "packets"),
}

ARGUS_DATASET_ARGS = {
    "lanl": "LANL",
    "optc": "OPTC",
    "cic": "O_cic",
    "0501": "O_0501",
}

@dataclass(frozen=True)
class EG4Job:
    dataset: str
    flow_mode: str
    seed: int

    @property
    def edge_features_enabled(self) -> bool:
        return self.flow_mode == "with_flows"

    @property
    def mode_token(self) -> str:
        return "flow" if self.edge_features_enabled else "noflow"

    @property
    def experiment_id(self) -> str:
        return f"eg4_argus_{self.dataset}_{self.mode_token}_s{self.seed:02d}"

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
    def featurization_root(self) -> Path:
        base = self.workspace_root / "output" / "modules" / "featurization" / "output" / "data_root" / "argus"
        if self.dataset in {"lanl", "optc", "cic"}:
            return base / self.dataset
        return base / "0501" / "0501"

def ensure_eg4_dirs() -> None:
    for path in (GENERATED_CONFIG_ROOT, ORCHESTRATOR_ROOT, REPORT_ROOT):
        path.mkdir(parents=True, exist_ok=True)

def detector_env_python() -> Path:
    return DETECTOR_ENV_PREFIX / "bin/python"

def _manifest_list(dataset: str) -> list[Path]:
    value = MANIFESTS[dataset]
    if isinstance(value, tuple):
        return list(value)
    return [value]

def _replace_job_dataset_arg(command: list[Any], dataset_value: str) -> list[str]:
    return replace_flag(command, "--dataset", dataset_value)

def _ensure_flag(command: list[Any], flag: str, value: str) -> list[str]:
    resolved = [str(part) for part in command]
    for index, part in enumerate(resolved):
        if part == flag:
            if index + 1 >= len(resolved):
                raise ValueError(f"Flag {flag} missing value in command: {resolved}")
            resolved[index + 1] = value
            return resolved
    return [*resolved, flag, value]

def dataset_list() -> list[str]:
    return list(DATASETS)

def flow_mode_list() -> list[str]:
    return list(FLOW_MODES)

def canonical_dataset(value: str) -> str:
    lowered = value.strip().lower()
    return DATASET_ALIASES.get(lowered, lowered)

def canonical_flow_mode(value: str) -> str:
    lowered = value.strip().lower()
    return FLOW_MODE_ALIASES.get(lowered, lowered)

def replace_flag(command: list[Any], flag: str, value: str) -> list[str]:
    resolved = [str(part) for part in command]
    for index, part in enumerate(resolved):
        if part == flag:
            if index + 1 >= len(resolved):
                raise ValueError(f"Flag {flag} missing value in command: {resolved}")
            updated = list(resolved)
            updated[index + 1] = value
            return updated
    raise ValueError(f"Flag {flag} not found in command: {resolved}")

def parse_csv_list(value: str | None, *, kind: str) -> list[str] | None:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if kind == "dataset":
        items = [canonical_dataset(item) for item in items]
    elif kind == "flow_mode":
        items = [canonical_flow_mode(item) for item in items]
    return list(dict.fromkeys(items))

def parse_seed_list(value: str | None) -> list[int]:
    if not value:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in value.split(",") if item.strip()]

def iter_jobs(
    datasets: Iterable[str] | None = None,
    flow_modes: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
) -> list[EG4Job]:
    selected_datasets = list(datasets or dataset_list())
    selected_flow_modes = list(flow_modes or flow_mode_list())
    selected_seeds = list(seeds or DEFAULT_SEEDS)
    jobs: list[EG4Job] = []
    for dataset in selected_datasets:
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        for flow_mode in selected_flow_modes:
            if flow_mode not in FLOW_MODES:
                raise ValueError(f"Unknown flow mode: {flow_mode}")
            for seed in selected_seeds:
                jobs.append(EG4Job(dataset=dataset, flow_mode=flow_mode, seed=seed))
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

def generate_config(job: EG4Job) -> Path:
    ensure_eg4_dirs()
    payload = load_yaml(BASE_CONFIG)
    generated = deepcopy(payload)
    generated["name"] = f"EG4 ARGUS {job.dataset} {job.flow_mode} seed {job.seed}"
    generated["experiment_id"] = job.experiment_id
    generated["run_label"] = RUN_LABEL
    generated["seed"] = job.seed
    generated.setdefault("runtime", {})
    generated["runtime"]["data_root"] = str(DATA_ROOT)
    generated["manifests"] = [str(path) for path in _manifest_list(job.dataset)]
    env_python = detector_env_python()
    if env_python.exists():
        generated["python_bin"] = str(env_python)

    featurization = generated["modules"]["featurization"]
    spec = PREPROCESS_SPECS[job.dataset]
    featurization["runner"] = "shell_command"
    featurization["inputs"] = {"data_root/raw_data": "${graph_construction_artifact_data_root}/raw_data"}
    featurization["workdir"] = spec["workdir"]
    featurization["env"] = {}
    featurization["command"] = list(spec["command"])

    detection = generated["modules"]["detection"]
    selected_job = _select_detection_job(detection["jobs"], JOB_IDS[job.dataset])
    selected_job["dataset"] = job.dataset
    command = _replace_job_dataset_arg(selected_job["command"], ARGUS_DATASET_ARGS[job.dataset])
    command = _ensure_flag(command, "--te_end", "all")
    selected_job["command"] = [
        selected_job["command"][0],
        "${repo_root}/EG4/legacy_entrypoint.py",
        *command[1:],
        "--edge-features",
        "on" if job.edge_features_enabled else "off",
    ]
    selected_job["env"] = dict(selected_job.get("env", {}))
    selected_job["env"]["EG4_FLOW_MODE"] = job.flow_mode
    if job.dataset == "0501":
        selected_job["env"]["ARGUS_RUN_NAME"] = "0501"
        selected_job["metrics_relpath"] = "results/0501/metrics.csv"
    detection["jobs"] = [selected_job]

    dump_yaml(job.config_path, generated)
    return job.config_path

def _select_detection_job(jobs: list[dict[str, Any]], target_job_id: str) -> dict[str, Any]:
    matches = [deepcopy(item) for item in jobs if item["id"] == target_job_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one detection job for {target_job_id}, found {len(matches)}")
    return matches[0]

def run_job(job: EG4Job, *, dry_run: bool = False, force: bool = False) -> str:
    config_path = generate_config(job)
    if not force:
        status = job_status(job)
        if status in {"success", "unsupported"}:
            return "skipped"
    env_python = detector_env_python()
    if not env_python.exists() and not dry_run:
        print(f"[EG4] missing detector environment: expected {env_python}", file=sys.stderr)
        return "failed"
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(EG4_CODE_ROOT.resolve()), str((REPO_ROOT / "src").resolve())]
    if pythonpath:
        pythonpath_parts.append(pythonpath)
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    if env_python.exists():
        env["PYTHON"] = str(env_python)
        env["PATH"] = f"{env_python.parent}:{env.get('PATH', '')}"
    command = [sys.executable, "-m", "gids_eval", "run"]
    if dry_run:
        command.append("--dry-run")
    command.append(str(config_path))
    completed = subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=False)
    if completed.returncode != 0:
        return "failed"
    return job_status(job)

def job_status(job: EG4Job) -> str:
    cost_rows = read_csv(job.cost_metrics_path)
    baseline_rows = read_csv(job.baseline_metrics_path)
    if cost_rows and baseline_rows:
        detection_rows = [row for row in cost_rows if row.get("phase") == "detection"]
        if detection_rows and all(row.get("status") in {"success", "cache_hit"} for row in detection_rows):
            return "success"
    if cost_rows:
        detection_rows = [row for row in cost_rows if row.get("phase") == "detection"]
        if detection_rows and all(row.get("status") == "unsupported" for row in detection_rows):
            return "unsupported"
    if job.run_summary_path.exists():
        try:
            with job.run_summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
        except json.JSONDecodeError:
            return "failed"
        records = summary.get("records", [])
        statuses = {str(record.get("status", "")) for record in records}
        if any(record.get("status") == "failed" for record in records):
            return "failed"
        if statuses and statuses.issubset({"dry_run"}):
            return "pending"
    if job.metadata_root.exists():
        return "partial"
    return "pending"

def build_status_rows(jobs: list[EG4Job]) -> list[dict[str, Any]]:
    rows = []
    for job in jobs:
        rows.append(
            {
                "dataset": job.dataset,
                "flow_mode": job.flow_mode,
                "seed": job.seed,
                "experiment_id": job.experiment_id,
                "status": job_status(job),
                "config_path": str(job.config_path),
                "baseline_metrics_path": str(job.baseline_metrics_path),
                "cost_metrics_path": str(job.cost_metrics_path),
                "run_summary_path": str(job.run_summary_path),
            }
        )
    return rows

def write_matrix_report(jobs: list[EG4Job], path: Path | None = None) -> Path:
    rows = [
        {
            "dataset": job.dataset,
            "flow_mode": job.flow_mode,
            "seed": job.seed,
            "experiment_id": job.experiment_id,
            "config_path": str(job.config_path),
        }
        for job in jobs
    ]
    path = path or (ORCHESTRATOR_ROOT / "job_matrix.csv")
    write_csv(path, rows, ["dataset", "flow_mode", "seed", "experiment_id", "config_path"])
    return path

def write_status_report(jobs: list[EG4Job], path: Path | None = None) -> Path:
    rows = build_status_rows(jobs)
    path = path or (ORCHESTRATOR_ROOT / "job_status.csv")
    write_csv(
        path,
        rows,
        [
            "dataset",
            "flow_mode",
            "seed",
            "experiment_id",
            "status",
            "config_path",
            "baseline_metrics_path",
            "cost_metrics_path",
            "run_summary_path",
        ],
    )
    return path

def write_progress_report(
    jobs: list[EG4Job],
    *,
    attempted_jobs: int | None = None,
    failed_jobs: int | None = None,
) -> tuple[Path, Path]:
    rows = build_status_rows(jobs)
    total = len(rows)
    counts = {status: sum(1 for row in rows if row["status"] == status) for status in ("pending", "partial", "failed", "unsupported", "success")}
    completed = counts["success"] + counts["failed"] + counts["unsupported"]
    payload = {
        "total_jobs": total,
        "completed_jobs": completed,
        "remaining_jobs": total - completed,
        "success_jobs": counts["success"],
        "failed_jobs": counts["failed"],
        "unsupported_jobs": counts["unsupported"],
        "partial_jobs": counts["partial"],
        "pending_jobs": counts["pending"],
        "completion_ratio": (completed / total) if total else 0.0,
        "attempted_jobs": attempted_jobs,
        "run_failures": failed_jobs,
    }
    PROGRESS_JSON_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# EG4 Progress",
        "",
        f"- total_jobs: {total}",
        f"- completed_jobs: {completed}",
        f"- remaining_jobs: {total - completed}",
        f"- success_jobs: {counts['success']}",
        f"- failed_jobs: {counts['failed']}",
        f"- unsupported_jobs: {counts['unsupported']}",
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

def summarize_jobs(jobs: list[EG4Job]) -> tuple[Path, Path]:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, Any]] = []
    success_jobs = [job for job in jobs if job_status(job) == "success"]
    for job in success_jobs:
        cost_rows = read_csv(job.cost_metrics_path)
        phase_rows = {str(row.get("phase", "")): row for row in cost_rows if row.get("phase")}
        detection_row = phase_rows.get("detection", {})
        featurization_row = phase_rows.get("featurization", {})
        total_wall_time_sec = sum(_to_float(row.get("wall_time_sec")) for row in cost_rows)
        run_peak_rss_kb = max((_to_float(row.get("peak_rss_kb")) for row in cost_rows), default=0.0)
        for metrics_row in read_csv(job.baseline_metrics_path):
            raw_rows.append(
                {
                    "detector": DETECTOR,
                    "dataset": job.dataset,
                    "reported_dataset": str(metrics_row.get("dataset", "")),
                    "flow_mode": job.flow_mode,
                    "edge_features_enabled": job.edge_features_enabled,
                    "seed": job.seed,
                    "experiment_id": job.experiment_id,
                    "auc": _to_float(metrics_row.get("auc")),
                    "ap": _to_float(metrics_row.get("ap")),
                    "threshold": _optional_float(metrics_row.get("threshold")),
                    "tpr": _optional_float(metrics_row.get("tpr")),
                    "fpr": _optional_float(metrics_row.get("fpr")),
                    "precision": _optional_float(metrics_row.get("precision")),
                    "recall": _optional_float(metrics_row.get("recall")),
                    "f1": _optional_float(metrics_row.get("f1")),
                    "cdr": _optional_float(metrics_row.get("cdr")),
                    "cdr_auc": _optional_float(metrics_row.get("cdr_auc")),
                    "campaign_count": _optional_float(metrics_row.get("campaign_count")),
                    "detected_campaigns": _optional_float(metrics_row.get("detected_campaigns")),
                    "cdr_meaningful": _to_bool(metrics_row.get("cdr_meaningful")),
                    "train_time": _optional_float(metrics_row.get("train_time")),
                    "test_time": _optional_float(metrics_row.get("test_time")),
                    "featurization_wall_time_sec": _optional_float(featurization_row.get("wall_time_sec")),
                    "detection_wall_time_sec": _optional_float(detection_row.get("wall_time_sec")),
                    "total_wall_time_sec": total_wall_time_sec,
                    "featurization_peak_rss_gib": _optional_kb_to_gib(featurization_row.get("peak_rss_kb")),
                    "detection_peak_rss_gib": _optional_kb_to_gib(detection_row.get("peak_rss_kb")),
                    "run_peak_rss_gib": _optional_kb_to_gib(run_peak_rss_kb),
                    "metrics_path": str(job.baseline_metrics_path),
                }
            )

    raw_path = REPORT_ROOT / "raw_results.csv"
    write_csv(raw_path, raw_rows, _raw_fieldnames())

    summary_rows = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        grouped.setdefault((str(row["dataset"]), str(row["flow_mode"])), []).append(row)
    for (dataset, flow_mode), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "dataset": dataset,
                "flow_mode": flow_mode,
                "edge_features_enabled": flow_mode == "with_flows",
                "seed_count": len(rows),
                "mean_ap": _aggregate_mean(rows, "ap"),
                "std_ap": _aggregate_std(rows, "ap"),
                "mean_auc": _aggregate_mean(rows, "auc"),
                "std_auc": _aggregate_std(rows, "auc"),
                "mean_cdr": _aggregate_mean(rows, "cdr"),
                "std_cdr": _aggregate_std(rows, "cdr"),
                "mean_train_time_sec": _aggregate_mean(rows, "train_time"),
                "mean_test_time_sec": _aggregate_mean(rows, "test_time"),
                "mean_total_wall_time_sec": _aggregate_mean(rows, "total_wall_time_sec"),
                "mean_run_peak_rss_gib": _aggregate_mean(rows, "run_peak_rss_gib"),
                "campaign_count": _aggregate_mean(rows, "campaign_count"),
                "detected_campaigns": _aggregate_mean(rows, "detected_campaigns"),
            }
        )
    summary_path = REPORT_ROOT / "summary_by_mode.csv"
    write_csv(summary_path, summary_rows, list(summary_rows[0].keys()) if summary_rows else _summary_fieldnames())

    overlap_lookup = {row["dataset"]: row for row in _collect_feature_overlap_rows(success_jobs)}
    paired_rows = []
    for dataset in dataset_list():
        without = next((row for row in summary_rows if row["dataset"] == dataset and row["flow_mode"] == "without_flows"), None)
        with_flows = next((row for row in summary_rows if row["dataset"] == dataset and row["flow_mode"] == "with_flows"), None)
        if not without or not with_flows:
            continue
        paired_rows.append(
            {
                "dataset": dataset,
                "without_ap": without["mean_ap"],
                "with_ap": with_flows["mean_ap"],
                "ap_delta": _delta(with_flows["mean_ap"], without["mean_ap"]),
                "without_auc": without["mean_auc"],
                "with_auc": with_flows["mean_auc"],
                "auc_delta": _delta(with_flows["mean_auc"], without["mean_auc"]),
                "without_cdr": without["mean_cdr"],
                "with_cdr": with_flows["mean_cdr"],
                "cdr_delta": _delta(with_flows["mean_cdr"], without["mean_cdr"]),
                "without_train_time_sec": without["mean_train_time_sec"],
                "with_train_time_sec": with_flows["mean_train_time_sec"],
                "train_time_delta_sec": _delta(with_flows["mean_train_time_sec"], without["mean_train_time_sec"]),
                "cosine_similarity": overlap_lookup.get(dataset, {}).get("cosine_similarity", ""),
            }
        )
    paired_path = REPORT_ROOT / "paired_deltas.csv"
    write_csv(paired_path, paired_rows, list(paired_rows[0].keys()) if paired_rows else _paired_fieldnames())

    overlap_path = REPORT_ROOT / "feature_overlap.csv"
    overlap_rows = list(overlap_lookup.values())
    write_csv(overlap_path, overlap_rows, list(overlap_rows[0].keys()) if overlap_rows else _overlap_fieldnames())

    _write_conclusion(summary_rows, paired_rows, REPORT_ROOT / "conclusion.md")
    return raw_path, summary_path

def _collect_feature_overlap_rows(success_jobs: list[EG4Job]) -> list[dict[str, Any]]:
    rows = []
    jobs_by_dataset: dict[str, list[EG4Job]] = {}
    for job in success_jobs:
        jobs_by_dataset.setdefault(job.dataset, []).append(job)
    for dataset, dataset_jobs in sorted(jobs_by_dataset.items()):
        preferred = next((job for job in dataset_jobs if job.edge_features_enabled), dataset_jobs[0])
        overlap = _feature_overlap_for_job(preferred)
        if overlap:
            rows.append(overlap)
    return rows

def _feature_overlap_for_job(job: EG4Job) -> dict[str, Any] | None:
    root = job.featurization_root
    if not root.exists():
        return None
    feature_names = FEATURE_SCHEMAS[job.dataset]
    benign_sum = [0.0, 0.0, 0.0]
    attack_sum = [0.0, 0.0, 0.0]
    benign_count = 0
    attack_count = 0
    for features, label in _iter_feature_rows(job.dataset, root):
        if label == 1:
            attack_sum = [a + b for a, b in zip(attack_sum, features)]
            attack_count += 1
        else:
            benign_sum = [a + b for a, b in zip(benign_sum, features)]
            benign_count += 1
    if attack_count == 0 or benign_count == 0:
        return None
    benign_mean = [value / benign_count for value in benign_sum]
    attack_mean = [value / attack_count for value in attack_sum]
    cosine_similarity = _cosine_similarity(benign_mean, attack_mean)
    return {
        "dataset": job.dataset,
        "feature_1_name": feature_names[0],
        "feature_2_name": feature_names[1],
        "feature_3_name": feature_names[2],
        "benign_count": benign_count,
        "attack_count": attack_count,
        "benign_mean_1": benign_mean[0],
        "benign_mean_2": benign_mean[1],
        "benign_mean_3": benign_mean[2],
        "attack_mean_1": attack_mean[0],
        "attack_mean_2": attack_mean[1],
        "attack_mean_3": attack_mean[2],
        "cosine_similarity": cosine_similarity,
        "featurization_root": str(root),
    }

def _iter_feature_rows(dataset: str, root: Path):
    if dataset == "lanl":
        files = sorted((root / "flows").glob("*.txt"))
        for path in files:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    cols = line.strip().split(",")
                    if len(cols) < 10:
                        continue
                    yield [float(cols[6]), float(cols[7]), float(cols[8])], int(float(cols[9]))
        return

    files = sorted(path for path in root.glob("*.txt") if path.is_file())
    for path in files:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                cols = line.strip().split(",")
                if dataset == "cic" and len(cols) >= 7:
                    yield [float(cols[3]), float(cols[4]), float(cols[5])], int(float(cols[6]))
                elif dataset == "optc" and len(cols) >= 9:
                    yield [float(cols[3]), float(cols[4]), float(cols[5])], int(float(cols[8]))
                elif dataset == "0501" and len(cols) >= 8:
                    yield [float(cols[3]), float(cols[4]), float(cols[5])], int(float(cols[6]))

def _cosine_similarity(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)

def _write_conclusion(summary_rows: list[dict[str, Any]], paired_rows: list[dict[str, Any]], path: Path) -> None:
    lines = ["# EG4 Conclusion", ""]
    if not paired_rows:
        lines.append("No completed paired results yet.")
    else:
        for row in paired_rows:
            direction = "improved" if _to_float(row["ap_delta"]) > 0 else "degraded"
            lines.append(
                f"- {row['dataset']}: edge features {direction} AP by {float(row['ap_delta']):.6f} "
                f"(without={float(row['without_ap']):.6f}, with={float(row['with_ap']):.6f}); "
                f"cosine_similarity={float(row['cosine_similarity']):.6f}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _aggregate_mean(rows: list[dict[str, Any]], key: str) -> float | str:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return statistics.mean(values) if values else ""

def _aggregate_std(rows: list[dict[str, Any]], key: str) -> float | str:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    if not values:
        return ""
    if len(values) == 1:
        return 0.0
    return statistics.stdev(values)

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
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}

def _delta(new_value: Any, old_value: Any) -> float | str:
    if new_value in (None, "") or old_value in (None, ""):
        return ""
    return float(new_value) - float(old_value)

def _raw_fieldnames() -> list[str]:
    return [
        "detector",
        "dataset",
        "reported_dataset",
        "flow_mode",
        "edge_features_enabled",
        "seed",
        "experiment_id",
        "auc",
        "ap",
        "threshold",
        "tpr",
        "fpr",
        "precision",
        "recall",
        "f1",
        "cdr",
        "cdr_auc",
        "campaign_count",
        "detected_campaigns",
        "cdr_meaningful",
        "train_time",
        "test_time",
        "featurization_wall_time_sec",
        "detection_wall_time_sec",
        "total_wall_time_sec",
        "featurization_peak_rss_gib",
        "detection_peak_rss_gib",
        "run_peak_rss_gib",
        "metrics_path",
    ]

def _summary_fieldnames() -> list[str]:
    return [
        "dataset",
        "flow_mode",
        "edge_features_enabled",
        "seed_count",
        "mean_ap",
        "std_ap",
        "mean_auc",
        "std_auc",
        "mean_cdr",
        "std_cdr",
        "mean_train_time_sec",
        "mean_test_time_sec",
        "mean_total_wall_time_sec",
        "mean_run_peak_rss_gib",
        "campaign_count",
        "detected_campaigns",
    ]

def _paired_fieldnames() -> list[str]:
    return [
        "dataset",
        "without_ap",
        "with_ap",
        "ap_delta",
        "without_auc",
        "with_auc",
        "auc_delta",
        "without_cdr",
        "with_cdr",
        "cdr_delta",
        "without_train_time_sec",
        "with_train_time_sec",
        "train_time_delta_sec",
        "cosine_similarity",
    ]

def _overlap_fieldnames() -> list[str]:
    return [
        "dataset",
        "feature_1_name",
        "feature_2_name",
        "feature_3_name",
        "benign_count",
        "attack_count",
        "benign_mean_1",
        "benign_mean_2",
        "benign_mean_3",
        "attack_mean_1",
        "attack_mean_2",
        "attack_mean_3",
        "cosine_similarity",
        "featurization_root",
    ]
