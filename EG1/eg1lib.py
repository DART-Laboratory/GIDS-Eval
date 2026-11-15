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
from gids_eval.status_codes import UNSUPPORTED_EXIT_CODE

_PATHS = get_paths()
REPO_ROOT = _PATHS.repo_root
EG1_CODE_ROOT = REPO_ROOT / "EG1"
DATA_ROOT = _PATHS.sc_data_root("eg1")
GENERATED_CONFIG_ROOT = DATA_ROOT / "generated-configs"
ORCHESTRATOR_ROOT = DATA_ROOT / "orchestrator"
REPORT_ROOT = DATA_ROOT / "reports"
SELECTION_REPORT_ROOT = REPORT_ROOT / "selection"
RUNTIME_ROOT = REPO_ROOT / "local_runtime"
RUN_LABEL = "eg1"
DEFAULT_SEEDS = (0,)
CONDA_ROOT = _PATHS.conda_root
MINIFORGE_ROOT = CONDA_ROOT / "miniforge3"
PIKACHU_PREPROCESS_SEED = 1337
PIKACHU_NEG_RATIO = 20
SENSITIVITY_THRESHOLD = 0.20
PROGRESS_JSON_PATH = ORCHESTRATOR_ROOT / "progress.json"
PROGRESS_MD_PATH = ORCHESTRATOR_ROOT / "progress.md"
MANUAL_STATUS_OVERRIDES_PATH = ORCHESTRATOR_ROOT / "manual_status_overrides.json"
DATASET_CAMPAIGN_MODES = {
    "0501": "family",
    "cic": "raw_label",
    "lanl": "entity_time_component",
    "optc": "entity_time_component",
}

DEFAULT_WINDOW_SECONDS = {
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
        "cic": 3600,
        "0501": 300,
    },
}

SVG_COLORS = {
    "argus": "#d1495b",
    "euler": "#2e6f95",
    "vgrnn": "#2a9d8f",
    "pikachu": "#6c584c",
}

DETECTOR_ENV_PREFIXES = {
    "argus": CONDA_ROOT / "envs/sc1-argus",
    "euler": CONDA_ROOT / "envs/sc1-euler",
    "vgrnn": CONDA_ROOT / "envs/sc1-vgrnn",
    "pikachu": CONDA_ROOT / "envs/sc1-pikachu",
}

DETECTOR_REQUIRED_IMPORTS = {
    "argus": ("numpy", "pandas", "sklearn", "torch", "torch_geometric", "networkx", "tqdm", "yaml"),
    "euler": ("numpy", "pandas", "sklearn", "torch", "torch_geometric", "networkx", "tqdm", "yaml"),
    "vgrnn": ("numpy", "pandas", "sklearn", "torch", "torch_geometric", "networkx", "tqdm", "yaml"),
    "pikachu": ("numpy", "pandas", "sklearn", "tensorflow", "gensim", "networkx", "tqdm", "yaml"),
}

BASE_CONFIGS = {
    "argus": REPO_ROOT / "configs/experiments/argus_pipeline.yaml",
    "euler": REPO_ROOT / "configs/experiments/euler_pipeline.yaml",
    "vgrnn": REPO_ROOT / "configs/experiments/vgrnn_pipeline.yaml",
    "pikachu": REPO_ROOT / "configs/experiments/pikachu_pipeline.yaml",
}

COMPONENT_MANIFESTS = {
    "lanl": REPO_ROOT / "configs/data/lanl.yaml",
    "optc": REPO_ROOT / "configs/data/optc.yaml",
    "cic": REPO_ROOT / "configs/data/cic.yaml",
    "oilrig": REPO_ROOT / "configs/data/0501_oilrig.yaml",
    "sandworm": REPO_ROOT / "configs/data/0501_sandworm.yaml",
    "wizard_spider": REPO_ROOT / "configs/data/0501_wizard_spider.yaml",
}

COMBINED_DATASET_COMPONENTS = {
    "0501": ("oilrig", "sandworm", "wizard_spider"),
}

DATASET_WINDOWS_SECONDS = {
    "lanl": (1800, 3600, 5400, 7200),
    "optc": (360, 1800, 3600, 9000),
    "cic": (600, 900, 1800, 3000),
    "0501": (75, 150, 300, 450),
}

DETECTOR_DATASET_JOB_IDS = {
    "argus": {
        "lanl": "lanl",
        "optc": "optc",
        "cic": "cic",
        "0501": "oilrig",
        "oilrig": "oilrig",
        "sandworm": "sandworm",
        "wizard_spider": "wizard",
    },
    "euler": {
        "lanl": "lanl",
        "optc": "optc",
        "cic": "cic",
        "0501": "oilrig",
        "oilrig": "oilrig",
        "sandworm": "sandworm",
        "wizard_spider": "wizard",
    },
    "vgrnn": {
        "lanl": "lanl",
        "optc": "optc",
        "cic": "cic",
        "0501": "oilrig",
        "oilrig": "oilrig",
        "sandworm": "sandworm",
        "wizard_spider": "wizard",
    },
    "pikachu": {
        "lanl": "lanl",
        "optc": "optc",
        "cic": "cic",
        "0501": "oilrig",
        "oilrig": "oilrig",
        "sandworm": "sandworm",
        "wizard_spider": "wizard_spider",
    },
}

PIKACHU_DATASET_ARGS = {
    "0501": "0501",
    "lanl": "lanl",
    "optc": "optc",
    "cic": "cic",
    "oilrig": "oilrig",
    "sandworm": "sandworm",
    "wizard_spider": "wizard_spider",
}

PIKACHU_BASE_ARGS = {
    "dimensions": "100",
    "alpha": "0.01",
    "support": "10",
    "gpu": "-1",
}

GRAPH_PREPROCESSOR_SPECS = {
    "argus": {
        "cic": {
            "workdir": "${runtime_root}/pre_processing/argus",
            "script": "process_CIC_argus.py",
            "input_relpath": "raw_data/CIC-new",
            "output_relpath": "argus/cic",
        },
        "lanl": {
            "workdir": "${runtime_root}/pre_processing/argus",
            "script": "process_lanl_argus.py",
            "input_relpath": "raw_data/LANL",
            "output_relpath": "argus/lanl",
        },
        "optc": {
            "workdir": "${runtime_root}/pre_processing/argus",
            "script": "process_optc_argus.py",
            "input_relpath": "raw_data/OpTC",
            "output_relpath": "argus/optc",
        },
    },
    "euler": {
        "cic": {
            "workdir": "${runtime_root}/pre_processing/euler",
            "script": "process_CIC_new_euler.py",
            "input_relpath": "raw_data/CIC-new",
            "output_relpath": "euler/cic",
        },
        "lanl": {
            "workdir": "${runtime_root}/pre_processing/euler",
            "script": "process_lanl_euler.py",
            "input_relpath": "raw_data/LANL",
            "output_relpath": "euler/lanl",
        },
        "optc": {
            "workdir": "${runtime_root}/pre_processing/euler",
            "script": "process_optc_euler.py",
            "input_relpath": "raw_data/OpTC",
            "output_relpath": "euler/optc",
        },
    },
    "vgrnn": {
        "cic": {
            "workdir": "${runtime_root}/pre_processing/VGRNN",
            "script": "process_CIC_vgrnn.py",
            "input_relpath": "raw_data/CIC-new",
            "output_relpath": "vgrnn/cic",
        },
        "lanl": {
            "workdir": "${runtime_root}/pre_processing/VGRNN",
            "script": "process_lanl_vgrnn.py",
            "input_relpath": "raw_data/LANL",
            "output_relpath": "vgrnn/lanl",
        },
        "optc": {
            "workdir": "${runtime_root}/pre_processing/VGRNN",
            "script": "process_optc_vgrnn.py",
            "input_relpath": "raw_data/OpTC",
            "output_relpath": "vgrnn/optc",
        },
    },
}

DATASET_ALIASES = {
    "0501": "0501",
    "wizard": "wizard_spider",
    "wizard_spider": "wizard_spider",
}

REQUESTED_DATASET_ALIASES = {
    "0501": "0501",
    "oilrig": "0501",
    "sandworm": "0501",
    "wizard": "0501",
    "wizard_spider": "0501",
}

MODULE_STAGE_ORDER = (
    "graph_construction",
    "featurization",
    "graph_encoder",
    "temporal_encoder",
    "loss",
    "detection",
)

SUMMARY_METRIC_SPECS = (
    {"slug": "ap", "raw": "ap", "mean": "mean_ap", "std": "std_ap"},
    {"slug": "auc", "raw": "auc", "mean": "mean_auc", "std": "std_auc"},
    {"slug": "tpr", "raw": "tpr", "mean": "mean_tpr", "std": "std_tpr"},
    {"slug": "fpr", "raw": "fpr", "mean": "mean_fpr", "std": "std_fpr"},
    {"slug": "precision", "raw": "precision", "mean": "mean_precision", "std": "std_precision"},
    {"slug": "recall", "raw": "recall", "mean": "mean_recall", "std": "std_recall"},
    {"slug": "f1", "raw": "f1", "mean": "mean_f1", "std": "std_f1"},
    {"slug": "cdr", "raw": "cdr", "mean": "mean_cdr", "std": "std_cdr"},
    {"slug": "cdr_auc", "raw": "cdr_auc", "mean": "mean_cdr_auc", "std": "std_cdr_auc"},
    {
        "slug": "campaign_count",
        "raw": "campaign_count",
        "mean": "mean_campaign_count",
        "std": "std_campaign_count",
    },
    {
        "slug": "detected_campaigns",
        "raw": "detected_campaigns",
        "mean": "mean_detected_campaigns",
        "std": "std_detected_campaigns",
    },
    {"slug": "train_time_sec", "raw": "train_time", "mean": "mean_train_time_sec", "std": "std_train_time_sec"},
    {"slug": "test_time_sec", "raw": "test_time", "mean": "mean_test_time_sec", "std": "std_test_time_sec"},
    {
        "slug": "featurization_wall_time_sec",
        "raw": "featurization_wall_time_sec",
        "mean": "mean_featurization_wall_time_sec",
        "std": "std_featurization_wall_time_sec",
    },
    {
        "slug": "detection_wall_time_sec",
        "raw": "detection_wall_time_sec",
        "mean": "mean_detection_wall_time_sec",
        "std": "std_detection_wall_time_sec",
    },
    {
        "slug": "total_wall_time_sec",
        "raw": "total_wall_time_sec",
        "mean": "mean_total_wall_time_sec",
        "std": "std_total_wall_time_sec",
    },
    {
        "slug": "featurization_peak_rss_gib",
        "raw": "featurization_peak_rss_gib",
        "mean": "mean_featurization_peak_rss_gib",
        "std": "std_featurization_peak_rss_gib",
    },
    {
        "slug": "detection_peak_rss_gib",
        "raw": "detection_peak_rss_gib",
        "mean": "mean_detection_peak_rss_gib",
        "std": "std_detection_peak_rss_gib",
    },
    {
        "slug": "run_peak_rss_gib",
        "raw": "run_peak_rss_gib",
        "mean": "mean_run_peak_rss_gib",
        "std": "std_run_peak_rss_gib",
    },
    {
        "slug": "featurization_mean_rss_gib",
        "raw": "featurization_mean_rss_gib",
        "mean": "mean_featurization_mean_rss_gib",
        "std": "std_featurization_mean_rss_gib",
    },
    {
        "slug": "detection_mean_rss_gib",
        "raw": "detection_mean_rss_gib",
        "mean": "mean_detection_mean_rss_gib",
        "std": "std_detection_mean_rss_gib",
    },
    {
        "slug": "run_mean_rss_gib",
        "raw": "run_mean_rss_gib",
        "mean": "mean_run_mean_rss_gib",
        "std": "std_run_mean_rss_gib",
    },
    {
        "slug": "featurization_mean_cpu_percent",
        "raw": "featurization_mean_cpu_percent",
        "mean": "mean_featurization_mean_cpu_percent",
        "std": "std_featurization_mean_cpu_percent",
    },
    {
        "slug": "detection_mean_cpu_percent",
        "raw": "detection_mean_cpu_percent",
        "mean": "mean_detection_mean_cpu_percent",
        "std": "std_detection_mean_cpu_percent",
    },
    {
        "slug": "run_mean_cpu_percent",
        "raw": "run_mean_cpu_percent",
        "mean": "mean_run_mean_cpu_percent",
        "std": "std_run_mean_cpu_percent",
    },
)

@dataclass(frozen=True)
class EG1Job:
    detector: str
    dataset: str
    window_seconds: int
    seed: int

    @property
    def window_label(self) -> str:
        return format_window_label(self.window_seconds)

    @property
    def experiment_id(self) -> str:
        return f"eg1_{self.detector}_{self.dataset}_{format_window_token(self.window_seconds)}_s{self.seed:02d}"

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

def ensure_eg1_dirs() -> None:
    for path in (GENERATED_CONFIG_ROOT, ORCHESTRATOR_ROOT, REPORT_ROOT, SELECTION_REPORT_ROOT):
        path.mkdir(parents=True, exist_ok=True)

def detector_env_prefix(detector: str) -> Path:
    override = os.environ.get(f"EG1_ENV_{detector.upper()}")
    if override:
        return Path(override)
    return DETECTOR_ENV_PREFIXES[detector]

def detector_env_python(detector: str) -> Path:
    return detector_env_prefix(detector) / "bin/python"

def format_window_label(seconds: int) -> str:
    if seconds % 3600 == 0:
        return f"{seconds / 3600:g}h"
    if seconds % 60 == 0:
        return f"{seconds / 60:g}m"
    return f"{seconds}s"

def format_window_token(seconds: int) -> str:
    return format_window_label(seconds).replace(".", "p")

def dataset_list() -> list[str]:
    return list(DATASET_WINDOWS_SECONDS)

def detector_list() -> list[str]:
    return list(BASE_CONFIGS)

def dataset_members(dataset: str) -> list[str]:
    return [dataset]

def manifest_members(dataset: str) -> list[str]:
    return list(COMBINED_DATASET_COMPONENTS.get(dataset, (dataset,)))

def all_manifest_paths() -> list[Path]:
    ordered: list[Path] = []
    seen: set[Path] = set()
    for dataset in dataset_list():
        for member in manifest_members(dataset):
            path = COMPONENT_MANIFESTS[member]
            if path in seen:
                continue
            ordered.append(path)
            seen.add(path)
    return ordered

def canonical_dataset(value: str) -> str:
    lowered = value.strip().lower()
    return DATASET_ALIASES.get(lowered, lowered)

def canonical_requested_dataset(value: str) -> str:
    lowered = canonical_dataset(value)
    return REQUESTED_DATASET_ALIASES.get(lowered, lowered)

def parse_csv_list(value: str | None, *, canonicalize: bool = False) -> list[str] | None:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if canonicalize:
        items = [canonical_requested_dataset(item) for item in items]
        items = list(dict.fromkeys(items))
    return items

def parse_seed_list(value: str | None) -> list[int]:
    if not value:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in value.split(",") if item.strip()]

def iter_jobs(
    detectors: Iterable[str] | None = None,
    datasets: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
) -> list[EG1Job]:
    selected_detectors = list(detectors or detector_list())
    selected_datasets = list(datasets or dataset_list())
    selected_seeds = list(seeds or DEFAULT_SEEDS)
    jobs: list[EG1Job] = []
    for detector in selected_detectors:
        if detector not in BASE_CONFIGS:
            raise ValueError(f"Unknown detector: {detector}")
        for dataset in selected_datasets:
            if dataset not in DATASET_WINDOWS_SECONDS:
                raise ValueError(f"Unknown dataset: {dataset}")
            members = dataset_members(dataset)
            if any(member not in DETECTOR_DATASET_JOB_IDS[detector] for member in members):
                continue
            for window_seconds in DATASET_WINDOWS_SECONDS[dataset]:
                for seed in selected_seeds:
                    jobs.append(
                        EG1Job(
                            detector=detector,
                            dataset=dataset,
                            window_seconds=window_seconds,
                            seed=seed,
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
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]

def load_manual_status_overrides() -> dict[str, dict[str, Any]]:
    if not MANUAL_STATUS_OVERRIDES_PATH.exists():
        return {}
    try:
        payload = json.loads(MANUAL_STATUS_OVERRIDES_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(payload, dict):
        return {}
    overrides: dict[str, dict[str, Any]] = {}
    for experiment_id, value in payload.items():
        if not isinstance(value, dict):
            continue
        status = str(value.get("status", "")).strip().lower()
        if not status:
            continue
        overrides[str(experiment_id)] = {
            "status": status,
            "reason": str(value.get("reason", "")).strip(),
            "recorded_at": str(value.get("recorded_at", "")).strip(),
            "source": str(value.get("source", "manual")).strip() or "manual",
        }
    return overrides

def manual_status_override(job: "EG1Job") -> dict[str, Any] | None:
    return load_manual_status_overrides().get(job.experiment_id)

def _save_manual_status_overrides(payload: dict[str, dict[str, Any]]) -> None:
    MANUAL_STATUS_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANUAL_STATUS_OVERRIDES_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def set_manual_status_override(
    experiment_id: str,
    *,
    status: str,
    reason: str,
    source: str = "manual",
) -> None:
    payload = load_manual_status_overrides()
    payload[str(experiment_id)] = {
        "status": str(status).strip().lower(),
        "reason": str(reason).strip(),
        "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": str(source).strip() or "manual",
    }
    _save_manual_status_overrides(payload)

def clear_manual_status_override(experiment_id: str, *, source: str | None = None) -> None:
    payload = load_manual_status_overrides()
    existing = payload.get(str(experiment_id))
    if existing is None:
        return
    if source is not None and str(existing.get("source", "")) != source:
        return
    payload.pop(str(experiment_id), None)
    _save_manual_status_overrides(payload)

def replace_flag(command: list[Any], flag: str, value: str) -> list[str]:
    resolved = [str(part) for part in command]
    for index, part in enumerate(resolved):
        if part == flag:
            if index + 1 >= len(resolved):
                raise ValueError(f"Flag {flag} missing value in command: {resolved}")
            resolved[index + 1] = value
            return resolved
    raise ValueError(f"Flag {flag} not found in command: {resolved}")

def detector_window_value(detector: str, dataset: str, window_seconds: int) -> str:
    if detector not in {"argus", "euler", "vgrnn"}:
        raise ValueError(f"Detector {detector} does not use runtime window conversion")
    if dataset in {"lanl", "optc"}:
        value = window_seconds / 3600
    else:
        value = window_seconds / 60
    return f"{value:g}"

def generate_config(job: EG1Job) -> Path:
    ensure_eg1_dirs()
    payload = load_yaml(BASE_CONFIGS[job.detector])
    generated = deepcopy(payload)
    generated["name"] = f"EG1 {job.detector.upper()} {job.dataset} {job.window_label} seed {job.seed}"
    generated["experiment_id"] = job.experiment_id
    generated["run_label"] = RUN_LABEL
    generated["seed"] = job.seed
    generated.setdefault("runtime", {})
    generated["runtime"]["data_root"] = str(DATA_ROOT)
    generated["manifests"] = [str(path) for path in all_manifest_paths()]
    env_python = detector_env_python(job.detector)
    if env_python.exists():
        generated["python_bin"] = str(env_python)

    if job.detector == "pikachu":
        _customize_pikachu(generated, job)
    else:
        _customize_runtime_window_detector(generated, job)

    dump_yaml(job.config_path, generated)
    return job.config_path

def _customize_runtime_window_detector(payload: dict[str, Any], job: EG1Job) -> None:
    if job.dataset == "0501":
        _customize_0501_graph_featurization(payload, job)
        detection = payload["modules"]["detection"]
        template_job_id = DETECTOR_DATASET_JOB_IDS[job.detector][job.dataset]
        selected_job = _select_detection_job(detection["jobs"], template_job_id, f"{job.detector}:{job.dataset}")
        selected_job["dataset"] = job.dataset
        selected_job["command"] = _replace_job_dataset_arg(
            selected_job["command"],
            "O_0501" if job.detector == "argus" else "0501",
        )
        flag = "--delta" if job.detector in {"argus", "euler"} else "--snapshot"
        command = replace_flag(
            selected_job["command"],
            flag,
            detector_window_value(job.detector, job.dataset, job.window_seconds),
        )
        if job.detector == "argus":
            command = _ensure_flag(command, "--te_end", "all")
        selected_job["command"] = [command[0], "${repo_root}/EG1/legacy_entrypoint.py", *command[1:]]
        selected_job["env"] = _tag_eg1_job_env(
            selected_job.get("env", {}),
            euler_local_mode=(job.detector == "euler"),
        )
        if job.detector == "argus":
            selected_job["env"]["ARGUS_RUN_NAME"] = "0501"
            selected_job["metrics_relpath"] = "results/0501/metrics.csv"
        elif job.detector == "euler":
            selected_job["metrics_relpath"] = "results/0501/0501/learning/metrics.csv"
        else:
            selected_job["env"]["VGRNN_RUN_NAME"] = "0501"
            selected_job["metrics_relpath"] = "results/0501/metrics.csv"
        detection["jobs"] = [selected_job]
        return

    preprocess_spec = GRAPH_PREPROCESSOR_SPECS[job.detector][job.dataset]
    featurization = payload["modules"]["featurization"]
    featurization["runner"] = "shell_command"
    featurization["inputs"] = {"data_root/raw_data": "${graph_construction_artifact_data_root}/raw_data"}
    featurization["workdir"] = preprocess_spec["workdir"]
    featurization["env"] = {}
    featurization["command"] = [
        "${python_bin}",
        preprocess_spec["script"],
        "-i",
        f"${{module_output_dir}}/data_root/{preprocess_spec['input_relpath']}",
        "-o",
        f"${{module_output_dir}}/data_root/{preprocess_spec['output_relpath']}",
    ]

    detection = payload["modules"]["detection"]
    flag = "--delta" if job.detector in {"argus", "euler"} else "--snapshot"
    target_job_id = DETECTOR_DATASET_JOB_IDS[job.detector][job.dataset]
    selected_job = _select_detection_job(detection["jobs"], target_job_id, f"{job.detector}:{job.dataset}")
    command = replace_flag(
        selected_job["command"],
        flag,
        detector_window_value(job.detector, job.dataset, job.window_seconds),
    )
    if job.detector == "argus":
        command = _ensure_flag(command, "--te_end", "all")
    selected_job["command"] = [command[0], "${repo_root}/EG1/legacy_entrypoint.py", *command[1:]]
    selected_job["env"] = _tag_eg1_job_env(
        selected_job.get("env", {}),
        euler_local_mode=(job.detector == "euler"),
    )
    detection["jobs"] = [selected_job]

def _customize_pikachu(payload: dict[str, Any], job: EG1Job) -> None:
    featurization = payload["modules"]["featurization"]
    detection = payload["modules"]["detection"]
    if job.dataset == "0501":
        featurization["runner"] = "shell_command"
        featurization["workdir"] = "${repo_root}"
        featurization["env"] = {}
        featurization["command"] = [
            "${python_bin}",
            "${repo_root}/EG1/build_0501_unified.py",
            "--mode",
            "pikachu",
            "--raw-root",
            "${module_output_dir}/data_root/raw_data",
            "--output-dir",
            "${module_output_dir}/data_root/pikachu_sc1/0501",
            "--snapshot-seconds",
            str(job.window_seconds),
            "--neg-ratio",
            str(PIKACHU_NEG_RATIO),
            "--seed",
            str(PIKACHU_PREPROCESS_SEED),
        ]
        selected_job = _select_detection_job(
            detection["jobs"],
            DETECTOR_DATASET_JOB_IDS["pikachu"][job.dataset],
            "pikachu:0501",
        )
        selected_job["dataset"] = "0501"
        selected_job["workdir"] = "${repo_root}"
        selected_job["command"] = [
            "${python_bin}",
            "${repo_root}/EG1/pikachu_run.py",
            "--dataset",
            "0501",
            "--metadata",
            "${module_output_dir}/data_root/pikachu_sc1/0501/metadata.json",
            "--detector-root",
            "${runtime_root}/detection_baselines/pikachu",
            "--output-dir",
            "${module_output_dir}/results/0501",
            "--metrics-csv",
            "${module_output_dir}/results/0501/metrics.csv",
            "--dimensions",
            PIKACHU_BASE_ARGS["dimensions"],
            "--alpha",
            PIKACHU_BASE_ARGS["alpha"],
            "--support",
            PIKACHU_BASE_ARGS["support"],
            "--gpu",
            PIKACHU_BASE_ARGS["gpu"],
        ]
        selected_job["metrics_relpath"] = "results/0501/metrics.csv"
        detection["jobs"] = [selected_job]
        return

    dataset_dir = job.dataset
    featurization["command"] = [
        "${python_bin}",
        "${repo_root}/EG1/pikachu_preprocess.py",
        "--dataset",
        PIKACHU_DATASET_ARGS[job.dataset],
        "--raw-root",
        "${module_output_dir}/data_root/raw_data",
        "--output-dir",
        f"${{module_output_dir}}/data_root/pikachu_sc1/{dataset_dir}",
        "--snapshot-seconds",
        str(job.window_seconds),
        "--neg-ratio",
        str(PIKACHU_NEG_RATIO),
        "--seed",
        str(PIKACHU_PREPROCESS_SEED),
    ]
    target_job_id = DETECTOR_DATASET_JOB_IDS["pikachu"][job.dataset]
    selected_job = _select_detection_job(detection["jobs"], target_job_id, f"pikachu:{job.dataset}")
    selected_job["workdir"] = "${repo_root}"
    selected_job["command"] = [
        "${python_bin}",
        "${repo_root}/EG1/pikachu_run.py",
        "--dataset",
        PIKACHU_DATASET_ARGS[job.dataset],
        "--metadata",
        f"${{module_output_dir}}/data_root/pikachu_sc1/{dataset_dir}/metadata.json",
        "--detector-root",
        "${runtime_root}/detection_baselines/pikachu",
        "--output-dir",
        f"${{module_output_dir}}/results/{target_job_id}",
        "--metrics-csv",
        f"${{module_output_dir}}/results/{target_job_id}/metrics.csv",
        "--dimensions",
        PIKACHU_BASE_ARGS["dimensions"],
        "--alpha",
        PIKACHU_BASE_ARGS["alpha"],
        "--support",
        PIKACHU_BASE_ARGS["support"],
        "--gpu",
        PIKACHU_BASE_ARGS["gpu"],
    ]
    detection["jobs"] = [selected_job]

def _select_detection_job(
    jobs: list[dict[str, Any]],
    target_job_id: str,
    label: str,
) -> dict[str, Any]:
    matching_jobs = [deepcopy(item) for item in jobs if item["id"] == target_job_id]
    if len(matching_jobs) != 1:
        raise ValueError(f"Expected one detection job for {label}, found {len(matching_jobs)}")
    return matching_jobs[0]

def _customize_0501_graph_featurization(payload: dict[str, Any], job: EG1Job) -> None:
    detector_root = payload["modules"]["featurization"]["inputs"]["data_root/raw_data"]
    featurization = payload["modules"]["featurization"]
    featurization["runner"] = "shell_command"
    featurization["inputs"] = {"data_root/raw_data": detector_root}
    featurization["workdir"] = "${repo_root}"
    featurization["env"] = {}
    featurization["command"] = [
        "${python_bin}",
        "${repo_root}/EG1/build_0501_unified.py",
        "--mode",
        "graph",
        "--raw-root",
        "${module_output_dir}/data_root/raw_data",
        "--output-root",
        f"${{module_output_dir}}/data_root/{job.detector}/0501/0501",
    ]

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

def _tag_eg1_job_env(
    env: dict[str, Any],
    *,
    combined_dataset: str | None = None,
    campaign_label: str | None = None,
    euler_local_mode: bool = False,
) -> dict[str, Any]:
    tagged = dict(env)
    if combined_dataset:
        tagged["EG1_COMBINED_DATASET"] = combined_dataset
    if campaign_label:
        tagged["EG1_CAMPAIGN_LABEL"] = campaign_label
    if euler_local_mode:
        tagged.setdefault("EG1_EULER_LOCAL_MODE", "1")
    return tagged

def run_job(job: EG1Job, *, dry_run: bool = False, force: bool = False) -> str:
    config_path = generate_config(job)
    if not force:
        status = job_status(job)
        if status in {"success", "unsupported"}:
            return "skipped"
    clear_manual_status_override(job.experiment_id, source="launcher_failure")
    env_python = detector_env_python(job.detector)
    if not env_python.exists() and not dry_run:
        print(
            f"[EG1] missing detector environment for {job.detector}: expected {env_python}",
            file=sys.stderr,
        )
        return "failed"
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [
        str(EG1_CODE_ROOT.resolve()),
        str((REPO_ROOT / "src").resolve()),
    ]
    if pythonpath:
        pythonpath_parts.append(pythonpath)
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    if env_python.exists():
        env["PYTHON"] = str(env_python)
        env["PATH"] = f"{env_python.parent}:{env.get('PATH', '')}"
        env["EG1_DETECTOR_PYTHON"] = str(env_python)
    command = [sys.executable, "-m", "gids_eval", "run"]
    if dry_run:
        command.append("--dry-run")
    command.append(str(config_path))
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        set_manual_status_override(
            job.experiment_id,
            status="failed",
            reason=f"launcher observed non-zero exit code {completed.returncode}",
            source="launcher_failure",
        )
        return "failed"
    return job_status(job)

def job_status(job: EG1Job) -> str:
    override = manual_status_override(job)
    if override is not None:
        return str(override.get("status", "pending"))
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
            records = summary.get("records", [])
            statuses = {str(record.get("status", "")) for record in records}
            if any(record.get("status") == "failed" for record in records):
                return "failed"
            if statuses and "unsupported" in statuses and statuses.issubset({"cache_hit", "success", "unsupported"}):
                return "unsupported"
            if statuses and statuses.issubset({"dry_run"}):
                return "pending"
        except json.JSONDecodeError:
            return "failed"
    if job.metadata_root.exists():
        return "partial"
    return "pending"

def build_status_rows(jobs: list[EG1Job]) -> list[dict[str, Any]]:
    rows = []
    for job in jobs:
        override = manual_status_override(job)
        rows.append(
            {
                "detector": job.detector,
                "dataset": job.dataset,
                "window_seconds": job.window_seconds,
                "window_label": job.window_label,
                "seed": job.seed,
                "experiment_id": job.experiment_id,
                "status": job_status(job),
                "config_path": str(job.config_path),
                "baseline_metrics_path": str(job.baseline_metrics_path),
                "cost_metrics_path": str(job.cost_metrics_path),
                "run_summary_path": str(job.run_summary_path),
                "status_reason": str((override or {}).get("reason", "")),
            }
        )
    return rows

def write_matrix_report(jobs: list[EG1Job], path: Path | None = None) -> Path:
    rows = []
    for job in jobs:
        rows.append(
            {
                "detector": job.detector,
                "dataset": job.dataset,
                "window_seconds": job.window_seconds,
                "window_label": job.window_label,
                "seed": job.seed,
                "experiment_id": job.experiment_id,
                "config_path": str(job.config_path),
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
            "seed",
            "experiment_id",
            "config_path",
        ],
    )
    return path

def write_status_report(jobs: list[EG1Job], path: Path | None = None) -> Path:
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
            "seed",
            "experiment_id",
            "status",
            "config_path",
            "baseline_metrics_path",
            "cost_metrics_path",
            "run_summary_path",
            "status_reason",
        ],
    )
    return path

def write_progress_report(
    jobs: list[EG1Job],
    *,
    json_path: Path | None = None,
    md_path: Path | None = None,
    scope_name: str = "global",
    attempted_jobs: int | None = None,
    failed_jobs: int | None = None,
) -> tuple[Path, Path]:
    rows = build_status_rows(jobs)
    total = len(rows)
    status_counts: dict[str, int] = {}
    for status in ("pending", "partial", "failed", "unsupported", "success"):
        status_counts[status] = sum(1 for row in rows if row["status"] == status)
    completed = status_counts["success"] + status_counts["failed"] + status_counts["unsupported"]
    completion_ratio = (completed / total) if total else 0.0
    success_ratio = (status_counts["success"] / total) if total else 0.0

    detector_progress: list[dict[str, Any]] = []
    for detector in detector_list():
        detector_rows = [row for row in rows if row["detector"] == detector]
        if not detector_rows:
            continue
        detector_total = len(detector_rows)
        detector_success = sum(1 for row in detector_rows if row["status"] == "success")
        detector_failed = sum(1 for row in detector_rows if row["status"] == "failed")
        detector_unsupported = sum(1 for row in detector_rows if row["status"] == "unsupported")
        detector_partial = sum(1 for row in detector_rows if row["status"] == "partial")
        detector_pending = sum(1 for row in detector_rows if row["status"] == "pending")
        detector_progress.append(
            {
                "detector": detector,
                "total": detector_total,
                "success": detector_success,
                "failed": detector_failed,
                "unsupported": detector_unsupported,
                "partial": detector_partial,
                "pending": detector_pending,
                "completed": detector_success + detector_failed + detector_unsupported,
                "completion_ratio": (
                    (detector_success + detector_failed + detector_unsupported) / detector_total if detector_total else 0.0
                ),
            }
        )

    partial_jobs = [row["experiment_id"] for row in rows if row["status"] == "partial"]
    failed_job_ids = [row["experiment_id"] for row in rows if row["status"] == "failed"]
    unsupported_job_ids = [row["experiment_id"] for row in rows if row["status"] == "unsupported"]

    payload = {
        "scope": scope_name,
        "total_jobs": total,
        "completed_jobs": completed,
        "remaining_jobs": total - completed,
        "success_jobs": status_counts["success"],
        "failed_jobs": status_counts["failed"],
        "unsupported_jobs": status_counts["unsupported"],
        "partial_jobs": status_counts["partial"],
        "pending_jobs": status_counts["pending"],
        "completion_ratio": completion_ratio,
        "success_ratio": success_ratio,
        "attempted_jobs": attempted_jobs,
        "run_failures": failed_jobs,
        "partial_experiment_ids": partial_jobs,
        "failed_experiment_ids": failed_job_ids,
        "unsupported_experiment_ids": unsupported_job_ids,
        "per_detector": detector_progress,
    }

    json_path = json_path or PROGRESS_JSON_PATH
    md_path = md_path or PROGRESS_MD_PATH
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        f"# EG1 Progress ({scope_name})",
        "",
        f"- total_jobs: {total}",
        f"- completed_jobs: {completed}",
        f"- remaining_jobs: {total - completed}",
        f"- success_jobs: {status_counts['success']}",
        f"- failed_jobs: {status_counts['failed']}",
        f"- unsupported_jobs: {status_counts['unsupported']}",
        f"- partial_jobs: {status_counts['partial']}",
        f"- pending_jobs: {status_counts['pending']}",
        f"- completion_ratio: {completion_ratio:.1%}",
        f"- success_ratio: {success_ratio:.1%}",
    ]
    if attempted_jobs is not None:
        lines.append(f"- attempted_jobs_in_current_run: {attempted_jobs}")
    if failed_jobs is not None:
        lines.append(f"- failures_in_current_run: {failed_jobs}")
    lines.extend(["", "## Per Detector"])
    for row in detector_progress:
        lines.append(
            (
                f"- {row['detector']}: {row['completed']}/{row['total']} complete "
                f"(success={row['success']}, failed={row['failed']}, unsupported={row['unsupported']}, "
                f"partial={row['partial']}, pending={row['pending']})"
            )
        )
    if partial_jobs:
        lines.extend(["", "## Partial Jobs"])
        for experiment_id in partial_jobs[:20]:
            lines.append(f"- {experiment_id}")
    if failed_job_ids:
        lines.extend(["", "## Failed Jobs"])
        for experiment_id in failed_job_ids[:20]:
            lines.append(f"- {experiment_id}")
    if unsupported_job_ids:
        lines.extend(["", "## Unsupported Jobs"])
        for experiment_id in unsupported_job_ids[:20]:
            lines.append(f"- {experiment_id}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path

def summarize_jobs(jobs: list[EG1Job], output_root: Path | None = None) -> tuple[Path, Path]:
    output_root = output_root or REPORT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, Any]] = []
    for job in jobs:
        if job_status(job) != "success":
            continue
        cost_rows = read_csv(job.cost_metrics_path)
        phase_rows = _phase_row_lookup(cost_rows)
        detection_row = phase_rows.get("detection", {})
        featurization_row = phase_rows.get("featurization", {})
        total_wall_time_sec = sum(_to_float(row.get("wall_time_sec")) for row in cost_rows)
        peak_values = [_optional_float(row.get("peak_rss_kb")) for row in cost_rows]
        peak_values = [value for value in peak_values if value != ""]
        run_peak_rss_kb = max(peak_values) if peak_values else ""
        run_mean_rss_kb = _weighted_mean_from_rows(cost_rows, value_key="mean_rss_kb", weight_key="sample_count")
        run_mean_cpu_percent = _weighted_mean_from_rows(
            cost_rows,
            value_key="mean_cpu_percent",
            weight_key="cpu_sample_count",
        )
        for metrics_row in read_csv(job.baseline_metrics_path):
            reported_dataset = canonical_dataset(str(metrics_row.get("dataset", job.dataset)))
            raw_rows.append(
                {
                    "detector": job.detector,
                    "dataset": job.dataset,
                    "reported_dataset": reported_dataset,
                    "campaign_mode": DATASET_CAMPAIGN_MODES.get(job.dataset, "unknown"),
                    "window_seconds": job.window_seconds,
                    "window_label": job.window_label,
                    "seed": job.seed,
                    "experiment_id": job.experiment_id,
                    "auc": _to_float(metrics_row.get("auc")),
                    "ap": _to_float(metrics_row.get("ap")),
                    "threshold": _to_float(metrics_row.get("threshold")),
                    "tpr": _to_float(metrics_row.get("tpr")),
                    "fpr": _to_float(metrics_row.get("fpr")),
                    "precision": _to_float(metrics_row.get("precision")),
                    "recall": _to_float(metrics_row.get("recall")),
                    "f1": _to_float(metrics_row.get("f1")),
                    "cdr": _to_float(metrics_row.get("cdr")),
                    "cdr_auc": _to_float(metrics_row.get("cdr_auc")),
                    "campaign_count": _optional_float(metrics_row.get("campaign_count")),
                    "detected_campaigns": _optional_float(metrics_row.get("detected_campaigns")),
                    "cdr_meaningful": _to_bool(metrics_row.get("cdr_meaningful")),
                    "train_time": _to_float(metrics_row.get("train_time")),
                    "test_time": _to_float(metrics_row.get("test_time")),
                    "detection_status": detection_row.get("status", ""),
                    "featurization_wall_time_sec": _to_float(featurization_row.get("wall_time_sec")),
                    "detection_wall_time_sec": _to_float(detection_row.get("wall_time_sec")),
                    "total_wall_time_sec": total_wall_time_sec,
                    "featurization_peak_rss_kb": _optional_float(featurization_row.get("peak_rss_kb")),
                    "detection_peak_rss_kb": _optional_float(detection_row.get("peak_rss_kb")),
                    "run_peak_rss_kb": run_peak_rss_kb,
                    "featurization_mean_rss_kb": _optional_float(featurization_row.get("mean_rss_kb")),
                    "detection_mean_rss_kb": _optional_float(detection_row.get("mean_rss_kb")),
                    "run_mean_rss_kb": run_mean_rss_kb,
                    "featurization_peak_rss_gib": _optional_kb_to_gib(featurization_row.get("peak_rss_kb")),
                    "detection_peak_rss_gib": _optional_kb_to_gib(detection_row.get("peak_rss_kb")),
                    "run_peak_rss_gib": _optional_kb_to_gib(run_peak_rss_kb),
                    "featurization_mean_rss_gib": _optional_kb_to_gib(featurization_row.get("mean_rss_kb")),
                    "detection_mean_rss_gib": _optional_kb_to_gib(detection_row.get("mean_rss_kb")),
                    "run_mean_rss_gib": _optional_kb_to_gib(run_mean_rss_kb),
                    "featurization_mean_cpu_percent": _optional_float(featurization_row.get("mean_cpu_percent")),
                    "detection_mean_cpu_percent": _optional_float(detection_row.get("mean_cpu_percent")),
                    "run_mean_cpu_percent": run_mean_cpu_percent,
                    "metrics_path": str(job.baseline_metrics_path),
                }
            )

    raw_path = output_root / "raw_results.csv"
    write_csv(raw_path, raw_rows, _raw_results_fieldnames())

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in raw_rows:
        key = (str(row["detector"]), str(row["dataset"]), int(row["window_seconds"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (detector, dataset, window_seconds), rows in sorted(grouped.items()):
        summary_row = {
            "detector": detector,
            "dataset": dataset,
            "campaign_mode": DATASET_CAMPAIGN_MODES.get(dataset, "unknown"),
            "window_seconds": window_seconds,
            "window_label": format_window_label(window_seconds),
            "seed_count": len(rows),
            "cdr_meaningful": all(bool(row.get("cdr_meaningful")) for row in rows),
        }
        for spec in SUMMARY_METRIC_SPECS:
            mean_value, std_value = _aggregate_metric(rows, spec["raw"])
            summary_row[spec["mean"]] = mean_value
            summary_row[spec["std"]] = std_value
        summary_rows.append(summary_row)

    summary_path = output_root / "summary_by_window.csv"
    write_csv(summary_path, summary_rows, _summary_fieldnames())
    pair_rows = build_pair_evidence_rows(summary_rows)
    pair_path = output_root / "evidence_table_pairs.csv"
    write_csv(
        pair_path,
        pair_rows,
        [
            "detector",
            "dataset",
            "window_count",
            "completed_seed_count",
            "default_window_seconds",
            "default_window_label",
            "default_window_observed",
            "best_window_seconds",
            "best_window_label",
            "best_mean_ap",
            "default_mean_ap",
            "ap_swing_abs",
            "ap_swing_rel",
            "best_mean_auc",
            "default_mean_auc",
            "auc_swing_abs",
            "auc_swing_rel",
            "sensitive_ap",
            "sensitive_auc",
        ],
    )

    overall_rows = build_overall_evidence_rows(pair_rows)
    overall_path = output_root / "evidence_table_overall.csv"
    write_csv(
        overall_path,
        overall_rows,
        [
            "group",
            "pair_count",
            "default_window_observed_pairs",
            "sensitive_ap_pairs",
            "sensitive_ap_ratio",
            "sensitive_auc_pairs",
            "sensitive_auc_ratio",
            "mean_ap_swing_rel",
            "median_ap_swing_rel",
            "max_ap_swing_rel",
            "mean_auc_swing_rel",
            "median_auc_swing_rel",
            "max_auc_swing_rel",
        ],
    )

    write_line_plot_svg(summary_rows, metric="mean_ap", title="EG1 AP by Window", output_path=output_root / "plot_ap_curves.svg")
    write_line_plot_svg(summary_rows, metric="mean_auc", title="EG1 AUC by Window", output_path=output_root / "plot_auc_curves.svg")
    write_heatmap_svg(pair_rows, metric_key="ap_swing_rel", title="EG1 Relative AP Swing", output_path=output_root / "plot_ap_swing_heatmap.svg")
    write_metric_tables(summary_rows, output_root / "metric_tables")
    write_conclusion_md(pair_rows, overall_rows, output_root / "conclusion.md")
    return raw_path, summary_path

def detector_env_status(detectors: Iterable[str] | None = None) -> tuple[Path, list[dict[str, Any]]]:
    ensure_eg1_dirs()
    rows: list[dict[str, Any]] = []
    for detector in list(detectors or detector_list()):
        env_prefix = detector_env_prefix(detector)
        env_python = detector_env_python(detector)
        row: dict[str, Any] = {
            "detector": detector,
            "env_prefix": str(env_prefix),
            "python_path": str(env_python),
            "env_exists": str(env_prefix.exists()).lower(),
            "python_exists": str(env_python.exists()).lower(),
            "imports_ok": "false",
            "missing_imports": "",
            "note": "",
        }
        if not env_python.exists():
            row["note"] = "python_missing"
            rows.append(row)
            continue
        imports = DETECTOR_REQUIRED_IMPORTS[detector]
        probe = (
            "import importlib.util\n"
            f"mods={list(imports)!r}\n"
            "missing=[]\n"
            "for name in mods:\n"
            "    if importlib.util.find_spec(name) is None:\n"
            "        missing.append(name)\n"
            "print(','.join(missing))\n"
        )
        completed = subprocess.run(
            [str(env_python), "-c", probe],
            cwd=str(REPO_ROOT),
            env=dict(os.environ),
            capture_output=True,
            text=True,
            check=False,
        )
        missing = completed.stdout.strip()
        if completed.returncode != 0:
            row["note"] = completed.stderr.strip() or "probe_failed"
            rows.append(row)
            continue
        row["imports_ok"] = "true" if not missing else "false"
        row["missing_imports"] = missing
        if missing:
            row["note"] = "missing_imports"
        rows.append(row)

    path = ORCHESTRATOR_ROOT / "env_status.csv"
    write_csv(
        path,
        rows,
        [
            "detector",
            "env_prefix",
            "python_path",
            "env_exists",
            "python_exists",
            "imports_ok",
            "missing_imports",
            "note",
        ],
    )
    return path, rows

def _to_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)

def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = _numeric_values(rows, key)
    return statistics.mean(values) if values else 0.0

def _std(rows: list[dict[str, Any]], key: str) -> float:
    values = _numeric_values(rows, key)
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)

def _aggregate_metric(rows: list[dict[str, Any]], key: str) -> tuple[float | str, float | str]:
    values = _numeric_values(rows, key)
    if not values:
        return "", ""
    if len(values) < 2:
        return statistics.mean(values), 0.0
    return statistics.mean(values), statistics.stdev(values)

def _numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        values.append(float(value))
    return values

def _optional_float(value: Any) -> float | str:
    if value in (None, ""):
        return ""
    return float(value)

def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}

def _phase_row_lookup(cost_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in cost_rows:
        phase = str(row.get("phase", "")).strip()
        if phase:
            lookup[phase] = row
    return lookup

def _kb_to_gib(value: Any) -> float:
    return _to_float(value) / (1024.0 * 1024.0)

def _optional_kb_to_gib(value: Any) -> float | str:
    if value in (None, ""):
        return ""
    return float(value) / (1024.0 * 1024.0)

def _weighted_mean_from_rows(
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    weight_key: str,
) -> float | str:
    weighted_total = 0.0
    total_weight = 0
    for row in rows:
        value = row.get(value_key)
        weight = row.get(weight_key)
        if value in (None, "") or weight in (None, ""):
            continue
        try:
            weight_int = int(float(weight))
        except (TypeError, ValueError):
            continue
        if weight_int <= 0:
            continue
        weighted_total += float(value) * weight_int
        total_weight += weight_int
    if total_weight == 0:
        return ""
    return weighted_total / total_weight

def _raw_results_fieldnames() -> list[str]:
    return [
        "detector",
        "dataset",
        "reported_dataset",
        "campaign_mode",
        "window_seconds",
        "window_label",
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
        "detection_status",
        "featurization_wall_time_sec",
        "detection_wall_time_sec",
        "total_wall_time_sec",
        "featurization_peak_rss_kb",
        "detection_peak_rss_kb",
        "run_peak_rss_kb",
        "featurization_mean_rss_kb",
        "detection_mean_rss_kb",
        "run_mean_rss_kb",
        "featurization_peak_rss_gib",
        "detection_peak_rss_gib",
        "run_peak_rss_gib",
        "featurization_mean_rss_gib",
        "detection_mean_rss_gib",
        "run_mean_rss_gib",
        "featurization_mean_cpu_percent",
        "detection_mean_cpu_percent",
        "run_mean_cpu_percent",
        "metrics_path",
    ]

def _summary_fieldnames() -> list[str]:
    fields = [
        "detector",
        "dataset",
        "campaign_mode",
        "window_seconds",
        "window_label",
        "seed_count",
        "cdr_meaningful",
    ]
    for spec in SUMMARY_METRIC_SPECS:
        fields.append(str(spec["mean"]))
        fields.append(str(spec["std"]))
    return fields

def write_metric_tables(summary_rows: list[dict[str, Any]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    for spec in SUMMARY_METRIC_SPECS:
        rows = build_metric_table_rows(summary_rows, spec)
        path = output_root / f"{spec['slug']}.csv"
        write_csv(path, rows, _metric_table_fieldnames(summary_rows))
        index_rows.append(
            {
                "metric": spec["slug"],
                "path": str(path),
                "row_count": len(rows),
            }
        )
    write_csv(output_root / "index.csv", index_rows, ["metric", "path", "row_count"])

def build_metric_table_rows(
    summary_rows: list[dict[str, Any]],
    spec: dict[str, str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        grouped.setdefault((str(row["detector"]), str(row["dataset"])), []).append(row)

    ordered_windows = _metric_table_window_seconds(summary_rows)
    result: list[dict[str, Any]] = []
    for (detector, dataset), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda item: int(item["window_seconds"]))
        default_window = DEFAULT_WINDOW_SECONDS.get(detector, {}).get(dataset)
        default_row = next((row for row in ordered if int(row["window_seconds"]) == default_window), None)
        numeric_rows = [row for row in ordered if row.get(spec["mean"]) not in ("", None)]
        min_row = min(numeric_rows, key=lambda item: float(item[spec["mean"]])) if numeric_rows else None
        max_row = max(numeric_rows, key=lambda item: float(item[spec["mean"]])) if numeric_rows else None
        min_value = float(min_row[spec["mean"]]) if min_row else ""
        max_value = float(max_row[spec["mean"]]) if max_row else ""
        table_row: dict[str, Any] = {
            "metric": spec["slug"],
            "detector": detector,
            "dataset": dataset,
            "window_count": len(ordered),
            "completed_seed_count": min(int(row["seed_count"]) for row in ordered),
            "default_window_seconds": default_window or "",
            "default_window_label": format_window_label(default_window) if default_window else "",
            "default_window_observed": str(default_row is not None).lower(),
            "default_value": float(default_row[spec["mean"]]) if default_row and default_row.get(spec["mean"]) not in ("", None) else "",
            "min_window_seconds": int(min_row["window_seconds"]) if min_row else "",
            "min_window_label": str(min_row["window_label"]) if min_row else "",
            "min_value": min_value,
            "max_window_seconds": int(max_row["window_seconds"]) if max_row else "",
            "max_window_label": str(max_row["window_label"]) if max_row else "",
            "max_value": max_value,
            "swing_abs": (max_value - min_value) if min_row and max_row else "",
            "swing_rel": ((max_value - min_value) / abs(max_value)) if min_row and max_row and max_value else "",
        }
        by_window = {int(row["window_seconds"]): row for row in ordered}
        for window_seconds in ordered_windows:
            window_label = format_window_label(window_seconds)
            row = by_window.get(window_seconds)
            table_row[f"window_{window_label}_mean"] = row.get(spec["mean"], "") if row else ""
            table_row[f"window_{window_label}_std"] = row.get(spec["std"], "") if row else ""
        result.append(table_row)
    return result

def _metric_table_fieldnames(summary_rows: list[dict[str, Any]]) -> list[str]:
    fields = [
        "metric",
        "detector",
        "dataset",
        "window_count",
        "completed_seed_count",
        "default_window_seconds",
        "default_window_label",
        "default_window_observed",
        "default_value",
        "min_window_seconds",
        "min_window_label",
        "min_value",
        "max_window_seconds",
        "max_window_label",
        "max_value",
        "swing_abs",
        "swing_rel",
    ]
    for window_seconds in _metric_table_window_seconds(summary_rows):
        window_label = format_window_label(window_seconds)
        fields.append(f"window_{window_label}_mean")
        fields.append(f"window_{window_label}_std")
    return fields

def _metric_table_window_seconds(summary_rows: list[dict[str, Any]]) -> list[int]:
    return sorted({int(row["window_seconds"]) for row in summary_rows})

def build_pair_evidence_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        grouped.setdefault((str(row["detector"]), str(row["dataset"])), []).append(row)

    pair_rows: list[dict[str, Any]] = []
    for (detector, dataset), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda item: int(item["window_seconds"]))
        best_ap_row = max(ordered, key=lambda item: float(item["mean_ap"]))
        best_auc_row = max(ordered, key=lambda item: float(item["mean_auc"]))
        default_window = DEFAULT_WINDOW_SECONDS.get(detector, {}).get(dataset)
        default_row = next((row for row in ordered if int(row["window_seconds"]) == default_window), None)

        ap_values = [float(row["mean_ap"]) for row in ordered]
        auc_values = [float(row["mean_auc"]) for row in ordered]
        ap_swing_abs = max(ap_values) - min(ap_values) if ap_values else 0.0
        auc_swing_abs = max(auc_values) - min(auc_values) if auc_values else 0.0
        ap_peak = max(ap_values) if ap_values else 0.0
        auc_peak = max(auc_values) if auc_values else 0.0
        ap_swing_rel = ap_swing_abs / ap_peak if ap_peak else 0.0
        auc_swing_rel = auc_swing_abs / auc_peak if auc_peak else 0.0

        pair_rows.append(
            {
                "detector": detector,
                "dataset": dataset,
                "window_count": len(ordered),
                "completed_seed_count": min(int(row["seed_count"]) for row in ordered),
                "default_window_seconds": default_window or "",
                "default_window_label": format_window_label(default_window) if default_window else "",
                "default_window_observed": str(default_row is not None).lower(),
                "best_window_seconds": int(best_ap_row["window_seconds"]),
                "best_window_label": best_ap_row["window_label"],
                "best_mean_ap": float(best_ap_row["mean_ap"]),
                "default_mean_ap": float(default_row["mean_ap"]) if default_row else "",
                "ap_swing_abs": ap_swing_abs,
                "ap_swing_rel": ap_swing_rel,
                "best_mean_auc": float(best_auc_row["mean_auc"]),
                "default_mean_auc": float(default_row["mean_auc"]) if default_row else "",
                "auc_swing_abs": auc_swing_abs,
                "auc_swing_rel": auc_swing_rel,
                "sensitive_ap": str(ap_swing_rel >= SENSITIVITY_THRESHOLD).lower(),
                "sensitive_auc": str(auc_swing_rel >= SENSITIVITY_THRESHOLD).lower(),
            }
        )
    return pair_rows

def build_overall_evidence_rows(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in pair_rows:
        groups.setdefault(str(row["detector"]), []).append(row)
    if pair_rows:
        groups["all"] = pair_rows

    overall_rows: list[dict[str, Any]] = []
    for group, rows in sorted(groups.items()):
        ap_values = [float(row["ap_swing_rel"]) for row in rows]
        auc_values = [float(row["auc_swing_rel"]) for row in rows]
        default_observed = sum(1 for row in rows if str(row["default_window_observed"]) == "true")
        sensitive_ap_pairs = sum(1 for row in rows if str(row["sensitive_ap"]) == "true")
        sensitive_auc_pairs = sum(1 for row in rows if str(row["sensitive_auc"]) == "true")
        count = len(rows)
        overall_rows.append(
            {
                "group": group,
                "pair_count": count,
                "default_window_observed_pairs": default_observed,
                "sensitive_ap_pairs": sensitive_ap_pairs,
                "sensitive_ap_ratio": sensitive_ap_pairs / count if count else 0.0,
                "sensitive_auc_pairs": sensitive_auc_pairs,
                "sensitive_auc_ratio": sensitive_auc_pairs / count if count else 0.0,
                "mean_ap_swing_rel": statistics.mean(ap_values) if ap_values else 0.0,
                "median_ap_swing_rel": statistics.median(ap_values) if ap_values else 0.0,
                "max_ap_swing_rel": max(ap_values) if ap_values else 0.0,
                "mean_auc_swing_rel": statistics.mean(auc_values) if auc_values else 0.0,
                "median_auc_swing_rel": statistics.median(auc_values) if auc_values else 0.0,
                "max_auc_swing_rel": max(auc_values) if auc_values else 0.0,
            }
        )
    return overall_rows

def write_conclusion_md(
    pair_rows: list[dict[str, Any]],
    overall_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not pair_rows:
        output_path.write_text(
            "# EG1 Conclusion\n\nNo successful EG1 jobs were found yet, so no conclusion can be drawn.\n",
            encoding="utf-8",
        )
        return

    overall = next((row for row in overall_rows if row["group"] == "all"), None)
    top_pairs = sorted(pair_rows, key=lambda row: float(row["ap_swing_rel"]), reverse=True)[:5]
    lines = [
        "# EG1 Conclusion",
        "",
        "## Overall",
    ]
    if overall:
        lines.append(
            (
                f"Across {int(overall['pair_count'])} detector-dataset pairs, "
                f"{int(overall['sensitive_ap_pairs'])} pairs exceed the "
                f"{SENSITIVITY_THRESHOLD:.0%} relative AP swing threshold."
            )
        )
        lines.append(
            (
                f"Mean relative AP swing is {float(overall['mean_ap_swing_rel']):.1%}; "
                f"mean relative AUC swing is {float(overall['mean_auc_swing_rel']):.1%}."
            )
        )
    lines.extend(
        [
            "",
            "## Highest AP Swing Pairs",
        ]
    )
    for row in top_pairs:
        lines.append(
            (
                f"- {row['detector']} / {row['dataset']}: "
                f"AP swing {float(row['ap_swing_rel']):.1%}, "
                f"best window {row['best_window_label']}, "
                f"default observed={row['default_window_observed']}"
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_line_plot_svg(
    summary_rows: list[dict[str, Any]],
    *,
    metric: str,
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    datasets = dataset_list()
    if not summary_rows:
        output_path.write_text(_empty_svg(title, "No successful jobs yet."), encoding="utf-8")
        return

    panel_width = 320
    panel_height = 220
    cols = 3
    rows_needed = max(1, math.ceil(len(datasets) / cols))
    width = panel_width * cols
    height = 60 + panel_height * rows_needed
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:monospace;font-size:12px}.title{font-size:18px;font-weight:bold}.axis{stroke:#444;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.label{fill:#222}.legend{font-size:11px}</style>',
        f'<text class="title" x="20" y="28">{title}</text>',
        '<text class="label" x="20" y="46">y-axis fixed to [0, 1] for comparability across datasets.</text>',
    ]

    for dataset_index, dataset in enumerate(datasets):
        x0 = (dataset_index % cols) * panel_width
        y0 = 70 + (dataset_index // cols) * panel_height
        panel_rows = [row for row in summary_rows if str(row["dataset"]) == dataset]
        x_min = min((int(row["window_seconds"]) for row in panel_rows), default=0)
        x_max = max((int(row["window_seconds"]) for row in panel_rows), default=1)
        if x_min == x_max:
            x_max = x_min + 1
        parts.extend(
            [
                f'<text class="label" x="{x0 + 16}" y="{y0 + 16}">{dataset}</text>',
                f'<line class="axis" x1="{x0 + 40}" y1="{y0 + 180}" x2="{x0 + 280}" y2="{y0 + 180}"/>',
                f'<line class="axis" x1="{x0 + 40}" y1="{y0 + 180}" x2="{x0 + 40}" y2="{y0 + 30}"/>',
            ]
        )
        for tick in range(6):
            y = y0 + 180 - tick * 30
            parts.append(f'<line class="grid" x1="{x0 + 40}" y1="{y}" x2="{x0 + 280}" y2="{y}"/>')
            parts.append(f'<text class="legend" x="{x0 + 8}" y="{y + 4}">{tick/5:.1f}</text>')
        if not panel_rows:
            parts.append(f'<text class="legend" x="{x0 + 70}" y="{y0 + 110}">no data</text>')
            continue
        for detector in detector_list():
            detector_rows = sorted(
                [row for row in panel_rows if str(row["detector"]) == detector],
                key=lambda row: int(row["window_seconds"]),
            )
            if not detector_rows:
                continue
            points = []
            for row in detector_rows:
                x_val = int(row["window_seconds"])
                y_val = max(0.0, min(1.0, float(row[metric])))
                x = x0 + 40 + ((x_val - x_min) / (x_max - x_min)) * 240
                y = y0 + 180 - y_val * 150
                points.append(f"{x:.1f},{y:.1f}")
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{SVG_COLORS[detector]}"/>'
                )
                parts.append(
                    f'<text class="legend" x="{x - 12:.1f}" y="{y0 + 198}">{format_window_label(x_val)}</text>'
                )
            parts.append(
                f'<polyline fill="none" stroke="{SVG_COLORS[detector]}" stroke-width="2" points="{" ".join(points)}"/>'
            )
        legend_y = y0 + 210
        for legend_index, detector in enumerate(detector_list()):
            lx = x0 + 16 + legend_index * 72
            parts.append(f'<line x1="{lx}" y1="{legend_y}" x2="{lx + 18}" y2="{legend_y}" stroke="{SVG_COLORS[detector]}" stroke-width="2"/>')
            parts.append(f'<text class="legend" x="{lx + 22}" y="{legend_y + 4}">{detector}</text>')
    parts.append("</svg>")
    output_path.write_text("".join(parts), encoding="utf-8")

def write_heatmap_svg(
    pair_rows: list[dict[str, Any]],
    *,
    metric_key: str,
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not pair_rows:
        output_path.write_text(_empty_svg(title, "No successful jobs yet."), encoding="utf-8")
        return

    datasets = dataset_list()
    detectors = detector_list()
    width = 140 + 120 * len(datasets)
    height = 120 + 50 * len(detectors)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:monospace;font-size:12px}.title{font-size:18px;font-weight:bold}.celltext{font-size:11px}</style>',
        f'<text class="title" x="20" y="28">{title}</text>',
    ]
    lookup = {(str(row["detector"]), str(row["dataset"])): row for row in pair_rows}
    for col, dataset in enumerate(datasets):
        parts.append(f'<text x="{170 + col * 120}" y="56">{dataset}</text>')
    for row_index, detector in enumerate(detectors):
        y = 80 + row_index * 50
        parts.append(f'<text x="20" y="{y + 18}">{detector}</text>')
        for col, dataset in enumerate(datasets):
            x = 140 + col * 120
            row = lookup.get((detector, dataset))
            value = float(row[metric_key]) if row else 0.0
            shade = 255 - int(min(1.0, value) * 155)
            fill = f"rgb(255,{shade},{shade})"
            parts.append(f'<rect x="{x}" y="{y}" width="96" height="34" fill="{fill}" stroke="#999"/>')
            label = f"{value:.1%}" if row else "n/a"
            parts.append(f'<text class="celltext" x="{x + 24}" y="{y + 21}">{label}</text>')
    parts.append("</svg>")
    output_path.write_text("".join(parts), encoding="utf-8")

def _empty_svg(title: str, message: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="180" viewBox="0 0 800 180">'
        '<style>text{font-family:monospace;font-size:14px}.title{font-size:18px;font-weight:bold}</style>'
        f'<text class="title" x="20" y="30">{title}</text>'
        f'<text x="20" y="70">{message}</text>'
        "</svg>"
    )

def same_job_set(left: Iterable[EG1Job], right: Iterable[EG1Job]) -> bool:
    left_ids = {job.experiment_id for job in left}
    right_ids = {job.experiment_id for job in right}
    return left_ids == right_ids
