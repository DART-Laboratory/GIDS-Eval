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
from pathlib import Path
from typing import Any, Iterable

import yaml

from gids_eval.paths import get_paths
from gids_eval.status_codes import UNSUPPORTED_EXIT_CODE

_PATHS = get_paths()
REPO_ROOT = _PATHS.repo_root
EG1_CODE_ROOT = REPO_ROOT / "EG1"
EG2_CODE_ROOT = REPO_ROOT / "EG2"
DATA_ROOT = _PATHS.sc_data_root("eg2")
GENERATED_CONFIG_ROOT = DATA_ROOT / "generated-configs"
ORCHESTRATOR_ROOT = DATA_ROOT / "orchestrator"
REPORT_ROOT = DATA_ROOT / "reports"
SELECTION_REPORT_ROOT = REPORT_ROOT / "selection"
RUNTIME_ROOT = REPO_ROOT / "local_runtime"
RUN_LABEL = "eg2"
CONDA_ROOT = _PATHS.conda_root
MINIFORGE_ROOT = CONDA_ROOT / "miniforge3"
PROGRESS_JSON_PATH = ORCHESTRATOR_ROOT / "progress.json"
PROGRESS_MD_PATH = ORCHESTRATOR_ROOT / "progress.md"
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
}

DEFAULT_SEEDS = tuple(range(10))

DETECTORS = ("argus", "euler")

SVG_COLORS = {
    "argus": "#d1495b",
    "euler": "#2e6f95",
}

DETECTOR_ENV_PREFIXES = {
    "argus": CONDA_ROOT / "envs/sc1-argus",
    "euler": CONDA_ROOT / "envs/sc1-euler",
}

DETECTOR_REQUIRED_IMPORTS = {
    "argus": ("numpy", "pandas", "sklearn", "torch", "torch_geometric", "networkx", "tqdm", "yaml"),
    "euler": ("numpy", "pandas", "sklearn", "torch", "torch_geometric", "networkx", "tqdm", "yaml"),
}

BASE_CONFIGS = {
    "argus": REPO_ROOT / "configs/experiments/argus_pipeline.yaml",
    "euler": REPO_ROOT / "configs/experiments/euler_pipeline.yaml",
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

DATASETS = ("lanl", "optc", "cic", "0501")

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

VARIANCE_METRICS = (
    "threshold", "tpr", "fpr", "precision", "recall", "f1", "ap", "auc",
    "cdr", "cdr_auc", "campaign_count", "detected_campaigns",
)

INSTABILITY_CV_THRESHOLD = 0.10

@dataclass(frozen=True)
class EG2Job:
    detector: str
    dataset: str
    window_seconds: int
    seed: int

    @property
    def experiment_id(self) -> str:
        return f"eg2_{self.detector}_{self.dataset}_s{self.seed:02d}"

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

def ensure_eg2_dirs() -> None:
    for path in (GENERATED_CONFIG_ROOT, ORCHESTRATOR_ROOT, REPORT_ROOT, SELECTION_REPORT_ROOT):
        path.mkdir(parents=True, exist_ok=True)

def detector_env_prefix(detector: str) -> Path:
    override = os.environ.get(f"EG2_ENV_{detector.upper()}")
    if override:
        return Path(override)
    return DETECTOR_ENV_PREFIXES[detector]

def detector_env_python(detector: str) -> Path:
    return detector_env_prefix(detector) / "bin/python"

def detector_list() -> list[str]:
    return list(DETECTORS)

def dataset_list() -> list[str]:
    return list(DATASETS)

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
) -> list[EG2Job]:
    selected_detectors = list(detectors or detector_list())
    selected_datasets = list(datasets or dataset_list())
    selected_seeds = list(seeds or DEFAULT_SEEDS)
    jobs: list[EG2Job] = []
    for detector in selected_detectors:
        if detector not in BASE_CONFIGS:
            raise ValueError(f"Unknown detector: {detector}")
        for dataset in selected_datasets:
            window = DEFAULT_WINDOW_SECONDS.get(detector, {}).get(dataset)
            if window is None:
                continue
            for seed in selected_seeds:
                jobs.append(EG2Job(
                    detector=detector,
                    dataset=dataset,
                    window_seconds=window,
                    seed=seed,
                ))
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

def _format_window_label(seconds: int) -> str:
    if seconds % 3600 == 0:
        return f"{seconds / 3600:g}h"
    if seconds % 60 == 0:
        return f"{seconds / 60:g}m"
    return f"{seconds}s"

def _format_window_token(seconds: int) -> str:
    return _format_window_label(seconds).replace(".", "p")

def _detector_window_value(detector: str, dataset: str, window_seconds: int) -> str:
    if dataset in {"lanl", "optc"}:
        value = window_seconds / 3600
    else:
        value = window_seconds / 60
    return f"{value:g}"

def _replace_flag(command: list[Any], flag: str, value: str) -> list[str]:
    resolved = [str(part) for part in command]
    for index, part in enumerate(resolved):
        if part == flag:
            if index + 1 >= len(resolved):
                raise ValueError(f"Flag {flag} missing value in command: {resolved}")
            resolved[index + 1] = value
            return resolved
    raise ValueError(f"Flag {flag} not found in command: {resolved}")

def _select_detection_job(
    jobs: list[dict[str, Any]],
    target_job_id: str,
    label: str,
) -> dict[str, Any]:
    matching = [deepcopy(item) for item in jobs if item["id"] == target_job_id]
    if len(matching) != 1:
        raise ValueError(f"Expected one detection job for {label}, found {len(matching)}")
    return matching[0]

def _replace_job_dataset_arg(command: list[Any], dataset_value: str) -> list[str]:
    return _replace_flag(command, "--dataset", dataset_value)

def _ensure_flag(command: list[Any], flag: str, value: str) -> list[str]:
    resolved = [str(part) for part in command]
    for index, part in enumerate(resolved):
        if part == flag:
            if index + 1 >= len(resolved):
                raise ValueError(f"Flag {flag} missing value in command: {resolved}")
            resolved[index + 1] = value
            return resolved
    return [*resolved, flag, value]

def _tag_eg2_job_env(
    env: dict[str, Any],
    *,
    euler_local_mode: bool = False,
) -> dict[str, Any]:
    tagged = dict(env)
    if euler_local_mode:
        tagged.setdefault("EG1_EULER_LOCAL_MODE", "1")
    return tagged

def _customize_0501_graph_featurization(payload: dict[str, Any], job: EG2Job) -> None:
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

def _customize_runtime_window_detector(payload: dict[str, Any], job: EG2Job) -> None:
    detection = payload["modules"]["detection"]

    if job.dataset == "0501":
        _customize_0501_graph_featurization(payload, job)
        template_job_id = DETECTOR_DATASET_JOB_IDS[job.detector][job.dataset]
        selected_job = _select_detection_job(detection["jobs"], template_job_id, f"{job.detector}:{job.dataset}")
        selected_job["dataset"] = job.dataset
        selected_job["command"] = _replace_job_dataset_arg(
            selected_job["command"],
            "O_0501" if job.detector == "argus" else "0501",
        )
        command = _replace_flag(
            selected_job["command"],
            "--delta",
            _detector_window_value(job.detector, job.dataset, job.window_seconds),
        )
        if job.detector == "argus":
            command = _ensure_flag(command, "--te_end", "all")
        selected_job["command"] = [command[0], "${repo_root}/EG1/legacy_entrypoint.py", *command[1:]]
        selected_job["env"] = _tag_eg2_job_env(
            selected_job.get("env", {}),
            euler_local_mode=(job.detector == "euler"),
        )
        if job.detector == "argus":
            selected_job["env"]["ARGUS_RUN_NAME"] = "0501"
            selected_job["metrics_relpath"] = "results/0501/metrics.csv"
        else:
            selected_job["metrics_relpath"] = "results/0501/0501/learning/metrics.csv"
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

    target_job_id = DETECTOR_DATASET_JOB_IDS[job.detector][job.dataset]
    selected_job = _select_detection_job(detection["jobs"], target_job_id, f"{job.detector}:{job.dataset}")
    command = _replace_flag(
        selected_job["command"],
        "--delta",
        _detector_window_value(job.detector, job.dataset, job.window_seconds),
    )
    if job.detector == "argus":
        command = _ensure_flag(command, "--te_end", "all")
    selected_job["command"] = [command[0], "${repo_root}/EG1/legacy_entrypoint.py", *command[1:]]
    selected_job["env"] = _tag_eg2_job_env(
        selected_job.get("env", {}),
        euler_local_mode=(job.detector == "euler"),
    )
    detection["jobs"] = [selected_job]

def generate_config(job: EG2Job) -> Path:
    ensure_eg2_dirs()
    payload = load_yaml(BASE_CONFIGS[job.detector])
    generated = deepcopy(payload)
    generated["name"] = f"EG2 {job.detector.upper()} {job.dataset} seed {job.seed}"
    generated["experiment_id"] = job.experiment_id
    generated["run_label"] = RUN_LABEL
    generated["seed"] = job.seed
    generated.setdefault("runtime", {})
    generated["runtime"]["repo_root"] = str(REPO_ROOT)
    generated["runtime"]["runtime_root"] = str(RUNTIME_ROOT)
    generated["runtime"]["data_root"] = str(DATA_ROOT)
    generated["manifests"] = [str(path) for path in all_manifest_paths()]
    env_python = detector_env_python(job.detector)
    if env_python.exists():
        generated["python_bin"] = str(env_python)

    for stage_name in ("graph_construction", "featurization", "graph_encoder",
                       "temporal_encoder", "loss"):
        stage_cfg = generated["modules"].get(stage_name, {})
        stage_cfg.setdefault("cache", {})["seed_independent"] = True

    _customize_runtime_window_detector(generated, job)

    dump_yaml(job.config_path, generated)
    return job.config_path

def run_job(job: EG2Job, *, dry_run: bool = False, force: bool = False) -> str:
    config_path = generate_config(job)
    if not force:
        status = job_status(job)
        if status == "success":
            return "skipped"
    env_python = detector_env_python(job.detector)
    if not env_python.exists() and not dry_run:
        print(
            f"[EG2] missing detector environment for {job.detector}: expected {env_python}",
            file=sys.stderr,
        )
        return "failed"
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [
        str(EG1_CODE_ROOT.resolve()),
        str(EG2_CODE_ROOT.resolve()),
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
    return "success" if completed.returncode == 0 else "failed"

def job_status(job: EG2Job) -> str:
    cost_rows = read_csv(job.cost_metrics_path)
    baseline_rows = read_csv(job.baseline_metrics_path)
    if cost_rows and baseline_rows:
        detection_rows = [row for row in cost_rows if row.get("phase") == "detection"]
        if detection_rows and all(row.get("status") == "unsupported" for row in detection_rows):
            return "unsupported"
        if detection_rows and all(row.get("status") in {"success", "cache_hit"} for row in detection_rows):
            return "success"
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

def build_status_rows(jobs: list[EG2Job]) -> list[dict[str, Any]]:
    rows = []
    for job in jobs:
        rows.append({
            "detector": job.detector,
            "dataset": job.dataset,
            "window_seconds": job.window_seconds,
            "seed": job.seed,
            "experiment_id": job.experiment_id,
            "status": job_status(job),
            "config_path": str(job.config_path),
            "baseline_metrics_path": str(job.baseline_metrics_path),
            "cost_metrics_path": str(job.cost_metrics_path),
            "run_summary_path": str(job.run_summary_path),
        })
    return rows

def write_matrix_report(jobs: list[EG2Job], path: Path | None = None) -> Path:
    rows = []
    for job in jobs:
        rows.append({
            "detector": job.detector,
            "dataset": job.dataset,
            "window_seconds": job.window_seconds,
            "seed": job.seed,
            "experiment_id": job.experiment_id,
            "config_path": str(job.config_path),
        })
    path = path or (ORCHESTRATOR_ROOT / "job_matrix.csv")
    write_csv(path, rows, ["detector", "dataset", "window_seconds", "seed", "experiment_id", "config_path"])
    return path

def write_status_report(jobs: list[EG2Job], path: Path | None = None) -> Path:
    rows = build_status_rows(jobs)
    path = path or (ORCHESTRATOR_ROOT / "job_status.csv")
    write_csv(
        path,
        rows,
        ["detector", "dataset", "window_seconds", "seed", "experiment_id", "status",
         "config_path", "baseline_metrics_path", "cost_metrics_path", "run_summary_path"],
    )
    return path

def write_progress_report(
    jobs: list[EG2Job],
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
        dt = len(detector_rows)
        ds = sum(1 for r in detector_rows if r["status"] == "success")
        df = sum(1 for r in detector_rows if r["status"] == "failed")
        du = sum(1 for r in detector_rows if r["status"] == "unsupported")
        dp = sum(1 for r in detector_rows if r["status"] == "partial")
        dn = sum(1 for r in detector_rows if r["status"] == "pending")
        detector_progress.append({
            "detector": detector, "total": dt, "success": ds, "failed": df,
            "unsupported": du, "partial": dp, "pending": dn,
            "completed": ds + df + du,
            "completion_ratio": (ds + df + du) / dt if dt else 0.0,
        })

    partial_jobs_list = [row["experiment_id"] for row in rows if row["status"] == "partial"]
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
        "partial_experiment_ids": partial_jobs_list,
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
        f"# EG2 Progress ({scope_name})",
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
            f"- {row['detector']}: {row['completed']}/{row['total']} complete "
            f"(success={row['success']}, failed={row['failed']}, unsupported={row['unsupported']}, "
            f"partial={row['partial']}, pending={row['pending']})"
        )
    if unsupported_job_ids:
        lines.extend(["", "## Unsupported Jobs"])
        for eid in unsupported_job_ids[:20]:
            lines.append(f"- {eid}")
    if partial_jobs_list:
        lines.extend(["", "## Partial Jobs"])
        for eid in partial_jobs_list[:20]:
            lines.append(f"- {eid}")
    if failed_job_ids:
        lines.extend(["", "## Failed Jobs"])
        for eid in failed_job_ids[:20]:
            lines.append(f"- {eid}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path

def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}

def _to_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)

def _optional_float(value: Any) -> float | str:
    if value in (None, ""):
        return ""
    return float(value)

def _numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        v = row.get(key)
        if v in (None, ""):
            continue
        values.append(float(v))
    return values

def _aggregate_metric(rows: list[dict[str, Any]], key: str) -> tuple[float | str, float | str]:
    values = _numeric_values(rows, key)
    if not values:
        return "", ""
    if len(values) < 2:
        return statistics.mean(values), 0.0
    return statistics.mean(values), statistics.stdev(values)

def _cv(mean: float, std: float) -> float | str:
    if mean == 0.0:
        return "" if std == 0.0 else "inf"
    return abs(std / mean)

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

def _phase_row_lookup(cost_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in cost_rows:
        phase = str(row.get("phase", "")).strip()
        if phase:
            lookup[phase] = row
    return lookup

def summarize_jobs(jobs: list[EG2Job], output_root: Path | None = None) -> tuple[Path, Path]:
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
        peak_values = [v for v in peak_values if v != ""]
        run_peak_rss_kb = max(peak_values) if peak_values else ""
        run_mean_rss_kb = _weighted_mean_from_rows(cost_rows, value_key="mean_rss_kb", weight_key="sample_count")
        run_mean_cpu_percent = _weighted_mean_from_rows(
            cost_rows,
            value_key="mean_cpu_percent",
            weight_key="cpu_sample_count",
        )
        for metrics_row in read_csv(job.baseline_metrics_path):
            reported_dataset = canonical_dataset(str(metrics_row.get("dataset", job.dataset)))
            raw_rows.append({
                "detector": job.detector,
                "dataset": job.dataset,
                "reported_dataset": reported_dataset,
                "campaign_mode": DATASET_CAMPAIGN_MODES.get(job.dataset, "unknown"),
                "window_seconds": job.window_seconds,
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
            })

    raw_fieldnames = [
        "detector", "dataset", "reported_dataset", "campaign_mode",
        "window_seconds", "seed",
        "experiment_id", "auc", "ap", "threshold", "tpr", "fpr",
        "precision", "recall", "f1", "cdr", "cdr_auc",
        "campaign_count", "detected_campaigns", "cdr_meaningful",
        "train_time", "test_time", "detection_status",
        "featurization_wall_time_sec", "detection_wall_time_sec", "total_wall_time_sec",
        "featurization_peak_rss_kb", "detection_peak_rss_kb", "run_peak_rss_kb",
        "featurization_mean_rss_kb", "detection_mean_rss_kb", "run_mean_rss_kb",
        "featurization_peak_rss_gib", "detection_peak_rss_gib", "run_peak_rss_gib",
        "featurization_mean_rss_gib", "detection_mean_rss_gib", "run_mean_rss_gib",
        "featurization_mean_cpu_percent", "detection_mean_cpu_percent", "run_mean_cpu_percent",
        "metrics_path",
    ]
    raw_path = output_root / "raw_results.csv"
    write_csv(raw_path, raw_rows, raw_fieldnames)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        key = (str(row["detector"]), str(row["dataset"]))
        grouped.setdefault(key, []).append(row)

    variance_rows: list[dict[str, Any]] = []
    for (detector, dataset), rows in sorted(grouped.items()):
        vrow: dict[str, Any] = {
            "detector": detector,
            "dataset": dataset,
            "campaign_mode": DATASET_CAMPAIGN_MODES.get(dataset, "unknown"),
            "window_seconds": rows[0]["window_seconds"],
            "seed_count": len(rows),
            "cdr_meaningful": all(bool(row.get("cdr_meaningful")) for row in rows),
        }
        for metric in VARIANCE_METRICS:
            mean_val, std_val = _aggregate_metric(rows, metric)
            vrow[f"mean_{metric}"] = mean_val
            vrow[f"std_{metric}"] = std_val
            values = _numeric_values(rows, metric)
            vrow[f"min_{metric}"] = min(values) if values else ""
            vrow[f"max_{metric}"] = max(values) if values else ""
            vrow[f"range_{metric}"] = (max(values) - min(values)) if values else ""
            if isinstance(mean_val, (int, float)) and isinstance(std_val, (int, float)):
                vrow[f"cv_{metric}"] = _cv(mean_val, std_val)
            else:
                vrow[f"cv_{metric}"] = ""
        variance_rows.append(vrow)

    variance_fieldnames = ["detector", "dataset", "campaign_mode", "window_seconds", "seed_count", "cdr_meaningful"]
    for metric in VARIANCE_METRICS:
        variance_fieldnames.extend([
            f"mean_{metric}", f"std_{metric}", f"min_{metric}", f"max_{metric}",
            f"range_{metric}", f"cv_{metric}",
        ])
    variance_path = output_root / "variance_by_group.csv"
    write_csv(variance_path, variance_rows, variance_fieldnames)

    evidence_rows = _build_evidence_rows(variance_rows)
    evidence_path = output_root / "evidence_table.csv"
    write_csv(evidence_path, evidence_rows, [
        "detector", "dataset", "campaign_mode", "seed_count",
        "cdr_meaningful",
        "unstable_metric_count", "unstable_metrics",
        "max_cv", "max_cv_metric",
        "max_range_tpr", "max_range_fpr", "max_range_precision",
        "max_range_threshold",
        "max_range_cdr", "max_range_detected_campaigns",
        "verdict",
    ])

    _write_box_plot_svg(raw_rows, output_root / "plot_threshold_spread.svg")
    _write_cv_heatmap_svg(variance_rows, output_root / "plot_cv_heatmap.svg")

    _write_conclusion_md(variance_rows, evidence_rows, output_root / "conclusion.md")

    return raw_path, variance_path

def _build_evidence_rows(variance_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for vrow in variance_rows:
        unstable: list[str] = []
        max_cv = 0.0
        max_cv_metric = ""
        for metric in VARIANCE_METRICS:
            cv = vrow.get(f"cv_{metric}", "")
            if cv in ("", "inf"):
                if cv == "inf":
                    unstable.append(metric)
                    max_cv = float("inf")
                    max_cv_metric = metric
                continue
            cv_val = float(cv)
            if cv_val > INSTABILITY_CV_THRESHOLD:
                unstable.append(metric)
            if cv_val > max_cv:
                max_cv = cv_val
                max_cv_metric = metric

        verdict = "unstable" if unstable else "stable"
        evidence.append({
            "detector": vrow["detector"],
            "dataset": vrow["dataset"],
            "campaign_mode": vrow.get("campaign_mode", "unknown"),
            "seed_count": vrow["seed_count"],
            "cdr_meaningful": vrow.get("cdr_meaningful", False),
            "unstable_metric_count": len(unstable),
            "unstable_metrics": ";".join(unstable),
            "max_cv": max_cv,
            "max_cv_metric": max_cv_metric,
            "max_range_tpr": vrow.get("range_tpr", ""),
            "max_range_fpr": vrow.get("range_fpr", ""),
            "max_range_precision": vrow.get("range_precision", ""),
            "max_range_threshold": vrow.get("range_threshold", ""),
            "max_range_cdr": vrow.get("range_cdr", ""),
            "max_range_detected_campaigns": vrow.get("range_detected_campaigns", ""),
            "verdict": verdict,
        })
    return evidence

def _write_box_plot_svg(raw_rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_rows:
        output_path.write_text(_empty_svg("EG2 Threshold Spread", "No successful jobs yet."), encoding="utf-8")
        return

    groups: dict[tuple[str, str], list[float]] = {}
    for row in raw_rows:
        key = (str(row["detector"]), str(row["dataset"]))
        groups.setdefault(key, []).append(float(row["threshold"]))

    sorted_keys = sorted(groups.keys())
    n = len(sorted_keys)
    panel_w = 120
    width = 100 + panel_w * n
    height = 280
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:monospace;font-size:11px}.title{font-size:16px;font-weight:bold}'
        '.label{fill:#222}.tick{stroke:#888;stroke-width:1}</style>',
        '<text class="title" x="20" y="24">EG2 Threshold Spread Across Seeds</text>',
    ]

    y_top = 50
    y_bot = 240
    plot_h = y_bot - y_top

    all_vals = [v for vals in groups.values() for v in vals]
    t_min = min(all_vals) if all_vals else 0
    t_max = max(all_vals) if all_vals else 1
    if t_min == t_max:
        t_min -= 0.5
        t_max += 0.5
    margin = (t_max - t_min) * 0.1
    t_min -= margin
    t_max += margin

    def y_pos(v: float) -> float:
        return y_bot - ((v - t_min) / (t_max - t_min)) * plot_h

    for i in range(5):
        frac = i / 4
        val = t_min + frac * (t_max - t_min)
        y = y_pos(val)
        parts.append(f'<line class="tick" x1="70" y1="{y:.1f}" x2="{70 + panel_w * n}" y2="{y:.1f}" stroke-dasharray="3,3"/>')
        parts.append(f'<text class="label" x="10" y="{y + 4:.1f}">{val:.3f}</text>')

    for idx, key in enumerate(sorted_keys):
        vals = sorted(groups[key])
        cx = 90 + idx * panel_w + panel_w // 2
        color = SVG_COLORS.get(key[0], "#666")

        q1 = _percentile(vals, 25)
        q3 = _percentile(vals, 75)
        med = _percentile(vals, 50)
        lo = min(vals)
        hi = max(vals)
        bw = 40
        y_q1 = y_pos(q1)
        y_q3 = y_pos(q3)
        y_med = y_pos(med)
        y_lo = y_pos(lo)
        y_hi = y_pos(hi)

        parts.append(f'<line x1="{cx}" y1="{y_hi:.1f}" x2="{cx}" y2="{y_q3:.1f}" stroke="{color}" stroke-width="1"/>')
        parts.append(f'<line x1="{cx}" y1="{y_q1:.1f}" x2="{cx}" y2="{y_lo:.1f}" stroke="{color}" stroke-width="1"/>')

        box_y = min(y_q1, y_q3)
        box_h = abs(y_q1 - y_q3)
        parts.append(f'<rect x="{cx - bw // 2}" y="{box_y:.1f}" width="{bw}" height="{max(box_h, 1):.1f}" fill="{color}" fill-opacity="0.25" stroke="{color}" stroke-width="1.5"/>')

        parts.append(f'<line x1="{cx - bw // 2}" y1="{y_med:.1f}" x2="{cx + bw // 2}" y2="{y_med:.1f}" stroke="{color}" stroke-width="2"/>')

        for v in vals:
            parts.append(f'<circle cx="{cx}" cy="{y_pos(v):.1f}" r="2.5" fill="{color}" fill-opacity="0.6"/>')

        parts.append(f'<text class="label" x="{cx - 30}" y="{y_bot + 16}">{key[0][:3]}/{key[1]}</text>')

    parts.append("</svg>")
    output_path.write_text("".join(parts), encoding="utf-8")

def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * pct / 100.0
    f = int(k)
    c = f + 1
    if c >= len(sorted_vals):
        return sorted_vals[-1]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])

def _write_cv_heatmap_svg(variance_rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not variance_rows:
        output_path.write_text(_empty_svg("EG2 CV Heatmap", "No data."), encoding="utf-8")
        return

    metrics = list(VARIANCE_METRICS)
    detectors = detector_list()
    datasets = dataset_list()
    row_labels = [f"{d}/{ds}" for d in detectors for ds in datasets]
    width = 160 + 90 * len(metrics)
    height = 80 + 36 * len(row_labels)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:monospace;font-size:11px}.title{font-size:15px;font-weight:bold}.cell{font-size:10px}</style>',
        '<text class="title" x="16" y="22">EG2 Coefficient of Variation by (Detector, Dataset, Metric)</text>',
    ]

    for ci, metric in enumerate(metrics):
        parts.append(f'<text x="{168 + ci * 90}" y="50">{metric}</text>')

    lookup = {(str(r["detector"]), str(r["dataset"])): r for r in variance_rows}
    for ri, (det, ds) in enumerate((d, ds) for d in detectors for ds in datasets):
        y = 60 + ri * 36
        parts.append(f'<text x="10" y="{y + 20}">{det}/{ds}</text>')
        vrow = lookup.get((det, ds))
        for ci, metric in enumerate(metrics):
            x = 150 + ci * 90
            cv = vrow.get(f"cv_{metric}", "") if vrow else ""
            if cv in ("", "inf"):
                fill = "#eee" if cv == "" else "#f99"
                label = "n/a" if cv == "" else "inf"
            else:
                cv_val = float(cv)
                intensity = min(1.0, cv_val / 0.5)
                shade = 255 - int(intensity * 155)
                fill = f"rgb(255,{shade},{shade})"
                label = f"{cv_val:.3f}"
            parts.append(f'<rect x="{x}" y="{y}" width="80" height="30" fill="{fill}" stroke="#ccc"/>')
            parts.append(f'<text class="cell" x="{x + 10}" y="{y + 20}">{label}</text>')

    parts.append("</svg>")
    output_path.write_text("".join(parts), encoding="utf-8")

def _write_conclusion_md(
    variance_rows: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not variance_rows:
        output_path.write_text(
            "# EG2 Conclusion\n\nNo successful EG2 jobs found yet.\n",
            encoding="utf-8",
        )
        return

    unstable_groups = [r for r in evidence_rows if r["verdict"] == "unstable"]
    total = len(evidence_rows)
    lines = [
        "# EG2 Conclusion",
        "",
        "## Summary",
        "",
        f"EG2 ran {total} (detector, dataset) groups with {int(variance_rows[0]['seed_count'])} seeds each.",
        f"{len(unstable_groups)}/{total} groups show threshold-dependent metric instability "
        f"(CV > {INSTABILITY_CV_THRESHOLD:.0%} on at least one metric).",
        "",
    ]

    if unstable_groups:
        lines.append("## Unstable Groups")
        lines.append("")
        for r in unstable_groups:
            cdr_info = ""
            if r.get("max_range_cdr") not in (None, ""):
                cdr_info = f", CDR range = {_fmt(r['max_range_cdr'])}"
            if r.get("max_range_detected_campaigns") not in (None, ""):
                cdr_info += f", detected_campaigns range = {_fmt(r['max_range_detected_campaigns'])}"
            lines.append(
                f"- **{r['detector']}/{r['dataset']}** (campaigns: {r.get('campaign_mode', 'unknown')}): "
                f"unstable on {r['unstable_metrics']} "
                f"(max CV = {float(r['max_cv']):.3f} on {r['max_cv_metric']}; "
                f"TPR range = {_fmt(r['max_range_tpr'])}, "
                f"FPR range = {_fmt(r['max_range_fpr'])}, "
                f"precision range = {_fmt(r['max_range_precision'])}"
                f"{cdr_info})"
            )
        lines.append("")

    cdr_unstable = [r for r in evidence_rows if "cdr" in str(r.get("unstable_metrics", ""))]
    if cdr_unstable:
        lines.append("## Campaign Detection Impact")
        lines.append("")
        lines.append(
            f"{len(cdr_unstable)}/{total} groups show CDR instability across seeds, "
            "meaning threshold randomness changes which attack campaigns are detected."
        )
        lines.append("")
        for r in cdr_unstable:
            lines.append(
                f"- **{r['detector']}/{r['dataset']}**: "
                f"CDR range = {_fmt(r.get('max_range_cdr'))}, "
                f"detected campaigns range = {_fmt(r.get('max_range_detected_campaigns'))}"
            )
        lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "The variance observed across seeds with identical configurations demonstrates that "
        "random validation sampling for threshold selection introduces non-trivial noise into "
        "threshold-dependent metrics (TPR, FPR, precision, recall, F1) and campaign-level "
        "detection rates (CDR). Threshold-free metrics (AUC, AP) should remain stable across "
        "seeds because they do not depend on the chosen operating point. Campaign-level "
        "metrics (CDR, detected_campaigns) are particularly important: threshold variance "
        "can cause entire attack campaigns to flip between detected and missed.",
        "",
        "## Recommendation",
        "",
        "Threshold computation must be deterministic. Use a fixed, chronologically separated "
        "validation set rather than randomly sampling events per snapshot. This eliminates "
        "threshold-selection variance without changing the underlying model.",
        "",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")

def _fmt(value: Any) -> str:
    if value in (None, ""):
        return "n/a"
    return f"{float(value):.4f}"

def _empty_svg(title: str, message: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="180" viewBox="0 0 800 180">'
        '<style>text{font-family:monospace;font-size:14px}.title{font-size:18px;font-weight:bold}</style>'
        f'<text class="title" x="20" y="30">{title}</text>'
        f'<text x="20" y="70">{message}</text>'
        "</svg>"
    )

def detector_env_status(detectors: Iterable[str] | None = None) -> tuple[Path, list[dict[str, Any]]]:
    ensure_eg2_dirs()
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
        path, rows,
        ["detector", "env_prefix", "python_path", "env_exists", "python_exists",
         "imports_ok", "missing_imports", "note"],
    )
    return path, rows

def same_job_set(left: Iterable[EG2Job], right: Iterable[EG2Job]) -> bool:
    return {j.experiment_id for j in left} == {j.experiment_id for j in right}
