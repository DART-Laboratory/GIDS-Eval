
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

_PATHS = get_paths()
REPO_ROOT = _PATHS.repo_root
EG3_CODE_ROOT = REPO_ROOT / "EG3"
DATA_ROOT = _PATHS.sc_data_root("eg3")
GENERATED_CONFIG_ROOT = DATA_ROOT / "generated-configs"
ORCHESTRATOR_ROOT = DATA_ROOT / "orchestrator"
REPORT_ROOT = DATA_ROOT / "reports"
MIRAGE_ROOT = _PATHS.mirage_official_root
CONDA_ROOT = _PATHS.conda_root
DEFAULT_EG1_RAW_RESULTS = _PATHS.sc_data_root("eg1") / "reports/raw_results.csv"
RUN_LABEL = "eg3"
DEFAULT_SEEDS = (0,)
PROGRESS_JSON_PATH = ORCHESTRATOR_ROOT / "progress.json"
PROGRESS_MD_PATH = ORCHESTRATOR_ROOT / "progress.md"

CALIBRATION_MODES = ("transductive", "inductive")

DATASET_WINDOW_SECONDS: dict[str, int] = {
    "lanl": 3600,
    "optc": 3600,
    "cic": 3000,
    "0501": 300,
}

INDUCTIVE_VAL_FRACTION = 0.5

DATASETS = list(DATASET_WINDOW_SECONDS)

PIKACHU_BASE_ARGS: dict[str, str] = {
    "dimensions": "100",
    "alpha": "0.01",
    "support": "10",
    "gpu": "-1",
}

PIKACHU_DATASET_ARGS: dict[str, str] = {
    "0501": "0501",
    "lanl": "lanl",
    "optc": "optc",
    "cic": "cic",
}

DETECTOR_ENV_PREFIX = CONDA_ROOT / "envs/sc1-pikachu"
PIKACHU_PREPROCESS_SEED = 1337
PIKACHU_NEG_RATIO = 20

BASE_CONFIG = REPO_ROOT / "configs/experiments/pikachu_pipeline.yaml"

COMPONENT_MANIFESTS: dict[str, Path] = {
    "lanl": REPO_ROOT / "configs/data/lanl.yaml",
    "optc": REPO_ROOT / "configs/data/optc.yaml",
    "cic": REPO_ROOT / "configs/data/cic.yaml",
    "oilrig": REPO_ROOT / "configs/data/0501_oilrig.yaml",
    "sandworm": REPO_ROOT / "configs/data/0501_sandworm.yaml",
    "wizard_spider": REPO_ROOT / "configs/data/0501_wizard_spider.yaml",
}

COMBINED_DATASET_COMPONENTS: dict[str, tuple[str, ...]] = {
    "0501": ("oilrig", "sandworm", "wizard_spider"),
}

DETECTOR_DATASET_JOB_IDS: dict[str, str] = {
    "lanl": "lanl",
    "optc": "optc",
    "cic": "cic",
    "0501": "oilrig",
}

SVG_COLORS: dict[str, str] = {
    "transductive": "#d1495b",
    "inductive": "#2e6f95",
}

@dataclass(frozen=True)
class EG3Job:
    dataset: str
    calibration: str
    seed: int

    @property
    def experiment_id(self) -> str:
        return f"eg3_pikachu_{self.dataset}_{self.calibration}_s{self.seed:02d}"

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

def ensure_eg3_dirs() -> None:
    for path in (GENERATED_CONFIG_ROOT, ORCHESTRATOR_ROOT, REPORT_ROOT):
        path.mkdir(parents=True, exist_ok=True)

def detector_env_python() -> Path:
    override = os.environ.get("EG3_ENV_PIKACHU")
    if override:
        return Path(override) / "bin/python"
    return DETECTOR_ENV_PREFIX / "bin/python"

def eg1_raw_results_path() -> Path:
    override = os.environ.get("EG3_EG1_RAW_RESULTS")
    if override:
        return Path(override)
    return DEFAULT_EG1_RAW_RESULTS

def iter_jobs(
    datasets: Iterable[str] | None = None,
    calibrations: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
) -> list[EG3Job]:
    selected_datasets = list(datasets or DATASETS)
    selected_calibrations = list(calibrations or CALIBRATION_MODES)
    selected_seeds = list(seeds or DEFAULT_SEEDS)
    jobs: list[EG3Job] = []
    for dataset in selected_datasets:
        if dataset not in DATASET_WINDOW_SECONDS:
            raise ValueError(f"Unknown dataset: {dataset}")
        for calibration in selected_calibrations:
            if calibration not in CALIBRATION_MODES:
                raise ValueError(f"Unknown calibration mode: {calibration}")
            for seed in selected_seeds:
                jobs.append(EG3Job(dataset=dataset, calibration=calibration, seed=seed))
    return jobs

def same_job_set(left: Iterable[EG3Job], right: Iterable[EG3Job]) -> bool:
    return {j.experiment_id for j in left} == {j.experiment_id for j in right}

def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload

def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)

def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f, "") for f in fieldnames})

def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]

def manifest_members(dataset: str) -> list[str]:
    return list(COMBINED_DATASET_COMPONENTS.get(dataset, (dataset,)))

def all_manifest_paths() -> list[Path]:
    ordered: list[Path] = []
    seen: set[Path] = set()
    for dataset in DATASETS:
        for member in manifest_members(dataset):
            p = COMPONENT_MANIFESTS[member]
            if p not in seen:
                ordered.append(p)
                seen.add(p)
    return ordered

def generate_config(job: EG3Job) -> Path:
    ensure_eg3_dirs()
    payload = load_yaml(BASE_CONFIG)
    gen = deepcopy(payload)

    gen["name"] = f"EG3 PIKACHU {job.dataset} {job.calibration} seed {job.seed}"
    gen["experiment_id"] = job.experiment_id
    gen["run_label"] = RUN_LABEL
    gen["seed"] = job.seed
    gen.setdefault("runtime", {})
    gen["runtime"]["repo_root"] = str(REPO_ROOT)
    gen["runtime"]["data_root"] = str(DATA_ROOT)

    gen["manifests"] = [str(p) for p in all_manifest_paths()]

    env_python = detector_env_python()
    if env_python.exists():
        gen["python_bin"] = str(env_python)

    window_seconds = DATASET_WINDOW_SECONDS[job.dataset]
    _customize_pikachu(gen, job, window_seconds)

    dump_yaml(job.config_path, gen)
    return job.config_path

def _select_detection_job(
    jobs: list[dict[str, Any]], target_id: str, label: str
) -> dict[str, Any]:
    matches = [deepcopy(j) for j in jobs if j["id"] == target_id]
    if len(matches) != 1:
        raise ValueError(f"Expected 1 detection job for {label}, found {len(matches)}")
    return matches[0]

def _customize_pikachu(payload: dict[str, Any], job: EG3Job, window_seconds: int) -> None:
    featurization = payload["modules"]["featurization"]
    detection = payload["modules"]["detection"]

    if job.dataset == "0501":
        featurization["runner"] = "shell_command"
        featurization["workdir"] = "${repo_root}"
        featurization["env"] = {}
        featurization["command"] = [
            "${python_bin}",
            "${repo_root}/EG3/eg3_build_0501.py",
            "--mode", "pikachu",
            "--raw-root", "${module_output_dir}/data_root/raw_data",
            "--output-dir", "${module_output_dir}/data_root/pikachu_sc3/0501",
            "--snapshot-seconds", str(window_seconds),
            "--neg-ratio", str(PIKACHU_NEG_RATIO),
            "--seed", str(PIKACHU_PREPROCESS_SEED),
        ]
        selected = _select_detection_job(
            detection["jobs"], DETECTOR_DATASET_JOB_IDS[job.dataset], f"pikachu:{job.dataset}"
        )
        selected["dataset"] = "0501"
        selected["workdir"] = "${repo_root}"
        selected["command"] = [
            "${python_bin}",
            "${repo_root}/EG3/pikachu_run_sc3.py",
            "--dataset", "0501",
            "--metadata", "${module_output_dir}/data_root/pikachu_sc3/0501/metadata.json",
            "--detector-root", str(MIRAGE_ROOT / "detection_baselines" / "pikachu"),
            "--output-dir", "${module_output_dir}/results/0501",
            "--metrics-csv", "${module_output_dir}/results/0501/metrics.csv",
            "--calibration", job.calibration,
            "--val-fraction", str(INDUCTIVE_VAL_FRACTION),
            "--dimensions", PIKACHU_BASE_ARGS["dimensions"],
            "--alpha", PIKACHU_BASE_ARGS["alpha"],
            "--support", PIKACHU_BASE_ARGS["support"],
            "--gpu", PIKACHU_BASE_ARGS["gpu"],
        ]
        selected["metrics_relpath"] = "results/0501/metrics.csv"
        detection["jobs"] = [selected]
        return

    dataset_dir = job.dataset
    featurization["runner"] = "shell_command"
    featurization["workdir"] = "${repo_root}"
    featurization["env"] = {}
    featurization["command"] = [
        "${python_bin}",
        "${repo_root}/EG3/eg3_preprocess.py",
        "--dataset", PIKACHU_DATASET_ARGS[job.dataset],
        "--raw-root", "${module_output_dir}/data_root/raw_data",
        "--output-dir", f"${{module_output_dir}}/data_root/pikachu_sc3/{dataset_dir}",
        "--snapshot-seconds", str(window_seconds),
        "--neg-ratio", str(PIKACHU_NEG_RATIO),
        "--seed", str(PIKACHU_PREPROCESS_SEED),
    ]

    target_job_id = DETECTOR_DATASET_JOB_IDS[job.dataset]
    selected = _select_detection_job(
        detection["jobs"], target_job_id, f"pikachu:{job.dataset}"
    )
    selected["workdir"] = "${repo_root}"
    selected["command"] = [
        "${python_bin}",
        "${repo_root}/EG3/pikachu_run_sc3.py",
        "--dataset", PIKACHU_DATASET_ARGS[job.dataset],
        "--metadata", f"${{module_output_dir}}/data_root/pikachu_sc3/{dataset_dir}/metadata.json",
        "--detector-root", str(MIRAGE_ROOT / "detection_baselines" / "pikachu"),
        "--output-dir", f"${{module_output_dir}}/results/{target_job_id}",
        "--metrics-csv", f"${{module_output_dir}}/results/{target_job_id}/metrics.csv",
        "--calibration", job.calibration,
        "--val-fraction", str(INDUCTIVE_VAL_FRACTION),
        "--dimensions", PIKACHU_BASE_ARGS["dimensions"],
        "--alpha", PIKACHU_BASE_ARGS["alpha"],
        "--support", PIKACHU_BASE_ARGS["support"],
        "--gpu", PIKACHU_BASE_ARGS["gpu"],
    ]
    detection["jobs"] = [selected]

def run_job(job: EG3Job, *, dry_run: bool = False, force: bool = False) -> str:
    if job.calibration == "transductive":
        generate_config(job)
        return "skipped"
    config_path = generate_config(job)
    if not force:
        status = job_status(job)
        if status == "success":
            return "skipped"
    env_python = detector_env_python()
    if not env_python.exists() and not dry_run:
        print(
            f"[EG3] missing pikachu environment: expected {env_python}",
            file=sys.stderr,
        )
        return "failed"

    env = dict(os.environ)
    pythonpath_parts = [
        str(EG3_CODE_ROOT.resolve()),
        str((REPO_ROOT / "src").resolve()),
    ]
    existing = env.get("PYTHONPATH", "")
    if existing:
        pythonpath_parts.append(existing)
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    if env_python.exists():
        env["PYTHON"] = str(env_python)
        env["PATH"] = f"{env_python.parent}:{env.get('PATH', '')}"
        env["EG1_DETECTOR_PYTHON"] = str(env_python)

    command = [sys.executable, "-m", "gids_eval", "run"]
    if dry_run:
        command.append("--dry-run")
    command.append(str(config_path))

    completed = subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=False)
    return "success" if completed.returncode == 0 else "failed"

def _has_successful_local_result(job: EG3Job) -> bool:
    cost_rows = read_csv(job.cost_metrics_path)
    baseline_rows = read_csv(job.baseline_metrics_path)
    if not cost_rows or not baseline_rows:
        return False
    detection_rows = [r for r in cost_rows if r.get("phase") == "detection"]
    return bool(detection_rows) and all(
        r.get("status") in {"success", "cache_hit"} for r in detection_rows
    )

def job_status(job: EG3Job) -> str:
    if job.calibration == "transductive":
        if external_eg1_transductive_row(job) or _has_successful_local_result(job):
            return "success"
    elif _has_successful_local_result(job):
        return "success"
    if job.run_summary_path.exists():
        try:
            with job.run_summary_path.open("r", encoding="utf-8") as fh:
                summary = json.load(fh)
            records = summary.get("records", [])
            if any(r.get("status") == "failed" for r in records):
                return "failed"
            statuses = {str(r.get("status", "")) for r in records}
            if statuses and statuses.issubset({"dry_run"}):
                return "pending"
        except json.JSONDecodeError:
            return "failed"
    if job.metadata_root.exists():
        return "partial"
    return "pending"

def _coerce_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None

def external_eg1_transductive_row(job: EG3Job) -> dict[str, Any] | None:
    if job.calibration != "transductive":
        return None
    expected_window = DATASET_WINDOW_SECONDS[job.dataset]
    candidates: list[dict[str, Any]] = []
    for row in read_csv(eg1_raw_results_path()):
        if str(row.get("detector", "")).strip() != "pikachu":
            continue
        if str(row.get("dataset", "")).strip() != job.dataset:
            continue
        if _coerce_int(row.get("seed")) != job.seed:
            continue
        if _coerce_int(row.get("window_seconds")) != expected_window:
            continue
        candidates.append(row)
    if not candidates:
        return None
    candidates.sort(key=lambda row: str(row.get("experiment_id", "")))
    return candidates[0]

def build_status_rows(jobs: list[EG3Job]) -> list[dict[str, Any]]:
    rows = []
    for job in jobs:
        rows.append({
            "dataset": job.dataset,
            "calibration": job.calibration,
            "seed": job.seed,
            "experiment_id": job.experiment_id,
            "status": job_status(job),
            "config_path": str(job.config_path),
        })
    return rows

STATUS_FIELDNAMES = [
    "dataset", "calibration", "seed", "experiment_id", "status", "config_path",
]

def write_matrix_report(jobs: list[EG3Job], path: Path | None = None) -> Path:
    rows = [
        {
            "dataset": j.dataset,
            "calibration": j.calibration,
            "seed": j.seed,
            "experiment_id": j.experiment_id,
            "config_path": str(j.config_path),
        }
        for j in jobs
    ]
    path = path or (ORCHESTRATOR_ROOT / "job_matrix.csv")
    write_csv(path, rows, ["dataset", "calibration", "seed", "experiment_id", "config_path"])
    return path

def write_status_report(jobs: list[EG3Job], path: Path | None = None) -> Path:
    rows = build_status_rows(jobs)
    path = path or (ORCHESTRATOR_ROOT / "job_status.csv")
    write_csv(path, rows, STATUS_FIELDNAMES)
    return path

def write_progress_report(
    jobs: list[EG3Job],
    *,
    json_path: Path | None = None,
    md_path: Path | None = None,
    scope_name: str = "global",
    attempted_jobs: int | None = None,
    failed_jobs: int | None = None,
) -> tuple[Path, Path]:
    rows = build_status_rows(jobs)
    total = len(rows)
    counts: dict[str, int] = {}
    for s in ("pending", "partial", "failed", "success"):
        counts[s] = sum(1 for r in rows if r["status"] == s)
    completed = counts["success"] + counts["failed"]
    completion_ratio = completed / total if total else 0.0
    success_ratio = counts["success"] / total if total else 0.0

    per_calibration: list[dict[str, Any]] = []
    for cal in CALIBRATION_MODES:
        cal_rows = [r for r in rows if r["calibration"] == cal]
        if not cal_rows:
            continue
        ct = len(cal_rows)
        per_calibration.append({
            "calibration": cal,
            "total": ct,
            "success": sum(1 for r in cal_rows if r["status"] == "success"),
            "failed": sum(1 for r in cal_rows if r["status"] == "failed"),
            "pending": sum(1 for r in cal_rows if r["status"] == "pending"),
            "partial": sum(1 for r in cal_rows if r["status"] == "partial"),
        })

    payload = {
        "scope": scope_name,
        "total_jobs": total,
        "completed_jobs": completed,
        "remaining_jobs": total - completed,
        "success_jobs": counts["success"],
        "failed_jobs": counts["failed"],
        "completion_ratio": completion_ratio,
        "success_ratio": success_ratio,
        "attempted_jobs": attempted_jobs,
        "run_failures": failed_jobs,
        "per_calibration": per_calibration,
    }

    json_path = json_path or PROGRESS_JSON_PATH
    md_path = md_path or PROGRESS_MD_PATH
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        f"# EG3 Progress ({scope_name})",
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
    lines.extend(["", "## Per Calibration Mode"])
    for row in per_calibration:
        lines.append(
            f"- {row['calibration']}: {row['success']}/{row['total']} success, "
            f"failed={row['failed']}, partial={row['partial']}, pending={row['pending']}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path

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

RAW_FIELDNAMES = [
    "dataset", "calibration", "seed", "experiment_id",
    "result_source",
    "auc", "ap", "threshold", "tpr", "fpr",
    "precision", "recall", "f1",
    "cdr", "cdr_auc",
    "train_time", "test_time",
    "featurization_wall_time_sec",
    "detection_wall_time_sec", "total_wall_time_sec",
    "featurization_peak_rss_kb", "detection_peak_rss_kb", "run_peak_rss_kb",
    "featurization_mean_rss_kb", "detection_mean_rss_kb", "run_mean_rss_kb",
    "featurization_peak_rss_gib", "detection_peak_rss_gib", "run_peak_rss_gib",
    "featurization_mean_rss_gib", "detection_mean_rss_gib", "run_mean_rss_gib",
    "featurization_mean_cpu_percent", "detection_mean_cpu_percent", "run_mean_cpu_percent",
    "metrics_path",
]

DELTA_METRICS = ("ap", "auc", "tpr", "fpr", "precision", "recall", "f1")

def _append_local_raw_rows(
    raw_rows: list[dict[str, Any]],
    job: EG3Job,
    *,
    result_source: str,
) -> None:
    cost_rows = read_csv(job.cost_metrics_path)
    phase_rows = _phase_row_lookup(cost_rows)
    detection_row = phase_rows.get("detection", {})
    featurization_row = phase_rows.get("featurization", {})
    total_wall = sum(_to_float(r.get("wall_time_sec")) for r in cost_rows)
    peak_values = [_optional_float(r.get("peak_rss_kb")) for r in cost_rows]
    peak_values = [v for v in peak_values if v != ""]
    run_peak_rss_kb = max(peak_values) if peak_values else ""
    run_mean_rss_kb = _weighted_mean_from_rows(
        cost_rows, value_key="mean_rss_kb", weight_key="sample_count",
    )
    run_mean_cpu_percent = _weighted_mean_from_rows(
        cost_rows, value_key="mean_cpu_percent", weight_key="cpu_sample_count",
    )

    for m_row in read_csv(job.baseline_metrics_path):
        raw_rows.append({
            "dataset": job.dataset,
            "calibration": job.calibration,
            "seed": job.seed,
            "experiment_id": job.experiment_id,
            "result_source": result_source,
            "auc": _to_float(m_row.get("auc")),
            "ap": _to_float(m_row.get("ap")),
            "threshold": _to_float(m_row.get("threshold")),
            "tpr": _to_float(m_row.get("tpr")),
            "fpr": _to_float(m_row.get("fpr")),
            "precision": _to_float(m_row.get("precision")),
            "recall": _to_float(m_row.get("recall")),
            "f1": _to_float(m_row.get("f1")),
            "cdr": _to_float(m_row.get("cdr")),
            "cdr_auc": _to_float(m_row.get("cdr_auc")),
            "train_time": _to_float(m_row.get("train_time")),
            "test_time": _to_float(m_row.get("test_time")),
            "featurization_wall_time_sec": _to_float(featurization_row.get("wall_time_sec")),
            "detection_wall_time_sec": _to_float(detection_row.get("wall_time_sec")),
            "total_wall_time_sec": total_wall,
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

def summarize_jobs(jobs: list[EG3Job], output_root: Path | None = None) -> tuple[Path, Path]:
    output_root = output_root or REPORT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, Any]] = []
    for job in jobs:
        if job.calibration == "transductive":
            eg1_row = external_eg1_transductive_row(job)
            if eg1_row:
                raw_rows.append({
                    "dataset": job.dataset,
                    "calibration": job.calibration,
                    "seed": job.seed,
                    "experiment_id": str(eg1_row.get("experiment_id", "")) or job.experiment_id,
                    "result_source": "eg1_transductive_baseline",
                    "auc": _to_float(eg1_row.get("auc")),
                    "ap": _to_float(eg1_row.get("ap")),
                    "threshold": _to_float(eg1_row.get("threshold")),
                    "tpr": _to_float(eg1_row.get("tpr")),
                    "fpr": _to_float(eg1_row.get("fpr")),
                    "precision": _to_float(eg1_row.get("precision")),
                    "recall": _to_float(eg1_row.get("recall")),
                    "f1": _to_float(eg1_row.get("f1")),
                    "cdr": _to_float(eg1_row.get("cdr")),
                    "cdr_auc": _to_float(eg1_row.get("cdr_auc")),
                    "train_time": _to_float(eg1_row.get("train_time")),
                    "test_time": _to_float(eg1_row.get("test_time")),
                    "featurization_wall_time_sec": _to_float(eg1_row.get("featurization_wall_time_sec")),
                    "detection_wall_time_sec": _to_float(eg1_row.get("detection_wall_time_sec")),
                    "total_wall_time_sec": _to_float(eg1_row.get("total_wall_time_sec")),
                    "featurization_peak_rss_kb": _optional_float(eg1_row.get("featurization_peak_rss_kb")),
                    "detection_peak_rss_kb": _optional_float(eg1_row.get("detection_peak_rss_kb")),
                    "run_peak_rss_kb": _optional_float(eg1_row.get("run_peak_rss_kb")),
                    "featurization_mean_rss_kb": _optional_float(eg1_row.get("featurization_mean_rss_kb")),
                    "detection_mean_rss_kb": _optional_float(eg1_row.get("detection_mean_rss_kb")),
                    "run_mean_rss_kb": _optional_float(eg1_row.get("run_mean_rss_kb")),
                    "featurization_peak_rss_gib": _optional_float(eg1_row.get("featurization_peak_rss_gib")),
                    "detection_peak_rss_gib": _optional_float(eg1_row.get("detection_peak_rss_gib")),
                    "run_peak_rss_gib": _optional_float(eg1_row.get("run_peak_rss_gib")),
                    "featurization_mean_rss_gib": _optional_float(eg1_row.get("featurization_mean_rss_gib")),
                    "detection_mean_rss_gib": _optional_float(eg1_row.get("detection_mean_rss_gib")),
                    "run_mean_rss_gib": _optional_float(eg1_row.get("run_mean_rss_gib")),
                    "featurization_mean_cpu_percent": _optional_float(eg1_row.get("featurization_mean_cpu_percent")),
                    "detection_mean_cpu_percent": _optional_float(eg1_row.get("detection_mean_cpu_percent")),
                    "run_mean_cpu_percent": _optional_float(eg1_row.get("run_mean_cpu_percent")),
                    "metrics_path": str(eg1_row.get("metrics_path", "")),
                })
                continue
            if _has_successful_local_result(job):
                _append_local_raw_rows(
                    raw_rows,
                    job,
                    result_source="eg3_local_transductive_fallback",
                )
            continue
        if job_status(job) != "success":
            continue
        _append_local_raw_rows(raw_rows, job, result_source="eg3_run")

    raw_path = output_root / "raw_results.csv"
    write_csv(raw_path, raw_rows, RAW_FIELDNAMES)

    delta_rows = _build_delta_rows(raw_rows)
    delta_path = output_root / "delta_table.csv"
    delta_fields = ["dataset", "seed"] + [f"delta_{m}" for m in DELTA_METRICS] + [
        f"rel_delta_{m}" for m in DELTA_METRICS
    ]
    write_csv(delta_path, delta_rows, delta_fields)

    evidence_rows = _build_evidence_rows(delta_rows)
    evidence_path = output_root / "evidence_table.csv"
    evidence_fields = (
        ["dataset", "seed_count"]
        + [f"mean_delta_{m}" for m in DELTA_METRICS]
        + [f"mean_rel_delta_{m}" for m in DELTA_METRICS]
    )
    write_csv(evidence_path, evidence_rows, evidence_fields)

    overall_rows = _build_overall_row(evidence_rows)
    overall_path = output_root / "evidence_table_overall.csv"
    write_csv(overall_path, overall_rows, evidence_fields)

    write_grouped_bar_svg(raw_rows, output_root / "plot_calibration_comparison.svg")
    write_delta_bar_svg(evidence_rows, output_root / "plot_delta_by_dataset.svg")
    write_conclusion_md(raw_rows, evidence_rows, output_root / "conclusion.md")

    return raw_path, delta_path

def _build_delta_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in raw_rows:
        key = (str(row["dataset"]), int(row["seed"]), str(row["calibration"]))
        by_key[key] = row

    datasets_seeds: dict[tuple[str, int], None] = {}
    for row in raw_rows:
        datasets_seeds[(str(row["dataset"]), int(row["seed"]))] = None

    deltas: list[dict[str, Any]] = []
    for dataset, seed in sorted(datasets_seeds):
        trans = by_key.get((dataset, seed, "transductive"))
        induc = by_key.get((dataset, seed, "inductive"))
        if not trans or not induc:
            continue
        row: dict[str, Any] = {"dataset": dataset, "seed": seed}
        for m in DELTA_METRICS:
            tv = float(trans[m])
            iv = float(induc[m])
            row[f"delta_{m}"] = iv - tv
            row[f"rel_delta_{m}"] = ((iv - tv) / tv) if tv != 0 else 0.0
        deltas.append(row)
    return deltas

def _build_evidence_rows(delta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in delta_rows:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)

    evidence: list[dict[str, Any]] = []
    for dataset in sorted(by_dataset):
        rows = by_dataset[dataset]
        entry: dict[str, Any] = {"dataset": dataset, "seed_count": len(rows)}
        for m in DELTA_METRICS:
            vals = [float(r[f"delta_{m}"]) for r in rows]
            entry[f"mean_delta_{m}"] = statistics.mean(vals) if vals else 0.0
            rel_vals = [float(r[f"rel_delta_{m}"]) for r in rows]
            entry[f"mean_rel_delta_{m}"] = statistics.mean(rel_vals) if rel_vals else 0.0
        evidence.append(entry)
    return evidence

def _build_overall_row(evidence_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not evidence_rows:
        return []
    entry: dict[str, Any] = {
        "dataset": "ALL",
        "seed_count": sum(int(r["seed_count"]) for r in evidence_rows),
    }
    for m in DELTA_METRICS:
        vals = [float(r[f"mean_delta_{m}"]) for r in evidence_rows]
        entry[f"mean_delta_{m}"] = statistics.mean(vals) if vals else 0.0
        rel_vals = [float(r[f"mean_rel_delta_{m}"]) for r in evidence_rows]
        entry[f"mean_rel_delta_{m}"] = statistics.mean(rel_vals) if rel_vals else 0.0
    return [entry]

def write_grouped_bar_svg(
    raw_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_rows:
        output_path.write_text(_empty_svg("EG3 Calibration Comparison", "No data."), encoding="utf-8")
        return

    metrics_to_plot = ("ap", "auc", "precision", "recall", "f1")
    panel_w = 260
    panel_h = 220
    cols = 3
    datasets_present = sorted({str(r["dataset"]) for r in raw_rows})
    rows_needed = max(1, math.ceil(len(datasets_present) / cols))
    width = panel_w * cols
    height = 60 + panel_h * rows_needed

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:monospace;font-size:11px}.title{font-size:16px;font-weight:bold}.val{font-size:9px}</style>',
        '<text class="title" x="20" y="24">EG3: Transductive vs Inductive Calibration</text>',
    ]

    agg: dict[tuple[str, str], dict[str, float]] = {}
    for row in raw_rows:
        key = (str(row["dataset"]), str(row["calibration"]))
        if key not in agg:
            agg[key] = {m: 0.0 for m in metrics_to_plot}
            agg[key]["_count"] = 0.0
        for m in metrics_to_plot:
            agg[key][m] += float(row[m])
        agg[key]["_count"] += 1.0
    for key in agg:
        cnt = agg[key]["_count"]
        for m in metrics_to_plot:
            agg[key][m] /= cnt

    bar_w = 18
    group_gap = 6

    for di, dataset in enumerate(datasets_present):
        x0 = (di % cols) * panel_w
        y0 = 50 + (di // cols) * panel_h
        parts.append(f'<text x="{x0 + 10}" y="{y0 + 14}">{dataset}</text>')

        ax_left = x0 + 30
        ax_bottom = y0 + 180
        ax_top = y0 + 30
        ax_right = x0 + panel_w - 10
        parts.append(f'<line x1="{ax_left}" y1="{ax_bottom}" x2="{ax_right}" y2="{ax_bottom}" stroke="#444" stroke-width="1"/>')
        parts.append(f'<line x1="{ax_left}" y1="{ax_bottom}" x2="{ax_left}" y2="{ax_top}" stroke="#444" stroke-width="1"/>')

        for tick in range(6):
            y = ax_bottom - tick * 30
            parts.append(f'<text x="{x0 + 4}" y="{y + 4}" font-size="9">{tick/5:.1f}</text>')
            parts.append(f'<line x1="{ax_left}" y1="{y}" x2="{ax_right}" y2="{y}" stroke="#eee" stroke-width="1"/>')

        for mi, metric in enumerate(metrics_to_plot):
            gx = ax_left + 8 + mi * (2 * bar_w + group_gap + 8)
            for ci, cal in enumerate(CALIBRATION_MODES):
                if (dataset, cal) not in agg:
                    continue
                val = agg[(dataset, cal)].get(metric, 0.0)
                bar_h = max(0, min(1.0, val)) * 150
                bx = gx + ci * (bar_w + 2)
                by = ax_bottom - bar_h
                color = SVG_COLORS[cal]
                parts.append(f'<rect x="{bx}" y="{by}" width="{bar_w}" height="{bar_h}" fill="{color}" stroke="#333" stroke-width="0.5"/>')
                parts.append(f'<text class="val" x="{bx}" y="{by - 2}">{val:.3f}</text>')
            parts.append(f'<text x="{gx}" y="{ax_bottom + 12}" font-size="9">{metric}</text>')

        ly = y0 + 198
        for ci, cal in enumerate(CALIBRATION_MODES):
            lx = x0 + 30 + ci * 120
            parts.append(f'<rect x="{lx}" y="{ly}" width="12" height="8" fill="{SVG_COLORS[cal]}"/>')
            parts.append(f'<text x="{lx + 16}" y="{ly + 8}" font-size="9">{cal}</text>')

    parts.append("</svg>")
    output_path.write_text("".join(parts), encoding="utf-8")

def write_delta_bar_svg(
    evidence_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not evidence_rows:
        output_path.write_text(_empty_svg("EG3 Metric Deltas", "No data."), encoding="utf-8")
        return

    metrics_to_plot = list(DELTA_METRICS)
    row_h = 28
    dataset_block_h = len(metrics_to_plot) * row_h + 30
    width = 700
    height = 60 + len(evidence_rows) * dataset_block_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:monospace;font-size:11px}.title{font-size:16px;font-weight:bold}</style>',
        '<text class="title" x="20" y="24">EG3: Metric Deltas (Inductive - Transductive)</text>',
        '<text x="20" y="42" font-size="10">Negative = inductive is worse.  Bar scale: [-0.5, +0.5]</text>',
    ]

    center_x = 400
    scale = 400

    for di, ev_row in enumerate(evidence_rows):
        y_base = 60 + di * dataset_block_h
        parts.append(f'<text x="20" y="{y_base + 14}" font-weight="bold">{ev_row["dataset"]}</text>')
        parts.append(f'<line x1="{center_x}" y1="{y_base + 20}" x2="{center_x}" y2="{y_base + 20 + len(metrics_to_plot) * row_h}" stroke="#999" stroke-width="1"/>')
        for mi, m in enumerate(metrics_to_plot):
            y = y_base + 22 + mi * row_h
            val = float(ev_row[f"mean_delta_{m}"])
            bar_len = val * scale
            if bar_len >= 0:
                parts.append(f'<rect x="{center_x}" y="{y}" width="{min(bar_len, 200)}" height="{row_h - 4}" fill="#2e6f95" opacity="0.7"/>')
            else:
                bw = min(abs(bar_len), 200)
                parts.append(f'<rect x="{center_x - bw}" y="{y}" width="{bw}" height="{row_h - 4}" fill="#d1495b" opacity="0.7"/>')
            parts.append(f'<text x="100" y="{y + 16}">{m}</text>')
            parts.append(f'<text x="{center_x + 210}" y="{y + 16}">{val:+.4f}</text>')

    parts.append("</svg>")
    output_path.write_text("".join(parts), encoding="utf-8")

def write_conclusion_md(
    raw_rows: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_rows:
        output_path.write_text(
            "# EG3 Conclusion\n\nNo successful EG3 jobs yet.\n", encoding="utf-8"
        )
        return

    lines = [
        "# EG3 Conclusion: Inductive Calibration",
        "",
        "## Summary",
        "",
        "EG3 compares PIKACHU's original **transductive** threshold calibration",
        "(threshold picked on all test-set scores) against a deployment-realistic",
        "**inductive** setup (threshold picked on a held-out validation split that",
        "precedes the test set chronologically).",
        "",
    ]

    agg: dict[tuple[str, str], dict[str, float]] = {}
    dataset_modes: dict[str, set[str]] = {}
    for row in raw_rows:
        key = (str(row["dataset"]), str(row["calibration"]))
        dataset_modes.setdefault(str(row["dataset"]), set()).add(str(row["calibration"]))
        if key not in agg:
            agg[key] = {}
            agg[key]["_count"] = 0.0
        for m in DELTA_METRICS:
            agg[key][m] = agg[key].get(m, 0.0) + float(row[m])
        agg[key]["_count"] += 1.0
    for key in agg:
        cnt = agg[key]["_count"]
        for m in DELTA_METRICS:
            agg[key][m] /= cnt

    complete_datasets = [
        dataset for dataset, modes in sorted(dataset_modes.items())
        if set(CALIBRATION_MODES).issubset(modes)
    ]
    incomplete_datasets = [
        dataset for dataset, modes in sorted(dataset_modes.items())
        if not set(CALIBRATION_MODES).issubset(modes)
    ]

    lines.extend([
        "## Paired Results",
        "",
    ])
    if complete_datasets:
        lines.extend([
            "| Dataset | Metric | Transductive | Inductive | Delta | Rel Delta |",
            "|---------|--------|-------------|-----------|-------|-----------|",
        ])
        for dataset in complete_datasets:
            for m in DELTA_METRICS:
                tv = agg[(dataset, "transductive")][m]
                iv = agg[(dataset, "inductive")][m]
                delta = iv - tv
                rel = (delta / tv * 100) if tv != 0 else 0.0
                lines.append(f"| {dataset} | {m} | {tv:.4f} | {iv:.4f} | {delta:+.4f} | {rel:+.1f}% |")
    else:
        lines.append("No dataset currently has both calibration modes available.")

    if incomplete_datasets:
        lines.extend([
            "",
            "## Incomplete Datasets",
            "",
            "| Dataset | Available Modes | Missing Modes |",
            "|---------|-----------------|---------------|",
        ])
        for dataset in incomplete_datasets:
            modes = dataset_modes[dataset]
            available = ", ".join(cal for cal in CALIBRATION_MODES if cal in modes)
            missing = ", ".join(cal for cal in CALIBRATION_MODES if cal not in modes)
            lines.append(f"| {dataset} | {available or '-'} | {missing or '-'} |")

    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    if evidence_rows:
        lines.extend([
            "Negative deltas in AP, AUC, precision, recall, and F1 indicate that",
            "PIKACHU's reported performance benefits from seeing test-set information",
            "during threshold selection.  The magnitude of the gap quantifies the",
            "realism bias in the original transductive evaluation.",
        ])
    else:
        lines.extend([
            "Paired deltas will appear once both calibration modes are available",
            "for at least one dataset.",
        ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _empty_svg(title: str, message: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="120" viewBox="0 0 600 120">'
        '<style>text{font-family:monospace;font-size:14px}.title{font-size:18px;font-weight:bold}</style>'
        f'<text class="title" x="20" y="30">{title}</text>'
        f'<text x="20" y="70">{message}</text>'
        "</svg>"
    )

def parse_csv_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]

def parse_seed_list(value: str | None) -> list[int]:
    if not value:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in value.split(",") if item.strip()]
