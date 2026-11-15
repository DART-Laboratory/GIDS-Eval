from __future__ import annotations

import csv
import json
import math
import statistics
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import yaml

from gids_eval.paths import get_paths

_PATHS = get_paths()
REPO_ROOT = _PATHS.repo_root
EG9_CODE_ROOT = REPO_ROOT / "EG9"
DATA_ROOT = _PATHS.sc_data_root("eg9")
GENERATED_CONFIG_ROOT = DATA_ROOT / "generated-configs"
ORCHESTRATOR_ROOT = DATA_ROOT / "orchestrator"
REPORT_ROOT = DATA_ROOT / "reports"
SELECTION_REPORT_ROOT = REPORT_ROOT / "selection"
PROGRESS_JSON_PATH = ORCHESTRATOR_ROOT / "progress.json"
PROGRESS_MD_PATH = ORCHESTRATOR_ROOT / "progress.md"
RUN_LABEL = "eg9"

def _newest_raw_results(track: str) -> Path:
    root = REPO_ROOT / track / "results"
    snapshots = sorted(
        (p for p in root.glob("*") if p.is_dir() and (p / "raw_results.csv").is_file()),
        key=lambda p: p.name,
    )
    if not snapshots:
        raise FileNotFoundError(f"no results snapshot with raw_results.csv under {root}")
    return snapshots[-1] / "raw_results.csv"


EG1_RAW_RESULTS_PATH = _newest_raw_results("EG1")
EG3_RAW_RESULTS_PATH = _newest_raw_results("EG3")
EG5_RAW_RESULTS_PATH = _newest_raw_results("EG5")

DETECTORS = ("argus", "euler", "vgrnn", "pikachu", "anomal")
DATASETS = ("lanl", "optc", "cic", "0501")
DEFAULT_RATES = (1, 2, 5)
COMMON_SUBSET_DATASETS = {"cic", "0501"}
REAL_TIME_BUFFER_MAX_SEC = 60.0
NEAR_REAL_TIME_BUFFER_MAX_SEC = 300.0

DATASET_ALIASES = {
    "wizard": "0501",
    "wizard_spider": "0501",
    "oilrig": "0501",
    "sandworm": "0501",
}

SUPPORTED_DATASETS = {
    "argus": ("lanl", "optc", "cic", "0501"),
    "euler": ("lanl", "optc", "cic", "0501"),
    "vgrnn": ("lanl", "optc", "cic", "0501"),
    "pikachu": ("lanl", "optc", "cic", "0501"),
    "anomal": ("cic", "0501"),
}

WINDOW_SECONDS = {
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
    "anomal": {
        "cic": None,
        "0501": None,
    },
}

ARCHITECTURES = {
    "argus": "snapshot_time",
    "euler": "snapshot_time",
    "vgrnn": "snapshot_time",
    "pikachu": "snapshot_time",
    "anomal": "static",
}

@dataclass(frozen=True)
class EG9Job:
    detector: str
    dataset: str
    replay_rate: int
    architecture: str
    window_seconds: int | None

    @property
    def rate_label(self) -> str:
        return f"{int(self.replay_rate)}x"

    @property
    def experiment_id(self) -> str:
        return f"eg9_{self.detector}_{self.dataset}_{self.rate_label}"

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
    def logs_root(self) -> Path:
        return self.metadata_root / "logs"

    @property
    def execution_config_path(self) -> Path:
        return self.metadata_root / "execution_config.yaml"

    @property
    def readiness_metrics_path(self) -> Path:
        return self.metadata_root / "readiness_metrics.csv"

    @property
    def latency_trace_path(self) -> Path:
        return self.metadata_root / "latency_trace.csv"

    @property
    def replay_progress_path(self) -> Path:
        return self.metadata_root / "replay_progress.json"

    @property
    def run_summary_path(self) -> Path:
        return self.metadata_root / "run_summary.json"

def ensure_eg9_dirs() -> None:
    for path in (GENERATED_CONFIG_ROOT, ORCHESTRATOR_ROOT, REPORT_ROOT, SELECTION_REPORT_ROOT):
        path.mkdir(parents=True, exist_ok=True)

def detector_list() -> list[str]:
    return list(DETECTORS)

def dataset_list() -> list[str]:
    return list(DATASETS)

def rate_list() -> list[int]:
    return list(DEFAULT_RATES)

def canonical_dataset(value: str) -> str:
    lowered = value.strip().lower()
    return DATASET_ALIASES.get(lowered, lowered)

def parse_csv_list(value: str | None, *, kind: str) -> list[Any] | None:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if kind == "dataset":
        items = [canonical_dataset(item) for item in items]
    elif kind == "rate":
        items = [int(float(item)) for item in items]
    return list(dict.fromkeys(items))

def iter_jobs(
    detectors: Iterable[str] | None = None,
    datasets: Iterable[str] | None = None,
    rates: Iterable[int] | None = None,
) -> list[EG9Job]:
    selected_detectors = list(detectors or detector_list())
    selected_datasets = list(datasets or dataset_list())
    selected_rates = list(rates or rate_list())
    jobs: list[EG9Job] = []
    for detector in selected_detectors:
        supported = SUPPORTED_DATASETS.get(detector)
        if supported is None:
            raise ValueError(f"Unknown detector: {detector}")
        for dataset in selected_datasets:
            if dataset not in supported:
                continue
            for replay_rate in selected_rates:
                jobs.append(
                    EG9Job(
                        detector=detector,
                        dataset=dataset,
                        replay_rate=int(replay_rate),
                        architecture=ARCHITECTURES[detector],
                        window_seconds=WINDOW_SECONDS[detector][dataset],
                    )
                )
    return jobs

def same_job_set(left: list[EG9Job], right: list[EG9Job]) -> bool:
    return {job.experiment_id for job in left} == {job.experiment_id for job in right}

def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping payload in {path}")
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

def _float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)

def _int(value: Any) -> int | None:
    parsed = _float(value)
    if parsed is None:
        return None
    return int(round(parsed))

@lru_cache(maxsize=1)
def _eg1_rows() -> list[dict[str, Any]]:
    return read_csv(EG1_RAW_RESULTS_PATH)

@lru_cache(maxsize=1)
def _eg3_rows() -> list[dict[str, Any]]:
    return read_csv(EG3_RAW_RESULTS_PATH)

@lru_cache(maxsize=1)
def _eg5_rows() -> list[dict[str, Any]]:
    return read_csv(EG5_RAW_RESULTS_PATH)

def _find_eg1_runtime(detector: str, dataset: str, window_seconds: int) -> dict[str, Any]:
    for row in _eg1_rows():
        if row["detector"] != detector or row["dataset"] != dataset:
            continue
        if _int(row["window_seconds"]) == int(window_seconds):
            return row
    raise KeyError(f"Missing EG1 runtime row for {detector}/{dataset}/{window_seconds}")

def _find_eg3_pikachu(dataset: str) -> dict[str, Any]:
    for row in _eg3_rows():
        if row["dataset"] == dataset and row["calibration"] == "inductive":
            return row
    raise KeyError(f"Missing EG3 inductive PIKACHU row for {dataset}")

def _find_eg5_row(detector: str, dataset: str, scale_percent: int = 100) -> dict[str, Any]:
    for row in _eg5_rows():
        if row["detector"] != detector or row["dataset"] != dataset:
            continue
        if _int(row["scale_percent"]) == int(scale_percent):
            return row
    raise KeyError(f"Missing EG5 row for {detector}/{dataset}/p{scale_percent}")

def _canonical_dataset_profile(dataset: str) -> dict[str, Any]:
    row = _find_eg5_row("argus", dataset, scale_percent=100)
    kept_post_units = max(1, _int(row.get("kept_post_units")) or _int(row.get("input_unit_count")) or 1)
    window_seconds = max(1, _int(row.get("window_seconds")) or WINDOW_SECONDS["argus"][dataset] or 1)
    duration_seconds = float(kept_post_units * window_seconds)
    input_rows = float(_float(row.get("input_rows")) or 0.0)
    arrival_rate_rows_per_sec = (input_rows / duration_seconds) if duration_seconds > 0 else 0.0
    return {
        "dataset": dataset,
        "canonical_detector": "argus",
        "duration_seconds": duration_seconds,
        "input_rows": input_rows,
        "arrival_rate_rows_per_sec": arrival_rate_rows_per_sec,
        "source_experiment_id": row.get("source_experiment_id", ""),
    }

def _input_stats_row(detector: str, dataset: str) -> dict[str, Any]:
    return _find_eg5_row(detector, dataset, scale_percent=100)

def _runtime_row(detector: str, dataset: str) -> tuple[dict[str, Any], str]:
    if detector in {"argus", "euler", "vgrnn"}:
        window_seconds = WINDOW_SECONDS[detector][dataset]
        return _find_eg1_runtime(detector, dataset, window_seconds), "eg1"
    if detector == "pikachu":
        return _find_eg3_pikachu(dataset), "eg3_inductive"
    return _find_eg5_row("anomal", dataset, scale_percent=100), "eg5_anomal"

def _job_profile(job: EG9Job) -> dict[str, Any]:
    input_stats = _input_stats_row(job.detector, job.dataset)
    runtime_row, runtime_source = _runtime_row(job.detector, job.dataset)
    canonical = _canonical_dataset_profile(job.dataset)

    if job.architecture == "snapshot_time":
        evaluation_units = max(
            1,
            _int(input_stats.get("kept_post_units"))
            or _int(input_stats.get("input_unit_count"))
            or 1,
        )
        buffering_unit_seconds = float(job.window_seconds or 0)
        effective_duration_seconds = float(evaluation_units) * buffering_unit_seconds
    else:
        evaluation_units = 1
        buffering_unit_seconds = float(canonical["duration_seconds"])
        effective_duration_seconds = float(canonical["duration_seconds"])

    estimated_input_rows = canonical["arrival_rate_rows_per_sec"] * effective_duration_seconds
    if estimated_input_rows <= 0:
        estimated_input_rows = float(_float(input_stats.get("input_rows")) or 0.0)

    featurization_wall = float(_float(runtime_row.get("featurization_wall_time_sec")) or 0.0)
    detection_wall = float(_float(runtime_row.get("detection_wall_time_sec")) or 0.0)
    total_wall = float(_float(runtime_row.get("total_wall_time_sec")) or (featurization_wall + detection_wall))
    if total_wall <= 0:
        total_wall = featurization_wall + detection_wall
    peak_rss_gib = _float(runtime_row.get("run_peak_rss_gib"))
    if peak_rss_gib is None:
        peak_rss_kb = _float(runtime_row.get("run_peak_rss_kb"))
        peak_rss_gib = (peak_rss_kb / (1024.0 ** 2)) if peak_rss_kb is not None else 0.0
    processing_time_per_unit_sec = total_wall / max(1, evaluation_units)
    pre_inference_time_ms = featurization_wall * 1000.0 / max(1, evaluation_units)
    inference_time_ms = detection_wall * 1000.0 / max(1, evaluation_units)

    return {
        "experiment_id": job.experiment_id,
        "detector": job.detector,
        "dataset": job.dataset,
        "replay_rate": int(job.replay_rate),
        "rate_label": job.rate_label,
        "architecture": job.architecture,
        "window_seconds": job.window_seconds,
        "canonical_dataset_duration_sec": canonical["duration_seconds"],
        "dataset_arrival_rate_rows_per_sec": canonical["arrival_rate_rows_per_sec"],
        "canonical_source_experiment_id": canonical["source_experiment_id"],
        "evaluation_units": int(evaluation_units),
        "effective_duration_sec": effective_duration_seconds,
        "buffering_unit_seconds": buffering_unit_seconds,
        "estimated_input_rows": estimated_input_rows,
        "pre_inference_total_sec": featurization_wall,
        "inference_total_sec": detection_wall,
        "total_wall_time_sec": total_wall,
        "processing_time_per_unit_sec": processing_time_per_unit_sec,
        "graph_build_time_ms": pre_inference_time_ms,
        "feature_time_ms": 0.0,
        "pre_inference_time_ms": pre_inference_time_ms,
        "inference_time_ms": inference_time_ms,
        "timing_granularity": "combined_featurization_from_archived_runtime",
        "peak_rss_gib": peak_rss_gib,
        "mean_cpu_percent": _float(runtime_row.get("run_mean_cpu_percent")),
        "offline_auc": _float(runtime_row.get("auc")),
        "offline_ap": _float(runtime_row.get("ap")),
        "offline_tpr": _float(runtime_row.get("tpr")),
        "offline_cdr": _float(runtime_row.get("cdr")),
        "offline_experiment_id": runtime_row.get("experiment_id", ""),
        "offline_metrics_path": runtime_row.get("metrics_path", ""),
        "runtime_source": runtime_source,
        "input_stats_source": "eg5_anchor",
        "input_stats_experiment_id": input_stats.get("experiment_id", ""),
        "input_stats_source_experiment_id": input_stats.get("source_experiment_id", ""),
    }

def generate_config(job: EG9Job) -> Path:
    ensure_eg9_dirs()
    profile = _job_profile(job)
    payload = {
        "name": f"EG9 {job.detector.upper()} {job.dataset} {job.rate_label}",
        "experiment_id": job.experiment_id,
        "detector": job.detector,
        "dataset": job.dataset,
        "replay_rate": job.replay_rate,
        "rate_label": job.rate_label,
        "architecture": job.architecture,
        "window_seconds": job.window_seconds,
        "profile": deepcopy(profile),
        "sources": {
            "eg1_raw_results": str(EG1_RAW_RESULTS_PATH),
            "eg3_raw_results": str(EG3_RAW_RESULTS_PATH),
            "eg5_raw_results": str(EG5_RAW_RESULTS_PATH),
        },
    }
    dump_yaml(job.config_path, payload)
    return job.config_path

def _append_log(log_handle, message: str) -> None:
    log_handle.write(message.rstrip() + "\n")
    log_handle.flush()

def _write_replay_progress(job: EG9Job, payload: dict[str, Any]) -> None:
    job.replay_progress_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def _simulate_latency_trace(job: EG9Job, profile: dict[str, Any], log_handle) -> dict[str, Any]:
    units = int(profile["evaluation_units"])
    replay_rate = int(profile["replay_rate"])
    total_wall_time_sec = float(profile["total_wall_time_sec"])
    processing_time_per_unit_sec = float(profile["processing_time_per_unit_sec"])
    estimated_input_rows = float(profile["estimated_input_rows"])
    rows_per_unit = estimated_input_rows / max(1, units)

    if profile["architecture"] == "static":
        interarrival_sec = float(profile["effective_duration_sec"]) / float(replay_rate)
    else:
        interarrival_sec = float(profile["buffering_unit_seconds"]) / float(replay_rate)

    end_replay_sec = interarrival_sec * float(units)
    peak_backlog_units = 0
    current_finish_sec = 0.0
    first_alert_sec: float | None = None
    trace_rows: list[dict[str, Any]] = []
    update_every = max(1, units // 20)

    for unit_index in range(1, units + 1):
        arrival_time_sec = interarrival_sec * unit_index
        start_process_sec = max(arrival_time_sec, current_finish_sec)
        finish_time_sec = start_process_sec + processing_time_per_unit_sec
        lag_sec = max(0.0, finish_time_sec - arrival_time_sec)
        current_finish_sec = finish_time_sec
        arrived_by_finish = min(units, int(math.floor((finish_time_sec / interarrival_sec) + 1e-9))) if interarrival_sec > 0 else units
        backlog_units_after_finish = max(0, arrived_by_finish - unit_index)
        peak_backlog_units = max(peak_backlog_units, backlog_units_after_finish)
        backlog_rows_after_finish = backlog_units_after_finish * rows_per_unit
        if first_alert_sec is None:
            first_alert_sec = finish_time_sec

        trace_rows.append(
            {
                "unit_index": unit_index,
                "arrival_time_sec": round(arrival_time_sec, 6),
                "start_process_sec": round(start_process_sec, 6),
                "finish_time_sec": round(finish_time_sec, 6),
                "processing_time_sec": round(processing_time_per_unit_sec, 6),
                "lag_sec": round(lag_sec, 6),
                "backlog_units_after_finish": backlog_units_after_finish,
                "backlog_rows_after_finish": round(backlog_rows_after_finish, 6),
            }
        )

        if unit_index == 1 or unit_index == units or (unit_index % update_every == 0):
            progress_payload = {
                "experiment_id": job.experiment_id,
                "status": "running",
                "detector": job.detector,
                "dataset": job.dataset,
                "replay_rate": replay_rate,
                "processed_units": unit_index,
                "total_units": units,
                "completion_ratio": (unit_index / units) if units else 1.0,
                "current_lag_sec": round(lag_sec, 6),
                "max_backlog_units": peak_backlog_units,
                "interarrival_sec": round(interarrival_sec, 6),
                "processing_time_per_unit_sec": round(processing_time_per_unit_sec, 6),
            }
            _write_replay_progress(job, progress_payload)
            _append_log(
                log_handle,
                (
                    f"[unit {unit_index}/{units}] arrival={arrival_time_sec:.3f}s "
                    f"finish={finish_time_sec:.3f}s lag={lag_sec:.3f}s "
                    f"peak_backlog_units={peak_backlog_units}"
                ),
            )

    final_lag_sec = max(0.0, current_finish_sec - end_replay_sec)
    final_backlog_units = int(trace_rows[-1]["backlog_units_after_finish"]) if trace_rows else 0
    keep_up = final_backlog_units == 0 and final_lag_sec <= max(1.0, interarrival_sec)

    write_csv(
        job.latency_trace_path,
        trace_rows,
        [
            "unit_index",
            "arrival_time_sec",
            "start_process_sec",
            "finish_time_sec",
            "processing_time_sec",
            "lag_sec",
            "backlog_units_after_finish",
            "backlog_rows_after_finish",
        ],
    )

    _write_replay_progress(
        job,
        {
            "experiment_id": job.experiment_id,
            "status": "success",
            "detector": job.detector,
            "dataset": job.dataset,
            "replay_rate": replay_rate,
            "processed_units": units,
            "total_units": units,
            "completion_ratio": 1.0,
            "final_lag_sec": round(final_lag_sec, 6),
            "max_backlog_units": peak_backlog_units,
            "keep_up": keep_up,
        },
    )

    return {
        "buffering_delay_sec": interarrival_sec,
        "end_to_first_alert_sec": float(first_alert_sec or 0.0),
        "end_to_drain_sec": current_finish_sec,
        "final_lag_sec": final_lag_sec,
        "peak_backlog_units": peak_backlog_units,
        "peak_backlog_rows": peak_backlog_units * rows_per_unit,
        "keep_up": keep_up,
    }

def run_job(job: EG9Job, *, force: bool = False) -> str:
    config_path = generate_config(job)
    if not force and job_status(job) == "success":
        return "skipped"

    config = load_yaml(config_path)
    profile = dict(config["profile"])
    job.metadata_root.mkdir(parents=True, exist_ok=True)
    job.logs_root.mkdir(parents=True, exist_ok=True)
    log_path = job.logs_root / "replay.stdout.log"

    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            _append_log(log_handle, f"experiment_id={job.experiment_id}")
            _append_log(log_handle, f"detector={job.detector} dataset={job.dataset} rate={job.rate_label}")
            _append_log(log_handle, f"runtime_source={profile['runtime_source']} input_stats_source={profile['input_stats_source']}")
            _append_log(
                log_handle,
                (
                    f"units={profile['evaluation_units']} "
                    f"buffering_unit_seconds={profile['buffering_unit_seconds']} "
                    f"processing_time_per_unit_sec={profile['processing_time_per_unit_sec']:.6f}"
                ),
            )
            dump_yaml(job.execution_config_path, config)
            replay = _simulate_latency_trace(job, profile, log_handle)

        arrival_rows_per_sec = float(profile["dataset_arrival_rate_rows_per_sec"]) * float(profile["replay_rate"])
        throughput_rows_per_sec = float(profile["estimated_input_rows"]) / max(1e-9, float(profile["total_wall_time_sec"]))
        metrics_row = {
            "experiment_id": job.experiment_id,
            "detector": job.detector,
            "dataset": job.dataset,
            "replay_rate": int(profile["replay_rate"]),
            "rate_label": profile["rate_label"],
            "architecture": profile["architecture"],
            "window_seconds": profile["window_seconds"] or "",
            "evaluation_units": int(profile["evaluation_units"]),
            "effective_duration_sec": round(float(profile["effective_duration_sec"]), 6),
            "estimated_input_rows": round(float(profile["estimated_input_rows"]), 6),
            "buffering_delay_sec": round(float(replay["buffering_delay_sec"]), 6),
            "graph_build_time_ms": round(float(profile["graph_build_time_ms"]), 6),
            "feature_time_ms": round(float(profile["feature_time_ms"]), 6),
            "pre_inference_time_ms": round(float(profile["pre_inference_time_ms"]), 6),
            "inference_time_ms": round(float(profile["inference_time_ms"]), 6),
            "processing_time_per_unit_sec": round(float(profile["processing_time_per_unit_sec"]), 6),
            "arrival_rows_per_sec": round(arrival_rows_per_sec, 6),
            "throughput_rows_per_sec": round(throughput_rows_per_sec, 6),
            "end_to_first_alert_sec": round(float(replay["end_to_first_alert_sec"]), 6),
            "end_to_drain_sec": round(float(replay["end_to_drain_sec"]), 6),
            "final_lag_sec": round(float(replay["final_lag_sec"]), 6),
            "peak_backlog_units": int(replay["peak_backlog_units"]),
            "peak_backlog_rows": round(float(replay["peak_backlog_rows"]), 6),
            "keep_up": str(bool(replay["keep_up"])).lower(),
            "offline_auc": profile["offline_auc"] if profile["offline_auc"] is not None else "",
            "offline_ap": profile["offline_ap"] if profile["offline_ap"] is not None else "",
            "offline_tpr": profile["offline_tpr"] if profile["offline_tpr"] is not None else "",
            "offline_cdr": profile["offline_cdr"] if profile["offline_cdr"] is not None else "",
            "peak_rss_gib": round(float(profile["peak_rss_gib"]), 6),
            "mean_cpu_percent": profile["mean_cpu_percent"] if profile["mean_cpu_percent"] is not None else "",
            "runtime_source": profile["runtime_source"],
            "input_stats_source": profile["input_stats_source"],
            "offline_experiment_id": profile["offline_experiment_id"],
            "offline_metrics_path": profile["offline_metrics_path"],
            "canonical_source_experiment_id": profile["canonical_source_experiment_id"],
            "timing_granularity": profile["timing_granularity"],
        }
        write_csv(job.readiness_metrics_path, [metrics_row], list(metrics_row.keys()))
        summary = {
            "experiment_id": job.experiment_id,
            "status": "success",
            "config_path": str(config_path),
            "execution_config_path": str(job.execution_config_path),
            "readiness_metrics_path": str(job.readiness_metrics_path),
            "latency_trace_path": str(job.latency_trace_path),
            "replay_progress_path": str(job.replay_progress_path),
            "stdout_log_path": str(log_path),
            "runtime_source": profile["runtime_source"],
            "input_stats_source": profile["input_stats_source"],
            "keep_up": bool(replay["keep_up"]),
        }
    except Exception as exc:
        summary = {
            "experiment_id": job.experiment_id,
            "status": "failed",
            "config_path": str(config_path),
            "execution_config_path": str(job.execution_config_path),
            "stdout_log_path": str(log_path),
            "error": repr(exc),
        }

    job.run_summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return "success" if summary["status"] == "success" else "failed"

def job_status(job: EG9Job) -> str:
    if job.run_summary_path.exists():
        try:
            summary = json.loads(job.run_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return "failed"
        if summary.get("status") == "success" and job.readiness_metrics_path.exists():
            return "success"
        if summary.get("status") == "failed":
            return "failed"
    if job.metadata_root.exists():
        return "partial"
    return "pending"

def build_status_rows(jobs: list[EG9Job]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        summary = {}
        if job.run_summary_path.exists():
            try:
                summary = json.loads(job.run_summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                summary = {}
        rows.append(
            {
                "detector": job.detector,
                "dataset": job.dataset,
                "replay_rate": job.replay_rate,
                "rate_label": job.rate_label,
                "architecture": job.architecture,
                "window_seconds": job.window_seconds or "",
                "experiment_id": job.experiment_id,
                "status": job_status(job),
                "config_path": str(job.config_path),
                "readiness_metrics_path": str(job.readiness_metrics_path),
                "latency_trace_path": str(job.latency_trace_path),
                "replay_progress_path": str(job.replay_progress_path),
                "run_summary_path": str(job.run_summary_path),
                "runtime_source": summary.get("runtime_source", ""),
                "input_stats_source": summary.get("input_stats_source", ""),
            }
        )
    return rows

def write_matrix_report(jobs: list[EG9Job], path: Path | None = None) -> Path:
    rows = [
        {
            "detector": job.detector,
            "dataset": job.dataset,
            "replay_rate": job.replay_rate,
            "rate_label": job.rate_label,
            "architecture": job.architecture,
            "window_seconds": job.window_seconds or "",
            "experiment_id": job.experiment_id,
            "config_path": str(job.config_path),
        }
        for job in jobs
    ]
    path = path or (ORCHESTRATOR_ROOT / "job_matrix.csv")
    write_csv(
        path,
        rows,
        [
            "detector",
            "dataset",
            "replay_rate",
            "rate_label",
            "architecture",
            "window_seconds",
            "experiment_id",
            "config_path",
        ],
    )
    return path

def write_status_report(jobs: list[EG9Job], path: Path | None = None) -> Path:
    rows = build_status_rows(jobs)
    path = path or (ORCHESTRATOR_ROOT / "job_status.csv")
    write_csv(
        path,
        rows,
        [
            "detector",
            "dataset",
            "replay_rate",
            "rate_label",
            "architecture",
            "window_seconds",
            "experiment_id",
            "status",
            "config_path",
            "readiness_metrics_path",
            "latency_trace_path",
            "replay_progress_path",
            "run_summary_path",
            "runtime_source",
            "input_stats_source",
        ],
    )
    return path

def write_progress_report(
    jobs: list[EG9Job],
    *,
    json_path: Path | None = None,
    md_path: Path | None = None,
    scope_name: str = "global",
    attempted_jobs: int | None = None,
    failed_jobs: int | None = None,
) -> tuple[Path, Path]:
    rows = build_status_rows(jobs)
    total = len(rows)
    status_counts = {
        status: sum(1 for row in rows if row["status"] == status)
        for status in ("pending", "partial", "failed", "success")
    }
    completed = status_counts["success"] + status_counts["failed"]
    completion_ratio = (completed / total) if total else 0.0
    success_ratio = (status_counts["success"] / total) if total else 0.0

    detector_progress = []
    for detector in detector_list():
        detector_rows = [row for row in rows if row["detector"] == detector]
        if not detector_rows:
            continue
        detector_progress.append(
            {
                "detector": detector,
                "total": len(detector_rows),
                "success": sum(1 for row in detector_rows if row["status"] == "success"),
                "failed": sum(1 for row in detector_rows if row["status"] == "failed"),
                "partial": sum(1 for row in detector_rows if row["status"] == "partial"),
                "pending": sum(1 for row in detector_rows if row["status"] == "pending"),
            }
        )

    rate_progress = []
    for replay_rate in rate_list():
        rate_rows = [row for row in rows if int(row["replay_rate"]) == int(replay_rate)]
        if not rate_rows:
            continue
        rate_progress.append(
            {
                "replay_rate": int(replay_rate),
                "success": sum(1 for row in rate_rows if row["status"] == "success"),
                "failed": sum(1 for row in rate_rows if row["status"] == "failed"),
                "partial": sum(1 for row in rate_rows if row["status"] == "partial"),
                "pending": sum(1 for row in rate_rows if row["status"] == "pending"),
                "total": len(rate_rows),
            }
        )

    payload = {
        "scope": scope_name,
        "total_jobs": total,
        "completed_jobs": completed,
        "remaining_jobs": total - completed,
        "success_jobs": status_counts["success"],
        "failed_jobs": status_counts["failed"],
        "partial_jobs": status_counts["partial"],
        "pending_jobs": status_counts["pending"],
        "completion_ratio": completion_ratio,
        "success_ratio": success_ratio,
        "attempted_jobs": attempted_jobs,
        "run_failures": failed_jobs,
        "per_detector": detector_progress,
        "per_rate": rate_progress,
    }

    json_path = json_path or PROGRESS_JSON_PATH
    md_path = md_path or PROGRESS_MD_PATH
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        f"# EG9 Progress ({scope_name})",
        "",
        f"- total_jobs: {total}",
        f"- completed_jobs: {completed}",
        f"- remaining_jobs: {total - completed}",
        f"- success_jobs: {status_counts['success']}",
        f"- failed_jobs: {status_counts['failed']}",
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
            f"- {row['detector']}: {row['success'] + row['failed']}/{row['total']} complete "
            f"(success={row['success']}, failed={row['failed']}, partial={row['partial']}, pending={row['pending']})"
        )
    lines.extend(["", "## Per Rate"])
    for row in rate_progress:
        lines.append(
            f"- {row['replay_rate']}x: {row['success'] + row['failed']}/{row['total']} complete "
            f"(success={row['success']}, failed={row['failed']}, partial={row['partial']}, pending={row['pending']})"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path

def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}

def _job_verdict(row: dict[str, Any]) -> str:
    keep_up = _to_bool(row["keep_up"])
    buffering_delay = float(row["buffering_delay_sec"])
    architecture = str(row["architecture"])
    if architecture == "static":
        return "batch_only"
    if keep_up and buffering_delay <= REAL_TIME_BUFFER_MAX_SEC:
        return "real_time_ready"
    if keep_up and buffering_delay <= NEAR_REAL_TIME_BUFFER_MAX_SEC:
        return "near_real_time"
    if keep_up:
        return "batch_gated"
    return "lagging"

def _summary_verdict(rows: list[dict[str, Any]]) -> str:
    by_rate = {int(row["replay_rate"]): row for row in rows}
    base = by_rate[min(by_rate)]
    if str(base["architecture"]) == "static":
        return "batch_only"
    keep_1x = _to_bool(by_rate.get(1, {}).get("keep_up", False))
    keep_2x = _to_bool(by_rate.get(2, {}).get("keep_up", False))
    keep_5x = _to_bool(by_rate.get(5, {}).get("keep_up", False))
    buffer_1x = float(by_rate.get(1, base)["buffering_delay_sec"])
    if keep_1x and keep_2x and keep_5x and buffer_1x <= REAL_TIME_BUFFER_MAX_SEC:
        return "real_time_ready"
    if keep_1x and buffer_1x <= NEAR_REAL_TIME_BUFFER_MAX_SEC:
        return "near_real_time"
    return "batch_only"

def _primary_bottleneck(rows: list[dict[str, Any]]) -> str:
    base = next((row for row in rows if int(row["replay_rate"]) == 1), rows[0])
    if str(base["architecture"]) == "static":
        return "static_graph_batching"
    if float(base["buffering_delay_sec"]) > float(base["processing_time_per_unit_sec"]) * 2.0:
        return "buffering_delay"
    if not _to_bool(base["keep_up"]):
        return "throughput_limit"
    if any(not _to_bool(row["keep_up"]) for row in rows):
        return "surge_backlog"
    return "buffering_delay"

def summarize_jobs(jobs: list[EG9Job], output_root: Path | None = None) -> tuple[Path, Path]:
    output_root = output_root or REPORT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, Any]] = []
    for job in jobs:
        if job_status(job) != "success":
            continue
        for row in read_csv(job.readiness_metrics_path):
            enriched = dict(row)
            enriched["job_verdict"] = _job_verdict(row)
            raw_rows.append(enriched)

    raw_path = output_root / "raw_results.csv"
    raw_fieldnames = [
        "experiment_id",
        "detector",
        "dataset",
        "replay_rate",
        "rate_label",
        "architecture",
        "window_seconds",
        "evaluation_units",
        "effective_duration_sec",
        "estimated_input_rows",
        "buffering_delay_sec",
        "graph_build_time_ms",
        "feature_time_ms",
        "pre_inference_time_ms",
        "inference_time_ms",
        "processing_time_per_unit_sec",
        "arrival_rows_per_sec",
        "throughput_rows_per_sec",
        "end_to_first_alert_sec",
        "end_to_drain_sec",
        "final_lag_sec",
        "peak_backlog_units",
        "peak_backlog_rows",
        "keep_up",
        "job_verdict",
        "offline_auc",
        "offline_ap",
        "offline_tpr",
        "offline_cdr",
        "peak_rss_gib",
        "mean_cpu_percent",
        "runtime_source",
        "input_stats_source",
        "offline_experiment_id",
        "offline_metrics_path",
        "canonical_source_experiment_id",
        "timing_granularity",
    ]
    write_csv(raw_path, raw_rows, raw_fieldnames)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        grouped.setdefault((row["detector"], row["dataset"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    common_subset_rows: list[dict[str, Any]] = []

    for (detector, dataset), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda item: int(item["replay_rate"]))
        keep_up_rates = [int(row["replay_rate"]) for row in rows if _to_bool(row["keep_up"])]
        max_keep_up_rate = max(keep_up_rates) if keep_up_rates else ""
        verdict = _summary_verdict(rows)
        bottleneck = _primary_bottleneck(rows)
        row_1x = next((row for row in rows if int(row["replay_rate"]) == 1), rows[0])
        row_2x = next((row for row in rows if int(row["replay_rate"]) == 2), None)
        row_5x = next((row for row in rows if int(row["replay_rate"]) == 5), None)
        summary_row = {
            "detector": detector,
            "dataset": dataset,
            "architecture": row_1x["architecture"],
            "window_seconds": row_1x["window_seconds"],
            "keep_up_1x": row_1x["keep_up"],
            "keep_up_2x": row_2x["keep_up"] if row_2x else "",
            "keep_up_5x": row_5x["keep_up"] if row_5x else "",
            "max_keep_up_rate": max_keep_up_rate,
            "verdict": verdict,
            "primary_bottleneck": bottleneck,
            "buffering_delay_sec_1x": row_1x["buffering_delay_sec"],
            "end_to_first_alert_sec_1x": row_1x["end_to_first_alert_sec"],
            "final_lag_sec_5x": row_5x["final_lag_sec"] if row_5x else "",
            "peak_backlog_units_5x": row_5x["peak_backlog_units"] if row_5x else "",
            "offline_auc": row_1x["offline_auc"],
            "offline_ap": row_1x["offline_ap"],
            "offline_tpr": row_1x["offline_tpr"],
            "offline_cdr": row_1x["offline_cdr"],
            "runtime_source": row_1x["runtime_source"],
        }
        summary_rows.append(summary_row)
        evidence_rows.append(summary_row)
        if dataset in COMMON_SUBSET_DATASETS:
            common_subset_rows.append(summary_row)

    summary_path = output_root / "readiness_summary.csv"
    evidence_path = output_root / "evidence_table.csv"
    common_subset_path = output_root / "evidence_table_common_subset.csv"
    summary_fields = [
        "detector",
        "dataset",
        "architecture",
        "window_seconds",
        "keep_up_1x",
        "keep_up_2x",
        "keep_up_5x",
        "max_keep_up_rate",
        "verdict",
        "primary_bottleneck",
        "buffering_delay_sec_1x",
        "end_to_first_alert_sec_1x",
        "final_lag_sec_5x",
        "peak_backlog_units_5x",
        "offline_auc",
        "offline_ap",
        "offline_tpr",
        "offline_cdr",
        "runtime_source",
    ]
    write_csv(summary_path, summary_rows, summary_fields)
    write_csv(evidence_path, evidence_rows, summary_fields)
    write_csv(common_subset_path, common_subset_rows, summary_fields)

    verdict_counts: dict[str, int] = {}
    for row in summary_rows:
        verdict_counts[row["verdict"]] = verdict_counts.get(row["verdict"], 0) + 1
    buffer_values = [float(row["buffering_delay_sec_1x"]) for row in summary_rows] if summary_rows else [0.0]
    lagging_rows = [row for row in summary_rows if row["verdict"] == "batch_only"]
    near_rt_rows = [row for row in summary_rows if row["verdict"] == "near_real_time"]
    near_rt_label = ", ".join(f"{row['detector']}/{row['dataset']}" for row in near_rt_rows) or "none"
    batch_only_label = ", ".join(f"{row['detector']}/{row['dataset']}" for row in lagging_rows) or "none"
    lines = [
        "# EG9 Conclusion",
        "",
        f"- total_groups: {len(summary_rows)}",
        f"- verdict_batch_only: {verdict_counts.get('batch_only', 0)}",
        f"- verdict_near_real_time: {verdict_counts.get('near_real_time', 0)}",
        f"- verdict_real_time_ready: {verdict_counts.get('real_time_ready', 0)}",
        f"- median_buffering_delay_sec_1x: {statistics.median(buffer_values):.3f}",
        "",
        "## Main Readout",
        "",
        "- `EG9` reuses archived detector runs to separate operational readiness from offline detection quality.",
        "- Old systems can still have non-trivial `AP/AUC` while remaining batching-gated in deployment terms.",
        f"- `near_real_time` groups: {near_rt_label}",
        f"- `batch_only` groups: {batch_only_label}",
    ]
    (output_root / "conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return raw_path, summary_path

def detector_env_status(
    detectors: list[str] | None = None,
    datasets: list[str] | None = None,
) -> tuple[Path, list[dict[str, str]]]:
    selected_detectors = detectors or detector_list()
    selected_datasets = datasets or dataset_list()
    rows: list[dict[str, str]] = []
    for detector in selected_detectors:
        for dataset in selected_datasets:
            if dataset not in SUPPORTED_DATASETS.get(detector, ()):
                continue
            runtime_ok = True
            input_ok = True
            note_parts: list[str] = []
            try:
                _runtime_row(detector, dataset)
            except Exception as exc:
                runtime_ok = False
                note_parts.append(f"runtime:{exc}")
            try:
                _input_stats_row(detector, dataset)
                _canonical_dataset_profile(dataset)
            except Exception as exc:
                input_ok = False
                note_parts.append(f"input:{exc}")
            rows.append(
                {
                    "detector": detector,
                    "dataset": dataset,
                    "runtime_source_ok": str(runtime_ok).lower(),
                    "input_stats_ok": str(input_ok).lower(),
                    "note": "; ".join(note_parts),
                }
            )
    path = ORCHESTRATOR_ROOT / "source_status.csv"
    write_csv(path, rows, ["detector", "dataset", "runtime_source_ok", "input_stats_ok", "note"])
    return path, rows
