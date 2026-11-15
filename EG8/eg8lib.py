from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from gids_eval.cdr import compute_cdr_summary, load_prediction_groups
from gids_eval.paths import get_paths
from gids_eval.process_monitor import run_command_logged

_PATHS = get_paths()
REPO_ROOT = _PATHS.repo_root
EG8_CODE_ROOT = REPO_ROOT / "EG8"
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DATA_ROOT = _PATHS.sc_data_root("eg8")
CANONICAL_ROOT = DATA_ROOT / "canonical" / "pikachu"
SHARED_GRAPH_ROOT = DATA_ROOT / "shared_graph"
ADAPTER_ROOT = DATA_ROOT / "adapters"
GENERATED_CONFIG_ROOT = DATA_ROOT / "generated-configs"
ORCHESTRATOR_ROOT = DATA_ROOT / "orchestrator"
REPORT_ROOT = DATA_ROOT / "reports"
PROGRESS_JSON_PATH = ORCHESTRATOR_ROOT / "progress.json"
PROGRESS_MD_PATH = ORCHESTRATOR_ROOT / "progress.md"
FULL_RUN_LOG_PATH = ORCHESTRATOR_ROOT / "full_run.log"
RUN_LABEL = "eg8"
CONDA_ROOT = _PATHS.conda_root

DETECTORS = ("argus", "euler", "pikachu", "vgrnn", "anomal")
GRAPH_DETECTORS = ("argus", "euler", "vgrnn")
DATASETS = ("0501", "cic", "lanl", "optc")
ANOMAL_DATASETS = ("0501", "cic")

WINDOW_SECONDS = {
    "0501": 300,
    "cic": 3000,
    "lanl": 3600,
    "optc": 3600,
}

WINDOW_LABELS = {
    "0501": "5m",
    "cic": "50m",
    "lanl": "1h",
    "optc": "1h",
}

CANONICAL_SOURCE_EXPERIMENTS = {
    "0501": "eg1_pikachu_0501_5m_s00",
    "cic": "eg1_pikachu_cic_50m_s00",
    "lanl": "eg1_pikachu_lanl_1h_s00",
    "optc": "eg1_pikachu_optc_1h_s00",
}

CANONICAL_SOURCE_ROOTS = {
    dataset: _PATHS.sc_data_root("eg1") / "experiments" / experiment_id / "eg1/output/modules/featurization/output/data_root/pikachu_sc1" / dataset
    for dataset, experiment_id in CANONICAL_SOURCE_EXPERIMENTS.items()
}

DETECTOR_ENV_PREFIXES = {
    "argus": CONDA_ROOT / "envs/sc1-argus",
    "euler": CONDA_ROOT / "envs/sc1-euler",
    "pikachu": CONDA_ROOT / "envs/sc1-pikachu",
    "vgrnn": CONDA_ROOT / "envs/sc1-vgrnn",
    "anomal": CONDA_ROOT / "envs/sc1-anomal",
}

GRAPH_FILE_DELTAS = {
    "0501": 100000,
    "cic": 100000,
    "lanl": 10000,
    "optc": 10000,
}

COMMON_REPORT_FIELDS = [
    "protocol",
    "source_kind",
    "detector",
    "dataset",
    "window_seconds",
    "window_label",
    "seed",
    "experiment_id",
    "status",
    "auc",
    "ap",
    "cdr",
    "cdr_auc",
    "campaign_count",
    "detected_campaigns",
    "cdr_meaningful",
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
    "prepare_wall_time_sec",
    "detection_wall_time_sec",
    "total_wall_time_sec",
    "run_peak_rss_gib",
    "input_rows",
    "positive_rows",
    "node_count",
    "metrics_path",
    "predictions_path",
]

@dataclass(frozen=True)
class EG8Job:
    detector: str
    dataset: str
    seed: int = 0

    @property
    def experiment_id(self) -> str:
        return f"eg8_{self.detector}_{self.dataset}_aligned_s{self.seed:02d}"

    @property
    def window_seconds(self) -> int:
        return int(WINDOW_SECONDS[self.dataset])

    @property
    def window_label(self) -> str:
        return str(WINDOW_LABELS[self.dataset])

    @property
    def workspace_root(self) -> Path:
        return DATA_ROOT / "experiments" / self.experiment_id / RUN_LABEL

    @property
    def input_root(self) -> Path:
        return self.workspace_root / "input"

    @property
    def output_root(self) -> Path:
        return self.workspace_root / "output"

    @property
    def metadata_root(self) -> Path:
        return self.workspace_root / "metadata"

    @property
    def logs_root(self) -> Path:
        return self.metadata_root / "logs"

    @property
    def results_root(self) -> Path:
        return self.output_root / "results"

    @property
    def models_root(self) -> Path:
        return self.output_root / "models"

    @property
    def baseline_metrics_path(self) -> Path:
        return self.metadata_root / "baseline_metrics.csv"

    @property
    def execution_config_path(self) -> Path:
        return self.metadata_root / "execution_config.json"

    @property
    def run_summary_path(self) -> Path:
        return self.metadata_root / "run_summary.json"

    @property
    def prepare_stdout_path(self) -> Path:
        return self.logs_root / "prepare.stdout.log"

    @property
    def prepare_stderr_path(self) -> Path:
        return self.logs_root / "prepare.stderr.log"

    @property
    def detection_stdout_path(self) -> Path:
        return self.logs_root / "detection.stdout.log"

    @property
    def detection_stderr_path(self) -> Path:
        return self.logs_root / "detection.stderr.log"

    @property
    def aligned_metadata_path(self) -> Path:
        return self.metadata_root / "aligned_input_metadata.json"

    @property
    def stage_usage_path(self) -> Path:
        return self.metadata_root / "resource_profile.json"

def ensure_eg8_dirs() -> None:
    for path in (
        DATA_ROOT,
        CANONICAL_ROOT,
        SHARED_GRAPH_ROOT,
        ADAPTER_ROOT,
        GENERATED_CONFIG_ROOT,
        ORCHESTRATOR_ROOT,
        REPORT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)

def detector_list() -> list[str]:
    return list(DETECTORS)

def parse_csv_list(raw: str | None, *, canonicalize: bool = False) -> list[str]:
    if not raw:
        return []
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not canonicalize:
        return values
    return [value.lower() for value in values]

def iter_jobs(*, detectors: Iterable[str] | None = None, datasets: Iterable[str] | None = None) -> list[EG8Job]:
    selected_detectors = [item.lower() for item in detectors] if detectors else list(DETECTORS)
    selected_datasets = [item.lower() for item in datasets] if datasets else list(DATASETS)
    ordered: list[EG8Job] = []
    for detector in DETECTORS:
        if detector not in selected_detectors:
            continue
        for dataset in DATASETS:
            if dataset not in selected_datasets:
                continue
            if detector == "anomal" and dataset not in ANOMAL_DATASETS:
                continue
            ordered.append(EG8Job(detector=detector, dataset=dataset, seed=0))
    return ordered

def _utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _load_run_summary(job: EG8Job) -> dict[str, Any] | None:
    if not job.run_summary_path.exists():
        return None
    try:
        return _read_json(job.run_summary_path)
    except Exception:
        return None

def _numeric(value: Any, digits: int = 6) -> float:
    try:
        return round(float(value), digits)
    except Exception:
        return 0.0

def _safe_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return None

def _safe_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(float(value))
    except Exception:
        return None

def _gib_from_kb(value: Any) -> float:
    return round(float(value or 0.0) / (1024.0 * 1024.0), 6)

def _stable_hash_token(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)

def _ensure_removed(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)

def _ensure_symlink(link_path: Path, target: Path) -> None:
    if link_path.is_symlink():
        if link_path.resolve() == target.resolve():
            return
        link_path.unlink()
    elif link_path.exists():
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target, target_is_directory=True)

def _job_python_bin(detector: str) -> str:
    prefix = DETECTOR_ENV_PREFIXES[detector]
    candidate = prefix / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable

def _append_log(handle, message: str) -> None:
    handle.write(message.rstrip() + "\n")
    handle.flush()

def _canonical_dataset_root(dataset: str) -> Path:
    return CANONICAL_ROOT / dataset

def _canonical_input_path(dataset: str) -> Path:
    return _canonical_dataset_root(dataset) / "input.csv"

def _canonical_metadata_path(dataset: str) -> Path:
    return _canonical_dataset_root(dataset) / "metadata.json"

def ensure_canonical_dataset(dataset: str, *, log_handle=None) -> dict[str, Any]:
    ensure_eg8_dirs()
    target_root = _canonical_dataset_root(dataset)
    target_root.mkdir(parents=True, exist_ok=True)
    input_path = target_root / "input.csv"
    metadata_path = target_root / "metadata.json"
    source_root = CANONICAL_SOURCE_ROOTS[dataset]
    source_metadata_path = source_root / "metadata.json"
    source_input_path = source_root / "input.csv"
    if input_path.exists() and metadata_path.exists():
        payload = _read_json(metadata_path)
        payload["root"] = str(target_root)
        return payload
    if log_handle is not None:
        _append_log(log_handle, f"[canonical] staging {dataset} from {source_root}")
    shutil.copy2(source_input_path, input_path)
    shutil.copy2(source_metadata_path, metadata_path)
    extra_name = "redteam_optc_small.txt"
    extra_path = source_root / extra_name
    if extra_path.exists():
        shutil.copy2(extra_path, target_root / extra_name)
    payload = _read_json(metadata_path)
    payload["root"] = str(target_root)
    payload["source_experiment_id"] = CANONICAL_SOURCE_EXPERIMENTS[dataset]
    payload["protocol"] = "pikachu_bounded_canonical"
    _write_json(metadata_path, payload)
    return payload

def _graph_root(name: str) -> Path:
    return SHARED_GRAPH_ROOT / name

def _detector_adapter_root(detector: str) -> Path:
    return ADAPTER_ROOT / detector

def _iter_canonical_rows(dataset: str):
    with _canonical_input_path(dataset).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row

def _build_shared_graph_root(dataset: str, variant: str, *, log_handle=None) -> tuple[Path, dict[str, Any]]:
    if dataset == "lanl" and variant == "argus":
        target_root = _graph_root("lanl_argus")
    elif dataset == "lanl":
        target_root = _graph_root("lanl")
    elif dataset == "0501":
        target_root = _graph_root("0501") / "0501"
    else:
        target_root = _graph_root(dataset)
    metadata_path = target_root / "eg8_adapter_metadata.json"
    if metadata_path.exists():
        return target_root, _read_json(metadata_path)

    target_root.mkdir(parents=True, exist_ok=True)
    canonical_meta = ensure_canonical_dataset(dataset, log_handle=log_handle)
    initial_time = int(canonical_meta.get("initial_time", 0) or 0)
    file_delta = int(GRAPH_FILE_DELTAS[dataset])
    node_map: dict[str, int] = {}
    reverse_map: list[str] = []
    snapshot_values: set[int] = set()
    row_count = 0
    positive_rows = 0
    current_bucket: int | None = None
    current_handle = None
    first_ts: int | None = None
    last_ts: int | None = None
    first_positive_ts: int | None = None

    def get_or_add(node: str) -> int:
        if node in node_map:
            return node_map[node]
        node_id = len(reverse_map)
        node_map[node] = node_id
        reverse_map.append(node)
        return node_id

    def open_bucket(bucket_value: int):
        nonlocal current_bucket, current_handle
        if current_handle is not None:
            current_handle.close()
        current_bucket = bucket_value
        current_path = target_root / f"{bucket_value}.txt"
        current_handle = current_path.open("w", encoding="utf-8")

    if log_handle is not None:
        _append_log(log_handle, f"[adapter] building graph root dataset={dataset} variant={variant} -> {target_root}")

    try:
        for row in _iter_canonical_rows(dataset):
            src = str(row.get("src_computer", ""))
            dst = str(row.get("dst_computer", ""))
            label = int(float(row.get("label", 0) or 0))
            campaign = str(row.get("campaign", "") if label == 1 else "")
            snapshot = int(float(row.get("snapshot", 0) or 0))
            raw_ts = int(float(row.get("timestamp", 0) or 0))
            graph_ts = raw_ts
            if dataset == "0501":
                graph_ts = int((raw_ts - initial_time) * 1000)
            if current_bucket is None or graph_ts // file_delta != current_bucket // file_delta:
                open_bucket(graph_ts - (graph_ts % file_delta))
            src_id = get_or_add(src)
            dst_id = get_or_add(dst)
            snapshot_values.add(snapshot)
            if variant == "argus" and dataset == "lanl":
                line = f"{graph_ts},{src_id},{dst_id},0,0,0,0,0,0,{label},{campaign}\n"
            elif dataset == "optc":
                line = f"{graph_ts},{src_id},{dst_id},0,0,0,0,0,{label},{campaign}\n"
            elif dataset == "cic":
                flow_a = 1 + snapshot
                flow_b = 1 + ((graph_ts // 60) % 17)
                flow_c = 1 + ((_stable_hash_token(src) + _stable_hash_token(dst)) % 23)
                line = f"{graph_ts},{src_id},{dst_id},{flow_a},{flow_b},{flow_c},{label},{campaign}\n"
            else:
                line = f"{graph_ts},{src_id},{dst_id},{label},{campaign}\n"
            current_handle.write(line)
            row_count += 1
            positive_rows += label
            first_ts = graph_ts if first_ts is None else min(first_ts, graph_ts)
            last_ts = graph_ts if last_ts is None else max(last_ts, graph_ts)
            if label == 1 and first_positive_ts is None:
                first_positive_ts = graph_ts
    finally:
        if current_handle is not None:
            current_handle.close()

    if first_ts is not None and last_ts is not None:
        first_bucket = first_ts - (first_ts % file_delta)
        last_bucket = last_ts - (last_ts % file_delta)
        bucket = first_bucket
        while bucket <= last_bucket:
            bucket_path = target_root / f"{bucket}.txt"
            if not bucket_path.exists():
                bucket_path.touch()
            bucket += file_delta

    with (target_root / "nmap.pkl").open("wb") as handle:
        pickle.dump(reverse_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if variant == "argus" and dataset == "lanl":
        with (target_root / "umap.pkl").open("wb") as handle:
            pickle.dump(["U0"], handle, protocol=pickle.HIGHEST_PROTOCOL)
    if dataset == "cic":
        _write_json(target_root / "meta.json", {"window_ms": file_delta})

    adapter_meta = {
        "dataset": dataset,
        "variant": variant,
        "target_root": str(target_root),
        "input_path": str(_canonical_input_path(dataset)),
        "row_count": int(row_count),
        "positive_rows": int(positive_rows),
        "node_count": int(len(reverse_map)),
        "snapshot_count": int(len(snapshot_values)),
        "first_timestamp": int(first_ts or 0),
        "last_timestamp": int(last_ts or 0),
        "first_positive_timestamp": int(first_positive_ts) if first_positive_ts is not None else int(last_ts or 0),
        "file_delta": int(file_delta),
        "source_experiment_id": CANONICAL_SOURCE_EXPERIMENTS[dataset],
    }
    _write_json(metadata_path, adapter_meta)
    return target_root, adapter_meta

def ensure_graph_adapter(detector: str, dataset: str, *, log_handle=None) -> tuple[Path, dict[str, Any]]:
    target_root = _detector_adapter_root(detector)
    target_root.mkdir(parents=True, exist_ok=True)
    if dataset == "0501":
        shared_root, meta = _build_shared_graph_root(dataset, "common", log_handle=log_handle)
        adapter_path = target_root / "0501" / "0501"
        _ensure_symlink(adapter_path, shared_root)
        return target_root, meta
    if dataset == "lanl":
        variant = "argus" if detector == "argus" else "common"
        shared_root, meta = _build_shared_graph_root(dataset, variant, log_handle=log_handle)
        adapter_path = target_root / "lanl"
        _ensure_symlink(adapter_path, shared_root)
        return target_root, meta
    shared_root, meta = _build_shared_graph_root(dataset, "common", log_handle=log_handle)
    adapter_path = target_root / dataset
    _ensure_symlink(adapter_path, shared_root)
    return target_root, meta

def ensure_anomal_adapter(dataset: str, *, log_handle=None) -> dict[str, Any]:
    target_root = ADAPTER_ROOT / "anomal" / dataset
    metadata_path = target_root / "metadata.json"
    if metadata_path.exists():
        return _read_json(metadata_path)

    target_root.mkdir(parents=True, exist_ok=True)
    canonical_meta = ensure_canonical_dataset(dataset, log_handle=log_handle)
    input_path = target_root / ("0501_aligned.csv" if dataset == "0501" else "cic_aligned.csv")
    row_count = 0
    positive_rows = 0
    split_index: int | None = None
    node_map: set[str] = set()

    if log_handle is not None:
        _append_log(log_handle, f"[adapter] building anomal root dataset={dataset} -> {input_path}")

    if dataset == "cic":
        with input_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "Src IP", "Dst IP", "duration", "bytes", "packets", "label", "campaign"])
            for row in _iter_canonical_rows(dataset):
                src = str(row.get("src_computer", ""))
                dst = str(row.get("dst_computer", ""))
                label = int(float(row.get("label", 0) or 0))
                timestamp = int(float(row.get("timestamp", 0) or 0))
                snapshot = int(float(row.get("snapshot", 0) or 0))
                campaign = str(row.get("campaign", "") if label == 1 else "")
                if split_index is None and label == 1:
                    split_index = row_count
                duration = max(1, snapshot + 1)
                bytes_v = 1 + ((_stable_hash_token(src) + _stable_hash_token(dst)) % 2048)
                packets = 1 + (timestamp % 17)
                writer.writerow([row_count, src, dst, duration, bytes_v, packets, label, campaign])
                row_count += 1
                positive_rows += label
                node_map.add(src)
                node_map.add(dst)
    else:
        with input_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            for row in _iter_canonical_rows(dataset):
                src = str(row.get("src_computer", ""))
                dst = str(row.get("dst_computer", ""))
                label = int(float(row.get("label", 0) or 0))
                timestamp = int(float(row.get("timestamp", 0) or 0))
                snapshot = int(float(row.get("snapshot", 0) or 0))
                campaign = str(row.get("campaign", "") if label == 1 else "")
                if split_index is None and label == 1:
                    split_index = row_count
                base = 1 + ((_stable_hash_token(src) ^ _stable_hash_token(dst)) % 256)
                writer.writerow(
                    [
                        src,
                        dst,
                        max(1, snapshot + 1),
                        base,
                        base + 1,
                        1,
                        1 + (timestamp % 7),
                        1 + (base % 31),
                        1 + ((base + timestamp) % 13),
                        1 + ((base + snapshot) % 29),
                        label,
                        campaign,
                    ]
                )
                row_count += 1
                positive_rows += label
                node_map.add(src)
                node_map.add(dst)

    meta = {
        "dataset": dataset,
        "input_path": str(input_path),
        "row_count": int(row_count),
        "positive_rows": int(positive_rows),
        "node_count": int(len(node_map)),
        "split_index": int(split_index if split_index is not None else max(int(row_count * 0.8), 1)),
        "source_experiment_id": CANONICAL_SOURCE_EXPERIMENTS[dataset],
        "trainwin": int(canonical_meta.get("trainwin", 0) or 0),
    }
    _write_json(metadata_path, meta)
    return meta

def _default_env() -> dict[str, str]:
    env = dict(os.environ)
    python_path_entries = [str(SRC_ROOT), str(REPO_ROOT)]
    current = env.get("PYTHONPATH", "").strip()
    if current:
        python_path_entries.append(current)
    env["PYTHONPATH"] = ":".join(python_path_entries)
    return env

def _prepare_execution(job: EG8Job, *, force: bool = False) -> dict[str, Any]:
    if force:
        _ensure_removed(job.results_root)
        _ensure_removed(job.models_root)
        if job.baseline_metrics_path.exists():
            job.baseline_metrics_path.unlink()
    job.logs_root.mkdir(parents=True, exist_ok=True)
    job.results_root.mkdir(parents=True, exist_ok=True)
    job.models_root.mkdir(parents=True, exist_ok=True)

    prepare_started = time.perf_counter()
    with job.prepare_stdout_path.open("a", encoding="utf-8") as stdout_handle:
        with job.prepare_stderr_path.open("a", encoding="utf-8") as stderr_handle:
            try:
                canonical_meta = ensure_canonical_dataset(job.dataset, log_handle=stdout_handle)
                input_rows = int(canonical_meta.get("row_count", 0) or 0)
                positive_rows = int(canonical_meta.get("positive_rows", 0) or 0)

                if job.detector == "pikachu":
                    metadata = dict(canonical_meta)
                    metadata["input_path"] = str(_canonical_input_path(job.dataset))
                    metadata["experiment_id"] = job.experiment_id
                    _write_json(job.aligned_metadata_path, metadata)
                    output_dir = job.results_root / job.dataset
                    metrics_path = output_dir / "metrics.csv"
                    predictions_path = output_dir / "predictions.csv"
                    command = [
                        _job_python_bin("pikachu"),
                        str(REPO_ROOT / "EG1" / "pikachu_run.py"),
                        "--dataset",
                        job.dataset,
                        "--metadata",
                        str(job.aligned_metadata_path),
                        "--detector-root",
                        str(REPO_ROOT / "local_runtime" / "detection_baselines" / "pikachu"),
                        "--output-dir",
                        str(output_dir),
                        "--metrics-csv",
                        str(metrics_path),
                        "--dimensions",
                        "100",
                        "--alpha",
                        "0.01",
                        "--support",
                        "10",
                        "--gpu",
                        "-1",
                    ]
                    env = _default_env()
                    env["GIDS_PREDICTIONS_PATH"] = str(predictions_path)
                    workdir = str(REPO_ROOT)
                    node_count = int(canonical_meta.get("row_count", 0) or 0)
                elif job.detector == "anomal":
                    anomal_meta = ensure_anomal_adapter(job.dataset, log_handle=stdout_handle)
                    _write_json(job.aligned_metadata_path, anomal_meta)
                    output_dir = job.results_root / job.dataset
                    metrics_path = output_dir / "metrics.csv"
                    predictions_path = output_dir / "predictions.csv"
                    command = [
                        _job_python_bin("anomal"),
                        str(REPO_ROOT / "EG5" / "anomal_run.py"),
                        "--dataset",
                        job.dataset,
                        "--metadata",
                        str(job.aligned_metadata_path),
                        "--detector-root",
                        str(REPO_ROOT / "local_runtime" / "detection_baselines" / "anomal"),
                        "--output-dir",
                        str(output_dir),
                        "--metrics-csv",
                        str(metrics_path),
                    ]
                    env = _default_env()
                    workdir = str(REPO_ROOT)
                    input_rows = int(anomal_meta.get("row_count", 0) or 0)
                    positive_rows = int(anomal_meta.get("positive_rows", 0) or 0)
                    node_count = int(anomal_meta.get("node_count", 0) or 0)
                else:
                    data_root, adapter_meta = ensure_graph_adapter(job.detector, job.dataset, log_handle=stdout_handle)
                    _write_json(job.aligned_metadata_path, adapter_meta)
                    env = _default_env()
                    input_rows = int(adapter_meta.get("row_count", 0) or 0)
                    positive_rows = int(adapter_meta.get("positive_rows", 0) or 0)
                    node_count = int(adapter_meta.get("node_count", 0) or 0)
                    if job.detector == "argus":
                        if job.dataset == "0501":
                            run_root = REPO_ROOT / "local_runtime" / "detection_baselines" / "argus" / "0501"
                            dataset_arg = "O_0501"
                            command = [
                                _job_python_bin("argus"),
                                str(REPO_ROOT / "EG1" / "legacy_entrypoint.py"),
                                str(run_root / "run.py"),
                                "--dataset",
                                dataset_arg,
                                "--hidden",
                                "32",
                                "-z",
                                "16",
                                "--rnn",
                                "GRU",
                                "--delta",
                                str(job.window_seconds / 60.0),
                                "--lr",
                                "0.01",
                                "--fpweight",
                                "0.48",
                                "--epochs",
                                "100",
                                "--patience",
                                "5",
                                "--te_end",
                                "all",
                            ]
                            metrics_path = job.results_root / "0501" / "metrics.csv"
                            predictions_path = job.results_root / "0501" / "predictions.csv"
                            env["ARGUS_RUN_NAME"] = "0501"
                        elif job.dataset == "cic":
                            run_root = REPO_ROOT / "local_runtime" / "detection_baselines" / "argus" / "CIC"
                            command = [
                                _job_python_bin("argus"),
                                str(REPO_ROOT / "EG1" / "legacy_entrypoint.py"),
                                str(run_root / "run.py"),
                                "--dataset",
                                "O_CIC",
                                "--delta",
                                str(job.window_seconds / 60.0),
                                "--lr",
                                "0.01",
                                "--hidden",
                                "32",
                                "-z",
                                "16",
                                "--fpweight",
                                "0.46",
                                "--epochs",
                                "100",
                                "--patience",
                                "3",
                            ]
                            metrics_path = job.results_root / "CIC" / "metrics.csv"
                            predictions_path = job.results_root / "CIC" / "predictions.csv"
                            env["ARGUS_RUN_NAME"] = "CIC"
                        else:
                            run_root = REPO_ROOT / "local_runtime" / "detection_baselines" / "argus" / "LANL_OpTC"
                            dataset_arg = "LANL" if job.dataset == "lanl" else "OPTC"
                            lr = "0.01" if job.dataset == "lanl" else "0.005"
                            patience = "3" if job.dataset == "lanl" else "10"
                            command = [
                                _job_python_bin("argus"),
                                str(REPO_ROOT / "EG1" / "legacy_entrypoint.py"),
                                str(run_root / "run.py"),
                                "--dataset",
                                dataset_arg,
                                "--hidden",
                                "32",
                                "-z",
                                "16",
                                "--rnn",
                                "GRU",
                                "--delta",
                                str(job.window_seconds / 3600.0),
                                "--lr",
                                lr,
                                "--fpweight",
                                "0.6",
                                "--epochs",
                                "100",
                                "--patience",
                                patience,
                                "--te_end",
                                "all",
                            ]
                            metrics_path = job.results_root / dataset_arg / "metrics.csv"
                            predictions_path = job.results_root / dataset_arg / "predictions.csv"
                            env["ARGUS_RUN_NAME"] = dataset_arg
                        env["ARGUS_DATA_ROOT"] = str(data_root)
                        env["ARGUS_RESULTS_ROOT"] = str(job.results_root)
                        env["ARGUS_MODELS_ROOT"] = str(job.models_root)
                    elif job.detector == "euler":
                        if job.dataset == "0501":
                            run_root = REPO_ROOT / "local_runtime" / "detection_baselines" / "euler" / "0501"
                            command = [
                                _job_python_bin("euler"),
                                str(REPO_ROOT / "EG1" / "legacy_entrypoint.py"),
                                str(run_root / "run.py"),
                                "--dataset",
                                "0501",
                                "--delta",
                                str(job.window_seconds / 60.0),
                                "--workers",
                                "1",
                                "--threads",
                                "1",
                                "--lr",
                                "0.005",
                                "--fpweight",
                                "0.48",
                                "--tests",
                                "1",
                            ]
                            metrics_path = job.results_root / "0501" / "0501" / "learning" / "metrics.csv"
                            predictions_path = job.results_root / "0501" / "0501" / "learning" / "predictions.csv"
                        elif job.dataset == "cic":
                            run_root = REPO_ROOT / "local_runtime" / "detection_baselines" / "euler" / "CIC"
                            command = [
                                _job_python_bin("euler"),
                                str(REPO_ROOT / "EG1" / "legacy_entrypoint.py"),
                                str(run_root / "run.py"),
                                "--dataset",
                                "CIC",
                                "--delta",
                                str(job.window_seconds / 60.0),
                                "--workers",
                                "1",
                                "--threads",
                                "1",
                                "--lr",
                                "0.005",
                                "--fpweight",
                                "0.7",
                                "--patience",
                                "5",
                                "--tests",
                                "1",
                                "--tr-start",
                                "0",
                                "--tr-end",
                                "50000",
                                "--val-end",
                                "55000",
                            ]
                            metrics_path = job.results_root / "cic" / "learning" / "metrics.csv"
                            predictions_path = job.results_root / "cic" / "learning" / "predictions.csv"
                        else:
                            run_root = REPO_ROOT / "local_runtime" / "detection_baselines" / "euler" / "LANL_OpTC"
                            dataset_arg = "LANL" if job.dataset == "lanl" else "OPTC"
                            lr = "0.005"
                            fpweight = "0.6"
                            command = [
                                _job_python_bin("euler"),
                                str(REPO_ROOT / "EG1" / "legacy_entrypoint.py"),
                                str(run_root / "run.py"),
                                "--dataset",
                                dataset_arg,
                                "--delta",
                                str(job.window_seconds / 3600.0),
                                "--workers",
                                "1",
                                "--threads",
                                "1",
                                "--lr",
                                lr,
                                "--fpweight",
                                fpweight,
                                "--patience",
                                "5",
                                "--tests",
                                "1",
                            ]
                            metrics_path = job.results_root / job.dataset / "learning" / "metrics.csv"
                            predictions_path = job.results_root / job.dataset / "learning" / "predictions.csv"
                        env["EULER_DATA_ROOT"] = str(data_root)
                        env["EULER_RESULTS_ROOT"] = str(job.results_root)
                    else:
                        if job.dataset == "0501":
                            run_root = REPO_ROOT / "local_runtime" / "detection_baselines" / "vgrnn" / "0501"
                            command = [
                                _job_python_bin("vgrnn"),
                                str(REPO_ROOT / "EG1" / "legacy_entrypoint.py"),
                                str(run_root / "run.py"),
                                "--dataset",
                                "0501",
                                "--snapshot",
                                str(job.window_seconds / 60.0),
                                "--dimension",
                                "32",
                                "--lr",
                                "0.01",
                                "--patience",
                                "10",
                                "--fw",
                                "0.43",
                                "--tests",
                                "1",
                            ]
                            metrics_path = job.results_root / "0501" / "metrics.csv"
                            predictions_path = job.results_root / "0501" / "predictions.csv"
                            env["VGRNN_RUN_NAME"] = "0501"
                        elif job.dataset == "cic":
                            run_root = REPO_ROOT / "local_runtime" / "detection_baselines" / "vgrnn" / "CIC"
                            command = [
                                _job_python_bin("vgrnn"),
                                str(REPO_ROOT / "EG1" / "legacy_entrypoint.py"),
                                str(run_root / "run.py"),
                                "--dataset",
                                "CIC",
                                "--snapshot",
                                str(job.window_seconds / 60.0),
                                "--dimension",
                                "32",
                                "--lr",
                                "0.01",
                                "--patience",
                                "10",
                                "--fw",
                                "0.43",
                                "--tests",
                                "1",
                            ]
                            metrics_path = job.results_root / "CIC" / "metrics.csv"
                            predictions_path = job.results_root / "CIC" / "predictions.csv"
                            env["VGRNN_RUN_NAME"] = "CIC"
                        else:
                            run_root = REPO_ROOT / "local_runtime" / "detection_baselines" / "vgrnn" / "LANL_OpTC"
                            dataset_arg = "LANL" if job.dataset == "lanl" else "OPTC"
                            lr = "0.01" if job.dataset == "lanl" else "0.005"
                            fw = "0.45" if job.dataset == "lanl" else "0.43"
                            command = [
                                _job_python_bin("vgrnn"),
                                str(REPO_ROOT / "EG1" / "legacy_entrypoint.py"),
                                str(run_root / "run.py"),
                                "--dataset",
                                dataset_arg,
                                "--snapshot",
                                str(job.window_seconds / 3600.0),
                                "--dimension",
                                "32",
                                "--lr",
                                lr,
                                "--patience",
                                "10",
                                "--fw",
                                fw,
                                "--tests",
                                "1",
                            ]
                            metrics_path = job.results_root / dataset_arg / "metrics.csv"
                            predictions_path = job.results_root / dataset_arg / "predictions.csv"
                            env["VGRNN_RUN_NAME"] = dataset_arg
                        env["VGRNN_DATA_ROOT"] = str(data_root)
                        env["VGRNN_RESULTS_ROOT"] = str(job.results_root)
                        env["VGRNN_MODELS_ROOT"] = str(job.models_root)

                    env["GIDS_PREDICTIONS_PATH"] = str(predictions_path)
                    workdir = str(run_root)

                env["GIDS_PHASE_PROFILE_PATH"] = str(job.stage_usage_path)
                execution_config = {
                    "experiment_id": job.experiment_id,
                    "detector": job.detector,
                    "dataset": job.dataset,
                    "window_seconds": job.window_seconds,
                    "window_label": job.window_label,
                    "command": command,
                    "workdir": workdir,
                    "metrics_path": str(metrics_path),
                    "predictions_path": str(predictions_path),
                    "input_rows": int(input_rows),
                    "positive_rows": int(positive_rows),
                    "node_count": int(node_count),
                }
                _write_json(job.execution_config_path, execution_config)
            except Exception as exc:
                _append_log(stderr_handle, f"[prepare] failed: {exc}")
                raise

    prepare_wall_time_sec = round(time.perf_counter() - prepare_started, 6)
    execution_config = _read_json(job.execution_config_path)
    execution_config["prepare_wall_time_sec"] = prepare_wall_time_sec
    _write_json(job.execution_config_path, execution_config)
    return execution_config

def _write_run_summary(job: EG8Job, payload: dict[str, Any]) -> None:
    merged = dict(payload)
    merged.setdefault("experiment_id", job.experiment_id)
    merged.setdefault("detector", job.detector)
    merged.setdefault("dataset", job.dataset)
    merged.setdefault("window_seconds", job.window_seconds)
    merged.setdefault("window_label", job.window_label)
    _write_json(job.run_summary_path, merged)

def _read_first_csv_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            return {str(key): str(value) for key, value in row.items()}
    return None

def _baseline_metric_row(job: EG8Job, execution: dict[str, Any], usage: Any) -> dict[str, Any]:
    metrics_path = Path(str(execution["metrics_path"]))
    predictions_path = Path(str(execution["predictions_path"]))
    metrics_row = _read_first_csv_row(metrics_path) or {}
    grouped_predictions = load_prediction_groups(predictions_path if predictions_path.exists() else None)
    flat_predictions: list[dict[str, Any]] = []
    for rows in grouped_predictions.values():
        flat_predictions.extend(rows)
    cdr_summary = compute_cdr_summary(flat_predictions) if flat_predictions else {
        "cdr": 0.0,
        "cdr_auc": 0.0,
        "campaign_count": 0,
        "detected_campaigns": 0,
        "cdr_meaningful": False,
    }

    row = {
        "detector": job.experiment_id,
        "job_id": job.dataset,
        "dataset": job.dataset,
        "seed": job.seed,
        "auc": metrics_row.get("auc", ""),
        "ap": metrics_row.get("ap", ""),
        "threshold": metrics_row.get("threshold", ""),
        "tp": metrics_row.get("tp", ""),
        "fp": metrics_row.get("fp", ""),
        "fn": metrics_row.get("fn", ""),
        "tn": metrics_row.get("tn", ""),
        "tpr": metrics_row.get("tpr", ""),
        "fpr": metrics_row.get("fpr", ""),
        "fnr": metrics_row.get("fnr", ""),
        "tnr": metrics_row.get("tnr", ""),
        "precision": metrics_row.get("precision", ""),
        "recall": metrics_row.get("recall", ""),
        "f1": metrics_row.get("f1", ""),
        "train_time": metrics_row.get("train_time", ""),
        "test_time": metrics_row.get("test_time", ""),
        "cdr": cdr_summary["cdr"],
        "cdr_auc": cdr_summary["cdr_auc"],
        "campaign_count": cdr_summary["campaign_count"],
        "detected_campaigns": cdr_summary["detected_campaigns"],
        "cdr_meaningful": cdr_summary["cdr_meaningful"],
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path) if predictions_path.exists() else "",
    }
    with job.baseline_metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return row

def run_job(job: EG8Job, *, force: bool = False) -> str:
    ensure_eg8_dirs()
    existing = _load_run_summary(job)
    if existing and existing.get("status") == "success" and job.baseline_metrics_path.exists() and not force:
        return "skipped"

    started_at = _utc_now()
    _write_run_summary(
        job,
        {
            "status": "running",
            "started_at": started_at,
            "finished_at": None,
            "last_error": "",
        },
    )

    try:
        execution = _prepare_execution(job, force=force)
        _write_run_summary(
            job,
            {
                "status": "running",
                "started_at": started_at,
                "execution_config_path": str(job.execution_config_path),
                "prepare_wall_time_sec": execution.get("prepare_wall_time_sec", 0.0),
                "metrics_path": execution.get("metrics_path", ""),
                "predictions_path": execution.get("predictions_path", ""),
                "input_rows": execution.get("input_rows", 0),
                "positive_rows": execution.get("positive_rows", 0),
                "node_count": execution.get("node_count", 0),
            },
        )

        env = _default_env()
        env.update({key: str(value) for key, value in _read_json(job.execution_config_path).get("env", {}).items()})
        config = _read_json(job.execution_config_path)
        usage = run_command_logged(
            config["command"],
            cwd=str(config["workdir"]),
            env=_materialize_env_for_command(job, config),
            stdout_path=job.detection_stdout_path,
            stderr_path=job.detection_stderr_path,
            poll_interval=0.5,
            monitor_resources=True,
        )
        metrics_path = Path(str(config["metrics_path"]))
        status = "success" if usage.exit_code == 0 and metrics_path.exists() else "failed"
        baseline_row = {}
        if status == "success":
            baseline_row = _baseline_metric_row(job, config, usage)
        finished_at = _utc_now()
        summary = {
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "command": config["command"],
            "workdir": config["workdir"],
            "metrics_path": str(metrics_path),
            "predictions_path": str(config["predictions_path"]),
            "input_rows": int(config.get("input_rows", 0) or 0),
            "positive_rows": int(config.get("positive_rows", 0) or 0),
            "node_count": int(config.get("node_count", 0) or 0),
            "prepare_wall_time_sec": _numeric(config.get("prepare_wall_time_sec", 0.0)),
            "detection_wall_time_sec": _numeric(usage.wall_time_sec),
            "total_wall_time_sec": _numeric(float(config.get("prepare_wall_time_sec", 0.0) or 0.0) + usage.wall_time_sec),
            "run_peak_rss_kb": int(usage.peak_rss_kb),
            "run_peak_rss_gib": _gib_from_kb(usage.peak_rss_kb),
            "run_mean_rss_kb": _numeric(usage.mean_rss_kb, digits=3),
            "run_mean_rss_gib": _gib_from_kb(usage.mean_rss_kb),
            "run_mean_cpu_percent": _numeric(usage.mean_cpu_percent, digits=3),
            "peak_process_count": int(usage.peak_process_count),
            "exit_code": int(usage.exit_code),
            "baseline_metrics_path": str(job.baseline_metrics_path) if job.baseline_metrics_path.exists() else "",
            "last_error": "" if status == "success" else f"exit_code={usage.exit_code} metrics_exists={metrics_path.exists()}",
            "auc": baseline_row.get("auc", ""),
            "ap": baseline_row.get("ap", ""),
            "cdr": baseline_row.get("cdr", ""),
        }
        _write_run_summary(job, summary)
        return status
    except Exception as exc:
        _write_run_summary(
            job,
            {
                "status": "failed",
                "started_at": started_at,
                "finished_at": _utc_now(),
                "last_error": str(exc),
            },
        )
        return "failed"

def _materialize_env_for_command(job: EG8Job, config: dict[str, Any]) -> dict[str, str]:
    env = _default_env()
    env["GIDS_PHASE_PROFILE_PATH"] = str(job.stage_usage_path)
    env["GIDS_PREDICTIONS_PATH"] = str(config["predictions_path"])
    detector = job.detector
    if detector == "pikachu":
        env.setdefault("EG1_PIKACHU_POOL_BACKEND", "thread")
        env.setdefault("EG1_PIKACHU_MAX_WORKERS", "4")
        return env
    if detector == "argus":
        env["ARGUS_DATA_ROOT"] = str(_detector_adapter_root("argus"))
        env["ARGUS_RESULTS_ROOT"] = str(job.results_root)
        env["ARGUS_MODELS_ROOT"] = str(job.models_root)
        if job.dataset == "0501":
            env["ARGUS_RUN_NAME"] = "0501"
        elif job.dataset == "cic":
            env["ARGUS_RUN_NAME"] = "CIC"
        elif job.dataset == "lanl":
            env["ARGUS_RUN_NAME"] = "LANL"
        else:
            env["ARGUS_RUN_NAME"] = "OPTC"
        return env
    if detector == "euler":
        env["EULER_DATA_ROOT"] = str(_detector_adapter_root("euler"))
        env["EULER_RESULTS_ROOT"] = str(job.results_root)
        env["EG1_EULER_LOCAL_MODE"] = "1"
        return env
    if detector == "vgrnn":
        env["VGRNN_DATA_ROOT"] = str(_detector_adapter_root("vgrnn"))
        env["VGRNN_RESULTS_ROOT"] = str(job.results_root)
        env["VGRNN_MODELS_ROOT"] = str(job.models_root)
        if job.dataset == "0501":
            env["VGRNN_RUN_NAME"] = "0501"
        elif job.dataset == "cic":
            env["VGRNN_RUN_NAME"] = "CIC"
        elif job.dataset == "lanl":
            env["VGRNN_RUN_NAME"] = "LANL"
        else:
            env["VGRNN_RUN_NAME"] = "OPTC"
        return env
    if detector == "anomal":
        return env
    return env

def _status_row(job: EG8Job) -> dict[str, Any]:
    summary = _load_run_summary(job) or {}
    return {
        "experiment_id": job.experiment_id,
        "detector": job.detector,
        "dataset": job.dataset,
        "window_seconds": job.window_seconds,
        "window_label": job.window_label,
        "seed": job.seed,
        "status": summary.get("status", "pending"),
        "started_at": summary.get("started_at", ""),
        "finished_at": summary.get("finished_at", ""),
        "auc": summary.get("auc", ""),
        "ap": summary.get("ap", ""),
        "cdr": summary.get("cdr", ""),
        "input_rows": summary.get("input_rows", ""),
        "positive_rows": summary.get("positive_rows", ""),
        "node_count": summary.get("node_count", ""),
        "total_wall_time_sec": summary.get("total_wall_time_sec", ""),
        "run_peak_rss_gib": summary.get("run_peak_rss_gib", ""),
        "baseline_metrics_path": summary.get("baseline_metrics_path", ""),
        "last_error": summary.get("last_error", ""),
    }

def write_matrix_report(jobs: list[EG8Job], *, path: Path | None = None) -> Path:
    output_path = path or (ORCHESTRATOR_ROOT / "job_matrix.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["experiment_id", "detector", "dataset", "window_seconds", "window_label", "seed"],
        )
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "experiment_id": job.experiment_id,
                    "detector": job.detector,
                    "dataset": job.dataset,
                    "window_seconds": job.window_seconds,
                    "window_label": job.window_label,
                    "seed": job.seed,
                }
            )
    return output_path

def write_status_report(jobs: list[EG8Job], *, path: Path | None = None) -> Path:
    output_path = path or (ORCHESTRATOR_ROOT / "job_status.csv")
    rows = [_status_row(job) for job in jobs]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["experiment_id"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path

def write_progress_report(
    jobs: list[EG8Job],
    *,
    json_path: Path | None = None,
    md_path: Path | None = None,
    attempted_jobs: int | None = None,
    failed_jobs: int | None = None,
) -> tuple[Path, Path]:
    json_target = json_path or PROGRESS_JSON_PATH
    md_target = md_path or PROGRESS_MD_PATH
    statuses = [_status_row(job) for job in jobs]
    counts = {"pending": 0, "running": 0, "success": 0, "failed": 0}
    for row in statuses:
        state = str(row["status"] or "pending")
        counts[state] = counts.get(state, 0) + 1
    total_jobs = len(jobs)
    remaining_jobs = counts.get("pending", 0) + counts.get("running", 0)
    payload = {
        "updated_at": _utc_now(),
        "total_jobs": total_jobs,
        "attempted_jobs": attempted_jobs if attempted_jobs is not None else counts.get("success", 0) + counts.get("failed", 0),
        "remaining_jobs": remaining_jobs,
        "success_jobs": counts.get("success", 0),
        "failed_jobs": failed_jobs if failed_jobs is not None else counts.get("failed", 0),
        "running_jobs": counts.get("running", 0),
        "status_counts": counts,
    }
    _write_json(json_target, payload)

    running = [row["experiment_id"] for row in statuses if row["status"] == "running"]
    failed = [row["experiment_id"] for row in statuses if row["status"] == "failed"]
    lines = [
        "# EG8 Progress",
        "",
        f"- updated_at: `{payload['updated_at']}`",
        f"- total_jobs: `{payload['total_jobs']}`",
        f"- success_jobs: `{payload['success_jobs']}`",
        f"- failed_jobs: `{payload['failed_jobs']}`",
        f"- running_jobs: `{payload['running_jobs']}`",
        f"- remaining_jobs: `{payload['remaining_jobs']}`",
        "",
        "## Running",
        "",
    ]
    if running:
        lines.extend([f"- `{item}`" for item in running])
    else:
        lines.append("- none")
    lines.extend(["", "## Failed", ""])
    if failed:
        lines.extend([f"- `{item}`" for item in failed])
    else:
        lines.append("- none")
    md_target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_target, md_target

def _load_aligned_report_row(job: EG8Job) -> dict[str, Any] | None:
    summary = _load_run_summary(job)
    if not summary or summary.get("status") != "success" or not job.baseline_metrics_path.exists():
        return None
    baseline = _read_first_csv_row(job.baseline_metrics_path)
    if not baseline:
        return None
    return {
        "protocol": "aligned",
        "source_kind": "eg8_aligned",
        "detector": job.detector,
        "dataset": job.dataset,
        "window_seconds": job.window_seconds,
        "window_label": job.window_label,
        "seed": job.seed,
        "experiment_id": job.experiment_id,
        "status": "success",
        "auc": baseline.get("auc", ""),
        "ap": baseline.get("ap", ""),
        "cdr": baseline.get("cdr", ""),
        "cdr_auc": baseline.get("cdr_auc", ""),
        "campaign_count": baseline.get("campaign_count", ""),
        "detected_campaigns": baseline.get("detected_campaigns", ""),
        "cdr_meaningful": baseline.get("cdr_meaningful", ""),
        "threshold": baseline.get("threshold", ""),
        "tp": baseline.get("tp", ""),
        "fp": baseline.get("fp", ""),
        "fn": baseline.get("fn", ""),
        "tn": baseline.get("tn", ""),
        "tpr": baseline.get("tpr", ""),
        "fpr": baseline.get("fpr", ""),
        "fnr": baseline.get("fnr", ""),
        "tnr": baseline.get("tnr", ""),
        "precision": baseline.get("precision", ""),
        "recall": baseline.get("recall", ""),
        "f1": baseline.get("f1", ""),
        "train_time": baseline.get("train_time", ""),
        "test_time": baseline.get("test_time", ""),
        "prepare_wall_time_sec": summary.get("prepare_wall_time_sec", ""),
        "detection_wall_time_sec": summary.get("detection_wall_time_sec", ""),
        "total_wall_time_sec": summary.get("total_wall_time_sec", ""),
        "run_peak_rss_gib": summary.get("run_peak_rss_gib", ""),
        "input_rows": summary.get("input_rows", ""),
        "positive_rows": summary.get("positive_rows", ""),
        "node_count": summary.get("node_count", ""),
        "metrics_path": baseline.get("metrics_path", ""),
        "predictions_path": baseline.get("predictions_path", ""),
    }

def _import_native_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    eg1_path = _PATHS.sc_data_root("eg1") / "reports/raw_results.csv"
    if eg1_path.exists():
        with eg1_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                detector = str(row.get("detector", "")).lower()
                dataset = str(row.get("dataset", "")).lower()
                window_label = str(row.get("window_label", ""))
                if detector not in {"argus", "euler", "pikachu", "vgrnn"}:
                    continue
                if WINDOW_LABELS.get(dataset) != window_label:
                    continue
                rows.append(
                    {
                        "protocol": "native",
                        "source_kind": "eg1",
                        "detector": detector,
                        "dataset": dataset,
                        "window_seconds": row.get("window_seconds", ""),
                        "window_label": row.get("window_label", ""),
                        "seed": row.get("seed", "0"),
                        "experiment_id": row.get("experiment_id", ""),
                        "status": row.get("detection_status", ""),
                        "auc": row.get("auc", ""),
                        "ap": row.get("ap", ""),
                        "cdr": row.get("cdr", ""),
                        "cdr_auc": row.get("cdr_auc", ""),
                        "campaign_count": row.get("campaign_count", ""),
                        "detected_campaigns": row.get("detected_campaigns", ""),
                        "cdr_meaningful": row.get("cdr_meaningful", ""),
                        "threshold": row.get("threshold", ""),
                        "tp": "",
                        "fp": "",
                        "fn": "",
                        "tn": "",
                        "tpr": row.get("tpr", ""),
                        "fpr": row.get("fpr", ""),
                        "fnr": "",
                        "tnr": "",
                        "precision": row.get("precision", ""),
                        "recall": row.get("recall", ""),
                        "f1": row.get("f1", ""),
                        "train_time": row.get("train_time", ""),
                        "test_time": row.get("test_time", ""),
                        "prepare_wall_time_sec": row.get("featurization_wall_time_sec", ""),
                        "detection_wall_time_sec": row.get("detection_wall_time_sec", ""),
                        "total_wall_time_sec": row.get("total_wall_time_sec", ""),
                        "run_peak_rss_gib": row.get("run_peak_rss_gib", ""),
                        "input_rows": "",
                        "positive_rows": "",
                        "node_count": "",
                        "metrics_path": row.get("metrics_path", ""),
                        "predictions_path": "",
                    }
                )
    eg5_path = _PATHS.sc_data_root("eg5") / "reports/raw_results.csv"
    if eg5_path.exists():
        with eg5_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if str(row.get("detector", "")).lower() != "anomal":
                    continue
                if str(row.get("scale_percent", "")) != "100":
                    continue
                dataset = str(row.get("dataset", "")).lower()
                if dataset not in ANOMAL_DATASETS:
                    continue
                rows.append(
                    {
                        "protocol": "native",
                        "source_kind": "eg5",
                        "detector": "anomal",
                        "dataset": dataset,
                        "window_seconds": row.get("window_seconds", ""),
                        "window_label": WINDOW_LABELS.get(dataset, ""),
                        "seed": row.get("seed", "0"),
                        "experiment_id": row.get("experiment_id", ""),
                        "status": row.get("status", ""),
                        "auc": row.get("auc", ""),
                        "ap": row.get("ap", ""),
                        "cdr": row.get("cdr", ""),
                        "cdr_auc": "",
                        "campaign_count": row.get("campaign_count", ""),
                        "detected_campaigns": row.get("detected_campaigns", ""),
                        "cdr_meaningful": row.get("cdr_meaningful", ""),
                        "threshold": "",
                        "tp": "",
                        "fp": "",
                        "fn": "",
                        "tn": "",
                        "tpr": "",
                        "fpr": "",
                        "fnr": "",
                        "tnr": "",
                        "precision": "",
                        "recall": "",
                        "f1": "",
                        "train_time": row.get("train_time", ""),
                        "test_time": row.get("test_time", ""),
                        "prepare_wall_time_sec": row.get("featurization_wall_time_sec", ""),
                        "detection_wall_time_sec": row.get("detection_wall_time_sec", ""),
                        "total_wall_time_sec": row.get("total_wall_time_sec", ""),
                        "run_peak_rss_gib": row.get("run_peak_rss_gib", ""),
                        "input_rows": row.get("input_rows", ""),
                        "positive_rows": row.get("positive_rows", ""),
                        "node_count": row.get("node_count", ""),
                        "metrics_path": row.get("metrics_path", ""),
                        "predictions_path": "",
                    }
                )
    return rows

def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

def summarize_jobs(jobs: list[EG8Job], *, output_root: Path | None = None) -> Path:
    root = output_root or REPORT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    aligned_rows = [row for row in (_load_aligned_report_row(job) for job in jobs) if row]
    native_rows = _import_native_rows()
    raw_rows = sorted(
        native_rows + aligned_rows,
        key=lambda row: (str(row.get("protocol", "")), str(row.get("detector", "")), str(row.get("dataset", ""))),
    )
    _write_csv(root / "raw_results.csv", raw_rows, COMMON_REPORT_FIELDS)

    native_map = {(row["detector"], row["dataset"]): row for row in native_rows}
    aligned_map = {(row["detector"], row["dataset"]): row for row in aligned_rows}
    delta_rows: list[dict[str, Any]] = []
    for key in sorted(set(native_map).intersection(aligned_map)):
        detector, dataset = key
        native = native_map[key]
        aligned = aligned_map[key]
        delta_rows.append(
            {
                "detector": detector,
                "dataset": dataset,
                "auc_native": native.get("auc", ""),
                "auc_aligned": aligned.get("auc", ""),
                "auc_delta": _delta(native.get("auc"), aligned.get("auc")),
                "ap_native": native.get("ap", ""),
                "ap_aligned": aligned.get("ap", ""),
                "ap_delta": _delta(native.get("ap"), aligned.get("ap")),
                "cdr_native": native.get("cdr", ""),
                "cdr_aligned": aligned.get("cdr", ""),
                "cdr_delta": _delta(native.get("cdr"), aligned.get("cdr")),
                "total_wall_time_native": native.get("total_wall_time_sec", ""),
                "total_wall_time_aligned": aligned.get("total_wall_time_sec", ""),
                "total_wall_time_delta": _delta(native.get("total_wall_time_sec"), aligned.get("total_wall_time_sec")),
                "run_peak_rss_native": native.get("run_peak_rss_gib", ""),
                "run_peak_rss_aligned": aligned.get("run_peak_rss_gib", ""),
                "run_peak_rss_delta": _delta(native.get("run_peak_rss_gib"), aligned.get("run_peak_rss_gib")),
            }
        )
    _write_csv(
        root / "delta_native_vs_aligned.csv",
        delta_rows,
        [
            "detector",
            "dataset",
            "auc_native",
            "auc_aligned",
            "auc_delta",
            "ap_native",
            "ap_aligned",
            "ap_delta",
            "cdr_native",
            "cdr_aligned",
            "cdr_delta",
            "total_wall_time_native",
            "total_wall_time_aligned",
            "total_wall_time_delta",
            "run_peak_rss_native",
            "run_peak_rss_aligned",
            "run_peak_rss_delta",
        ],
    )

    _write_csv(
        root / "summary_by_detector.csv",
        _aggregate_delta(delta_rows, "detector"),
        ["detector", "pairs", "mean_auc_delta", "mean_ap_delta", "mean_cdr_delta", "mean_total_wall_time_delta", "mean_run_peak_rss_delta"],
    )
    _write_csv(
        root / "summary_by_dataset.csv",
        _aggregate_delta(delta_rows, "dataset"),
        ["dataset", "pairs", "mean_auc_delta", "mean_ap_delta", "mean_cdr_delta", "mean_total_wall_time_delta", "mean_run_peak_rss_delta"],
    )

    evidence_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        meta_path = _canonical_metadata_path(dataset)
        if not meta_path.exists():
            continue
        meta = _read_json(meta_path)
        evidence_rows.append(
            {
                "dataset": dataset,
                "window_seconds": WINDOW_SECONDS[dataset],
                "window_label": WINDOW_LABELS[dataset],
                "source_experiment_id": CANONICAL_SOURCE_EXPERIMENTS[dataset],
                "canonical_input_path": meta.get("input_path", ""),
                "row_count": meta.get("row_count", ""),
                "positive_rows": meta.get("positive_rows", ""),
                "snapshot_count": meta.get("snapshot_count", ""),
                "trainwin": meta.get("trainwin", ""),
            }
        )
    _write_csv(
        root / "evidence_table.csv",
        evidence_rows,
        ["dataset", "window_seconds", "window_label", "source_experiment_id", "canonical_input_path", "row_count", "positive_rows", "snapshot_count", "trainwin"],
    )

    _write_conclusion(root / "conclusion.md", aligned_rows, delta_rows)
    return root / "raw_results.csv"

def _delta(left: Any, right: Any) -> str:
    left_val = _safe_float(left)
    right_val = _safe_float(right)
    if left_val is None or right_val is None:
        return ""
    return str(round(right_val - left_val, 6))

def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0

def _aggregate_delta(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group = str(row.get(key, ""))
        groups.setdefault(group, []).append(row)
    aggregated: list[dict[str, Any]] = []
    for group in sorted(groups):
        members = groups[group]
        aggregated.append(
            {
                key: group,
                "pairs": len(members),
                "mean_auc_delta": _mean([_safe_float(item.get("auc_delta")) or 0.0 for item in members]),
                "mean_ap_delta": _mean([_safe_float(item.get("ap_delta")) or 0.0 for item in members]),
                "mean_cdr_delta": _mean([_safe_float(item.get("cdr_delta")) or 0.0 for item in members]),
                "mean_total_wall_time_delta": _mean([_safe_float(item.get("total_wall_time_delta")) or 0.0 for item in members]),
                "mean_run_peak_rss_delta": _mean([_safe_float(item.get("run_peak_rss_delta")) or 0.0 for item in members]),
            }
        )
    return aggregated

def _write_conclusion(path: Path, aligned_rows: list[dict[str, Any]], delta_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# EG8 Conclusion",
        "",
        f"- aligned_successes: `{len(aligned_rows)}`",
        f"- comparable_pairs: `{len(delta_rows)}`",
        "",
    ]
    if aligned_rows:
        best_ap = max(aligned_rows, key=lambda row: _safe_float(row.get("ap")) or float("-inf"))
        lines.append(
            f"- highest aligned AP: `{best_ap['detector']}/{best_ap['dataset']}` -> `{best_ap.get('ap', '')}`"
        )
    if delta_rows:
        biggest_drop = min(delta_rows, key=lambda row: _safe_float(row.get("ap_delta")) or 0.0)
        lines.append(
            f"- largest AP shift vs native: `{biggest_drop['detector']}/{biggest_drop['dataset']}` -> `{biggest_drop.get('ap_delta', '')}`"
        )
    lines.append("")
    lines.append("EG8 compares native results against one unified PIKACHU-bounded preprocessing contract.")
    lines.append("Interpretation should focus on the aligned rows and on the native-to-aligned deltas.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
