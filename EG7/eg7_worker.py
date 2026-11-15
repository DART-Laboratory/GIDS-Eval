from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from gids_eval.cdr import build_prediction_rows, compute_cdr_summary, write_prediction_rows
from gids_eval.io_utils import dump_json, ensure_dir, expand_templates, load_yaml, write_csv_rows
from gids_eval.paths import expand_template_vars, get_paths
from gids_eval.process_monitor import maybe_profile_phase

PATHS = get_paths()
RAW_ROOT = PATHS.raw_data_root
EG7_CODE_ROOT = PATHS.repo_root / "EG7"
EG3_CODE_ROOT = PATHS.repo_root / "EG3"

BASELINE_FIELDS = [
    "detector",
    "job_id",
    "dataset",
    "seed",
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
    "mcc",
    "train_time",
    "test_time",
    "cdr",
    "cdr_auc",
    "campaign_count",
    "detected_campaigns",
    "cdr_meaningful",
    "metrics_path",
    "predictions_path",
]

COST_FIELDS = [
    "detector",
    "phase",
    "job_id",
    "dataset",
    "seed",
    "status",
    "exit_code",
    "started_at",
    "finished_at",
    "wall_time_sec",
    "peak_rss_kb",
    "mean_rss_kb",
    "peak_cpu_percent",
    "mean_cpu_percent",
    "peak_process_count",
    "sample_count",
    "cpu_sample_count",
    "cache_hit",
    "cache_key",
    "train_time",
    "test_time",
    "command",
    "workdir",
    "stdout_log",
    "stderr_log",
    "metrics_path",
]

MAX_TRAIN_ROWS = 50000
MIN_TRAIN_EDGES = 100
FALLBACK_TRAIN_FRACTION = float(os.environ.get("EG7_FALLBACK_TRAIN_FRACTION", "0.4"))
EDGE_SCALAR_DIM = 8
FLOAT16_CACHE_ABS_MAX = float(np.finfo(np.float16).max)

@dataclass
class DatasetBundle:
    frame: pd.DataFrame
    metadata: dict[str, Any]
    cache_hit: bool
    cache_key: str
    cache_dir: Path

@dataclass
class RepresentationBundle:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    val_campaigns: np.ndarray
    test_campaigns: np.ndarray
    rep_dim: int
    scalar_dim: int
    train_rows: int
    val_rows: int
    test_rows: int
    cache_hit: bool
    cache_key: str
    cache_dir: Path
    split_summary: dict[str, Any]

class TaskProgressTracker:
    def __init__(self, metadata_root: Path, experiment_id: str) -> None:
        self.json_path = metadata_root / "task_progress.json"
        self.md_path = metadata_root / "task_progress.md"
        self.experiment_id = experiment_id
        self.started_at = _iso_now()
        self.history: list[dict[str, Any]] = []

    def update(
        self,
        *,
        status: str,
        phase: str,
        message: str,
        dataset: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "experiment_id": self.experiment_id,
            "status": status,
            "phase": phase,
            "message": message,
            "dataset": dataset,
            "started_at": self.started_at,
            "updated_at": _iso_now(),
            "history": self.history + [
                {
                    "at": _iso_now(),
                    "phase": phase,
                    "status": status,
                    "message": message,
                }
            ],
        }
        if extra:
            payload["extra"] = extra
        self.history = list(payload["history"])
        dump_json(self.json_path, payload)
        lines = [
            "# EG7 Task Progress",
            "",
            f"- experiment_id: {self.experiment_id}",
            f"- dataset: {dataset}",
            f"- status: {status}",
            f"- phase: {phase}",
            f"- started_at: {self.started_at}",
            f"- updated_at: {payload['updated_at']}",
            f"- message: {message}",
        ]
        if extra:
            lines.append("")
            lines.append("## Details")
            for key, value in sorted(extra.items()):
                lines.append(f"- {key}: {value}")
        lines.append("")
        lines.append("## History")
        for item in self.history:
            lines.append(f"- {item['at']}: [{item['status']}] {item['phase']} - {item['message']}")
        self.md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EG7 unified worker")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    config = expand_templates(config, expand_template_vars(PATHS))
    paths = dict(config.get("paths") or {})
    workspace_root = Path(str(paths.get("workspace_root")))
    metadata_root = Path(str(paths.get("metadata_root")))
    result_root = Path(str(paths.get("result_root")))
    cache_root = Path(str(paths.get("cache_root")))
    ensure_dir(workspace_root)
    ensure_dir(metadata_root)
    ensure_dir(result_root)
    ensure_dir(cache_root)

    config_copy_path = metadata_root / "job_config.yaml"
    config_copy_path.write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")

    experiment_id = str(config["experiment_id"])
    dataset = str(config["dataset"])
    tracker = TaskProgressTracker(metadata_root, experiment_id)
    tracker.update(
        status="running",
        phase="bootstrap",
        message="Loaded config and prepared workspace",
        dataset=dataset,
        extra={
            "config_path": str(args.config),
            "workspace_root": str(workspace_root),
            "metadata_root": str(metadata_root),
            "result_root": str(result_root),
        },
    )

    summary_path = metadata_root / "run_summary.json"
    baseline_path = metadata_root / "baseline_metrics.csv"
    cost_path = metadata_root / "cost_metrics.csv"
    predictions_path = result_root / "predictions.csv"

    if args.dry_run:
        dump_json(
            summary_path,
            {
                "status": "dry_run",
                "experiment_id": experiment_id,
                "dataset": dataset,
                "updated_at": _iso_now(),
            },
        )
        tracker.update(
            status="dry_run",
            phase="bootstrap",
            message="Dry run requested; no compute executed",
            dataset=dataset,
        )
        return 0

    phase_statuses: dict[str, dict[str, Any]] = {}
    try:
        with maybe_profile_phase("preprocess"):
            tracker.update(
                status="running",
                phase="preprocess",
                message="Preparing dataset cache",
                dataset=dataset,
            )
            dataset_bundle = prepare_dataset(config)
        phase_statuses["preprocess"] = {
            "status": "cache_hit" if dataset_bundle.cache_hit else "success",
            "cache_hit": dataset_bundle.cache_hit,
            "cache_key": dataset_bundle.cache_key,
        }
        tracker.update(
            status="running",
            phase="preprocess",
            message="Dataset cache ready",
            dataset=dataset,
            extra={
                "cache_hit": dataset_bundle.cache_hit,
                "cache_dir": str(dataset_bundle.cache_dir),
                "rows": len(dataset_bundle.frame),
                "positive_rows": int((dataset_bundle.frame["label"] == 1).sum()),
                "snapshots": int(dataset_bundle.frame["snapshot"].nunique()),
                "trainwin": int(dataset_bundle.metadata["trainwin"]),
            },
        )

        with maybe_profile_phase("representation"):
            tracker.update(
                status="running",
                phase="representation",
                message="Building or loading representation cache",
                dataset=dataset,
            )
            representation_bundle = build_representation(config, dataset_bundle)
        phase_statuses["representation"] = {
            "status": "cache_hit" if representation_bundle.cache_hit else "success",
            "cache_hit": representation_bundle.cache_hit,
            "cache_key": representation_bundle.cache_key,
        }
        tracker.update(
            status="running",
            phase="representation",
            message="Representation bundle ready",
            dataset=dataset,
            extra={
                "cache_hit": representation_bundle.cache_hit,
                "cache_dir": str(representation_bundle.cache_dir),
                "train_rows": representation_bundle.train_rows,
                "val_rows": representation_bundle.val_rows,
                "test_rows": representation_bundle.test_rows,
                "rep_dim": representation_bundle.rep_dim,
                "scalar_dim": representation_bundle.scalar_dim,
                **representation_bundle.split_summary,
            },
        )

        with maybe_profile_phase("detection"):
            tracker.update(
                status="running",
                phase="detection",
                message="Training scorer and evaluating test split",
                dataset=dataset,
            )
            metrics_row, prediction_rows, detection_summary = run_detection(
                config=config,
                bundle=representation_bundle,
                dataset=dataset,
                experiment_id=experiment_id,
                predictions_path=predictions_path,
                metrics_path=baseline_path,
            )
        phase_statuses["detection"] = {
            "status": "success",
            "cache_hit": False,
            "cache_key": stable_key(
                {
                    "experiment_id": experiment_id,
                    "loss": config["loss"],
                    "code_version": config["code_version"],
                }
            ),
        }
        write_prediction_rows(predictions_path, prediction_rows, overwrite=True)
        write_csv_rows(baseline_path, [metrics_row], BASELINE_FIELDS)
        dump_json(result_root / "detection_summary.json", detection_summary)
        tracker.update(
            status="running",
            phase="detection",
            message="Detection metrics written",
            dataset=dataset,
            extra={
                "ap": round(float(metrics_row["ap"]), 6),
                "cdr_auc": round(float(metrics_row["cdr_auc"]), 6),
                "mcc": round(float(metrics_row["mcc"]), 6),
                "threshold": round(float(metrics_row["threshold"]), 6),
                "predictions_path": str(predictions_path),
            },
        )

        cost_rows = build_cost_rows(
            config=config,
            phase_statuses=phase_statuses,
            metadata_root=metadata_root,
            baseline_path=baseline_path,
            metrics_row=metrics_row,
        )
        write_csv_rows(cost_path, cost_rows, COST_FIELDS)
        dump_json(
            summary_path,
            {
                "status": "success",
                "experiment_id": experiment_id,
                "dataset": dataset,
                "updated_at": _iso_now(),
                "baseline_metrics_path": str(baseline_path),
                "cost_metrics_path": str(cost_path),
                "predictions_path": str(predictions_path),
                "phase_statuses": phase_statuses,
                "summary": {
                    "ap": float(metrics_row["ap"]),
                    "cdr_auc": float(metrics_row["cdr_auc"]),
                    "mcc": float(metrics_row["mcc"]),
                    "threshold": float(metrics_row["threshold"]),
                },
            },
        )
        tracker.update(
            status="success",
            phase="complete",
            message="EG7 job finished successfully",
            dataset=dataset,
            extra={
                "baseline_metrics_path": str(baseline_path),
                "cost_metrics_path": str(cost_path),
                "predictions_path": str(predictions_path),
            },
        )
        return 0
    except Exception as exc:
        partial_cost_rows = build_cost_rows(
            config=config,
            phase_statuses=phase_statuses,
            metadata_root=metadata_root,
            baseline_path=baseline_path,
            metrics_row=None,
        )
        if partial_cost_rows:
            write_csv_rows(cost_path, partial_cost_rows, COST_FIELDS)
        dump_json(
            summary_path,
            {
                "status": "failed",
                "experiment_id": experiment_id,
                "dataset": dataset,
                "updated_at": _iso_now(),
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "phase_statuses": phase_statuses,
            },
        )
        tracker.update(
            status="failed",
            phase="error",
            message=str(exc),
            dataset=dataset,
            extra={"traceback": traceback.format_exc()},
        )
        raise

def prepare_dataset(config: dict[str, Any]) -> DatasetBundle:
    dataset = str(config["dataset"])
    window_seconds = int(config["window_seconds"])
    neg_ratio = int(config["neg_ratio"])
    seed = int(config["preprocess_seed"])
    cache_root = Path(str((config.get("paths") or {}).get("cache_root")))
    cache_key = stable_key(
        {
            "dataset": dataset,
            "window_seconds": window_seconds,
            "neg_ratio": neg_ratio,
            "seed": seed,
            "code_version": config["code_version"],
        }
    )[:16]
    dataset_cache_dir = cache_root / "datasets" / dataset / f"w{window_seconds}_nr{neg_ratio}_s{seed}_{cache_key}"
    input_path = dataset_cache_dir / "input.csv"
    metadata_path = dataset_cache_dir / "metadata.json"
    cache_hit = input_path.exists() and metadata_path.exists()
    if not cache_hit:
        ensure_dir(dataset_cache_dir)
        cache_hit = hydrate_dataset_cache_from_existing(
            dataset_cache_dir=dataset_cache_dir,
            dataset=dataset,
            window_seconds=window_seconds,
            neg_ratio=neg_ratio,
            seed=seed,
            cache_root=cache_root,
        )
    if not cache_hit:
        if dataset == "0501":
            command = [
                sys.executable,
                str(EG3_CODE_ROOT / "eg3_build_0501.py"),
                "--mode",
                "pikachu",
                "--raw-root",
                str(RAW_ROOT),
                "--work-root",
                str(dataset_cache_dir / "_merge_work"),
                "--output-dir",
                str(dataset_cache_dir),
                "--snapshot-seconds",
                str(window_seconds),
                "--neg-ratio",
                str(neg_ratio),
                "--seed",
                str(seed),
            ]
        else:
            command = [
                sys.executable,
                str(EG3_CODE_ROOT / "eg3_preprocess.py"),
                "--dataset",
                dataset,
                "--raw-root",
                str(RAW_ROOT),
                "--output-dir",
                str(dataset_cache_dir),
                "--snapshot-seconds",
                str(window_seconds),
                "--neg-ratio",
                str(neg_ratio),
                "--seed",
                str(seed),
            ]
        subprocess.run(command, check=True)

    frame = pd.read_csv(input_path)
    if frame.empty:
        raise ValueError(f"Dataset cache is empty: {input_path}")
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce").fillna(0).astype(int)
    frame["snapshot"] = pd.to_numeric(frame["snapshot"], errors="coerce").fillna(0).astype(int)
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce").fillna(0).astype(int)
    frame["src_computer"] = frame["src_computer"].astype(str)
    frame["dst_computer"] = frame["dst_computer"].astype(str)
    if "campaign" not in frame.columns:
        frame["campaign"] = ""
    frame["campaign"] = frame["campaign"].fillna("").astype(str)
    frame = frame.sort_values(by=["timestamp", "snapshot", "src_computer", "dst_computer"]).reset_index(drop=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return DatasetBundle(
        frame=frame,
        metadata=metadata,
        cache_hit=cache_hit,
        cache_key=cache_key,
        cache_dir=dataset_cache_dir,
    )

def build_representation(config: dict[str, Any], dataset_bundle: DatasetBundle) -> RepresentationBundle:
    dataset = str(config["dataset"])
    cache_root = Path(str((config.get("paths") or {}).get("cache_root")))
    cache_key = stable_key(
        {
            "dataset": dataset,
            "featurization": config["featurization"],
            "graph_encoder": config["graph_encoder"],
            "temporal_encoder": config["temporal_encoder"],
            "code_version": config["code_version"],
            "dataset_cache_key": dataset_bundle.cache_key,
            "one_hot_dim": config["one_hot_dim"],
            "projection_dim": config["projection_dim"],
            "val_fraction": config["val_fraction"],
            "split_rule": f"guard{MIN_TRAIN_EDGES}_frac{FALLBACK_TRAIN_FRACTION}",
        }
    )[:16]
    representation_dir = cache_root / "representations" / dataset / cache_key
    npz_path = representation_dir / "features.npz"
    meta_path = representation_dir / "meta.json"
    cache_hit = npz_path.exists() and meta_path.exists()
    if cache_hit and representation_cache_is_finite(npz_path):
        bundle = load_representation_bundle(npz_path=npz_path, meta_path=meta_path)
        if not representation_bundle_has_nonfinite_values(bundle):
            bundle.cache_hit = True
            bundle.cache_key = cache_key
            bundle.cache_dir = representation_dir
            return bundle

    ensure_dir(representation_dir)
    bundle = compute_representation(config, dataset_bundle)
    np.savez_compressed(
        npz_path,
        x_train=bundle.x_train.astype(np.float16),
        x_val=bundle.x_val.astype(np.float16),
        x_test=bundle.x_test.astype(np.float16),
        y_val=bundle.y_val.astype(np.int8),
        y_test=bundle.y_test.astype(np.int8),
        val_campaigns=bundle.val_campaigns.astype(str),
        test_campaigns=bundle.test_campaigns.astype(str),
    )
    dump_json(
        meta_path,
        {
            "rep_dim": bundle.rep_dim,
            "scalar_dim": bundle.scalar_dim,
            "train_rows": bundle.train_rows,
            "val_rows": bundle.val_rows,
            "test_rows": bundle.test_rows,
            "split_summary": bundle.split_summary,
        },
    )
    bundle.cache_hit = False
    bundle.cache_key = cache_key
    bundle.cache_dir = representation_dir
    return bundle

def representation_cache_is_finite(npz_path: Path) -> bool:
    try:
        payload = np.load(npz_path, allow_pickle=False)
    except (OSError, ValueError):
        return False
    for name in ("x_train", "x_val", "x_test"):
        if name not in payload.files:
            return False
        if not np.isfinite(np.asarray(payload[name], dtype=np.float32)).all():
            return False
    return True


def load_representation_bundle(*, npz_path: Path, meta_path: Path) -> RepresentationBundle:
    payload = np.load(npz_path, allow_pickle=False)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    x_train, _ = sanitize_feature_matrix(payload["x_train"])
    x_val, _ = sanitize_feature_matrix(payload["x_val"])
    x_test, _ = sanitize_feature_matrix(payload["x_test"])
    return RepresentationBundle(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_val=np.asarray(payload["y_val"], dtype=np.int32),
        y_test=np.asarray(payload["y_test"], dtype=np.int32),
        val_campaigns=np.asarray(payload["val_campaigns"], dtype=str),
        test_campaigns=np.asarray(payload["test_campaigns"], dtype=str),
        rep_dim=int(meta["rep_dim"]),
        scalar_dim=int(meta["scalar_dim"]),
        train_rows=int(meta["train_rows"]),
        val_rows=int(meta["val_rows"]),
        test_rows=int(meta["test_rows"]),
        cache_hit=True,
        cache_key="",
        cache_dir=npz_path.parent,
        split_summary=dict(meta.get("split_summary") or {}),
    )

def compute_representation(config: dict[str, Any], dataset_bundle: DatasetBundle) -> RepresentationBundle:
    frame = dataset_bundle.frame.copy()
    featurization = str(config["featurization"])
    graph_encoder = str(config["graph_encoder"])
    temporal_encoder = str(config["temporal_encoder"])
    one_hot_dim = int(config["one_hot_dim"])
    projection_dim = int(config["projection_dim"])
    val_fraction = float(config["val_fraction"])

    nodes = sorted(set(frame["src_computer"]).union(frame["dst_computer"]))
    if not nodes:
        raise ValueError("No nodes available after preprocessing")
    node_to_idx = {node: index for index, node in enumerate(nodes)}
    frame["src_idx"] = frame["src_computer"].map(node_to_idx).astype(int)
    frame["dst_idx"] = frame["dst_computer"].map(node_to_idx).astype(int)

    snapshot_groups = list(frame.groupby("snapshot", sort=True))
    snapshots = [int(snapshot) for snapshot, _ in snapshot_groups]
    trainwin = int(dataset_bundle.metadata["trainwin"])
    train_snapshots = [snapshot for snapshot in snapshots if snapshot < trainwin]
    post_snapshots = [snapshot for snapshot in snapshots if snapshot >= trainwin]
    if not train_snapshots:
        split_at = max(1, len(snapshots) // 3)
        train_snapshots = snapshots[:split_at]
        post_snapshots = snapshots[split_at:]

    snapshot_sizes = {int(snapshot): len(group) for snapshot, group in snapshot_groups}
    trainwin_edge_count = sum(snapshot_sizes[s] for s in train_snapshots)
    edge_split_boundaries: tuple[int, int] | None = None
    if trainwin_edge_count < MIN_TRAIN_EDGES:
        total_edges = int(len(frame))
        train_edges = int(round(total_edges * FALLBACK_TRAIN_FRACTION))
        post_edges = total_edges - train_edges
        val_edges = max(1, min(post_edges - 1, int(math.ceil(post_edges * val_fraction))))
        edge_split_boundaries = (train_edges, train_edges + val_edges)
        train_snapshots = []
        val_snapshots = []
        test_snapshots = []
    else:
        if len(post_snapshots) < 2:
            raise ValueError(
                f"Dataset {config['dataset']} does not have enough post-train snapshots for validation/test split"
            )
        val_count = max(1, min(len(post_snapshots) - 1, int(math.ceil(len(post_snapshots) * val_fraction))))
        val_snapshots = post_snapshots[:val_count]
        test_snapshots = post_snapshots[val_count:]
        if not test_snapshots:
            test_snapshots = [val_snapshots[-1]]
            val_snapshots = val_snapshots[:-1]
        if not val_snapshots or not test_snapshots:
            raise ValueError(f"Could not derive validation/test splits for dataset {config['dataset']}")

    id_features = build_identity_features(nodes, one_hot_dim)
    active_hidden: dict[int, np.ndarray] = {}
    active_cell: dict[int, np.ndarray] = {}

    hist_out: Counter[int] = Counter()
    hist_in: Counter[int] = Counter()
    hist_total: Counter[int] = Counter()
    pair_hist: Counter[tuple[int, int]] = Counter()
    last_seen: dict[int, int] = {}
    last_pair_seen: dict[tuple[int, int], int] = {}
    unique_out: defaultdict[int, set[int]] = defaultdict(set)
    unique_in: defaultdict[int, set[int]] = defaultdict(set)
    hist_neighbors: defaultdict[int, set[int]] = defaultdict(set)
    activity_ema: dict[int, float] = defaultdict(float)

    train_features: list[np.ndarray] = []
    val_features: list[np.ndarray] = []
    test_features: list[np.ndarray] = []
    val_labels: list[int] = []
    test_labels: list[int] = []
    edge_cursor = 0
    train_snapshots_seen: set[int] = set()
    val_snapshots_seen: set[int] = set()
    test_snapshots_seen: set[int] = set()
    val_campaigns: list[str] = []
    test_campaigns: list[str] = []
    rep_dim: int | None = None

    for snapshot, group in snapshot_groups:
        snapshot_value = int(snapshot)
        src_nodes = group["src_idx"].astype(int).to_numpy()
        dst_nodes = group["dst_idx"].astype(int).to_numpy()
        labels = group["label"].astype(int).to_numpy()
        campaigns = group["campaign"].astype(str).to_numpy()

        snap_out: Counter[int] = Counter()
        snap_in: Counter[int] = Counter()
        snap_out_neighbors: defaultdict[int, set[int]] = defaultdict(set)
        snap_in_neighbors: defaultdict[int, set[int]] = defaultdict(set)
        snap_pair_count: Counter[tuple[int, int]] = Counter()
        for src, dst in zip(src_nodes, dst_nodes):
            snap_out[src] += 1
            snap_in[dst] += 1
            snap_out_neighbors[src].add(dst)
            snap_in_neighbors[dst].add(src)
            snap_pair_count[(src, dst)] += 1
        active_nodes = sorted(set(src_nodes).union(dst_nodes))
        active_index = {node: idx for idx, node in enumerate(active_nodes)}
        snap_neighbors: dict[int, set[int]] = {}
        for node in active_nodes:
            snap_neighbors[node] = set(snap_out_neighbors.get(node, set())) | set(snap_in_neighbors.get(node, set()))

        base_features: dict[int, np.ndarray] = {}
        for node in active_nodes:
            base_features[node] = build_base_feature(
                node=node,
                snapshot=snapshot_value,
                featurization=featurization,
                id_features=id_features,
                hist_out=hist_out,
                hist_in=hist_in,
                hist_total=hist_total,
                snap_out=snap_out,
                snap_in=snap_in,
                snap_neighbors=snap_neighbors,
                snap_out_neighbors=snap_out_neighbors,
                snap_in_neighbors=snap_in_neighbors,
                hist_neighbors=hist_neighbors,
                unique_out=unique_out,
                unique_in=unique_in,
                last_seen=last_seen,
                activity_ema=activity_ema,
            )

        graph_encoded = apply_graph_encoder(
            base_features=base_features,
            active_nodes=active_nodes,
            snap_neighbors=snap_neighbors,
            graph_encoder=graph_encoder,
            projection_dim=projection_dim,
        )
        temporal_encoded = apply_temporal_encoder(
            graph_encoded=graph_encoded,
            active_nodes=active_nodes,
            snap_neighbors=snap_neighbors,
            temporal_encoder=temporal_encoder,
            hidden_state=active_hidden,
            cell_state=active_cell,
        )
        rep_dim = rep_dim or next(iter(temporal_encoded.values())).shape[0]

        for src, dst, label, campaign in zip(src_nodes, dst_nodes, labels, campaigns):
            edge_feature = build_edge_feature(
                src_state=temporal_encoded[src],
                dst_state=temporal_encoded[dst],
                scalar_values=np.array(
                    [
                        math.log1p(pair_hist.get((src, dst), 0)),
                        math.log1p(
                            snapshot_value - last_pair_seen[(src, dst)] if (src, dst) in last_pair_seen else snapshot_value + 1
                        ),
                        math.log1p(hist_out.get(src, 0)),
                        math.log1p(hist_in.get(dst, 0)),
                        math.log1p(snap_pair_count.get((src, dst), 0)),
                        math.log1p(len(snap_neighbors.get(src, set()).intersection(snap_neighbors.get(dst, set())))),
                        1.0 if dst in hist_neighbors.get(src, set()) else 0.0,
                        abs(hist_out.get(src, 0) - hist_in.get(dst, 0)) / (hist_out.get(src, 0) + hist_in.get(dst, 0) + 1.0),
                    ],
                    dtype=np.float32,
                ),
            )
            if edge_split_boundaries is not None:
                if edge_cursor < edge_split_boundaries[0]:
                    split_bucket = "train"
                elif edge_cursor < edge_split_boundaries[1]:
                    split_bucket = "val"
                else:
                    split_bucket = "test"
                edge_cursor += 1
            elif snapshot_value in train_snapshots:
                split_bucket = "train"
            elif snapshot_value in val_snapshots:
                split_bucket = "val"
            elif snapshot_value in test_snapshots:
                split_bucket = "test"
            else:
                split_bucket = ""
            if split_bucket == "train":
                train_features.append(edge_feature)
                train_snapshots_seen.add(snapshot_value)
            elif split_bucket == "val":
                val_features.append(edge_feature)
                val_labels.append(int(label))
                val_campaigns.append(str(campaign))
                val_snapshots_seen.add(snapshot_value)
            elif split_bucket == "test":
                test_features.append(edge_feature)
                test_labels.append(int(label))
                test_campaigns.append(str(campaign))
                test_snapshots_seen.add(snapshot_value)

        for node in active_nodes:
            snap_total = snap_out.get(node, 0) + snap_in.get(node, 0)
            activity_ema[node] = 0.7 * float(activity_ema.get(node, 0.0)) + 0.3 * float(snap_total)
            last_seen[node] = snapshot_value
            hist_out[node] += snap_out.get(node, 0)
            hist_in[node] += snap_in.get(node, 0)
            hist_total[node] += snap_total
            hist_neighbors[node].update(snap_neighbors.get(node, set()))
            unique_out[node].update(snap_out_neighbors.get(node, set()))
            unique_in[node].update(snap_in_neighbors.get(node, set()))
        for pair, count in snap_pair_count.items():
            pair_hist[pair] += int(count)
            last_pair_seen[pair] = snapshot_value

    if not train_features or not val_features or not test_features:
        raise ValueError(
            f"Representation build produced empty split: train={len(train_features)} val={len(val_features)} test={len(test_features)}"
        )
    if rep_dim is None:
        raise ValueError("Representation dimension could not be inferred")

    x_train = np.asarray(train_features, dtype=np.float32)
    x_val = np.asarray(val_features, dtype=np.float32)
    x_test = np.asarray(test_features, dtype=np.float32)
    if len(x_train) > MAX_TRAIN_ROWS:
        sample_idx = evenly_spaced_indices(len(x_train), MAX_TRAIN_ROWS)
        x_train = x_train[sample_idx]
    x_train, train_sanitize = sanitize_feature_matrix(x_train)
    x_val, val_sanitize = sanitize_feature_matrix(x_val)
    x_test, test_sanitize = sanitize_feature_matrix(x_test)
    if edge_split_boundaries is not None:
        train_snapshots = sorted(train_snapshots_seen)
        val_snapshots = sorted(val_snapshots_seen)
        test_snapshots = sorted(test_snapshots_seen)
    split_summary = {
        "train_snapshot_count": len(train_snapshots),
        "val_snapshot_count": len(val_snapshots),
        "test_snapshot_count": len(test_snapshots),
        "train_snapshot_min": min(train_snapshots),
        "train_snapshot_max": max(train_snapshots),
        "val_snapshot_min": min(val_snapshots),
        "val_snapshot_max": max(val_snapshots),
        "test_snapshot_min": min(test_snapshots),
        "test_snapshot_max": max(test_snapshots),
    }
    if edge_split_boundaries is not None:
        split_summary.update(
            {
                "split_mode": "chronological_edge_fallback",
                "fallback_train_fraction": FALLBACK_TRAIN_FRACTION,
                "train_edge_boundary": edge_split_boundaries[0],
                "val_edge_boundary": edge_split_boundaries[1],
                "trainwin_edge_count": trainwin_edge_count,
            }
        )
    if any(
        stats["nonfinite_count"] or stats["clipped_count"]
        for stats in (train_sanitize, val_sanitize, test_sanitize)
    ):
        split_summary.update(
            {
                "train_nonfinite_fixed": train_sanitize["nonfinite_count"],
                "train_clipped_values": train_sanitize["clipped_count"],
                "val_nonfinite_fixed": val_sanitize["nonfinite_count"],
                "val_clipped_values": val_sanitize["clipped_count"],
                "test_nonfinite_fixed": test_sanitize["nonfinite_count"],
                "test_clipped_values": test_sanitize["clipped_count"],
            }
        )
    return RepresentationBundle(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_val=np.asarray(val_labels, dtype=np.int32),
        y_test=np.asarray(test_labels, dtype=np.int32),
        val_campaigns=np.asarray(val_campaigns, dtype=str),
        test_campaigns=np.asarray(test_campaigns, dtype=str),
        rep_dim=rep_dim,
        scalar_dim=EDGE_SCALAR_DIM,
        train_rows=len(train_features),
        val_rows=len(val_features),
        test_rows=len(test_features),
        cache_hit=False,
        cache_key="",
        cache_dir=Path(),
        split_summary=split_summary,
    )

def build_identity_features(nodes: list[str], one_hot_dim: int) -> np.ndarray:
    matrix = np.zeros((len(nodes), one_hot_dim), dtype=np.float32)
    for index, node in enumerate(nodes):
        rng = np.random.default_rng(stable_seed("node", node, one_hot_dim))
        matrix[index] = rng.standard_normal(one_hot_dim).astype(np.float32)
    return matrix

def build_base_feature(
    *,
    node: int,
    snapshot: int,
    featurization: str,
    id_features: np.ndarray,
    hist_out: Counter[int],
    hist_in: Counter[int],
    hist_total: Counter[int],
    snap_out: Counter[int],
    snap_in: Counter[int],
    snap_neighbors: dict[int, set[int]],
    snap_out_neighbors: defaultdict[int, set[int]],
    snap_in_neighbors: defaultdict[int, set[int]],
    hist_neighbors: defaultdict[int, set[int]],
    unique_out: defaultdict[int, set[int]],
    unique_in: defaultdict[int, set[int]],
    last_seen: dict[int, int],
    activity_ema: dict[int, float],
) -> np.ndarray:
    if featurization == "one_hot":
        return id_features[node]
    structural = build_structural_feature(
        node=node,
        snap_out=snap_out,
        snap_in=snap_in,
        snap_neighbors=snap_neighbors,
        snap_out_neighbors=snap_out_neighbors,
        snap_in_neighbors=snap_in_neighbors,
        hist_total=hist_total,
        hist_neighbors=hist_neighbors,
        unique_out=unique_out,
        unique_in=unique_in,
    )
    if featurization == "structural":
        return structural
    if featurization == "flow_features":
        return np.asarray(
            [
                math.log1p(hist_out.get(node, 0)),
                math.log1p(hist_in.get(node, 0)),
                math.log1p(snap_out.get(node, 0)),
                math.log1p(snap_in.get(node, 0)),
                math.log1p(len(unique_out.get(node, set()))),
                math.log1p(len(unique_in.get(node, set()))),
                math.log1p(snapshot - last_seen[node]) if node in last_seen else math.log1p(snapshot + 1),
                math.log1p(max(0.0, float(activity_ema.get(node, 0.0)))),
            ],
            dtype=np.float32,
        )
    if featurization == "one_hot_structural":
        return np.concatenate([id_features[node], structural], axis=0).astype(np.float32)
    raise ValueError(f"Unsupported featurization: {featurization}")

def build_structural_feature(
    *,
    node: int,
    snap_out: Counter[int],
    snap_in: Counter[int],
    snap_neighbors: dict[int, set[int]],
    snap_out_neighbors: defaultdict[int, set[int]],
    snap_in_neighbors: defaultdict[int, set[int]],
    hist_total: Counter[int],
    hist_neighbors: defaultdict[int, set[int]],
    unique_out: defaultdict[int, set[int]],
    unique_in: defaultdict[int, set[int]],
) -> np.ndarray:
    node_neighbors = snap_neighbors.get(node, set())
    reciprocal = len(set(snap_out_neighbors.get(node, set())).intersection(snap_in_neighbors.get(node, set())))
    snap_degree = len(node_neighbors)
    neighbor_hist_mean = 0.0
    neighbor_snap_mean = 0.0
    if node_neighbors:
        neighbor_hist_mean = sum(math.log1p(hist_total.get(neighbor, 0)) for neighbor in node_neighbors) / len(node_neighbors)
        neighbor_snap_mean = sum(
            snap_out.get(neighbor, 0) + snap_in.get(neighbor, 0) for neighbor in node_neighbors
        ) / len(node_neighbors)
    persistent_neighbors = len(node_neighbors.intersection(hist_neighbors.get(node, set())))
    snap_total = snap_out.get(node, 0) + snap_in.get(node, 0)
    active_scale = max(1.0, float(len(node_neighbors) + 1))
    return np.asarray(
        [
            math.log1p(snap_degree),
            reciprocal / active_scale,
            float(neighbor_hist_mean),
            math.log1p(max(0.0, float(neighbor_snap_mean))),
            math.log1p(len(hist_neighbors.get(node, set()))),
            math.log1p(persistent_neighbors),
            (snap_out.get(node, 0) - snap_in.get(node, 0)) / (snap_total + 1.0),
            math.log1p(len(unique_out.get(node, set())) + len(unique_in.get(node, set()))),
        ],
        dtype=np.float32,
    )

def apply_graph_encoder(
    *,
    base_features: dict[int, np.ndarray],
    active_nodes: list[int],
    snap_neighbors: dict[int, set[int]],
    graph_encoder: str,
    projection_dim: int,
) -> dict[int, np.ndarray]:
    sample = next(iter(base_features.values()))
    feature_dim = int(sample.shape[0])
    matrix = np.stack([base_features[node] for node in active_nodes], axis=0)
    local_index = {node: idx for idx, node in enumerate(active_nodes)}
    neigh_mean = np.zeros_like(matrix)
    neigh_max = np.zeros_like(matrix)
    for node in active_nodes:
        idx = local_index[node]
        neighbors = [local_index[nbr] for nbr in snap_neighbors.get(node, set()) if nbr in local_index]
        if not neighbors:
            continue
        neighbor_matrix = matrix[neighbors]
        neigh_mean[idx] = neighbor_matrix.mean(axis=0)
        neigh_max[idx] = neighbor_matrix.max(axis=0)

    if graph_encoder == "none":
        encoded = matrix
    elif graph_encoder == "linear":
        weight = projection_matrix(feature_dim, projection_dim, "linear")
        encoded = np.tanh(matrix @ weight)
    elif graph_encoder == "gcn":
        weight = projection_matrix(feature_dim, projection_dim, "gcn")
        encoded = np.tanh((0.5 * matrix + 0.5 * neigh_mean) @ weight)
    elif graph_encoder == "graphsage":
        stacked = np.concatenate([matrix, neigh_mean], axis=1)
        weight = projection_matrix(stacked.shape[1], projection_dim, "graphsage")
        encoded = np.tanh(stacked @ weight)
    elif graph_encoder == "mpnn":
        stacked = np.concatenate([matrix, neigh_mean, neigh_max, np.abs(matrix - neigh_mean)], axis=1)
        weight = projection_matrix(stacked.shape[1], projection_dim, "mpnn")
        encoded = np.tanh(stacked @ weight)
    else:
        raise ValueError(f"Unsupported graph encoder: {graph_encoder}")
    return {node: encoded[local_index[node]].astype(np.float32) for node in active_nodes}

def apply_temporal_encoder(
    *,
    graph_encoded: dict[int, np.ndarray],
    active_nodes: list[int],
    snap_neighbors: dict[int, set[int]],
    temporal_encoder: str,
    hidden_state: dict[int, np.ndarray],
    cell_state: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    sample = next(iter(graph_encoded.values()))
    zeros = np.zeros_like(sample)
    updated: dict[int, np.ndarray] = {}
    for node in active_nodes:
        current = graph_encoded[node]
        prev_hidden = hidden_state.get(node, zeros)
        if temporal_encoder == "none":
            state = current
        elif temporal_encoder == "gru":
            state = np.tanh(0.65 * prev_hidden + 0.35 * current)
        elif temporal_encoder == "lstm":
            prev_cell = cell_state.get(node, zeros)
            cell = 0.7 * prev_cell + 0.3 * current
            cell_state[node] = cell.astype(np.float32)
            state = np.tanh(cell)
        elif temporal_encoder == "gc_lstm":
            prev_cell = cell_state.get(node, zeros)
            neighbor_hidden = [hidden_state.get(neighbor, zeros) for neighbor in snap_neighbors.get(node, set())]
            if neighbor_hidden:
                neigh_mean = np.mean(np.stack(neighbor_hidden, axis=0), axis=0)
            else:
                neigh_mean = zeros
            candidate = 0.5 * current + 0.5 * neigh_mean
            cell = 0.7 * prev_cell + 0.3 * candidate
            cell_state[node] = cell.astype(np.float32)
            state = np.tanh(cell)
        else:
            raise ValueError(f"Unsupported temporal encoder: {temporal_encoder}")
        hidden_state[node] = state.astype(np.float32)
        updated[node] = hidden_state[node]
    return updated

def build_edge_feature(*, src_state: np.ndarray, dst_state: np.ndarray, scalar_values: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [src_state, dst_state, np.abs(src_state - dst_state), src_state * dst_state, scalar_values.astype(np.float32)],
        axis=0,
    ).astype(np.float32)

def projection_matrix(input_dim: int, output_dim: int, label: str) -> np.ndarray:
    rng = np.random.default_rng(stable_seed("projection", label, input_dim, output_dim))
    scale = 1.0 / max(math.sqrt(float(input_dim)), 1.0)
    return rng.normal(loc=0.0, scale=scale, size=(input_dim, output_dim)).astype(np.float32)

def evenly_spaced_indices(total: int, keep: int) -> np.ndarray:
    if keep >= total:
        return np.arange(total, dtype=np.int64)
    return np.linspace(0, total - 1, num=keep, dtype=np.int64)

def run_detection(
    *,
    config: dict[str, Any],
    bundle: RepresentationBundle,
    dataset: str,
    experiment_id: str,
    predictions_path: Path,
    metrics_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    loss_name = str(config["loss"])
    rep_dim = int(bundle.rep_dim)
    scaler = StandardScaler()
    x_train_raw, _ = sanitize_feature_matrix(bundle.x_train)
    x_val_raw, _ = sanitize_feature_matrix(bundle.x_val)
    x_test_raw, _ = sanitize_feature_matrix(bundle.x_test)
    x_train = scaler.fit_transform(x_train_raw)
    x_val = scaler.transform(x_val_raw)
    x_test = scaler.transform(x_test_raw)

    train_t0 = time.perf_counter()
    if loss_name == "reconstruction":
        model = train_reconstruction_head(x_train)
        train_time = round(time.perf_counter() - train_t0, 6)
        val_scores = score_reconstruction(model, x_val)
        test_scores = score_reconstruction(model, x_test)
    elif loss_name == "contrastive":
        model = train_contrastive_head(x_train, rep_dim)
        train_time = round(time.perf_counter() - train_t0, 6)
        val_scores = score_linear_head(model, x_val)
        test_scores = score_linear_head(model, x_test)
    elif loss_name == "link_prediction":
        model = train_link_prediction_head(x_train, rep_dim)
        train_time = round(time.perf_counter() - train_t0, 6)
        val_scores = score_link_prediction_head(model, x_val, rep_dim)
        test_scores = score_link_prediction_head(model, x_test, rep_dim)
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")

    test_t0 = time.perf_counter()
    threshold = select_threshold(val_scores, bundle.y_val, int(config["mcc_grid_size"]))
    predicted = (test_scores >= threshold).astype(int)
    test_time = round(time.perf_counter() - test_t0, 6)

    tp = int(np.sum((predicted == 1) & (bundle.y_test == 1)))
    fp = int(np.sum((predicted == 1) & (bundle.y_test == 0)))
    fn = int(np.sum((predicted == 0) & (bundle.y_test == 1)))
    tn = int(np.sum((predicted == 0) & (bundle.y_test == 0)))

    ap = safe_metric(lambda: average_precision_score(bundle.y_test, test_scores), default=0.0)
    auc = safe_metric(lambda: roc_auc_score(bundle.y_test, test_scores), default=0.0)
    precision = safe_metric(lambda: precision_score(bundle.y_test, predicted, zero_division=0), default=0.0)
    recall = safe_metric(lambda: recall_score(bundle.y_test, predicted, zero_division=0), default=0.0)
    f1 = safe_metric(lambda: f1_score(bundle.y_test, predicted, zero_division=0), default=0.0)
    mcc = safe_metric(lambda: matthews_corrcoef(bundle.y_test, predicted), default=0.0)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0

    prediction_rows = build_prediction_rows(
        dataset=dataset,
        scores=test_scores.tolist(),
        labels=bundle.y_test.tolist(),
        predicted=predicted.tolist(),
        campaign=bundle.test_campaigns.tolist(),
        eval_index=0,
    )
    cdr_summary = compute_cdr_summary(prediction_rows)
    metrics_row = {
        "detector": "eg7_reference",
        "job_id": experiment_id,
        "dataset": dataset,
        "seed": "",
        "auc": round(float(auc), 6),
        "ap": round(float(ap), 6),
        "threshold": round(float(threshold), 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "tpr": round(float(tpr), 6),
        "fpr": round(float(fpr), 6),
        "fnr": round(float(fnr), 6),
        "tnr": round(float(tnr), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "mcc": round(float(mcc), 6),
        "train_time": train_time,
        "test_time": test_time,
        "cdr": round(float(cdr_summary["cdr"]), 6),
        "cdr_auc": round(float(cdr_summary["cdr_auc"]), 6),
        "campaign_count": int(cdr_summary["campaign_count"]),
        "detected_campaigns": int(cdr_summary["detected_campaigns"]),
        "cdr_meaningful": bool(cdr_summary["cdr_meaningful"]),
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
    }
    detection_summary = {
        "loss": loss_name,
        "threshold": float(threshold),
        "val_positive_rows": int(np.sum(bundle.y_val == 1)),
        "test_positive_rows": int(np.sum(bundle.y_test == 1)),
        "score_min": float(np.min(test_scores)),
        "score_max": float(np.max(test_scores)),
    }
    return metrics_row, prediction_rows, detection_summary

def train_reconstruction_head(x_train: np.ndarray) -> dict[str, Any]:
    n_components = max(1, min(12, x_train.shape[1] - 1, x_train.shape[0] - 1))
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=0)
    pca.fit(x_train)
    return {"pca": pca}

def score_reconstruction(model: dict[str, Any], x_eval: np.ndarray) -> np.ndarray:
    pca: PCA = model["pca"]
    reconstructed = pca.inverse_transform(pca.transform(x_eval))
    error = np.square(x_eval - reconstructed).mean(axis=1)
    return error.astype(np.float32)

def train_contrastive_head(x_train: np.ndarray, rep_dim: int) -> SGDClassifier:
    negative = corrupt_edge_features(x_train, rep_dim=rep_dim)
    features = np.concatenate([x_train, negative], axis=0)
    labels = np.concatenate(
        [np.ones(len(x_train), dtype=np.int32), np.zeros(len(negative), dtype=np.int32)],
        axis=0,
    )
    model = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=1000,
        tol=1e-3,
        random_state=0,
        class_weight="balanced",
    )
    model.fit(features, labels)
    return model

def train_link_prediction_head(x_train: np.ndarray, rep_dim: int) -> dict[str, Any]:
    negative = corrupt_edge_features(x_train, rep_dim=rep_dim)
    train_lp = extract_link_prediction_features(x_train, rep_dim)
    negative_lp = extract_link_prediction_features(negative, rep_dim)
    features = np.concatenate([train_lp, negative_lp], axis=0)
    labels = np.concatenate(
        [np.ones(len(train_lp), dtype=np.int32), np.zeros(len(negative_lp), dtype=np.int32)],
        axis=0,
    )
    model = SGDClassifier(
        loss="log_loss",
        alpha=5e-4,
        max_iter=1200,
        tol=1e-3,
        random_state=0,
        class_weight="balanced",
    )
    model.fit(features, labels)
    return {"classifier": model}

def score_linear_head(model: SGDClassifier, x_eval: np.ndarray) -> np.ndarray:
    return (-model.decision_function(x_eval)).astype(np.float32)

def score_link_prediction_head(model: dict[str, Any], x_eval: np.ndarray, rep_dim: int) -> np.ndarray:
    classifier: SGDClassifier = model["classifier"]
    features = extract_link_prediction_features(x_eval, rep_dim)
    return (-classifier.decision_function(features)).astype(np.float32)

def corrupt_edge_features(x_train: np.ndarray, *, rep_dim: int) -> np.ndarray:
    if len(x_train) <= 1:
        return x_train.copy()
    src = x_train[:, :rep_dim]
    dst = x_train[:, rep_dim : 2 * rep_dim]
    scalars = x_train[:, 4 * rep_dim :]
    shift = np.roll(np.arange(len(x_train)), 1)
    dst_perm = dst[shift]
    scalars_perm = scalars[shift]
    scalar_block = np.stack(
        [
            np.zeros(len(x_train), dtype=np.float32),
            np.zeros(len(x_train), dtype=np.float32),
            scalars[:, 2],
            scalars_perm[:, 3],
            np.zeros(len(x_train), dtype=np.float32),
            np.zeros(len(x_train), dtype=np.float32),
            np.zeros(len(x_train), dtype=np.float32),
            np.abs(scalars[:, 2] - scalars_perm[:, 3]) / (scalars[:, 2] + scalars_perm[:, 3] + 1.0),
        ],
        axis=1,
    )
    return np.concatenate([src, dst_perm, np.abs(src - dst_perm), src * dst_perm, scalar_block], axis=1).astype(np.float32)

def extract_link_prediction_features(x: np.ndarray, rep_dim: int) -> np.ndarray:
    src = x[:, :rep_dim]
    dst = x[:, rep_dim : 2 * rep_dim]
    scalars = x[:, 4 * rep_dim :]
    similarity = np.sum(src * dst, axis=1, keepdims=True)
    diff_norm = np.linalg.norm(src - dst, axis=1, keepdims=True)
    history = scalars[:, [0, 2, 3, 5, 6, 7]]
    return np.concatenate([similarity, diff_norm, history], axis=1).astype(np.float32)

def select_threshold(scores: np.ndarray, labels: np.ndarray, grid_size: int) -> float:
    if len(scores) == 0:
        return 0.0
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    if len(np.unique(labels)) < 2:
        return float(np.median(scores))
    eps = 1e-6
    unique_scores = np.unique(scores)
    if len(unique_scores) <= grid_size:
        candidates = unique_scores
    else:
        quantiles = np.linspace(0.0, 1.0, num=grid_size, dtype=np.float32)
        candidates = np.quantile(scores, quantiles)
    candidates = np.concatenate(
        [
            np.array([scores.min() - eps], dtype=np.float32),
            np.asarray(candidates, dtype=np.float32),
            np.array([scores.max() + eps], dtype=np.float32),
        ]
    )
    best_threshold = float(candidates[0])
    best_score = -float("inf")
    best_recall = -float("inf")
    for threshold in candidates:
        predicted = (scores >= threshold).astype(int)
        score = safe_metric(lambda: matthews_corrcoef(labels, predicted), default=0.0)
        recall = safe_metric(lambda: recall_score(labels, predicted, zero_division=0), default=0.0)
        if score > best_score or (math.isclose(score, best_score) and recall > best_recall):
            best_threshold = float(threshold)
            best_score = float(score)
            best_recall = float(recall)
    return best_threshold

def safe_metric(fn: Any, *, default: float) -> float:
    try:
        value = fn()
    except Exception:
        return float(default)
    if isinstance(value, np.ndarray):
        return float(value.item())
    return float(value)

def build_cost_rows(
    *,
    config: dict[str, Any],
    phase_statuses: dict[str, dict[str, Any]],
    metadata_root: Path,
    baseline_path: Path,
    metrics_row: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    profile_path = Path(str(metadata_root / "resource_profile.json"))
    phase_usage: dict[str, Any] = {}
    if profile_path.exists():
        try:
            phase_usage = json.loads(profile_path.read_text(encoding="utf-8")).get("phases") or {}
        except (OSError, ValueError):
            phase_usage = {}
    rows: list[dict[str, Any]] = []
    for phase in ("preprocess", "representation", "detection"):
        status_info = phase_statuses.get(phase)
        if not status_info:
            continue
        usage = dict(phase_usage.get(phase) or {})
        rows.append(
            {
                "detector": "eg7_reference",
                "phase": phase,
                "job_id": str(config["experiment_id"]),
                "dataset": str(config["dataset"]),
                "seed": "",
                "status": status_info["status"],
                "exit_code": usage.get("exit_code", 0),
                "started_at": "",
                "finished_at": "",
                "wall_time_sec": round(float(usage.get("wall_time_sec", 0.0) or 0.0), 6),
                "peak_rss_kb": int(usage.get("peak_rss_kb", 0) or 0),
                "mean_rss_kb": round(float(usage.get("mean_rss_kb", 0.0) or 0.0), 3),
                "peak_cpu_percent": round(float(usage.get("peak_cpu_percent", 0.0) or 0.0), 3),
                "mean_cpu_percent": round(float(usage.get("mean_cpu_percent", 0.0) or 0.0), 3),
                "peak_process_count": int(usage.get("peak_process_count", 0) or 0),
                "sample_count": int(usage.get("sample_count", 0) or 0),
                "cpu_sample_count": int(usage.get("cpu_sample_count", 0) or 0),
                "cache_hit": bool(status_info.get("cache_hit", False)),
                "cache_key": str(status_info.get("cache_key", "")),
                "train_time": metrics_row.get("train_time", "") if metrics_row and phase == "detection" else "",
                "test_time": metrics_row.get("test_time", "") if metrics_row and phase == "detection" else "",
                "command": "internal_eg7_worker",
                "workdir": str(metadata_root.parent.parent),
                "stdout_log": str(metadata_root / "logs" / "worker.stdout.log"),
                "stderr_log": str(metadata_root / "logs" / "worker.stderr.log"),
                "metrics_path": str(baseline_path),
            }
        )
    return rows

def sanitize_feature_matrix(matrix: Any, *, clip_abs: float = FLOAT16_CACHE_ABS_MAX) -> tuple[np.ndarray, dict[str, int]]:
    array = np.asarray(matrix, dtype=np.float32)
    if array.size == 0:
        return array, {"nonfinite_count": 0, "clipped_count": 0}
    nonfinite_mask = ~np.isfinite(array)
    nonfinite_count = int(nonfinite_mask.sum())
    sanitized = np.nan_to_num(
        array,
        copy=True,
        nan=0.0,
        posinf=clip_abs,
        neginf=-clip_abs,
    )
    clipped_mask = np.abs(sanitized) > clip_abs
    clipped_count = int(clipped_mask.sum())
    if clipped_count:
        np.clip(sanitized, -clip_abs, clip_abs, out=sanitized)
    return sanitized.astype(np.float32, copy=False), {
        "nonfinite_count": nonfinite_count,
        "clipped_count": clipped_count,
    }

def representation_bundle_has_nonfinite_values(bundle: RepresentationBundle) -> bool:
    return any(
        not np.isfinite(matrix).all()
        for matrix in (bundle.x_train, bundle.x_val, bundle.x_test)
    )

def hydrate_dataset_cache_from_existing(
    *,
    dataset_cache_dir: Path,
    dataset: str,
    window_seconds: int,
    neg_ratio: int,
    seed: int,
    cache_root: Path,
) -> bool:
    input_path = dataset_cache_dir / "input.csv"
    metadata_path = dataset_cache_dir / "metadata.json"
    for candidate in iter_dataset_cache_candidates(
        dataset=dataset,
        window_seconds=window_seconds,
        neg_ratio=neg_ratio,
        seed=seed,
        cache_root=cache_root,
    ):
        if candidate.resolve() == dataset_cache_dir.resolve():
            continue
        shutil.copy2(candidate / "input.csv", input_path)
        metadata = json.loads((candidate / "metadata.json").read_text(encoding="utf-8"))
        metadata["input_path"] = str(input_path)
        dump_json(metadata_path, metadata)
        optc_sidecar = candidate / "redteam_optc_small.txt"
        if optc_sidecar.exists():
            shutil.copy2(optc_sidecar, dataset_cache_dir / optc_sidecar.name)
        return True
    return False

def iter_dataset_cache_candidates(
    *,
    dataset: str,
    window_seconds: int,
    neg_ratio: int,
    seed: int,
    cache_root: Path,
) -> list[Path]:
    candidates: list[Path] = []

    local_root = cache_root / "datasets" / dataset
    if local_root.exists():
        candidates.extend(sorted(local_root.glob(f"w{window_seconds}_nr{neg_ratio}_s{seed}_*")))

    work_root = PATHS.work_root
    external_patterns = [
        (work_root / "eg3/cache/experiments/featurization", f"*/output/data_root/pikachu_sc3/{dataset}"),
        (
            work_root / "eg3/experiments",
            f"*/eg3/output/modules/featurization/output/data_root/pikachu_sc3/{dataset}",
        ),
        (work_root / "eg1/cache/experiments/featurization", f"*/output/data_root/pikachu_sc1/{dataset}"),
        (
            work_root / "eg1/experiments",
            f"*/eg1/output/modules/featurization/output/data_root/pikachu_sc1/{dataset}",
        ),
        (work_root / "eg8/canonical/pikachu", f"{dataset}"),
    ]
    for root, pattern in external_patterns:
        if root.exists():
            candidates.extend(sorted(root.glob(pattern)))

    matched: list[Path] = []
    for candidate in candidates:
        if dataset_cache_matches(
            candidate,
            dataset=dataset,
            window_seconds=window_seconds,
            neg_ratio=neg_ratio,
            seed=seed,
        ):
            matched.append(candidate)
    return matched

def dataset_cache_matches(
    candidate: Path,
    *,
    dataset: str,
    window_seconds: int,
    neg_ratio: int,
    seed: int,
) -> bool:
    input_path = candidate / "input.csv"
    metadata_path = candidate / "metadata.json"
    if not input_path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        str(metadata.get("dataset", "")).strip() == dataset
        and int(metadata.get("snapshot_seconds", -1) or -1) == window_seconds
        and int(metadata.get("neg_ratio", -1) or -1) == neg_ratio
        and int(metadata.get("preprocess_seed", -1) or -1) == seed
    )

def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)

def stable_key(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")

if __name__ == "__main__":
    sys.exit(main())
