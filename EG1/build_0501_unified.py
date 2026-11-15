from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import pandas as pd

from merge_0501 import build_sorted_combined_events, iter_sorted_events

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a true one-run EG1 0501 dataset")
    parser.add_argument("--mode", choices=["graph", "pikachu"], required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--work-root", default=None)
    parser.add_argument("--output-root", help="Target graph detector folder like .../argus/0501/0501")
    parser.add_argument("--output-dir", help="Target PIKACHU preprocessing folder")
    parser.add_argument("--snapshot-seconds", type=int, default=None)
    parser.add_argument("--neg-ratio", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--background-family", default="oilrig")
    parser.add_argument("--oilrig-start", type=int, default=8 * 3600)
    parser.add_argument("--sandworm-start", type=int, default=14 * 3600)
    parser.add_argument("--wizard-start", type=int, default=18 * 3600)
    parser.add_argument("--day-seconds", type=int, default=24 * 3600)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    if args.mode == "graph" and not args.output_root:
        raise SystemExit("--output-root is required for --mode graph")
    if args.mode == "pikachu" and (not args.output_dir or not args.snapshot_seconds):
        raise SystemExit("--output-dir and --snapshot-seconds are required for --mode pikachu")

    output_path = Path(args.output_root or args.output_dir)
    work_root = Path(args.work_root) if args.work_root else output_path / "_merge_work"
    insertion_starts = {
        "oilrig": int(args.oilrig_start),
        "sandworm": int(args.sandworm_start),
        "wizard_spider": int(args.wizard_start),
    }
    metadata = build_sorted_combined_events(
        raw_root=raw_root,
        work_root=work_root,
        background_family=args.background_family,
        day_seconds=int(args.day_seconds),
        insertion_starts=insertion_starts,
    )

    if args.mode == "graph":
        write_graph_dataset(Path(args.output_root), metadata)
    else:
        write_pikachu_dataset(
            Path(args.output_dir),
            metadata,
            int(args.snapshot_seconds),
            neg_ratio=int(args.neg_ratio),
            seed=int(args.seed),
        )

def write_graph_dataset(output_root: Path, metadata: dict[str, object]) -> None:
    import numpy as np

    output_root.mkdir(parents=True, exist_ok=True)
    sorted_path = Path(str(metadata["sorted_path"]))
    node_map: dict[str, int] = {}
    reverse_map: list[str] = []
    current_floor = 0
    current_file = (output_root / "0.txt").open("w", encoding="utf-8")
    first_ts: float | None = None
    first_positive_ms: int | None = None
    first_positive_index: int | None = None
    row_index = 0
    last_ms = 0

    def get_or_add(node: str) -> int:
        if node in node_map:
            return node_map[node]
        node_id = len(reverse_map)
        node_map[node] = node_id
        reverse_map.append(node)
        return node_id

    try:
        for ts, src, dst, label, campaign in iter_sorted_events(sorted_path):
            if first_ts is None:
                first_ts = ts
            rel_ms = int((ts - first_ts) * 1000.0)
            while rel_ms >= current_floor + 100000:
                current_file.close()
                current_floor += 100000
                current_file = (output_root / f"{current_floor}.txt").open("w", encoding="utf-8")
            src_id = get_or_add(src)
            dst_id = get_or_add(dst)
            if label == 1 and first_positive_ms is None:
                first_positive_ms = rel_ms
                first_positive_index = row_index
            current_file.write(f"{rel_ms},{src_id},{dst_id},{label},{campaign}\n")
            row_index += 1
            last_ms = rel_ms
    finally:
        current_file.close()

    if first_ts is None:
        raise SystemExit("Unified 0501 build produced no rows")

    with (output_root / "nmap.pkl").open("wb") as handle:
        pickle.dump(reverse_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

    np.save(output_root / "time.npy", np.array((first_positive_ms or 0, last_ms), dtype=np.int32))
    np.save(output_root / "data_info.npy", np.array((len(reverse_map), first_positive_index or 0), dtype=np.int32))

    payload = _metadata_payload(metadata)
    payload.update(
        {
            "format": "graph",
            "output_root": str(output_root),
            "row_count": int(row_index),
            "node_count": int(len(reverse_map)),
            "first_positive_ms": int(first_positive_ms or 0),
            "last_ms": int(last_ms),
        }
    )
    with (output_root / "eg1_0501_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

def downsample_negatives_preserve_snapshots(
    frame: pd.DataFrame,
    *,
    neg_ratio: int,
    seed: int,
) -> pd.DataFrame:
    if neg_ratio <= 0 or frame.empty:
        return frame.sort_values(by=["timestamp", "snapshot"]).reset_index(drop=True)

    pos = frame[frame["label"] == 1]
    neg = frame[frame["label"] == 0]
    if pos.empty or neg.empty:
        return frame.sort_values(by=["timestamp", "snapshot"]).reset_index(drop=True)

    target_neg = min(len(neg), len(pos) * neg_ratio)
    if target_neg >= len(neg):
        return frame.sort_values(by=["timestamp", "snapshot"]).reset_index(drop=True)

    mandatory_idx = (
        neg.sort_values(by=["snapshot", "timestamp", "src_computer", "dst_computer"])
        .groupby("snapshot", sort=True)
        .head(1)
        .index
        .tolist()
    )
    target_neg = max(target_neg, len(mandatory_idx))
    if target_neg >= len(neg):
        return frame.sort_values(by=["timestamp", "snapshot"]).reset_index(drop=True)

    rng = pd.Series(neg.index.tolist()).sample(frac=1.0, random_state=seed).tolist()
    mandatory = set(mandatory_idx)
    remaining_idx = [idx for idx in rng if idx not in mandatory]
    extra_count = target_neg - len(mandatory_idx)
    chosen_idx = sorted(mandatory.union(remaining_idx[:extra_count]))
    neg_sampled = neg.loc[chosen_idx]
    sampled = pd.concat([pos, neg_sampled], ignore_index=True)
    return sampled.sort_values(by=["timestamp", "snapshot", "src_computer", "dst_computer"]).reset_index(drop=True)

def densify_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    ordered = sorted(int(value) for value in result["snapshot"].drop_duplicates().tolist())
    mapping = {snapshot: index for index, snapshot in enumerate(ordered)}
    result["snapshot"] = result["snapshot"].map(mapping).astype(int)
    return result


MIN_TRAIN_EDGES = 100
FALLBACK_TRAIN_FRACTION = 0.4


def compute_guarded_trainwin(frame: pd.DataFrame, first_positive_snapshot: int) -> int:
    sizes = frame.groupby("snapshot").size().sort_index()
    pre_attack_edges = int(sizes.loc[sizes.index < first_positive_snapshot].sum())
    if pre_attack_edges >= MIN_TRAIN_EDGES:
        return int(first_positive_snapshot)
    target = int(round(len(frame) * FALLBACK_TRAIN_FRACTION))
    cumulative = sizes.cumsum()
    boundary_snapshot = int(cumulative.index[(cumulative >= target).argmax()])
    return boundary_snapshot + 1


def write_pikachu_dataset(
    output_dir: Path,
    metadata: dict[str, object],
    snapshot_seconds: int,
    *,
    neg_ratio: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_path = Path(str(metadata["sorted_path"]))
    output_csv = output_dir / "input.csv"
    first_ts: float | None = None

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "src_computer", "dst_computer", "label", "campaign", "snapshot"],
        )
        writer.writeheader()
        for ts, src, dst, label, campaign in iter_sorted_events(sorted_path):
            if first_ts is None:
                first_ts = ts
            snapshot = int((ts - first_ts) // float(snapshot_seconds))
            writer.writerow(
                {
                    "timestamp": int(ts),
                    "src_computer": src,
                    "dst_computer": dst,
                    "label": label,
                    "campaign": campaign,
                    "snapshot": snapshot,
                }
            )

    if first_ts is None:
        raise SystemExit("Unified 0501 PIKACHU build produced no positive rows")

    frame = pd.read_csv(output_csv)
    if frame.empty or not (frame["label"] == 1).any():
        raise SystemExit("Unified 0501 PIKACHU build produced no positive rows")
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce").fillna(0).astype(int)
    frame["snapshot"] = pd.to_numeric(frame["snapshot"], errors="coerce").fillna(0).astype(int)
    frame["src_computer"] = frame["src_computer"].astype(str)
    frame["dst_computer"] = frame["dst_computer"].astype(str)
    frame["campaign"] = frame["campaign"].fillna("").astype(str)
    original_snapshot_count = int(frame["snapshot"].nunique())
    original_first_attack_snapshot = int(frame.loc[frame["label"] == 1, "snapshot"].min())
    frame = downsample_negatives_preserve_snapshots(frame, neg_ratio=neg_ratio, seed=seed)
    frame = densify_snapshots(frame)
    frame = frame.sort_values(by=["timestamp", "snapshot", "src_computer", "dst_computer"]).reset_index(drop=True)
    frame.to_csv(output_csv, index=False)

    first_positive_snapshot = int(frame.loc[frame["label"] == 1, "snapshot"].min())
    positive_rows = int((frame["label"] == 1).sum())
    row_count = int(len(frame))

    payload = _metadata_payload(metadata)
    payload.update(
        {
            "format": "pikachu",
            "dataset": "0501",
            "snapshot_seconds": int(snapshot_seconds),
            "input_path": str(output_csv),
            "row_count": int(row_count),
            "positive_rows": int(positive_rows),
            "campaign_mode": "family",
            "first_attack_snapshot": int(first_positive_snapshot),
            "original_first_attack_snapshot": int(original_first_attack_snapshot),
            "snapshot_count": int(frame["snapshot"].nunique()),
            "original_snapshot_count": int(original_snapshot_count),
            "trainwin": compute_guarded_trainwin(frame, first_positive_snapshot),
            "initial_time": int(first_ts),
            "first_attack_timestamp": int(frame.loc[frame["label"] == 1, "timestamp"].min()),
            "neg_ratio": int(neg_ratio),
            "preprocess_seed": int(seed),
        }
    )
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

def _metadata_payload(metadata: dict[str, object]) -> dict[str, object]:
    stats = metadata["stats"]
    return {
        "background_family": str(metadata["background_family"]),
        "day_seconds": int(metadata["day_seconds"]),
        "insertion_starts": {str(k): int(v) for k, v in dict(metadata["insertion_starts"]).items()},
        "source_stats": {
            str(name): {
                "rows": int(stat.rows),
                "positive_rows": int(stat.positive_rows),
                "min_ts": float(stat.min_ts),
                "max_ts": float(stat.max_ts),
                "first_positive_ts": float(stat.first_positive_ts) if stat.first_positive_ts is not None else None,
                "last_positive_ts": float(stat.last_positive_ts) if stat.last_positive_ts is not None else None,
            }
            for name, stat in dict(stats).items()
        },
    }

if __name__ == "__main__":
    main()
