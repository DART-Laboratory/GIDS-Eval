
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

FAMILY_FILES = {
    "oilrig": "label_oilrig.log",
    "sandworm": "label_sandworm.log",
    "wizard_spider": "label_wizard_spider.log",
}

DEFAULT_ATTACK_INSERTION_SECONDS = {
    "oilrig": 8 * 3600,
    "sandworm": 14 * 3600,
    "wizard_spider": 18 * 3600,
}

DEFAULT_BACKGROUND_FAMILY = "oilrig"
DEFAULT_DAY_SECONDS = 24 * 3600

@dataclass(frozen=True)
class FileStats:
    rows: int
    positive_rows: int
    min_ts: float
    max_ts: float
    first_positive_ts: float | None
    last_positive_ts: float | None

def family_log_path(raw_root: Path, family: str) -> Path:
    return raw_root / "0501" / FAMILY_FILES[family]

def scan_file_stats(path: Path) -> FileStats:
    rows = 0
    positive_rows = 0
    min_ts: float | None = None
    max_ts: float | None = None
    first_positive_ts: float | None = None
    last_positive_ts: float | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue
            ts = float(row[0])
            label = int(float(row[-1]))
            rows += 1
            min_ts = ts if min_ts is None or ts < min_ts else min_ts
            max_ts = ts if max_ts is None or ts > max_ts else max_ts
            if label == 1:
                positive_rows += 1
                first_positive_ts = ts if first_positive_ts is None or ts < first_positive_ts else first_positive_ts
                last_positive_ts = ts if last_positive_ts is None or ts > last_positive_ts else last_positive_ts
    if rows == 0 or min_ts is None or max_ts is None:
        raise ValueError(f"0501 source file is empty: {path}")
    return FileStats(
        rows=rows,
        positive_rows=positive_rows,
        min_ts=min_ts,
        max_ts=max_ts,
        first_positive_ts=first_positive_ts,
        last_positive_ts=last_positive_ts,
    )

def build_sorted_combined_events(
    *,
    raw_root: Path,
    work_root: Path,
    background_family: str = DEFAULT_BACKGROUND_FAMILY,
    day_seconds: int = DEFAULT_DAY_SECONDS,
    insertion_starts: dict[str, int] | None = None,
) -> dict[str, object]:
    insertion_starts = dict(DEFAULT_ATTACK_INSERTION_SECONDS if insertion_starts is None else insertion_starts)
    stats = {family: scan_file_stats(family_log_path(raw_root, family)) for family in FAMILY_FILES}
    base_stats = stats[background_family]
    day_start = base_stats.min_ts
    day_end = day_start + float(day_seconds)

    work_root.mkdir(parents=True, exist_ok=True)
    unsorted_path = work_root / "combined_0501_events.unsorted.tsv"
    sorted_path = work_root / "combined_0501_events.sorted.tsv"

    with unsorted_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        _write_background_rows(
            writer=writer,
            path=family_log_path(raw_root, background_family),
            day_start=day_start,
            day_end=day_end,
        )
        for family, start_offset in insertion_starts.items():
            _write_shifted_attack_rows(
                writer=writer,
                path=family_log_path(raw_root, family),
                family=family,
                segment_start=stats[family].first_positive_ts,
                segment_end=stats[family].last_positive_ts,
                target_start=day_start + float(start_offset),
            )

    subprocess.run(
        ["sort", "-s", "-g", "-t", "\t", "-k1,1", str(unsorted_path), "-o", str(sorted_path)],
        check=True,
    )

    return {
        "sorted_path": sorted_path,
        "day_start": day_start,
        "day_end": day_end,
        "background_family": background_family,
        "day_seconds": day_seconds,
        "insertion_starts": insertion_starts,
        "stats": stats,
    }

def _write_background_rows(
    *,
    writer: csv.writer,
    path: Path,
    day_start: float,
    day_end: float,
) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue
            ts = float(row[0])
            if ts < day_start or ts >= day_end:
                continue
            label = int(float(row[-1]))
            if label != 0:
                continue
            writer.writerow([f"{ts:.6f}", row[2], row[4], 0, ""])

def _write_shifted_attack_rows(
    *,
    writer: csv.writer,
    path: Path,
    family: str,
    segment_start: float | None,
    segment_end: float | None,
    target_start: float,
) -> None:
    if segment_start is None or segment_end is None:
        raise ValueError(f"0501 family {family} has no positive rows in {path}")
    shift = target_start - segment_start
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue
            ts = float(row[0])
            if ts < segment_start or ts > segment_end:
                continue
            label = int(float(row[-1]))
            campaign = family if label == 1 else ""
            writer.writerow([f"{ts + shift:.6f}", row[2], row[4], label, campaign])

def iter_sorted_events(sorted_path: Path) -> Iterable[tuple[float, str, str, int, str]]:
    with sorted_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue
            yield float(row[0]), str(row[1]), str(row[2]), int(float(row[3])), str(row[4])

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

    rng = random.Random(seed)
    remaining_idx = [idx for idx in neg.index.tolist() if idx not in set(mandatory_idx)]
    extra_count = target_neg - len(mandatory_idx)
    sampled_extra = rng.sample(remaining_idx, extra_count) if extra_count < len(remaining_idx) else remaining_idx
    chosen_idx = sorted(set(mandatory_idx).union(sampled_extra))
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
        raise SystemExit("Unified 0501 PIKACHU build produced no rows")

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

    stats = metadata["stats"]
    payload = {
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
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EG3 standalone 0501 unified dataset builder")
    parser.add_argument("--mode", choices=["pikachu"], required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--work-root", default=None)
    parser.add_argument("--output-dir", required=True, help="Target PIKACHU preprocessing folder")
    parser.add_argument("--snapshot-seconds", type=int, required=True)
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
    if not args.output_dir or not args.snapshot_seconds:
        raise SystemExit("--output-dir and --snapshot-seconds are required")

    output_path = Path(args.output_dir)
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

    write_pikachu_dataset(
        output_path,
        metadata,
        int(args.snapshot_seconds),
        neg_ratio=int(args.neg_ratio),
        seed=int(args.seed),
    )

if __name__ == "__main__":
    main()
