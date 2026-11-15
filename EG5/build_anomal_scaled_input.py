from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "EG1"))

from merge_0501 import DEFAULT_ATTACK_INSERTION_SECONDS, family_log_path, scan_file_stats
from merge_0501 import select_background_window_start

RAW_0501_INDICES = {
    "timestamp": 0,
    "src": 2,
    "dst": 4,
    "duration": 8,
    "orig_bytes": 9,
    "resp_bytes": 10,
    "missed_bytes": 14,
    "orig_pkts": 16,
    "orig_ip_bytes": 17,
    "resp_pkts": 18,
    "resp_ip_bytes": 19,
    "label": 21,
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SC5-scaled Anomal-E inputs")
    parser.add_argument("--dataset", choices=["cic", "0501"], required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scale-percent", required=True, type=int)
    parser.add_argument("--source-experiment-id", default="")
    parser.add_argument("--background-family", default="oilrig")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "cic":
        build_cic_scaled_input(
            raw_root=raw_root,
            output_dir=output_dir,
            scale_percent=int(args.scale_percent),
            source_experiment_id=str(args.source_experiment_id),
        )
        return

    build_unified_0501_scaled_input(
        raw_root=raw_root,
        output_dir=output_dir,
        scale_percent=int(args.scale_percent),
        source_experiment_id=str(args.source_experiment_id),
        background_family=str(args.background_family),
    )

def build_cic_scaled_input(
    *,
    raw_root: Path,
    output_dir: Path,
    scale_percent: int,
    source_experiment_id: str,
) -> None:
    input_dir = raw_root / "CIC-new"
    frame = pd.concat(
        [_load_cic_day(input_dir / f"{day}.csv") for day in ("monday", "tuesday", "wednesday", "thursday", "friday")],
        ignore_index=True,
    )
    frame = frame.sort_values(by="time_int").reset_index(drop=True)
    frame["time_int"] = frame["time_int"] - int(frame["time_int"].iloc[0])
    frame["raw_label"] = frame["Label"].fillna("").astype(str).str.strip()
    frame["label"] = frame["raw_label"].map(lambda value: 0 if value == "BENIGN" else 1).astype(int)
    frame["campaign"] = frame["raw_label"].where(frame["label"] == 1, "")
    split_index = _first_positive_index(frame["label"].tolist())
    scaled = _scale_frame_by_phase(frame, split_index=split_index, scale_percent=scale_percent)
    output_path = output_dir / "cic_2017_1.csv"
    scaled.to_csv(output_path, encoding="utf-8", index=False)
    src_col = "Source IP" if "Source IP" in scaled.columns else "Src IP"
    dst_col = "Destination IP" if "Destination IP" in scaled.columns else "Dst IP"
    metadata = {
        "dataset": "cic",
        "scale_percent": scale_percent,
        "source_experiment_id": source_experiment_id,
        "input_path": str(output_path),
        "split_index": int(_first_positive_index(scaled["label"].tolist())),
        "row_count": int(len(scaled)),
        "positive_rows": int((scaled["label"] == 1).sum()),
        "node_count": int(len(set(scaled[src_col].astype(str)).union(set(scaled[dst_col].astype(str))))),
        "campaign_count": int(scaled.loc[scaled["label"] == 1, "campaign"].astype(str).replace("", pd.NA).dropna().nunique()),
        "source_pre_rows": int(split_index),
        "source_post_rows": int(len(frame) - split_index),
        "kept_pre_rows": int(_scaled_count(split_index, scale_percent)),
        "kept_post_rows": int(_scaled_count(len(frame) - split_index, scale_percent)),
    }
    _dump_json(output_dir / "metadata.json", metadata)

def build_unified_0501_scaled_input(
    *,
    raw_root: Path,
    output_dir: Path,
    scale_percent: int,
    source_experiment_id: str,
    background_family: str,
) -> None:
    work_root = output_dir / "_build_work"
    work_root.mkdir(parents=True, exist_ok=True)
    sorted_path = _build_sorted_0501_feature_events(
        raw_root=raw_root,
        work_root=work_root,
        background_family=background_family,
    )
    split_index, total_rows = _scan_0501_split(sorted_path)
    keep_pre = _scaled_count(split_index, scale_percent)
    keep_post = _scaled_count(total_rows - split_index, scale_percent)
    output_path = output_dir / "0501_unified.csv"
    row_count, positive_rows, node_count, campaign_count = _write_scaled_0501_csv(
        sorted_path=sorted_path,
        output_path=output_path,
        keep_pre=keep_pre,
        keep_post=keep_post,
    )
    metadata = {
        "dataset": "0501",
        "scale_percent": scale_percent,
        "source_experiment_id": source_experiment_id,
        "input_path": str(output_path),
        "split_index": int(keep_pre),
        "row_count": int(row_count),
        "positive_rows": int(positive_rows),
        "node_count": int(node_count),
        "campaign_count": int(campaign_count),
        "source_pre_rows": int(split_index),
        "source_post_rows": int(total_rows - split_index),
        "kept_pre_rows": int(keep_pre),
        "kept_post_rows": int(keep_post),
        "background_family": background_family,
    }
    _dump_json(output_dir / "metadata.json", metadata)

def _load_cic_day(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["time_int"] = [
        int(datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M:%S.%f").timestamp())
        for ts in frame["Timestamp"].astype(str)
    ]
    return frame

def _first_positive_index(labels: list[int]) -> int:
    for index, label in enumerate(labels):
        if int(label) == 1:
            return index
    return len(labels)

def _scale_frame_by_phase(frame: pd.DataFrame, *, split_index: int, scale_percent: int) -> pd.DataFrame:
    pre = frame.iloc[:split_index]
    post = frame.iloc[split_index:]
    keep_pre = _scaled_count(len(pre), scale_percent)
    keep_post = _scaled_count(len(post), scale_percent)
    scaled = pd.concat([pre.iloc[:keep_pre], post.iloc[:keep_post]], ignore_index=True)
    return scaled.reset_index(drop=True)

def _build_sorted_0501_feature_events(
    *,
    raw_root: Path,
    work_root: Path,
    background_family: str,
) -> Path:
    stats = {family: scan_file_stats(family_log_path(raw_root, family)) for family in DEFAULT_ATTACK_INSERTION_SECONDS}
    day_start = select_background_window_start(
        family_log_path(raw_root, background_family),
        day_seconds=24 * 3600,
        preferred_pre_attack_seconds=float(min(DEFAULT_ATTACK_INSERTION_SECONDS.values())),
    )
    day_end = day_start + 24 * 3600
    unsorted_path = work_root / "combined_0501_anomal.unsorted.tsv"
    sorted_path = work_root / "combined_0501_anomal.sorted.tsv"

    with unsorted_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        _write_background_0501_rows(
            writer=writer,
            path=family_log_path(raw_root, background_family),
            day_start=day_start,
            day_end=day_end,
        )
        for family, start_offset in DEFAULT_ATTACK_INSERTION_SECONDS.items():
            stat = stats[family]
            _write_shifted_0501_rows(
                writer=writer,
                path=family_log_path(raw_root, family),
                family=family,
                segment_start=stat.first_positive_ts,
                segment_end=stat.last_positive_ts,
                target_start=day_start + float(start_offset),
            )

    subprocess.run(
        ["sort", "-s", "-g", "-t", "\t", "-k1,1", str(unsorted_path), "-o", str(sorted_path)],
        check=True,
    )
    return sorted_path

def _write_background_0501_rows(*, writer: csv.writer, path: Path, day_start: float, day_end: float) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) <= RAW_0501_INDICES["label"]:
                continue
            ts = float(row[RAW_0501_INDICES["timestamp"]])
            if ts < day_start or ts >= day_end:
                continue
            label = int(float(row[RAW_0501_INDICES["label"]]))
            if label != 0:
                continue
            writer.writerow(_normalized_0501_record(ts, row, label=0, campaign=""))

def _write_shifted_0501_rows(
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
            if len(row) <= RAW_0501_INDICES["label"]:
                continue
            ts = float(row[RAW_0501_INDICES["timestamp"]])
            if ts < segment_start or ts > segment_end:
                continue
            label = int(float(row[RAW_0501_INDICES["label"]]))
            campaign = family if label == 1 else ""
            writer.writerow(_normalized_0501_record(ts + shift, row, label=label, campaign=campaign))

def _normalized_0501_record(ts: float, row: list[str], *, label: int, campaign: str) -> list[str]:
    def value(index: int) -> str:
        if index >= len(row):
            return "0"
        raw = str(row[index]).strip()
        return "0" if raw in {"", "-"} else raw

    return [
        f"{ts:.6f}",
        value(RAW_0501_INDICES["src"]),
        value(RAW_0501_INDICES["dst"]),
        value(RAW_0501_INDICES["duration"]),
        value(RAW_0501_INDICES["orig_bytes"]),
        value(RAW_0501_INDICES["resp_bytes"]),
        value(RAW_0501_INDICES["missed_bytes"]),
        value(RAW_0501_INDICES["orig_pkts"]),
        value(RAW_0501_INDICES["orig_ip_bytes"]),
        value(RAW_0501_INDICES["resp_pkts"]),
        value(RAW_0501_INDICES["resp_ip_bytes"]),
        str(label),
        campaign,
    ]

def _scan_0501_split(sorted_path: Path) -> tuple[int, int]:
    total = 0
    split = None
    with sorted_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 13:
                continue
            if split is None and int(float(row[11])) == 1:
                split = total
            total += 1
    return int(split or total), total

def _write_scaled_0501_csv(
    *,
    sorted_path: Path,
    output_path: Path,
    keep_pre: int,
    keep_post: int,
) -> tuple[int, int, int, int]:
    kept_pre = 0
    kept_post = 0
    row_count = 0
    positive_rows = 0
    nodes: set[str] = set()
    campaigns: set[str] = set()
    in_post = False

    with sorted_path.open("r", encoding="utf-8", newline="") as handle, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as out_handle:
        reader = csv.reader(handle, delimiter="\t")
        writer = csv.writer(out_handle)
        for row in reader:
            if len(row) < 13:
                continue
            label = int(float(row[11]))
            if label == 1:
                in_post = True
            if not in_post:
                if kept_pre >= keep_pre:
                    continue
                kept_pre += 1
            else:
                if kept_post >= keep_post:
                    continue
                kept_post += 1

            writer.writerow(row[1:])
            row_count += 1
            if label == 1:
                positive_rows += 1
                if row[12]:
                    campaigns.add(row[12])
            nodes.add(row[1])
            nodes.add(row[2])

    return row_count, positive_rows, len(nodes), len(campaigns)

def _scaled_count(total: int, scale_percent: int) -> int:
    if total <= 0:
        return 0
    return max(1, int(math.ceil(total * float(scale_percent) / 100.0)))

def _dump_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

if __name__ == "__main__":
    main()
