from __future__ import annotations

import argparse
import csv
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from EG1.merge_0501 import (
    DEFAULT_ATTACK_INSERTION_SECONDS,
    DEFAULT_BACKGROUND_FAMILY,
    DEFAULT_DAY_SECONDS,
    FAMILY_FILES,
    family_log_path,
    scan_file_stats,
)

FILE_DELTA_MS = 100000

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified 0501 ARGUS data with campaign + flow features for EG4.")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--work-root", default=None)
    parser.add_argument("--background-family", default=DEFAULT_BACKGROUND_FAMILY)
    parser.add_argument("--oilrig-start", type=int, default=DEFAULT_ATTACK_INSERTION_SECONDS["oilrig"])
    parser.add_argument("--sandworm-start", type=int, default=DEFAULT_ATTACK_INSERTION_SECONDS["sandworm"])
    parser.add_argument("--wizard-start", type=int, default=DEFAULT_ATTACK_INSERTION_SECONDS["wizard_spider"])
    parser.add_argument("--day-seconds", type=int, default=DEFAULT_DAY_SECONDS)
    return parser.parse_args()

def _safe_float(tokens: list[str], index: int, default: float = 0.0) -> float:
    try:
        return float(tokens[index])
    except Exception:
        return default

def _safe_int(tokens: list[str], index: int, default: int = 0) -> int:
    try:
        return int(float(tokens[index]))
    except Exception:
        return default

def _parse_raw_0501_row(row: list[str]) -> tuple[float, str, str, float, int, int, int] | None:
    if len(row) < 22:
        return None
    timestamp = _safe_float(row, 0, default=-1.0)
    if timestamp < 0:
        return None
    src = str(row[2]).strip()
    dst = str(row[4]).strip()
    if not src or not dst:
        return None
    duration = _safe_float(row, 8)
    total_pkts = _safe_int(row, 16) + _safe_int(row, 18)
    total_bytes = _safe_int(row, 17) + _safe_int(row, 19)
    label = _safe_int(row, 21)
    return (timestamp, src, dst, duration, total_bytes, total_pkts, label)

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
            parsed = _parse_raw_0501_row(row)
            if parsed is None:
                continue
            ts, src, dst, duration, total_bytes, total_pkts, label = parsed
            if ts < day_start or ts >= day_end or label != 0:
                continue
            writer.writerow([f"{ts:.6f}", src, dst, f"{duration:.6f}", total_bytes, total_pkts, 0, ""])

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
            parsed = _parse_raw_0501_row(row)
            if parsed is None:
                continue
            ts, src, dst, duration, total_bytes, total_pkts, label = parsed
            if ts < segment_start or ts > segment_end:
                continue
            campaign = family if label == 1 else ""
            writer.writerow(
                [f"{ts + shift:.6f}", src, dst, f"{duration:.6f}", total_bytes, total_pkts, label, campaign]
            )

def _sort_events(input_path: Path, output_path: Path) -> None:
    subprocess.run(
        [
            "sort",
            "-s",
            "-g",
            "-t",
            "\t",
            "-k1,1",
            str(input_path),
            "-o",
            str(output_path),
        ],
        check=True,
    )

def _iter_sorted_rows(sorted_path: Path):
    with sorted_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 8:
                continue
            yield (
                float(row[0]),
                str(row[1]),
                str(row[2]),
                float(row[3]),
                int(float(row[4])),
                int(float(row[5])),
                int(float(row[6])),
                str(row[7]),
            )

def _build_sorted_combined_events(
    *,
    raw_root: Path,
    work_root: Path,
    background_family: str,
    day_seconds: int,
    insertion_starts: dict[str, int],
) -> dict[str, object]:
    stats = {family: scan_file_stats(family_log_path(raw_root, family)) for family in FAMILY_FILES}
    base_stats = stats[background_family]
    day_start = base_stats.min_ts
    day_end = day_start + float(day_seconds)

    work_root.mkdir(parents=True, exist_ok=True)
    unsorted_path = work_root / "combined_0501_argus.unsorted.tsv"
    sorted_path = work_root / "combined_0501_argus.sorted.tsv"

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

    _sort_events(unsorted_path, sorted_path)
    return {
        "sorted_path": sorted_path,
        "day_start": day_start,
        "day_end": day_end,
        "background_family": background_family,
        "day_seconds": day_seconds,
        "insertion_starts": insertion_starts,
        "stats": stats,
    }

def build_unified_argus_dataset(
    *,
    raw_root: Path,
    output_root: Path,
    work_root: Path,
    background_family: str,
    day_seconds: int,
    insertion_starts: dict[str, int],
) -> None:
    metadata = _build_sorted_combined_events(
        raw_root=raw_root,
        work_root=work_root,
        background_family=background_family,
        day_seconds=day_seconds,
        insertion_starts=insertion_starts,
    )

    output_root.mkdir(parents=True, exist_ok=True)
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
        for ts, src, dst, duration, total_bytes, total_pkts, label, campaign in _iter_sorted_rows(
            Path(str(metadata["sorted_path"]))
        ):
            if first_ts is None:
                first_ts = ts
            rel_ms = int((ts - first_ts) * 1000.0)
            while rel_ms >= current_floor + FILE_DELTA_MS:
                current_file.close()
                current_floor += FILE_DELTA_MS
                current_file = (output_root / f"{current_floor}.txt").open("w", encoding="utf-8")
            src_id = get_or_add(src)
            dst_id = get_or_add(dst)
            if label == 1 and first_positive_ms is None:
                first_positive_ms = rel_ms
                first_positive_index = row_index
            current_file.write(
                f"{rel_ms},{src_id},{dst_id},{duration:.6f},{int(total_bytes)},{int(total_pkts)},{label},{campaign}\n"
            )
            row_index += 1
            last_ms = rel_ms
    finally:
        current_file.close()

    if first_ts is None:
        raise SystemExit("Unified 0501 ARGUS build produced no rows")

    with (output_root / "nmap.pkl").open("wb") as handle:
        pickle.dump(reverse_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

    np.save(output_root / "time.npy", np.array((first_positive_ms or 0, last_ms), dtype=np.int32))
    np.save(output_root / "data_info.npy", np.array((len(reverse_map), first_positive_index or 0), dtype=np.int32))

    payload = {
        "format": "argus_eg4_unified_0501",
        "dataset": "0501",
        "campaign_mode": "family",
        "campaigns": ["oilrig", "sandworm", "wizard_spider"],
        "output_root": str(output_root),
        "row_count": int(row_index),
        "node_count": int(len(reverse_map)),
        "first_positive_ms": int(first_positive_ms or 0),
        "last_ms": int(last_ms),
        "background_family": str(metadata["background_family"]),
        "day_seconds": int(metadata["day_seconds"]),
        "insertion_starts": metadata["insertion_starts"],
    }
    with (output_root / "eg4_0501_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    work_root = Path(args.work_root) if args.work_root else output_root / "_merge_work"
    insertion_starts = {
        "oilrig": int(args.oilrig_start),
        "sandworm": int(args.sandworm_start),
        "wizard_spider": int(args.wizard_start),
    }
    build_unified_argus_dataset(
        raw_root=raw_root,
        output_root=output_root,
        work_root=work_root,
        background_family=str(args.background_family),
        day_seconds=int(args.day_seconds),
        insertion_starts=insertion_starts,
    )

if __name__ == "__main__":
    main()
