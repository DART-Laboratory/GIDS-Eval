from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
    day_start = select_background_window_start(
        family_log_path(raw_root, background_family),
        day_seconds=float(day_seconds),
        preferred_pre_attack_seconds=float(min(insertion_starts.values())) if insertion_starts else 0.0,
    )
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

def select_background_window_start(
    path: Path,
    *,
    day_seconds: float,
    preferred_pre_attack_seconds: float = 0.0,
) -> float:
    benign_timestamps: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue
            ts = float(row[0])
            label = int(float(row[-1]))
            if label == 0:
                benign_timestamps.append(ts)

    if not benign_timestamps:
        raise ValueError(f"0501 background source has no benign rows: {path}")

    benign_timestamps.sort()
    if len(benign_timestamps) == 1:
        return float(benign_timestamps[0])

    best_start = float(benign_timestamps[0])
    best_total = -1
    best_pre_attack = -1
    right_total = 0
    right_pre_attack = 0
    total_rows = len(benign_timestamps)
    pre_attack_span = max(0.0, float(preferred_pre_attack_seconds))

    for left, start in enumerate(benign_timestamps):
        while right_total < total_rows and benign_timestamps[right_total] < start + float(day_seconds):
            right_total += 1
        while right_pre_attack < total_rows and benign_timestamps[right_pre_attack] < start + pre_attack_span:
            right_pre_attack += 1

        total_count = right_total - left
        pre_attack_count = total_count if pre_attack_span <= 0 else max(0, right_pre_attack - left)
        if (
            pre_attack_count > best_pre_attack
            or (pre_attack_count == best_pre_attack and total_count > best_total)
        ):
            best_start = float(start)
            best_total = total_count
            best_pre_attack = pre_attack_count

    return best_start

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

def iter_sorted_events(sorted_path: Path) -> Iterable[tuple[float, str, str, int, str]]:
    with sorted_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue
            yield float(row[0]), str(row[1]), str(row[2]), int(float(row[3])), str(row[4])
