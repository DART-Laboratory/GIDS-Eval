from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import shutil
from pathlib import Path

GRAPH_LABEL_INDEX = {
    "lanl": 9,
    "optc": 8,
    "cic": 6,
    "0501": 3,
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scale EG1 detector-ready inputs for EG5")
    parser.add_argument("--kind", choices=["graph", "pikachu"], required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scale-percent", required=True, type=int)
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--source-metadata", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-experiment-id", default="")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.kind == "graph":
        if not args.source_root:
            raise SystemExit("--source-root is required for graph scaling")
        scale_graph_input(
            detector=args.detector,
            dataset=args.dataset,
            scale_percent=int(args.scale_percent),
            source_root=Path(args.source_root),
            output_root=output_root,
            source_experiment_id=str(args.source_experiment_id),
        )
        return

    if not args.source_metadata:
        raise SystemExit("--source-metadata is required for pikachu scaling")
    scale_pikachu_input(
        dataset=args.dataset,
        scale_percent=int(args.scale_percent),
        source_metadata_path=Path(args.source_metadata),
        output_root=output_root,
        source_experiment_id=str(args.source_experiment_id),
    )

def scale_graph_input(
    *,
    detector: str,
    dataset: str,
    scale_percent: int,
    source_root: Path,
    output_root: Path,
    source_experiment_id: str,
) -> None:
    numeric_files = _numeric_txt_files(source_root)
    if not numeric_files:
        raise SystemExit(f"No numeric shard files found under {source_root}")

    label_index = _infer_graph_label_index(numeric_files, fallback=GRAPH_LABEL_INDEX[dataset])
    file_delta = _infer_file_delta(numeric_files)
    first_positive_file, first_positive_offset = _first_positive_shard(
        numeric_files,
        label_index=label_index,
    )
    if first_positive_file is None:
        raise SystemExit(f"No positive shard found for {source_root}")

    pre_files = [item for item in numeric_files if item[0] < first_positive_file]
    post_files = [item for item in numeric_files if item[0] >= first_positive_file]
    keep_pre = _scaled_count(len(pre_files), scale_percent)
    keep_post = _scaled_count(len(post_files), scale_percent)
    selected = [*pre_files[:keep_pre], *post_files[:keep_post]]
    if not selected:
        raise SystemExit("EG5 scaling selected zero graph shards")

    _copy_graph_auxiliary(source_root, output_root)

    row_count = 0
    positive_rows = 0
    active_nodes: set[str] = set()
    first_positive_ms: int | None = None
    first_positive_row_index: int | None = None
    last_ms = 0

    for target_index, (source_base, source_path) in enumerate(selected):
        target_base = target_index * file_delta
        target_path = output_root / f"{target_base}.txt"
        with source_path.open("r", encoding="utf-8", errors="replace") as src, target_path.open(
            "w", encoding="utf-8"
        ) as dst:
            for raw_line in src:
                stripped = raw_line.rstrip("\n")
                if not stripped:
                    continue
                parts = stripped.split(",")
                if len(parts) <= label_index:
                    continue
                try:
                    old_ts = int(float(parts[0]))
                except ValueError:
                    continue
                offset = old_ts - source_base
                new_ts = target_base + offset
                parts[0] = str(new_ts)
                label = _parse_label(parts, label_index)
                if len(parts) > 2:
                    active_nodes.add(parts[1])
                    active_nodes.add(parts[2])
                if label == 1:
                    positive_rows += 1
                    if first_positive_ms is None:
                        first_positive_ms = new_ts
                        first_positive_row_index = row_count
                row_count += 1
                last_ms = new_ts
                dst.write(",".join(parts))
                dst.write("\n")

    if first_positive_ms is None:
        first_positive_ms = keep_pre * file_delta + first_positive_offset
    if first_positive_row_index is None:
        first_positive_row_index = 0

    metadata = {
        "kind": "graph",
        "detector": detector,
        "dataset": dataset,
        "scale_percent": scale_percent,
        "source_root": str(source_root),
        "source_experiment_id": source_experiment_id,
        "label_index": label_index,
        "file_delta_ms": file_delta,
        "source_file_count": len(numeric_files),
        "file_count": len(selected),
        "source_pre_file_count": len(pre_files),
        "source_post_file_count": len(post_files),
        "kept_pre_file_count": keep_pre,
        "kept_post_file_count": keep_post,
        "row_count": row_count,
        "positive_rows": positive_rows,
        "node_count": len(active_nodes),
        "first_positive_ms": int(first_positive_ms),
        "last_ms": int(last_ms),
    }
    _dump_json(output_root / "meta.json", metadata)
    _dump_json(output_root / "eg5_scale_metadata.json", metadata)

    if dataset == "0501":
        _write_0501_graph_arrays(
            output_root=output_root,
            node_count=len(active_nodes),
            first_positive_ms=int(first_positive_ms),
            first_positive_row_index=int(first_positive_row_index),
            last_ms=int(last_ms),
            source_root=source_root,
            row_count=row_count,
        )

def scale_pikachu_input(
    *,
    dataset: str,
    scale_percent: int,
    source_metadata_path: Path,
    output_root: Path,
    source_experiment_id: str,
) -> None:
    with source_metadata_path.open("r", encoding="utf-8") as handle:
        source_metadata = json.load(handle)
    input_path = Path(str(source_metadata["input_path"]))
    rows = list(csv.DictReader(input_path.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise SystemExit(f"PIKACHU input is empty: {input_path}")

    first_attack_snapshot = int(source_metadata.get("trainwin", source_metadata["first_attack_snapshot"]))
    pre_snapshots = sorted({int(row["snapshot"]) for row in rows if int(row["snapshot"]) < first_attack_snapshot})
    post_snapshots = sorted({int(row["snapshot"]) for row in rows if int(row["snapshot"]) >= first_attack_snapshot})
    keep_pre = _scaled_count(len(pre_snapshots), scale_percent)
    keep_post = _scaled_count(len(post_snapshots), scale_percent)
    selected_pre = pre_snapshots[:keep_pre]
    selected_post = post_snapshots[:keep_post]
    mapping = {snapshot: index for index, snapshot in enumerate([*selected_pre, *selected_post])}

    output_csv = output_root / "input.csv"
    fieldnames = list(rows[0].keys())
    row_count = 0
    positive_rows = 0
    node_count: set[str] = set()
    campaigns: set[str] = set()
    first_attack_ts: int | None = None

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            snapshot = int(row["snapshot"])
            if snapshot not in mapping:
                continue
            updated = dict(row)
            updated["snapshot"] = str(mapping[snapshot])
            writer.writerow(updated)
            row_count += 1
            label = int(float(str(row["label"]).strip() or "0"))
            if label == 1:
                positive_rows += 1
                campaign = str(row.get("campaign", "")).strip()
                if campaign:
                    campaigns.add(campaign)
                if first_attack_ts is None:
                    first_attack_ts = int(float(str(row["timestamp"]).strip() or "0"))
            node_count.add(str(row["src_computer"]))
            node_count.add(str(row["dst_computer"]))

    scaled_metadata = dict(source_metadata)
    scaled_metadata.update(
        {
            "input_path": str(output_csv),
            "scale_percent": scale_percent,
            "source_experiment_id": source_experiment_id,
            "row_count": row_count,
            "positive_rows": positive_rows,
            "snapshot_count": len(mapping),
            "node_count": len(node_count),
            "first_attack_snapshot": len(selected_pre),
            "trainwin": len(selected_pre),
            "first_attack_timestamp": first_attack_ts or source_metadata.get("first_attack_timestamp", 0),
            "source_snapshot_count": int(source_metadata.get("snapshot_count", 0)),
            "kept_pre_snapshot_count": keep_pre,
            "kept_post_snapshot_count": keep_post,
            "positive_campaign_count": len(campaigns),
        }
    )
    _dump_json(output_root / "metadata.json", scaled_metadata)
    _dump_json(
        output_root / "eg5_scale_metadata.json",
        {
            "kind": "pikachu",
            "dataset": dataset,
            "scale_percent": scale_percent,
            "source_experiment_id": source_experiment_id,
            "row_count": row_count,
            "positive_rows": positive_rows,
            "snapshot_count": len(mapping),
            "node_count": len(node_count),
            "source_snapshot_count": int(source_metadata.get("snapshot_count", 0)),
            "source_pre_snapshot_count": len(pre_snapshots),
            "source_post_snapshot_count": len(post_snapshots),
            "kept_pre_snapshot_count": keep_pre,
            "kept_post_snapshot_count": keep_post,
            "first_attack_snapshot": len(selected_pre),
        },
    )

def _numeric_txt_files(root: Path) -> list[tuple[int, Path]]:
    result: list[tuple[int, Path]] = []
    for path in sorted(root.glob("*.txt")):
        stem = path.stem
        if stem.isdigit():
            result.append((int(stem), path))
    return result

def _infer_file_delta(files: list[tuple[int, Path]]) -> int:
    if len(files) < 2:
        return 10000
    deltas = [right - left for (left, _), (right, _) in zip(files, files[1:]) if right > left]
    if not deltas:
        return 10000
    return min(deltas)

def _parse_label(parts: list[str], label_index: int) -> int:
    try:
        return int(float(parts[label_index].strip() or "0"))
    except Exception:
        return 0

def _infer_graph_label_index(files: list[tuple[int, Path]], *, fallback: int) -> int:
    for _, path in files:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                stripped = raw_line.rstrip("\n")
                if not stripped:
                    continue
                parts = stripped.split(",")

                effective_len = len(parts)
                while effective_len > 0 and parts[effective_len - 1] == "":
                    effective_len -= 1
                if effective_len <= 0:
                    continue

                if effective_len >= 4:
                    return effective_len - 1
                return fallback
    return fallback

def _first_positive_shard(files: list[tuple[int, Path]], *, label_index: int) -> tuple[int | None, int]:
    for base, path in files:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                stripped = raw_line.rstrip("\n")
                if not stripped:
                    continue
                parts = stripped.split(",")
                if len(parts) <= label_index:
                    continue
                if _parse_label(parts, label_index) == 1:
                    try:
                        ts = int(float(parts[0]))
                    except ValueError:
                        ts = base
                    return base, max(0, ts - base)
    return None, 0

def _scaled_count(total: int, scale_percent: int) -> int:
    if total <= 0:
        return 0
    return max(1, int(math.ceil(total * float(scale_percent) / 100.0)))

def _copy_graph_auxiliary(source_root: Path, output_root: Path) -> None:
    skip_names = {
        "meta.json",
        "eg5_scale_metadata.json",
        "time.npy",
        "data_info.npy",
        "eg1_0501_metadata.json",
    }
    for entry in source_root.iterdir():
        if entry.name in skip_names:
            continue
        if entry.is_file() and entry.suffix == ".txt" and entry.stem.isdigit():
            continue
        if entry.is_dir() and entry.name == "_merge_work":
            continue
        target = output_root / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, target)

def _write_0501_graph_arrays(
    *,
    output_root: Path,
    node_count: int,
    first_positive_ms: int,
    first_positive_row_index: int,
    last_ms: int,
    source_root: Path,
    row_count: int,
) -> None:
    import numpy as np

    np.save(output_root / "time.npy", np.array((first_positive_ms, last_ms), dtype=np.int32))
    np.save(output_root / "data_info.npy", np.array((node_count, first_positive_row_index), dtype=np.int32))

    source_metadata_path = source_root / "eg1_0501_metadata.json"
    payload = {}
    if source_metadata_path.exists():
        with source_metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    payload.update(
        {
            "output_root": str(output_root),
            "row_count": int(row_count),
            "node_count": int(node_count),
            "first_positive_ms": int(first_positive_ms),
            "last_ms": int(last_ms),
        }
    )
    _dump_json(output_root / "eg1_0501_metadata.json", payload)

def _dump_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

if __name__ == "__main__":
    main()
