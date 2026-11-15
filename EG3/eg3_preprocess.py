
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

LANL_CAMPAIGN_WINDOW_SECONDS = 30 * 60
OPTC_CAMPAIGN_WINDOW_SECONDS = 5 * 60

def _to_int(value: Any) -> int | None:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None

def assign_temporal_entity_campaigns(
    events: list[dict[str, Any]],
    *,
    dataset: str,
    entity_keys: tuple[str, ...],
    time_window_seconds: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    if not events:
        return [], []

    prepared: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        timestamp = _to_int(event.get("timestamp", event.get("ts")))
        if timestamp is None:
            continue
        normalized = dict(event)
        normalized["_orig_index"] = index
        normalized["timestamp"] = timestamp
        prepared.append(normalized)

    if not prepared:
        return [""] * len(events), []

    prepared.sort(key=lambda event: (int(event["timestamp"]), int(event["_orig_index"])))
    parent = list(range(len(prepared)))
    rank = [0] * len(prepared)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            parent[left_root] = right_root
        elif rank[left_root] > rank[right_root]:
            parent[right_root] = left_root
        else:
            parent[right_root] = left_root
            rank[left_root] += 1

    recent_by_entity = {
        entity_key: defaultdict(deque)
        for entity_key in entity_keys
    }

    for event_index, event in enumerate(prepared):
        timestamp = int(event["timestamp"])
        for entity_key in entity_keys:
            entity_value = str(event.get(entity_key, "")).strip()
            if entity_value == "":
                continue
            recent = recent_by_entity[entity_key][entity_value]
            while recent and timestamp - int(prepared[recent[0]]["timestamp"]) > time_window_seconds:
                recent.popleft()
            for previous_index in recent:
                union(event_index, previous_index)
            recent.append(event_index)

    groups: dict[int, list[int]] = defaultdict(list)
    for event_index in range(len(prepared)):
        groups[find(event_index)].append(event_index)

    ordered_roots = sorted(
        groups,
        key=lambda root: (
            min(int(prepared[event_index]["timestamp"]) for event_index in groups[root]),
            -len(groups[root]),
        ),
    )

    labels = [""] * len(events)
    stats: list[dict[str, Any]] = []
    for ordinal, root in enumerate(ordered_roots, start=1):
        group_indices = groups[root]
        campaign = f"{dataset}_campaign_{ordinal:03d}"
        timestamps = [int(prepared[event_index]["timestamp"]) for event_index in group_indices]
        entity_summary = {
            entity_key: sorted(
                {
                    str(prepared[event_index].get(entity_key, "")).strip()
                    for event_index in group_indices
                    if str(prepared[event_index].get(entity_key, "")).strip()
                }
            )
            for entity_key in entity_keys
        }
        for event_index in group_indices:
            labels[int(prepared[event_index]["_orig_index"])] = campaign
        stats.append(
            {
                "campaign": campaign,
                "event_count": len(group_indices),
                "start_time": min(timestamps),
                "end_time": max(timestamps),
                "duration_seconds": max(timestamps) - min(timestamps),
                "entities": entity_summary,
            }
        )
    return labels, stats

def build_lanl_campaign_lookup(
    redteam_path: str,
    *,
    time_window_seconds: int = LANL_CAMPAIGN_WINDOW_SECONDS,
) -> tuple[dict[tuple[int, str, str, str], str], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    with open(redteam_path, "r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 4:
                continue
            timestamp = _to_int(row[0])
            if timestamp is None:
                continue
            src_user = str(row[1]).strip()
            src_computer = str(row[2]).strip()
            dst_computer = str(row[3]).strip()
            if src_user == "" or src_computer == "" or dst_computer == "":
                continue
            events.append(
                {
                    "timestamp": timestamp,
                    "src_user": src_user,
                    "src_computer": src_computer,
                    "dst_computer": dst_computer,
                }
            )
    campaigns, stats = assign_temporal_entity_campaigns(
        events,
        dataset="lanl",
        entity_keys=("src_computer",),
        time_window_seconds=time_window_seconds,
    )
    lookup = {
        (
            int(event["timestamp"]),
            str(event["src_user"]),
            str(event["src_computer"]),
            str(event["dst_computer"]),
        ): campaign
        for event, campaign in zip(events, campaigns)
    }
    return lookup, stats

def build_optc_campaign_lookup(
    events: list[dict[str, Any]],
    *,
    time_window_seconds: int = OPTC_CAMPAIGN_WINDOW_SECONDS,
) -> tuple[dict[tuple[int, str, str], str], list[dict[str, Any]]]:
    normalized: list[dict[str, Any]] = []
    for event in events:
        timestamp = _to_int(event.get("timestamp", event.get("ts")))
        src = str(event.get("src", event.get("src_computer", ""))).strip()
        dst = str(event.get("dst", event.get("dst_computer", ""))).strip()
        if timestamp is None or src == "" or dst == "":
            continue
        normalized.append(
            {
                "timestamp": timestamp,
                "src_computer": src,
                "dst_computer": dst,
            }
        )
    campaigns, stats = assign_temporal_entity_campaigns(
        normalized,
        dataset="optc",
        entity_keys=("src_computer",),
        time_window_seconds=time_window_seconds,
    )
    lookup = {
        (
            int(event["timestamp"]),
            str(event["src_computer"]),
            str(event["dst_computer"]),
        ): campaign
        for event, campaign in zip(normalized, campaigns)
    }
    return lookup, stats

def _read_optc_events(path: str) -> list[dict[str, Any]]:
    events = []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            ts = _to_int(row[0])
            src = str(row[1]).strip()
            dst = str(row[2]).strip()
            if ts is None or src == "" or dst == "":
                continue
            events.append({"ts": ts, "src": src, "dst": dst, "row": row})
    return events

def _optc_components(events: list[dict[str, Any]]) -> list[set[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    nodes: set[str] = set()
    for event in events:
        src = event["src"]
        dst = event["dst"]
        nodes.add(src)
        nodes.add(dst)
        adj[src].add(dst)
        adj[dst].add(src)
    comps = []
    seen: set[str] = set()
    for start in sorted(nodes):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: set[str] = set()
        while stack:
            node = stack.pop()
            comp.add(node)
            for nxt in adj.get(node, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(comp)
    comps.sort(key=lambda c: (-len(c), sorted(c)))
    return comps

def _component_edge_count(events: list[dict[str, Any]], node_set: set[str]) -> int:
    edges = {(e["src"], e["dst"]) for e in events if e["src"] in node_set and e["dst"] in node_set}
    return len(edges)

def _select_component_nodes(events: list[dict[str, Any]], target_nodes: int = 8) -> set[str]:
    comps = _optc_components(events)
    candidates = [comp for comp in comps if len(comp) == target_nodes]
    if not candidates:
        stats = [{"nodes": len(comp), "edges": _component_edge_count(events, comp)} for comp in comps]
        raise ValueError(f"no component with {target_nodes} nodes, components={stats}")
    candidates.sort(key=lambda c: (-_component_edge_count(events, c), sorted(c)))
    return set(candidates[0])

def _filter_events_by_nodes(events: list[dict[str, Any]], nodes: set[str]) -> list[dict[str, Any]]:
    return [event for event in events if event["src"] in nodes and event["dst"] in nodes]

def _write_optc_events(path: str, events: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for event in events:
            writer.writerow(event["row"])

def load_small_optc_component(redteam_path: str, target_nodes: int = 8) -> dict[str, Any]:
    all_events = _read_optc_events(redteam_path)
    nodes = _select_component_nodes(all_events, target_nodes=target_nodes)
    selected = _filter_events_by_nodes(all_events, nodes)
    unique_edges = {(event["src"], event["dst"]) for event in selected}
    return {
        "all_events": all_events,
        "selected_nodes": sorted(nodes),
        "selected_events": selected,
        "selected_unique_edges": sorted(unique_edges),
    }

def assign_snapshots(frame: pd.DataFrame, snapshot_seconds: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    initial_time = int(result["timestamp"].min())
    result["snapshot"] = ((result["timestamp"] - initial_time) // snapshot_seconds).astype(int)
    return result

def densify_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    ordered = sorted(int(value) for value in result["snapshot"].drop_duplicates().tolist())
    mapping = {snapshot: index for index, snapshot in enumerate(ordered)}
    result["snapshot"] = result["snapshot"].map(mapping).astype(int)
    return result

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

def compute_trainwin(frame: pd.DataFrame) -> int:
    return int(frame.loc[frame["label"] == 1, "snapshot"].min())

def campaign_mode(dataset: str) -> str:
    if dataset == "cic":
        return "raw_label"
    if dataset in {"lanl", "optc"}:
        return "entity_time_component"
    return "src_dst_edge"

def build_campaign_column(frame: pd.DataFrame, dataset: str) -> pd.Series:
    if dataset == "cic" and "raw_label" in frame.columns:
        labels = frame["raw_label"].fillna("").astype(str)
        return labels.where(frame["label"] == 1, "")
    return frame["src_computer"].astype(str) + "__" + frame["dst_computer"].astype(str)

def preprocess_cic(raw_root: Path, snapshot_seconds: int, neg_ratio: int, seed: int) -> pd.DataFrame:
    in_dir = raw_root / "CIC-new"
    frame = []
    for day in ("monday", "tuesday", "wednesday", "thursday", "friday"):
        part = pd.read_csv(in_dir / f"{day}.csv")
        part["time_int"] = [int(datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M:%S.%f").timestamp()) for ts in part["Timestamp"]]
        frame.append(part.sort_values(by="time_int"))
    result = pd.concat(frame).sort_values(by="time_int").reset_index(drop=True)
    result["time_int"] = result["time_int"] - int(result["time_int"].iloc[0])
    result["raw_label"] = result["Label"].fillna("").astype(str)
    result["label"] = [0 if item == "BENIGN" else 1 for item in result["raw_label"].values]
    result["campaign"] = result["raw_label"].where(result["label"] == 1, "")
    result = assign_snapshots(result.rename(columns={"time_int": "timestamp"}), snapshot_seconds)
    result = downsample_negatives_preserve_snapshots(result, neg_ratio=neg_ratio, seed=seed)
    result = result[["timestamp", "Src IP", "Dst IP", "label", "campaign", "raw_label", "snapshot"]]
    result.columns = ["timestamp", "src_computer", "dst_computer", "label", "campaign", "raw_label", "snapshot"]
    return result

def preprocess_lanl(raw_root: Path, snapshot_seconds: int, neg_ratio: int, seed: int) -> pd.DataFrame:
    src = raw_root / "LANL" / "auth.txt"
    red = raw_root / "LANL" / "redteam.txt"
    campaign_lookup, _ = build_lanl_campaign_lookup(str(red))
    rt_df = pd.read_csv(
        red,
        header=None,
        names=["timestamp", "src_user", "src_computer", "dst_computer"],
        dtype={"timestamp": "int32", "src_user": str, "src_computer": str, "dst_computer": str},
    )
    attack_keys = (
        rt_df["timestamp"].astype(str)
        + "|"
        + rt_df["src_user"].astype(str)
        + "|"
        + rt_df["src_computer"].astype(str)
        + "|"
        + rt_df["dst_computer"].astype(str)
    )
    attack_key_set = set(attack_keys.tolist())
    anom_users = sorted(set(rt_df["src_user"].astype(str).tolist()))
    anom_user_set = set(anom_users)

    def iter_filtered_chunks() -> Any:
        for chunk_df in pd.read_csv(
            src,
            sep=",",
            header=None,
            names=[
                "timestamp",
                "src_user",
                "dst_user",
                "src_computer",
                "dst_computer",
                "auth_type",
                "logon_type",
                "auth_ori",
                "success",
            ],
            dtype={
                "timestamp": "int32",
                "src_user": str,
                "dst_user": str,
                "src_computer": str,
                "dst_computer": str,
                "auth_type": "category",
                "logon_type": "category",
            },
            chunksize=2_000_000,
        ):
            chunk_df = chunk_df[~((chunk_df["auth_type"] == "?") | (chunk_df["logon_type"] == "?"))]
            chunk_df = chunk_df[
                ~(
                    (chunk_df["src_user"].str.contains(r"ANONYMOUS(?!$)"))
                    | (chunk_df["src_user"].str.contains(r"LOCAL(?!$)"))
                    | (chunk_df["src_user"].str.contains(r"NETWORK(?!$)"))
                    | (chunk_df["src_user"].str.contains(r"ADMIN(?!$)"))
                )
            ]
            chunk_df = chunk_df[chunk_df["src_computer"] != chunk_df["dst_computer"]]
            yield chunk_df[["timestamp", "src_user", "src_computer", "dst_computer"]].reset_index(drop=True)

    def row_keys(frame: pd.DataFrame) -> pd.Series:
        return (
            frame["timestamp"].astype(str)
            + "|"
            + frame["src_user"].astype(str)
            + "|"
            + frame["src_computer"].astype(str)
            + "|"
            + frame["dst_computer"].astype(str)
        )

    initial_time: int | None = None
    normal_first_seen: dict[str, int] = {}
    snapshot_anchor_users: dict[int, str] = {}

    for chunk_df in iter_filtered_chunks():
        if chunk_df.empty:
            continue
        if initial_time is None:
            initial_time = int(chunk_df["timestamp"].iloc[0])

        labels = row_keys(chunk_df).isin(attack_key_set)
        chunk_df = chunk_df.copy()
        chunk_df["label"] = labels.astype(int)
        chunk_df["snapshot"] = ((chunk_df["timestamp"] - initial_time) // snapshot_seconds).astype(int)

        benign = chunk_df[chunk_df["label"] == 0]
        if benign.empty:
            continue

        anchor_rows = (
            benign.sort_values(by=["snapshot", "timestamp", "src_user"])
            .groupby("snapshot", sort=True)
            .head(1)
        )
        for snapshot, src_user in zip(
            anchor_rows["snapshot"].tolist(),
            anchor_rows["src_user"].astype(str).tolist(),
        ):
            snapshot_anchor_users.setdefault(int(snapshot), src_user)

        normal = benign[~benign["src_user"].isin(anom_user_set)][["src_user", "timestamp"]]
        if normal.empty:
            continue
        normal = normal.drop_duplicates(subset=["src_user"]).sort_values(by=["timestamp", "src_user"])
        for src_user, timestamp in zip(normal["src_user"].astype(str).tolist(), normal["timestamp"].tolist()):
            normal_first_seen.setdefault(src_user, int(timestamp))

    if initial_time is None:
        return pd.DataFrame(columns=["timestamp", "src_computer", "dst_computer", "label", "campaign", "snapshot"])

    normal_user_cap = len(anom_users) * 20
    selected_norm_users = [
        user
        for user, _ in sorted(normal_first_seen.items(), key=lambda item: (item[1], item[0]))[:normal_user_cap]
    ]
    keep_users = set(anom_users)
    keep_users.update(selected_norm_users)
    keep_users.update(snapshot_anchor_users.values())

    selected_chunks = []
    for chunk_df in iter_filtered_chunks():
        if chunk_df.empty:
            continue
        chunk_df = chunk_df[chunk_df["src_user"].isin(keep_users)].reset_index(drop=True)
        if chunk_df.empty:
            continue

        labels = row_keys(chunk_df).isin(attack_key_set)
        chunk_df["label"] = labels.astype(int)
        chunk_df["campaign"] = [
            campaign_lookup.get((int(timestamp), str(src_user), str(src_computer), str(dst_computer)), "")
            if int(label) == 1
            else ""
            for timestamp, src_user, src_computer, dst_computer, label in zip(
                chunk_df["timestamp"].tolist(),
                chunk_df["src_user"].tolist(),
                chunk_df["src_computer"].tolist(),
                chunk_df["dst_computer"].tolist(),
                chunk_df["label"].tolist(),
            )
        ]
        chunk_df["snapshot"] = ((chunk_df["timestamp"] - initial_time) // snapshot_seconds).astype(int)
        selected_chunks.append(
            chunk_df[["timestamp", "src_user", "src_computer", "dst_computer", "label", "campaign", "snapshot"]]
        )

    if not selected_chunks:
        return pd.DataFrame(columns=["timestamp", "src_computer", "dst_computer", "label", "campaign", "snapshot"])

    auth_df = pd.concat(selected_chunks, ignore_index=True)
    auth_df = downsample_negatives_preserve_snapshots(auth_df, neg_ratio=neg_ratio, seed=seed)

    return auth_df[["timestamp", "src_computer", "dst_computer", "label", "campaign", "snapshot"]]

def preprocess_optc(
    raw_root: Path,
    snapshot_seconds: int,
    neg_ratio: int,
    seed: int,
    output_dir: Path,
) -> pd.DataFrame:
    src = raw_root / "OpTC" / "auth_optc.txt"
    redteam_path = raw_root / "OpTC" / "redteam_optc.txt"
    sel = load_small_optc_component(str(redteam_path), target_nodes=8)
    selected_events = sel["selected_events"]
    optc_campaign_lookup, _ = build_optc_campaign_lookup(selected_events)
    attack_keys = {f"{int(e['ts'])}|{e['src']}|{e['dst']}" for e in selected_events}

    frame = pd.read_csv(
        src,
        header=None,
        names=["timestamp", "src_computer", "dst_computer", "label", "1", "2", "3", "4", "5"],
    )
    frame = frame.drop(["1", "2", "3", "4", "5"], axis=1).reset_index(drop=True)
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce").fillna(0).astype(int)
    frame["src_computer"] = frame["src_computer"].astype(str)
    frame["dst_computer"] = frame["dst_computer"].astype(str)
    row_keys = frame["timestamp"].astype(str) + "|" + frame["src_computer"] + "|" + frame["dst_computer"]
    frame["label"] = row_keys.isin(attack_keys).astype(int)
    frame["campaign"] = [
        optc_campaign_lookup.get((int(timestamp), str(src_c), str(dst_c)), "") if int(label) == 1 else ""
        for timestamp, src_c, dst_c, label in zip(
            frame["timestamp"].tolist(),
            frame["src_computer"].tolist(),
            frame["dst_computer"].tolist(),
            frame["label"].tolist(),
        )
    ]

    frame = assign_snapshots(frame, snapshot_seconds)
    frame = downsample_negatives_preserve_snapshots(frame, neg_ratio=neg_ratio, seed=seed)
    frame = frame[["timestamp", "src_computer", "dst_computer", "label", "campaign", "snapshot"]]

    _write_optc_events(str(output_dir / "redteam_optc_small.txt"), selected_events)
    return frame

def preprocess_0501_single(raw_root: Path, dataset: str, snapshot_seconds: int, neg_ratio: int, seed: int) -> pd.DataFrame:
    dataset_to_file = {
        "oilrig": "label_oilrig.log",
        "sandworm": "label_sandworm.log",
        "wizard_spider": "label_wizard_spider.log",
    }
    input_file = raw_root / "0501" / dataset_to_file[dataset]
    rows: list[dict[str, Any]] = []
    with input_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            rows.append(
                {
                    "timestamp": int(float(parts[0])),
                    "src_computer": str(parts[2]),
                    "dst_computer": str(parts[4]),
                    "label": int(float(parts[-1])),
                }
            )
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(by="timestamp", ascending=True, ignore_index=True)
    frame = frame.iloc[frame[frame["timestamp"] < 1682870100].shape[0]:].reset_index(drop=True)
    frame = assign_snapshots(frame, snapshot_seconds)
    frame = downsample_negatives_preserve_snapshots(frame, neg_ratio=neg_ratio, seed=seed)
    return frame[["timestamp", "src_computer", "dst_computer", "label", "snapshot"]]

DATASET_CHOICES = ["lanl", "optc", "cic", "oilrig", "sandworm", "wizard_spider"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EG3 standalone PIKACHU preprocessing")
    parser.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    parser.add_argument("--raw-root", required=True, help="Path to staged raw_data root")
    parser.add_argument("--output-dir", required=True, help="Output directory for preprocessed CSV and metadata")
    parser.add_argument("--snapshot-seconds", required=True, type=int, help="Snapshot size in seconds")
    parser.add_argument("--neg-ratio", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "lanl":
        frame = preprocess_lanl(raw_root, args.snapshot_seconds, args.neg_ratio, args.seed)
    elif args.dataset == "optc":
        frame = preprocess_optc(raw_root, args.snapshot_seconds, args.neg_ratio, args.seed, output_dir)
    elif args.dataset == "cic":
        frame = preprocess_cic(raw_root, args.snapshot_seconds, args.neg_ratio, args.seed)
    else:
        frame = preprocess_0501_single(raw_root, args.dataset, args.snapshot_seconds, args.neg_ratio, args.seed)

    if frame.empty:
        raise SystemExit(f"No rows produced for dataset={args.dataset}")
    if not (frame["label"] == 1).any():
        raise SystemExit(f"No positive rows available for dataset={args.dataset}")

    if "campaign" not in frame.columns:
        frame["campaign"] = build_campaign_column(frame, args.dataset)
    frame = frame.sort_values(by=["timestamp", "snapshot", "src_computer", "dst_computer"]).reset_index(drop=True)
    original_snapshot_count = int(frame["snapshot"].nunique())
    original_first_attack_snapshot = int(frame.loc[frame["label"] == 1, "snapshot"].min())
    frame = densify_snapshots(frame)
    trainwin = compute_trainwin(frame)
    first_attack_snapshot = int(frame.loc[frame["label"] == 1, "snapshot"].min())
    initial_time = int(frame["timestamp"].min())

    output_csv = output_dir / "input.csv"
    frame.to_csv(output_csv, index=False)

    metadata = {
        "dataset": args.dataset,
        "snapshot_seconds": args.snapshot_seconds,
        "input_path": str(output_csv),
        "row_count": int(len(frame)),
        "positive_rows": int((frame["label"] == 1).sum()),
        "positive_campaign_count": int(frame.loc[frame["label"] == 1, "campaign"].nunique()),
        "campaign_mode": campaign_mode(args.dataset),
        "initial_time": initial_time,
        "first_attack_timestamp": int(frame.loc[frame["label"] == 1, "timestamp"].iloc[0]),
        "first_attack_snapshot": first_attack_snapshot,
        "original_first_attack_snapshot": original_first_attack_snapshot,
        "snapshot_count": int(frame["snapshot"].nunique()),
        "original_snapshot_count": original_snapshot_count,
        "trainwin": int(trainwin),
        "neg_ratio": int(args.neg_ratio),
        "preprocess_seed": int(args.seed),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

if __name__ == "__main__":
    main()
