from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from runtime_shims import load_eg1_sitecustomize

load_eg1_sitecustomize()

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
OPT_PREPROC_ROOT = REPO_ROOT / "local_runtime" / "pre_processing"
if str(OPT_PREPROC_ROOT) not in sys.path:
    sys.path.insert(0, str(OPT_PREPROC_ROOT))

from campaign_grouping import build_lanl_campaign_lookup, build_optc_campaign_lookup
from optc_small_component import load_small_optc_component, write_events

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SC1-specific PIKACHU preprocessing wrapper")
    parser.add_argument("--dataset", required=True, choices=[
        "lanl",
        "optc",
        "cic",
        "oilrig",
        "sandworm",
        "wizard_spider",
    ])
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
        frame = preprocess_0501(raw_root, args.dataset, args.snapshot_seconds, args.neg_ratio, args.seed)

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

MIN_TRAIN_EDGES = 100
FALLBACK_TRAIN_FRACTION = 0.4


def compute_trainwin(frame: pd.DataFrame) -> int:
    first_positive = int(frame.loc[frame["label"] == 1, "snapshot"].min())
    sizes = frame.groupby("snapshot").size().sort_index()
    pre_attack_edges = int(sizes.loc[sizes.index < first_positive].sum())
    if pre_attack_edges >= MIN_TRAIN_EDGES:
        return first_positive
    target = int(round(len(frame) * FALLBACK_TRAIN_FRACTION))
    cumulative = sizes.cumsum()
    boundary_snapshot = int(cumulative.index[(cumulative >= target).argmax()])
    return boundary_snapshot + 1

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

def densify_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    ordered = sorted(int(value) for value in result["snapshot"].drop_duplicates().tolist())
    mapping = {snapshot: index for index, snapshot in enumerate(ordered)}
    result["snapshot"] = result["snapshot"].map(mapping).astype(int)
    return result

def assign_snapshots(frame: pd.DataFrame, snapshot_seconds: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    initial_time = int(result["timestamp"].min())
    result["snapshot"] = ((result["timestamp"] - initial_time) // snapshot_seconds).astype(int)
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

def select_lanl_users(frame: pd.DataFrame) -> pd.DataFrame:
    anom_users = sorted(set(frame.loc[frame["label"] == 1, "src_user"].astype(str).tolist()))
    if not anom_users:
        return frame.reset_index(drop=True)

    anom_user_set = set(anom_users)
    normal_candidates = (
        frame.loc[~frame["src_user"].isin(anom_user_set), ["src_user", "timestamp"]]
        .drop_duplicates(subset=["src_user"])
        .sort_values(by=["timestamp", "src_user"])
    )
    normal_user_cap = len(anom_users) * 20
    selected_norm_users = normal_candidates["src_user"].astype(str).head(normal_user_cap).tolist()

    snapshot_anchor_users = (
        frame.loc[frame["label"] == 0]
        .sort_values(by=["snapshot", "timestamp", "src_user"])
        .groupby("snapshot", sort=True)["src_user"]
        .first()
        .astype(str)
        .tolist()
    )

    keep_users = set(anom_users)
    keep_users.update(selected_norm_users)
    keep_users.update(snapshot_anchor_users)
    return frame[frame["src_user"].isin(keep_users)].reset_index(drop=True)

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
    campaign_lookup, _ = build_optc_campaign_lookup(selected_events)
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
        campaign_lookup.get((int(timestamp), str(src), str(dst)), "") if int(label) == 1 else ""
        for timestamp, src, dst, label in zip(
            frame["timestamp"].tolist(),
            frame["src_computer"].tolist(),
            frame["dst_computer"].tolist(),
            frame["label"].tolist(),
        )
    ]

    frame = assign_snapshots(frame, snapshot_seconds)
    frame = downsample_negatives_preserve_snapshots(frame, neg_ratio=neg_ratio, seed=seed)
    frame = frame[["timestamp", "src_computer", "dst_computer", "label", "campaign", "snapshot"]]

    write_events(str(output_dir / "redteam_optc_small.txt"), selected_events)
    return frame

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

def preprocess_0501(raw_root: Path, dataset: str, snapshot_seconds: int, neg_ratio: int, seed: int) -> pd.DataFrame:
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

if __name__ == "__main__":
    main()
