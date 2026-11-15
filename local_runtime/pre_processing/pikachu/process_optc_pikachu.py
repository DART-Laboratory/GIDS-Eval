import argparse
import os
import sys

import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from campaign_grouping import build_optc_campaign_lookup
from optc_small_component import load_small_optc_component, write_events

parser = argparse.ArgumentParser(description="OPTC preprocessing with small-component anomaly relabeling.")
parser.add_argument("-i", "--input", type=str, default="auth_optc.txt", help="Path to auth_optc.txt")
parser.add_argument("-o", "--output", type=str, default="optc.csv", help="Path to output CSV")
parser.add_argument("--redteam", type=str, default=None, help="Path to redteam_optc.txt")
parser.add_argument("--target_nodes", type=int, default=8, help="Connected-component node count to keep")
parser.add_argument("--snapshot", type=int, default=3600, help="Snapshot size in seconds")
parser.add_argument("--neg_ratio", type=int, default=20, help="Max negative-to-positive ratio (0 to disable downsampling)")
parser.add_argument("--seed", type=int, default=42, help="Random seed for downsampling")
args = parser.parse_args()

SRC = args.input
DST = args.output
snapshot = args.snapshot

if args.redteam is None:
    redteam_path = os.path.join(os.path.dirname(os.path.abspath(SRC)), "redteam_optc.txt")
else:
    redteam_path = args.redteam

if not os.path.exists(redteam_path):
    raise SystemExit(f"redteam file not found: {redteam_path}")

sel = load_small_optc_component(redteam_path, target_nodes=args.target_nodes)
selected_events = sel["selected_events"]
selected_nodes = sel["selected_nodes"]
campaign_lookup, _ = build_optc_campaign_lookup(selected_events)
attack_keys = {f"{int(e['ts'])}|{e['src']}|{e['dst']}" for e in selected_events}

optc = pd.read_csv(SRC, header=None, names=["timestamp", "src_computer", "dst_computer", "label", "1", "2", "3", "4", "5"])
optc = optc.drop(["1", "2", "3", "4", "5"], axis=1).reset_index(drop=True)
optc["timestamp"] = pd.to_numeric(optc["timestamp"], errors="coerce").fillna(0).astype(int)
optc["src_computer"] = optc["src_computer"].astype(str)
optc["dst_computer"] = optc["dst_computer"].astype(str)
row_keys = optc["timestamp"].astype(str) + "|" + optc["src_computer"] + "|" + optc["dst_computer"]
optc["label"] = row_keys.isin(attack_keys).astype(int)
optc["campaign"] = [
    campaign_lookup.get((int(timestamp), str(src), str(dst)), "") if int(label) == 1 else ""
    for timestamp, src, dst, label in zip(
        optc["timestamp"].tolist(),
        optc["src_computer"].tolist(),
        optc["dst_computer"].tolist(),
        optc["label"].tolist(),
    )
]

if args.neg_ratio and args.neg_ratio > 0:
    pos = optc[optc["label"] == 1]
    neg = optc[optc["label"] == 0]
    if len(pos) > 0:
        target_neg = min(len(neg), len(pos) * args.neg_ratio)
        neg_sampled = neg.sample(n=target_neg, random_state=args.seed, replace=False)
        optc = pd.concat([pos, neg_sampled], ignore_index=True)
        optc = optc.sort_values(by="timestamp").reset_index(drop=True)

initial_time = int(optc["timestamp"].min()) if len(optc) > 0 else 0
optc["delta"] = optc["timestamp"] - initial_time
optc["snapshot"] = (optc["delta"] // snapshot).astype(int)
optc = optc.drop(["delta"], axis=1).reset_index(drop=True)

print("trainwin:")
pos_rows = optc[optc["label"] == 1]
if len(pos_rows) > 0:
    sp_time = int(pos_rows.iloc[0]["timestamp"])
    print(int(optc[optc["timestamp"] < sp_time]["snapshot"].nunique()))
else:
    print(0)

out_dir = os.path.dirname(DST)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    write_events(os.path.join(out_dir, "redteam_optc_small.txt"), selected_events)
optc.to_csv(DST, index=False)

print(f"selected_nodes={len(selected_nodes)} selected_events={len(selected_events)} selected_unique_edges={len(sel['selected_unique_edges'])}")
