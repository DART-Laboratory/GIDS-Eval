import os
import pickle
from collections import defaultdict
from pathlib import Path

import json
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True)
    parser.add_argument("-o", "--output_dir", required=True)
    parser.add_argument("--win-rows", type=int, default=10000)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    win_rows = args.win_rows

    order = ["monday.csv", "tuesday.csv", "wednesday.csv", "thursday.csv", "friday.csv"]
    available = {p.name: p for p in in_dir.glob("*.csv")}
    csv_files = [available[name] for name in order if name in available]
    csv_files += [p for p in sorted(in_dir.glob("*.csv")) if p not in csv_files]
    if not csv_files:
        raise SystemExit(f"No CSV files found in {in_dir}")
    dfs = []
    for fp in csv_files:
        tmp = pd.read_csv(fp)
        tmp["_src_file"] = fp.name
        dfs.append(tmp)
    df = pd.concat(dfs, ignore_index=True)

    src_col = "Src IP"
    dst_col = "Dst IP"
    dur_col = "Flow Duration"
    bytes_col = None
    pkts_col = None
    for candidate in ("Total Length of Fwd Packets", "Total Length of Fwd Packet"):
        if candidate in df.columns:
            bytes_col = candidate
            break
    for candidate in ("Total Fwd Packets", "Total Fwd Packet"):
        if candidate in df.columns:
            pkts_col = candidate
            break
    label_col = "Label"

    cols = [src_col, dst_col, label_col, "_src_file"]
    if dur_col in df.columns:
        cols.append(dur_col)
    if bytes_col:
        cols.append(bytes_col)
    if pkts_col:
        cols.append(pkts_col)
    if "Timestamp" in df.columns:
        cols.append("Timestamp")
    elif " Timestamp" in df.columns:
        cols.append(" Timestamp")
    df = df[cols].copy()

    if label_col in df.columns:
        labels = df[label_col].fillna("").astype(str).str.strip()
        df["campaign"] = labels.where(labels.ne("BENIGN"), "")
        df["label_bin"] = labels.ne("BENIGN").astype(int)
    else:
        df["campaign"] = ""
        df["label_bin"] = 0

    nmap = {}
    nid = 0

    def get_or_add(n):
        nonlocal nid
        if n not in nmap:
            nmap[n] = nid
            nid += 1
        return nmap[n]

    first_attack_ts = None

    ts_col = None
    for candidate in ("Timestamp", " Timestamp"):
        if candidate in df.columns:
            ts_col = candidate
            break
    if not ts_col:
        raise SystemExit("Timestamp column not found in CIC-new CSVs.")

    dfs_out = []
    cumulative = 0
    global_min = None
    for fp in csv_files:
        part = df[df["_src_file"] == fp.name].copy()
        ts = pd.to_datetime(part[ts_col], errors="coerce")
        part = part[~ts.isna()].copy()
        ts = ts.loc[part.index]
        shift = (ts.dt.hour < 8).astype(int)
        ts = ts + pd.to_timedelta(shift * 43200, unit="s")
        if global_min is None:
            global_min = ts.min()
        part["_ts_int"] = (ts - global_min).dt.total_seconds().astype(int)
        part = part.sort_values(by="_ts_int").reset_index(drop=True)
        start = int(part["_ts_int"].min())
        part["_ts_int"] = part["_ts_int"] - start + cumulative
        cumulative = int(part["_ts_int"].max()) + 1
        dfs_out.append(part)

    df = pd.concat(dfs_out, ignore_index=True)
    df = df.sort_values(by="_ts_int").reset_index(drop=True)

    windows = defaultdict(list)
    for _, row in df.iterrows():
        lbl = int(row["label_bin"])
        if lbl == 1 and first_attack_ts is None:
            first_attack_ts = int(row["_ts_int"])

        ts_int = int(row["_ts_int"])
        win_start = (ts_int // win_rows) * win_rows
        src_id = get_or_add(row[src_col])
        dst_id = get_or_add(row[dst_col])

        dur = int(row[dur_col]) if dur_col in row.index else 0
        byt = int(row[bytes_col]) if bytes_col and bytes_col in row.index else 0
        pkts = int(row[pkts_col]) if pkts_col and pkts_col in row.index else 0

        ts_out = ts_int
        campaign = str(row.get("campaign", "")) if lbl == 1 else ""
        windows[win_start].append([ts_out, src_id, dst_id, dur, byt, pkts, lbl, campaign])

    os.makedirs(out_dir, exist_ok=True)
    for fname in os.listdir(out_dir):
        if fname.endswith(".txt") or fname == "nmap.pkl":
            try:
                os.remove(out_dir / fname)
            except Exception:
                pass

    for win_start in sorted(windows.keys()):
        entries = windows[win_start]
        out_f = out_dir / f"{win_start}.txt"
        with out_f.open("w+") as f_out:
            for entry in entries:
                ts, src_id, dst_id, dur, byt, pkts, lbl, campaign = entry
                f_out.write(f"{ts},{src_id},{dst_id},{dur},{byt},{pkts},{lbl},{campaign}\n")

    nmap_rev = [None] * len(nmap)
    for k, v in nmap.items():
        nmap_rev[v] = k
    with open(out_dir / "nmap.pkl", "wb") as f:
        pickle.dump(nmap_rev, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "window_ms": win_rows,
        "first_attack_ts": first_attack_ts,
        "num_windows": len(windows),
        "num_nodes": len(nmap),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Output dir: {out_dir}")
    print(f"Windows written: {len(windows)}")
    print(f"Nodes: {len(nmap)}")
    if first_attack_ts is None:
        print("No attacks found.")
    else:
        print(f"First attack ts (ms from start): {first_attack_ts}")

if __name__ == "__main__":
    main()
