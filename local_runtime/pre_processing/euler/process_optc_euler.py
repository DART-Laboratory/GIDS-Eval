import argparse
import csv
import os
import pickle
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from campaign_grouping import build_optc_campaign_lookup
from optc_small_component import build_anomaly_index, load_small_optc_component, write_events

parser = argparse.ArgumentParser(description="OpTC preprocessing with small-component anomaly labeling.")
parser.add_argument("-i", "--input_dir", type=str, default=".", help="Directory containing auth_optc.txt and redteam_optc.txt")
parser.add_argument("-o", "--output_dir", type=str, default=".", help="Directory to write split files")
parser.add_argument("--target_nodes", type=int, default=8, help="Connected-component node count to keep")
args = parser.parse_args()

IN_DIR = args.input_dir
OUT_DIR = args.output_dir
os.makedirs(OUT_DIR, exist_ok=True)

SRC = os.path.join(IN_DIR, "auth_optc.txt")
RED = os.path.join(IN_DIR, "redteam_optc.txt")
DST = OUT_DIR
DELTA = 100000

def save_map(m, fname):
    m_rev = [None] * (max(m.values()) + 1) if m else []
    for key, val in m.items():
        m_rev[val] = key
    with open(os.path.join(DST, fname), "wb") as f:
        pickle.dump(m_rev, f, protocol=pickle.HIGHEST_PROTOCOL)

def get_or_add(node, nmap, nid):
    if node not in nmap:
        nmap[node] = nid[0]
        nid[0] += 1
    return nmap[node]

def split_auth():
    if not os.path.exists(SRC):
        raise SystemExit(f"missing auth file: {SRC}")
    if not os.path.exists(RED):
        raise SystemExit(f"missing redteam file: {RED}")

    selected = load_small_optc_component(RED, target_nodes=args.target_nodes)
    selected_events = selected["selected_events"]
    anom_dict = build_anomaly_index(selected_events)
    campaign_lookup, _ = build_optc_campaign_lookup(selected_events)
    write_events(os.path.join(DST, "redteam_optc_small.txt"), selected_events)

    cur_time = 0
    out_path = os.path.join(DST, f"{cur_time}.txt")
    f_out = open(out_path, "w", encoding="utf-8")

    nmap = {}
    nid = [0]

    with open(SRC, "r", encoding="utf-8", errors="replace", newline="") as f_in:
        reader = csv.reader(f_in)
        for row in reader:
            if len(row) < 9:
                continue
            try:
                ts = int(float(row[0]))
            except Exception:
                continue
            src = str(row[1]).strip()
            dst = str(row[2]).strip()
            if src == "" or dst == "":
                continue
            pid = str(row[4]).strip()
            ppid = str(row[5]).strip()
            dest_port = str(row[6]).strip()
            l4protocol = str(row[7]).strip()
            img_path = str(row[8]).strip()

            while ts >= cur_time + DELTA:
                f_out.close()
                cur_time += DELTA
                out_path = os.path.join(DST, f"{cur_time}.txt")
                f_out = open(out_path, "w", encoding="utf-8")

            label = 1 if ts in anom_dict.get((src, dst), set()) else 0
            campaign = campaign_lookup.get((ts, src, dst), "") if label == 1 else ""
            f_out.write(
                f"{ts},{get_or_add(src, nmap, nid)},{get_or_add(dst, nmap, nid)},{pid},{ppid},{dest_port},{l4protocol},{img_path},{label},{campaign}\n"
            )

    f_out.close()
    save_map(nmap, "nmap.pkl")
    print(f"selected_nodes={len(selected['selected_nodes'])} selected_events={len(selected_events)} selected_unique_edges={len(selected['selected_unique_edges'])}")

if __name__ == "__main__":
    split_auth()
