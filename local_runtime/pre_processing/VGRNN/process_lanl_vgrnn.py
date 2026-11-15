import csv
import os
import pickle
import argparse
import sys
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from campaign_grouping import build_lanl_campaign_lookup

parser = argparse.ArgumentParser(description="LANL preprocessing with configurable input/output directories.")
parser.add_argument("-i", "--input_dir", type=str, default=".", help="Directory containing auth.txt and redteam.txt")
parser.add_argument("-o", "--output_dir", type=str, default=".", help="Directory to write outputs (auth_all.csv and all_20.csv)")
args = parser.parse_args()
IN_DIR = args.input_dir
OUT_DIR = args.output_dir
os.makedirs(OUT_DIR, exist_ok=True)

SRC = os.path.join(IN_DIR, 'auth.txt')
RED = os.path.join(IN_DIR, 'redteam.txt')

DST = OUT_DIR

assert RED and SRC and DST, 'Please download the LANL data set, and mark in the code where it is:\nLines 13-15 of /lanl_experiments/loaders/split.py'

DELTA = 10000
DAY = 60 ** 2 * 24

def mark_anoms():
    with open(RED, 'r', encoding='utf-8', errors='replace', newline='') as f:
        red_events = list(csv.reader(f))

    def add_ts(d, val, ts):
        if len(val) < 4:
            return
        val = (str(val[2]).strip(), str(val[3]).strip())
        if val in d:
            d[val].append(ts)
        else:
            d[val] = [ts]

    anom_dict = {}
    for event in red_events:
        if len(event) < 4:
            continue
        try:
            ts = int(str(event[0]).strip())
        except ValueError:
            continue
        add_ts(anom_dict, event, ts)

    return anom_dict

def is_anomalous(d, src, dst, ts):
    if ts < 150885 or (src, dst) not in d:
        return False

    times = d[(src, dst)]
    for time in times:
        if ts == time:
            return True

    return False

def split():
    anom_dict = mark_anoms()
    campaign_lookup, _ = build_lanl_campaign_lookup(RED)

    last_time = 1
    cur_time = 0

    f_in = open(SRC, 'r')
    f_out = open(os.path.join(DST, f"{cur_time}.txt"), 'w+')

    line = f_in.readline()

    nmap = {}
    nid = [0]

    def get_or_add(n):
        if n not in nmap:
            nmap[n] = nid[0]
            nid[0] += 1

        return nmap[n]

    prog = tqdm(desc='Seconds parsed', total=5011199)

    fmt_src = lambda x:\
        x.split('@')[0].replace('$', '')

    fmt_label = lambda ts, src, dst:\
        1 if is_anomalous(anom_dict, src, dst, ts)\
            else 0

    def fmt_line(ts, src, dst, src_user):
        ts_int = int(ts)
        label = fmt_label(ts_int, src, dst)
        campaign = campaign_lookup.get((ts_int, str(src_user).strip(), str(src).strip(), str(dst).strip()), "") if label == 1 else ""
        return (
            '%s,%s,%s,%s,%s\n' % (
                ts,
                get_or_add(src),
                get_or_add(dst),
                label,
                campaign,
            ),
            ts_int,
        )

    while line:
        if 'NTLM' not in line.upper():
            line = f_in.readline()
            continue

        tokens = line.split(',')
        l, ts = fmt_line(tokens[0], tokens[3], tokens[4], tokens[1])

        if ts != last_time:
            prog.update(ts - last_time)
            last_time = ts

        while ts >= cur_time + DELTA:
            cur_time += DELTA
            f_out.close()
            f_out = open(os.path.join(DST, f"{cur_time}.txt"), 'w+')

        f_out.write(l)
        line = f_in.readline()

    f_out.close()
    f_in.close()

    nmap_rev = [None] * (max(nmap.values()) + 1)
    for (k, v) in nmap.items():
        nmap_rev[v] = k

    with open(DST + 'nmap.pkl', 'wb+') as f:
        pickle.dump(nmap_rev, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    split()
