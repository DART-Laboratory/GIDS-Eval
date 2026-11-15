import argparse
import os
import pickle
from datetime import datetime

import pandas as pd

parser = argparse.ArgumentParser(
    description="Process CIC-IDS2017 (new format) from an input directory and write output to an output directory."
)
parser.add_argument("-i", "--input_dir", type=str, default=".", help="Directory containing the CSV files")
parser.add_argument("-o", "--output_dir", type=str, default=".", help="Directory to write the output CSV")
args = parser.parse_args()
IN_DIR = args.input_dir
OUT_DIR = args.output_dir
os.makedirs(OUT_DIR, exist_ok=True)

def parse_time_column(series):
    return [int(datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M:%S.%f").timestamp()) for ts in series]

def load_day(filename: str):
    df = pd.read_csv(os.path.join(IN_DIR, filename))
    df["time_int"] = parse_time_column(df["Timestamp"])
    return df.sort_values(by="time_int")

mon = load_day("monday.csv")
tue = load_day("tuesday.csv")
wed = load_day("wednesday.csv")
thu = load_day("thursday.csv")
fri = load_day("friday.csv")

frame = [mon, tue, wed, thu, fri]
result = pd.concat(frame)
result = result.reset_index(drop=True)
result = result.sort_values(by="time_int")
result = result.reset_index(drop=True)
result["time_int"] = result["time_int"] - result["time_int"].iloc[0]

result["raw_label"] = result["Label"].fillna("").astype(str).str.strip()
label = [0 if i == "BENIGN" else 1 for i in result["raw_label"].values]
result["label"] = label
result["campaign"] = result["raw_label"].where(result["label"] == 1, "")

data1 = result.values
DELTA = 100000
DST = OUT_DIR

nmap = {}
nid = [0]

def get_or_add(n):
    if n not in nmap:
        nmap[n] = nid[0]
        nid[0] += 1
    return nmap[n]

fmt_line = lambda ts, src, dst, dur, bytes_, pkts, label_val, campaign: (
    "%s,%s,%s,%s,%s,%s,%s,%s\n"
    % (
        int(ts),
        get_or_add(src),
        get_or_add(dst),
        dur,
        bytes_,
        pkts,
        label_val,
        campaign,
    ),
    int(ts),
)

cur_time = 0
f_out = open(os.path.join(DST, f"{cur_time}.txt"), "w+")

for tokens in data1:
    l, ts = fmt_line(tokens[-4], tokens[2], tokens[4], tokens[8], tokens[9], tokens[10], tokens[-2], tokens[-1])

    if ts >= cur_time + DELTA:
        cur_time += DELTA
        f_out.close()
        f_out = open(os.path.join(DST, f"{cur_time}.txt"), "w+")

    f_out.write(l)
f_out.close()

nmap_rev = [None] * (max(nmap.values()) + 1)
for (k, v) in nmap.items():
    nmap_rev[v] = k

with open(os.path.join(DST, "nmap.pkl"), "wb+") as f:
    pickle.dump(nmap_rev, f, protocol=pickle.HIGHEST_PROTOCOL)
