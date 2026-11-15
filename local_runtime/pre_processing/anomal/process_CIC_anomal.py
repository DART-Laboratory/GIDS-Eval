import argparse
import os
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

result.to_csv(os.path.join(OUT_DIR, "cic_2017_1.csv"), encoding="utf-8", index=False)
