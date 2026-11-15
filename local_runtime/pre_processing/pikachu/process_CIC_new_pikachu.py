import pandas as pd
from datetime import datetime
import os
import argparse

parser = argparse.ArgumentParser(description="Process CIC-IDS2017 (new CIC format) from an input directory and write output to an output directory.")
parser.add_argument("-i", "--input_dir", type=str, default=".", help="Directory containing the CSV files")
parser.add_argument("-o", "--output_dir", type=str, default=".", help="Directory to write the output CSV")
args = parser.parse_args()
IN_DIR = args.input_dir
OUT_DIR = args.output_dir
os.makedirs(OUT_DIR, exist_ok=True)

def parse_time_column(series):
    return [int(datetime.strptime(ts.strip(), '%Y-%m-%d %H:%M:%S.%f').timestamp()) for ts in series]

mon = pd.read_csv(os.path.join(IN_DIR, 'monday.csv'))
mon['time_int'] = parse_time_column(mon['Timestamp'])
mon1 = mon.sort_values(by='time_int')

tue = pd.read_csv(os.path.join(IN_DIR, 'tuesday.csv'))
tue['time_int'] = parse_time_column(tue['Timestamp'])
tue1 = tue.sort_values(by='time_int')

wed = pd.read_csv(os.path.join(IN_DIR, 'wednesday.csv'))
wed['time_int'] = parse_time_column(wed['Timestamp'])
wed1 = wed.sort_values(by='time_int')

thu = pd.read_csv(os.path.join(IN_DIR, 'thursday.csv'))
thu['time_int'] = parse_time_column(thu['Timestamp'])
thu1 = thu.sort_values(by='time_int')

fri = pd.read_csv(os.path.join(IN_DIR, 'friday.csv'))
fri['time_int'] = parse_time_column(fri['Timestamp'])
fri1 = fri.sort_values(by='time_int')

frame = [mon1, tue1, wed1, thu1, fri1]
result = pd.concat(frame)
result = result.reset_index(drop=True)
result = result.sort_values(by='time_int')
result = result.reset_index(drop=True)
result['time_int'] = result['time_int'] - result['time_int'].iloc[0]

result['raw_label'] = result['Label'].fillna('').astype(str).str.strip()
label = [0 if i == 'BENIGN' else 1 for i in result['raw_label'].values]
result['label'] = label
result['campaign'] = result['raw_label'].where(result['label'] == 1, '')

ti = 60
result['snapshot'] = result['time_int'] // (60 * ti)

cic = result[['time_int', 'Src IP', 'Dst IP', 'label', 'campaign', 'raw_label', 'snapshot']]
cic.columns = ['timestamp', 'src_computer', 'dst_computer', 'label', 'campaign', 'raw_label', 'snapshot']

out_path = os.path.join(OUT_DIR, 'cic_20.csv')
cic.to_csv(out_path, index=False)

print("trainwin:")
sp_time = cic[cic['label'] == 1].iloc[0]['timestamp']
trainwin = cic[cic['timestamp'] < sp_time]['snapshot'].nunique()
print(trainwin)
