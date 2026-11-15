import pandas as pd
from datetime import datetime
import numpy as np
import pickle
import time
import csv
import os
import argparse

parser = argparse.ArgumentParser(description="Process CIC-IDS2017 daily CSVs from an input directory and write output to an output directory.")
parser.add_argument("-i", "--input_dir", type=str, default=".", help="Directory containing the daily CSV files")
parser.add_argument("-o", "--output_dir", type=str, default=".", help="Directory to write the output CSV")
args = parser.parse_args()
IN_DIR = args.input_dir
OUT_DIR = args.output_dir
os.makedirs(OUT_DIR, exist_ok=True)

mon = pd.read_csv(os.path.join(IN_DIR, 'Monday-WorkingHours.pcap_ISCX.csv'))
time1 = mon[' Timestamp'].values
time2 = []
for i in time1:
    tmp = datetime.strptime(i, '%d/%m/%Y %H:%M:%S')
    if tmp.hour < 8:
        time2.append(int(tmp.timestamp()) + 43200)
    else:
        time2.append(int(tmp.timestamp()))
mon['time_int'] = time2
mon1 = mon.sort_values(by='time_int')

tue = pd.read_csv(os.path.join(IN_DIR, 'Tuesday-WorkingHours.pcap_ISCX.csv'))
time1 = tue[' Timestamp'].values
time2 = []
for i in time1:
    tmp = datetime.strptime(i, '%d/%m/%Y %H:%M')
    if tmp.hour < 8:
        time2.append(int(tmp.timestamp()) + 43200)
    else:
        time2.append(int(tmp.timestamp()))
tue['time_int'] = time2
tue1 = tue.sort_values(by='time_int')

wed = pd.read_csv(os.path.join(IN_DIR, 'Wednesday-workingHours.pcap_ISCX.csv'))
time1 = wed[' Timestamp'].values
time2 = []
for i in time1:
    tmp = datetime.strptime(i, '%d/%m/%Y %H:%M')
    if tmp.hour < 8:
        time2.append(int(tmp.timestamp()) + 43200)
    else:
        time2.append(int(tmp.timestamp()))
wed['time_int'] = time2
wed1 = wed.sort_values(by='time_int')

Thu_1 = pd.read_csv(os.path.join(IN_DIR, 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'))
Thu_1 = Thu_1.iloc[:170365]
time1 = Thu_1[' Timestamp'].values
time2 = []
for i in time1:
    tmp = datetime.strptime(i, '%d/%m/%Y %H:%M')
    time2.append(int(tmp.timestamp()))
Thu_1['time_int'] = time2
Thu_11 = Thu_1.sort_values(by='time_int')

Thu_2 = pd.read_csv(os.path.join(IN_DIR, 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv'))
time1 = Thu_2[' Timestamp'].values
time2 = []
for i in time1:
    tmp = datetime.strptime(i, '%d/%m/%Y %H:%M')
    time2.append(int(tmp.timestamp()) + 43200)
Thu_2['time_int'] = time2
Thu_21 = Thu_2.sort_values(by='time_int')

Fri_1 = pd.read_csv(os.path.join(IN_DIR, 'Friday-WorkingHours-Morning.pcap_ISCX.csv'))
time1 = Fri_1[' Timestamp'].values
time2 = []
for i in time1:
    tmp = datetime.strptime(i, '%d/%m/%Y %H:%M')
    time2.append(int(tmp.timestamp()))
Fri_1['time_int'] = time2
Fri_11 = Fri_1.sort_values(by='time_int')

Fri_2 = pd.read_csv(os.path.join(IN_DIR, 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'))
time1 = Fri_2[' Timestamp'].values
time2 = []
for i in time1:
    tmp = datetime.strptime(i, '%d/%m/%Y %H:%M')
    time2.append(int(tmp.timestamp()) + 43200)
Fri_2['time_int'] = time2
Fri_21 = Fri_2.sort_values(by='time_int')

Fri_3 = pd.read_csv(os.path.join(IN_DIR, 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'))
time1 = Fri_3[' Timestamp'].values
time2 = []
for i in time1:
    tmp = datetime.strptime(i, '%d/%m/%Y %H:%M')
    time2.append(int(tmp.timestamp()) + 43200)
Fri_3['time_int'] = time2
Fri_31 = Fri_3.sort_values(by='time_int')

frame = [mon1, tue1, wed1, Thu_11, Thu_21, Fri_11, Fri_21, Fri_31]

result = pd.concat(frame)
result = result.reset_index(drop=True)
result = result.sort_values(by='time_int')
result = result.reset_index(drop=True)
result['time_int'] = result['time_int'] - result['time_int'].iloc[0]
la = result[' Label'].values
label = []
for i in la:
    if i != 'BENIGN':
        label.append(1)
    else:
        label.append(0)
result['label'] = label

ti = 20
result['snapshot'] = result['time_int'] // (60 * ti)

cic = result[['time_int', ' Source IP', ' Destination IP', 'label', 'snapshot']]
cic.columns = ['timestamp', 'src_computer', 'dst_computer', 'label', 'snapshot']

out_path = os.path.join(OUT_DIR, 'cic_20.csv')
cic.to_csv(out_path)

print("trainwin")
print(result.iloc[529917]['snapshot'] + 1)
