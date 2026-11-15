import pickle
import networkx
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import os
import argparse
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from campaign_grouping import build_lanl_campaign_lookup

parser = argparse.ArgumentParser(description="LANL preprocessing with configurable input/output directories.")
parser.add_argument("-i", "--input_dir", type=str, default=".", help="Directory containing auth.txt and redteam.txt")
parser.add_argument("-o", "--output_dir", type=str, default=".", help="Directory to write outputs (auth_all.csv and all_20.csv)")
parser.add_argument("--neg_ratio", type=int, default=20, help="Max negative-to-positive ratio (0 to disable downsampling)")
parser.add_argument("--seed", type=int, default=42, help="Random seed for downsampling")
args = parser.parse_args()
IN_DIR = args.input_dir
OUT_DIR = args.output_dir
os.makedirs(OUT_DIR, exist_ok=True)

SRC = os.path.join(IN_DIR, 'auth.txt')
RED = os.path.join(IN_DIR, 'redteam.txt')
DST = os.path.join(OUT_DIR, 'all_20.csv')

snapshot = 3600

def preprocess_lanl_data(data_folder, red):
    campaign_lookup, _ = build_lanl_campaign_lookup(red)
    auth_df = pd.DataFrame()
    for chunk_df in tqdm(pd.read_csv(
        data_folder,
        sep=',',
        header=None,
        names=['timestamp', 'src_user', 'dst_user', 'src_computer', 'dst_computer', 'auth_type', 'logon_type', 'auth_ori', 'success'],
        dtype={
            'timestamp': np.int32,
            'src_user': str,
            'dst_user': str,
            'src_computer': str,
            'dst_computer': str,
            'auth_type': 'category',
            'logon_type': 'category'
        },
        chunksize=10000000
    )):
        chunk_df = chunk_df[~((chunk_df['auth_type'] == '?') | (chunk_df['logon_type'] == '?'))]
        chunk_df = chunk_df[~(
            (chunk_df['src_user'].str.contains(r'ANONYMOUS(?!$)')) |
            (chunk_df['src_user'].str.contains(r'LOCAL(?!$)')) |
            (chunk_df['src_user'].str.contains(r'NETWORK(?!$)')) |
            (chunk_df['src_user'].str.contains(r'ADMIN(?!$)'))
        )]
        chunk_df = chunk_df[chunk_df['src_computer'] != chunk_df['dst_computer']]
        chunk_df = chunk_df.drop(['dst_user', 'auth_type', 'logon_type', 'auth_ori', 'success'], axis=1).reset_index(drop=True)
        auth_df = pd.concat([auth_df, chunk_df], ignore_index=True)

    rt_df = pd.read_csv(red, header=None)
    rt_df.columns = ['timestamp', 'src_user', 'src_computer', 'dst_computer']
    filter_col_name = ['timestamp', 'src_user', 'src_computer', 'dst_computer']
    comm_df = pd.merge(auth_df.reset_index(), rt_df.reset_index(), how='inner', on=filter_col_name)

    anom_row_index = comm_df.index_x.to_list()

    auth_df['label'] = 0
    auth_df.loc[anom_row_index, 'label'] = 1
    auth_df['campaign'] = [
        campaign_lookup.get((int(ts), str(src_user), str(src), str(dst)), '') if int(label) == 1 else ''
        for ts, src_user, src, dst, label in zip(
            auth_df['timestamp'].tolist(),
            auth_df['src_user'].tolist(),
            auth_df['src_computer'].tolist(),
            auth_df['dst_computer'].tolist(),
            auth_df['label'].tolist(),
        )
    ]
    initial_time = min(auth_df['timestamp'])
    auth_df['delta'] = auth_df['timestamp'] - initial_time
    auth_df['snapshot'] = auth_df['delta'] // snapshot
    auth_df = auth_df.drop(['delta'], axis=1).reset_index(drop=True)

    if args.neg_ratio and args.neg_ratio > 0:
        pos = auth_df[auth_df['label'] == 1]
        neg = auth_df[auth_df['label'] == 0]
        target_neg = min(len(neg), len(pos) * args.neg_ratio)
        if target_neg < len(neg):
            neg = neg.sample(n=target_neg, random_state=args.seed, replace=False)
            auth_df = pd.concat([pos, neg], ignore_index=True)
            auth_df = auth_df.sort_values(by='timestamp').reset_index(drop=True)

    print("trainwin")
    sp_time = auth_df[auth_df['label'] == 1].iloc[0]['timestamp']
    print(auth_df[auth_df['timestamp'] < sp_time]['snapshot'].nunique())
    auth_df.to_csv(os.path.join(OUT_DIR, "auth_all.csv"))

def lanl_user_subset(data_folder, dst):
    lanl_df = pd.read_csv(
        data_folder,
        header=0,
        index_col=0,
        dtype={
            'timestamp': np.int32,
            'src_user': str,
            'src_computer': str,
            'dst_computer': str,
            'label': bool,
            'campaign': str,
            'snapshot': int
        }
    )
    anom_user_df = lanl_df[lanl_df['label'] == 1]
    anom_row_index = anom_user_df.index.to_list()
    print("Total Anom Edges: ", len(anom_row_index))
    print("Anom rows index: ", anom_row_index)

    anom_user = list(set(lanl_df.loc[anom_row_index, 'src_user'].tolist()))
    print("Anomalous Users: ", len(anom_user), anom_user)
    all_user = lanl_df.src_user.unique()
    print("total users: ", len(all_user))

    norm_users = np.setdiff1d(all_user, anom_user).tolist()
    print("Norm users: ", len(norm_users))
    sample_size = min(len(norm_users), len(anom_user) * 20)
    norm_users = random.sample(norm_users, sample_size)
    all_users = norm_users + anom_user
    print("all users: ", len(all_users), all_users)
    all_user_df = lanl_df[lanl_df['src_user'].isin(all_users)]
    all_user_df.to_csv(dst)

preprocess_lanl_data(data_folder=SRC, red=RED)
lanl_user_subset(os.path.join(OUT_DIR, "auth_all.csv"), DST)
