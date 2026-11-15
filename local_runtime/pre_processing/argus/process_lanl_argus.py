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

SRC_DIR = IN_DIR
FLOWS_SRC = os.path.join(IN_DIR, 'flows.txt')

assert RED and SRC and DST, 'Please download the LANL data set, and mark in the code where it is'

DELTA = 10000
DAY = 60**2*24

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

def mark_anoms_node():
    with open(RED, 'r', encoding='utf-8', errors='replace', newline='') as f:
        red_events = list(csv.reader(f))
    def add_ts(d, val, ts):
        if len(val) < 4:
            return
        src = str(val[2]).strip()
        dst = str(val[3]).strip()
        if src in d:
            d[src].append(ts)
        else:
            d[src] = [ts]
        if dst in d:
            d[dst].append(ts)
        else:
            d[dst] = [ts]

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
    times = d[(src,dst)]
    for time in times:
        if ts == time:
            return True
    return False

def is_anomalous_range(d, src, dst, ts):
    if ts < 150885 or (src, dst) not in d:
        return False
    times = d[(src,dst)]
    for time in times:
        if abs(ts-time) <= 300:
            return True
    return False

def is_anomalous_node_range(d, node, ts):
    if ts < 150885 or node not in d:
        return False

    times = d[node]
    for time in times:
        if abs(ts-time) <= 300:
            return True

    return False

def save_map(m, fname):
    if not m:
        m_rev = []
    else:
        m_rev = [None] * (max(m.values()) + 1)
    for (k,v) in m.items():
        m_rev[v] = k

    with open(os.path.join(DST, fname), 'wb') as f:
        pickle.dump(m_rev, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(os.path.join(DST, fname) + ' saved')

def get_or_add(n, m, id):
    if n not in m:
        m[n] = id[0]
        id[0] += 1

    return m[n]

def split_auth():
    anom_dict = mark_anoms()
    campaign_lookup, _ = build_lanl_campaign_lookup(RED)

    last_time = 1
    cur_time = 0

    f_in = open(SRC,'r')
    f_out = open(os.path.join(DST, str(cur_time) + '.txt'), 'w')

    line = f_in.readline()

    nmap = {}
    nid = [0]
    umap = {}
    uid = [0]
    atmap = {}
    atid = [0]
    ltmap = {}
    ltid = [0]
    aomap = {}
    aoid = [0]
    smap = {}
    sid = [0]
    prog = tqdm(desc='Seconds parsed', total=5011199)

    fmt_src = lambda x :\
        x.split('@')[0].replace('$', '')

    fmt_label = lambda ts,src,dst :\
        1 if is_anomalous(anom_dict, src, dst, ts)\
        else 0

    def fmt_line(ts, src, dst, src_u, dst_u, auth_t, logon_t, auth_o, success):
        ts_int = int(ts)
        label = fmt_label(ts_int, src, dst)
        campaign = campaign_lookup.get((ts_int, str(src_u).strip(), str(src).strip(), str(dst).strip()), "") if label == 1 else ""
        return (
            '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (
                ts,
                get_or_add(src, nmap, nid),
                get_or_add(dst, nmap, nid),
                get_or_add(fmt_src(src_u), umap, uid),
                get_or_add(fmt_src(dst_u), umap, uid),
                get_or_add(auth_t, atmap, atid),
                get_or_add(logon_t, ltmap, ltid),
                get_or_add(auth_o, aomap, aoid),
                get_or_add(success, smap, sid),
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
        l, ts = fmt_line(tokens[0], tokens[3], tokens[4], tokens[1], tokens[2], tokens[5], tokens[6], tokens[7], tokens[8][:-1])

        if ts != last_time:
            prog.update(ts-last_time)
            last_time = ts

        while ts >= cur_time + DELTA:
            f_out.close()
            cur_time += DELTA
            f_out = open(os.path.join(DST, str(cur_time) + '.txt'), 'w')

        f_out.write(l)
        line = f_in.readline()

    f_out.close()
    f_in.close()

    save_map(nmap, 'nmap.pkl')
    save_map(umap, 'umap.pkl')
    save_map(atmap, 'atmap.pkl')
    save_map(ltmap, 'ltmap.pkl')
    save_map(aomap, 'aomap.pkl')
    save_map(smap, 'smap.pkl')

def reverse_load_map(fname):
    m = {}

    with open(os.path.join(DST, fname), 'rb') as f:
        l = pickle.load(f)
        for i in range(0, len(l)):
            m[l[i]] = i
    return m

def split_flows():
    anom_dict = mark_anoms()

    last_time = 1
    cur_time = 0

    f_in = open(FLOWS_SRC, 'r')
    flows_folder = os.path.join(DST, 'flows')
    if not os.path.exists(flows_folder):
        os.makedirs(flows_folder)
        print(flows_folder + " is created!")

    f_out = open(os.path.join(flows_folder, str(cur_time) + '.txt'), 'w')

    line = f_in.readline()

    nmap = reverse_load_map('nmap.pkl')

    port_map = {}
    port_id = [0]
    proto_map = {}
    proto_id = [0]

    prog = tqdm(desc='Seconds parsed', total=3126928)

    fmt_label = lambda ts,src,dst :\
        1 if is_anomalous_range(anom_dict, src, dst, ts)\
        else 0

    fmt_line = lambda ts,duration,src,src_p,dst,dst_p,proto,pkt_cnt,byte_cnt: (
        '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (
            ts, nmap[src], nmap[dst],get_or_add(src_p, port_map, port_id),
            get_or_add(dst_p, port_map, port_id), get_or_add(proto, proto_map, proto_id),
            duration, pkt_cnt, byte_cnt, fmt_label(int(ts),src,dst)
        ),
        int(ts)
    )

    while line:
        if '?' in line:
            line = f_in.readline()
            continue

        tokens = line.split(',')

        if not tokens[2] in nmap or not tokens[4] in nmap:
            line = f_in.readline()
            continue

        l, ts = fmt_line(tokens[0], tokens[1], tokens[2], tokens[3], tokens[4], tokens[5], tokens[6], tokens[7], tokens[8][:-1])

        if ts != last_time:
            prog.update(ts-last_time)
            last_time = ts

        while ts >= cur_time + DELTA:
            f_out.close()
            cur_time += DELTA
            f_out = open(os.path.join(flows_folder, str(cur_time) + '.txt'), 'w')

        f_out.write(l)
        line = f_in.readline()

    f_out.close()
    f_in.close()

    save_map(port_map, 'pomap.pkl')
    save_map(proto_map, 'prmap.pkl')

if __name__ == '__main__':
    split_auth()
    split_flows()
