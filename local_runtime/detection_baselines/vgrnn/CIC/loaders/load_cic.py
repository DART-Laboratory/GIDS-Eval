from copy import deepcopy
import os
import pickle
from bisect import bisect_right
from joblib import Parallel, delayed

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from .tdata import TData
from .load_utils import edge_tv_split, std_edge_w, standardized

DATE_OF_EVIL_LANL = 29136
FILE_DELTA = 100000

DATA_ROOT = os.environ.get("VGRNN_DATA_ROOT") or os.path.join(os.environ.get("MIRAGE_INPUT_ROOT", ""), "vgrnn")
LANL_FOLDER = os.path.join(DATA_ROOT, 'cic') + os.sep
assert LANL_FOLDER, 'Please fill in the LANL_FOLDER variable: /loaders/load_cic.py'

AVAILABLE_SLICES = sorted([
    int(filename.split('.')[0])
    for filename in os.listdir(LANL_FOLDER)
    if filename.endswith('.txt') and filename.split('.')[0].isdigit()
]) if os.path.isdir(LANL_FOLDER) else []

COMP = 0
USR = 1
SPEC = 2
X_DIM = 17688

TIMES = {

    'all'  : 374762
}

torch.set_num_threads(1)

def empty_lanl():
    return make_data_obj([],None,None)

def _slice_index_for_ts(timestamp):
    if not AVAILABLE_SLICES:
        return None
    idx = bisect_right(AVAILABLE_SLICES, int(timestamp)) - 1
    return max(idx, 0)

def load_lanl_dist(workers, start=0, end=635015, delta=8640, is_test=False, ew_fn=std_edge_w):
    if start == None or end == None:
        return empty_lanl()

    num_slices = ((end - start) // delta)
    remainder = (end-start) % delta
    num_slices = num_slices + 1 if remainder else num_slices
    workers = min(num_slices, workers)

    if workers <= 1:
        return load_partial_lanl(start, end, delta, is_test, ew_fn)

    per_worker = [num_slices // workers] * workers
    remainder = num_slices % workers

    if remainder:
        for i in range(workers, workers-remainder, -1):
            per_worker[i-1] += 1

    kwargs = []
    prev = start
    for i in range(workers):
        end_t = prev + delta*per_worker[i]
        kwargs.append({
            'start': prev,
            'end': min(end_t-1, end),
            'delta': delta,
            'is_test': is_test,
            'ew_fn': ew_fn
        })
        prev = end_t

    datas = Parallel(n_jobs=workers, prefer='processes')(
        delayed(load_partial_lanl_job)(i, kwargs[i]) for i in range(workers)
    )

    data_reduce = lambda x : sum([getattr(datas[i], x) for i in range(workers)], [])

    print("Joining Data objects")
    x = datas[0].xs
    eis = data_reduce('eis')
    masks = data_reduce('masks')
    ews = data_reduce('ews')
    node_map = datas[0].node_map
    campaigns = data_reduce('campaigns') if hasattr(datas[0], 'campaigns') else []

    if is_test:
        ys = data_reduce('ys')
        cnt = data_reduce('cnt')
    else:
        ys = None
        cnt = None

    print("Done")
    return TData(
        eis, x, ys, masks, ews=ews, node_map=node_map, cnt=cnt, campaigns=campaigns
    )

def load_partial_lanl_job(pid, args):
    data = load_partial_lanl(**args)
    return data

def make_data_obj(eis, ys, ew_fn, ews=None, campaigns=None, **kwargs):
    if 'node_map' in kwargs:
        nm = kwargs['node_map']
    else:
        with open(LANL_FOLDER + 'nmap.pkl', 'rb') as handle:
            nm = pickle.load(handle)

    cl_cnt = len(nm)
    x = torch.eye(cl_cnt+1)

    eis_t = []
    masks = []

    for i in range(len(eis)):
        ei = torch.tensor(eis[i])
        eis_t.append(ei)

        if isinstance(ys, None.__class__):
            masks.append(edge_tv_split(ei)[0])

    if not isinstance(ews, None.__class__):
        cnt = deepcopy(ews)
        ews = ew_fn(ews)
    else:
        cnt = None

    return TData(
        eis_t, x, ys, masks, ews=ews, cnt=cnt, node_map=nm, campaigns=campaigns or []
    )

'''
Equivilant to load_cyber.load_lanl but uses the sliced LANL files
for faster scanning to the correct lines
'''
def load_partial_lanl(start=140000, end=156659, delta=8640, is_test=False, ew_fn=standardized):
    start_idx = _slice_index_for_ts(start)
    if start_idx is None:
        return empty_lanl()

    cur_idx = start_idx
    cur_slice = AVAILABLE_SLICES[cur_idx]
    if cur_slice > end:
        return empty_lanl()
    start_f = str(cur_slice) + '.txt'
    in_f = open(LANL_FOLDER + start_f, 'r')

    edges = []
    ews = []
    edges_t = {}
    ys = []
    campaigns = []

    with open(LANL_FOLDER + 'nmap.pkl', 'rb') as handle:
        node_map = pickle.load(handle)

    fmt_line = lambda x : (
        int(x[0]),
        int(x[1]),
        int(x[2]),
        int(x[6].strip()),
        x[7].strip() if len(x) > 7 else "",
    )

    def add_edge(et, is_anom=0):
        if et in edges_t:
            val = edges_t[et]
            edges_t[et] = (max(is_anom, val[0]), val[1]+1)
        else:
            edges_t[et] = (is_anom, 1)

    scan_prog = tqdm(desc='Finding start', total=start-cur_slice-1)
    prog = tqdm(desc='Seconds read', total=end-start-1)

    anom_marked = False
    keep_reading = True
    next_split = start+delta

    line = in_f.readline()
    curtime = fmt_line(line.split(','))[0]
    old_ts = curtime
    while keep_reading:
        while line:
            l = line.split(',')

            ts = int(l[0])
            if ts < start:
                line = in_f.readline()
                scan_prog.update(ts-old_ts)
                old_ts = ts
                curtime = ts
                continue

            ts, src, dst, label, campaign = fmt_line(l)
            et = (src, dst)

            prog.update(ts-old_ts)
            old_ts = ts

            if ts >= next_split:
                if len(edges_t):
                    edge_keys = list(edges_t.keys())
                    ei = list(zip(*edge_keys))
                    edges.append(ei)
                    campaigns.append(["" for _ in edge_keys])

                    y,ew = list(zip(*edges_t.values()))
                    ews.append(torch.tensor(ew))

                    if is_test:
                        ys.append(torch.tensor(y))
                    edges_t = {}

                curtime = next_split
                next_split += delta

                if ts >= end:
                    keep_reading = False
                    break

            if et[0] == et[1]:
                line = in_f.readline()
                continue

            add_edge(et, is_anom=label)
            line = in_f.readline()

        in_f.close()
        cur_idx += 1
        if cur_idx < len(AVAILABLE_SLICES):
            cur_slice = AVAILABLE_SLICES[cur_idx]
            in_f = open(LANL_FOLDER + str(cur_slice) + '.txt', 'r')
            line = in_f.readline()
        else:
            keep_reading=False
            break

    ys = ys if is_test else None

    scan_prog.close()
    prog.close()

    return make_data_obj(
        edges, ys, ew_fn,
        ews=ews, node_map=node_map, campaigns=campaigns
    )

if __name__ == '__main__':
    data = load_lanl_dist(2, start=0, end=21600, delta=21600)
    print(data)
