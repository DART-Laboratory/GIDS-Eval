from copy import deepcopy
import os
import pickle
from joblib import Parallel, delayed

import torch

from tqdm import tqdm

from .tdata import TData
from .load_utils import edge_tv_split, std_edge_w, standardized, std_edge_a

import numpy as np

DATE_OF_EVIL_LANL = 573290
FILE_DELTA = 10000

LANL_FOLDER = os.path.join(os.environ.get("ARGUS_DATA_ROOT") or os.path.join(os.environ.get("MIRAGE_INPUT_ROOT", ""), "argus"), "optc") + os.sep

def _load_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)

TIMES = {
    '20'      : 573383,
    '100'     : 573751,
    '500'     : 575885,
    'all'  : 745983,
    'test' :  745983
}

def empty_lanl(use_flows=False):
    return make_data_obj(None,[],None,None,None,use_flows=use_flows)

def load_optc_dist(start=0, end=635015, delta=8640, is_test=False, use_flows=False, ew_fn=std_edge_w, ea_fn=std_edge_a):
    if start == None or end == None:
        return empty_lanl(use_flows)

    num_slices = ((end - start) // delta)
    remainder = (end-start) % delta
    num_slices = num_slices + 1 if remainder else num_slices

    return load_partial_lanl(start, end, delta, is_test, use_flows, ew_fn, ea_fn)

def load_partial_lanl_job(pid, args):
    data = load_partial_lanl(**args)
    return data

def make_data_obj(cur_slice, eis, ys, ew_fn, ea_fn, ews=None, eas=None, use_flows=False, campaigns=None, **kwargs):
    if 'node_map' in kwargs:
        nm = kwargs['node_map']
    else:
        nm = _load_pickle(LANL_FOLDER + 'nmap.pkl')

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
        cur_slice,
        eis_t,
        x,
        ys,
        masks,
        ews=ews,
        eas=eas,
        use_flows=use_flows,
        cnt=cnt,
        node_map=nm,
        campaigns=campaigns or [],
    )

'''
Read a file in flows and return the edge features

'''
def load_flows(fname, start, end):

    eas_flows = {}
    temp_flows = {}
    if not os.path.exists(fname):
        return eas_flows
    in_f = open(fname)
    line = in_f.readline()

    fmt_line = lambda x : (int(x[0]), int(x[1]), int(x[2]), int(x[6]), int(x[7]), int(x[8]))

    while line:
        l = line.split(',')
        ts = int(l[0])
        if ts < start:
            line = in_f.readline()
            continue
        if ts > end:
            break
        ts, src, dst, duration, pck_cnt, byte_cnt = fmt_line(l)
        et = (src,dst)
        if et in temp_flows:
            temp_flows[et][0].append(duration)
            temp_flows[et][1].append(pck_cnt)
            temp_flows[et][2].append(byte_cnt)
        else:
            temp_flows[et] = [[duration], [pck_cnt], [byte_cnt]]
        line = in_f.readline()
    in_f.close()

    for et in temp_flows.keys():
        eas_flows[et] = [len(temp_flows[et][0]), np.mean(temp_flows[et][0]), np.std(temp_flows[et][0]),\
        np.mean(temp_flows[et][1]), np.std(temp_flows[et][1]), np.mean(temp_flows[et][2]), np.std(temp_flows[et][2])]
    return eas_flows

'''
Equivilant to load_cyber.load_lanl but uses the sliced LANL files
for faster scanning to the correct lines
'''
def load_partial_lanl(start=140000, end=156659, delta=8640, is_test=False, use_flows=False, ew_fn=standardized, ea_fn=std_edge_a):
    print('start:' + str(start) + ', end:' + str(end))
    cur_slice = int(start - (start % FILE_DELTA))
    start_f = str(cur_slice) + '.txt'
    in_f = open(LANL_FOLDER + start_f, 'r')

    edges = []
    ews = []
    edges_t = {}
    ys = []
    slices = []
    eas = []
    campaigns = []

    node_map = _load_pickle(LANL_FOLDER + 'nmap.pkl')

    fmt_line = lambda x : (
        int(x[0]),
        int(x[1]),
        int(x[2]),
        int(x[3]),
        int(x[4]),
        int(x[5]),
        int(x[6]),
        int(x[7]),
        int(x[8].strip()),
        x[9].strip() if len(x) > 9 else "",
    )

    def add_edge(et, ea, is_anom=0):
        if et in edges_t:
            val = edges_t[et]
            edges_t[et] = (max(is_anom, val[0]), val[1]+1, ea)
        else:
            edges_t[et] = (is_anom, 1, ea)

    def flush_edges(flush_ts):
        nonlocal edges_t
        if not edges_t:
            return
        edge_keys = list(edges_t.keys())
        ei = list(zip(*[(src, dst) for src, dst, _ in edge_keys]))
        edges.append(ei)
        campaigns.append([campaign for _, _, campaign in edge_keys])

        y, ew, ea = list(zip(*edges_t.values()))
        ews.append(torch.tensor(ew))

        if use_flows:
            ea = [list(elem) if len(elem) == 5 else list(elem[0]) for elem in ea]
            ea = np.array(ea)
            if ea.ndim == 3:
                ea = ea[0]
            eas.append(torch.tensor(ea).transpose(1, 0))
        if is_test:
            ys.append(torch.tensor(y))

        slices.append(str(flush_ts))
        edges_t = {}

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

            ts, src, dst, pid, ppid, dest_port, l4protocol, img_path, label, campaign = fmt_line(l)
            ea = (int(pid), int(ppid), int(dest_port), int(l4protocol), int(img_path))

            et = (src, dst, campaign if label == 1 else "")

            prog.update(ts-old_ts)
            old_ts = ts

            if ts >= next_split:
                flush_edges(ts)

                curtime = next_split
                next_split += delta

                if ts >= end:
                    keep_reading = False
                    break

            if et[0] == et[1]:
                line = in_f.readline()
                continue
            add_edge(et, ea, is_anom=label)

            line = in_f.readline()

        in_f.close()
        cur_slice += FILE_DELTA

        if os.path.exists(LANL_FOLDER + str(cur_slice) + '.txt'):
            in_f = open(LANL_FOLDER + str(cur_slice) + '.txt', 'r')
            line = in_f.readline()
        else:
            keep_reading=False
            break

    flush_edges(min(old_ts, end))
    ys = ys if is_test else None

    scan_prog.close()
    prog.close()
    return make_data_obj(
        slices, edges, ys, ew_fn, ea_fn,
        ews=ews, eas=eas, use_flows=use_flows, node_map=node_map, campaigns=campaigns
    )
