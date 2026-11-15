from copy import deepcopy
import os
import pickle
from joblib import Parallel, delayed

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from .tdata import TData
from .load_utils import edge_tv_split, std_edge_w, standardized, std_edge_a

import random
import numpy as np

DATE_OF_EVIL_LANL = 72643741
FILE_DELTA = 100000

LANL_FOLDER = os.path.join(os.environ.get("ARGUS_DATA_ROOT") or os.path.join(os.environ.get("MIRAGE_INPUT_ROOT", ""), "argus"), "0501/wizard_spider") + os.sep
assert LANL_FOLDER, 'Please fill in the LANL_FOLDER in loaders/load_wizard_flow.py'

TIMES = {

    'all'  : 86693493,

}

def empty_lanl(use_flows=False):
    return make_data_obj(None,[],None,None,None,use_flows=use_flows)

def load_lanl_dist(start=0, end=635015, delta=8640, is_test=False, use_flows=False, ew_fn=std_edge_w, ea_fn=std_edge_a):
    if start == None or end == None:
        return empty_lanl(use_flows)

    num_slices = ((end - start) // delta)
    remainder = (end-start) % delta
    num_slices = num_slices + 1 if remainder else num_slices
    return load_partial_lanl(start, end, delta, is_test, use_flows, ew_fn, ea_fn)

def load_partial_lanl_job(pid, args):
    data = load_partial_lanl(**args)
    return data

def make_data_obj(cur_slice, eis, ys, ew_fn, ea_fn, ews=None, eas=None, use_flows=False, **kwargs):
    if 'node_map' in kwargs:
        nm = kwargs['node_map']
    else:
        nm = pickle.load(open(LANL_FOLDER+'nmap.pkl', 'rb'))

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

    if not isinstance(eas, None.__class__):
        eas = ea_fn(eas)

    return TData(cur_slice, eis_t, x, ys, masks, ews=ews, eas=eas, use_flows=use_flows, cnt=cnt, node_map=nm)

def load_flows(fname, start, end):
    eas_flows = {}
    temp_flows = {}
    if not os.path.exists(fname):
        return eas_flows
    in_f = open(fname)
    line = in_f.readline()

    fmt_line = lambda x : (int(x[0]), int(x[1]), int(x[2]), int(x[6]), int(x[7]), int(x[8][-1]))

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
    eas_flows = {}
    temp_flows = {}

    node_map = pickle.load(open(LANL_FOLDER+'nmap.pkl', 'rb'))

    fmt_line = lambda x : (int(x[0]), int(x[1]), int(x[2]), float(x[3]),float(x[4]),int(x[5]),int(x[6][:-1]))

    def add_edge(et,  is_anom=0):
        if et in edges_t:
            val = edges_t[et]
            edges_t[et] = [max(is_anom, val[0]), val[1]+1]
        else:
            edges_t[et] = [is_anom, 1]
    def add_flow(et,dur,bytes,pkts):
        if et in temp_flows:
            temp_flows[et][0].append(dur)
            temp_flows[et][1].append(pkts)
            temp_flows[et][2].append(bytes)
        else:
            temp_flows[et] = [[dur], [pkts], [bytes]]

    scan_prog = tqdm(desc='Finding start', total=start-cur_slice-1)
    prog = tqdm(desc='Seconds read', total=end-start-1)

    anom_marked = False
    keep_reading = True
    next_split = start+delta

    line = in_f.readline()
    curtime = fmt_line(line.split(','))[0]
    old_ts = curtime

    i=0
    j=0

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

            ts,src,dst,dur,bytes,pkts,label= fmt_line(l)

            if(label==1):
                i+=1

            et = (src,dst)

            prog.update(ts-old_ts)
            old_ts = ts

            if ts >= next_split:
                if len(edges_t):
                    ei = list(zip(*edges_t.keys()))
                    edges.append(ei)

                    y,ew = list(zip(*edges_t.values()))
                    ews.append(torch.tensor(ew))

                    if use_flows:

                        for et in temp_flows.keys():
                            eas_flows[et] = [ np.mean(temp_flows[et][0]), np.std(temp_flows[et][0]), np.mean(temp_flows[et][1]), np.std(temp_flows[et][1]), np.mean(temp_flows[et][2]), np.std(temp_flows[et][2])]
                        eas_flows_dim = 6

                        fs = {}

                        for eij in edges_t.keys():
                            if eij in eas_flows:
                                fs[eij] = eas_flows[eij]
                            else:
                                fs[eij] = [0] * eas_flows_dim

                        mean_duration,std_duration,mean_pkt_cnt,std_pkt_cnt,mean_byte_cnt,std_byte_cnt = list(zip(*fs.values()))
                        eas.append(torch.tensor([mean_duration,std_duration,mean_pkt_cnt,std_pkt_cnt,mean_byte_cnt,std_byte_cnt]))

                    if is_test:
                        j+=np.sum(list(y))
                        ys.append(torch.tensor(y))

                    slices.append(str(ts))

                    edges_t = {}
                    eas_flows = {}
                    temp_flows = {}

                curtime = next_split
                next_split += delta

                if ts >= end:
                    keep_reading = False
                    break

            if et[0] == et[1]:
                line = in_f.readline()
                continue

            add_flow(et,dur,bytes,pkts)
            add_edge(et, is_anom=label)
            line = in_f.readline()

        in_f.close()
        cur_slice += FILE_DELTA

        if os.path.exists(LANL_FOLDER + str(cur_slice) + '.txt'):
            in_f = open(LANL_FOLDER + str(cur_slice) + '.txt', 'r')
            line = in_f.readline()
        else:
            keep_reading=False
            break
    print('sds',i)
    print('sdsd',j)

    ys = ys if is_test else None

    scan_prog.close()
    prog.close()

    return make_data_obj(slices, edges, ys, ew_fn, ea_fn, ews=ews, eas=eas, use_flows=use_flows, node_map=node_map)
