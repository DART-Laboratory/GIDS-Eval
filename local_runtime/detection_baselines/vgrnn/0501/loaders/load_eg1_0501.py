from copy import deepcopy
import os
import pickle
from joblib import Parallel, delayed

import torch

from .tdata import TData
from .load_utils import edge_tv_split, std_edge_w, standardized

DATE_OF_EVIL_LANL = 28800000
FILE_DELTA = 100000

DATA_ROOT = os.environ.get("VGRNN_DATA_ROOT") or os.path.join(os.environ.get("MIRAGE_INPUT_ROOT", ""), "vgrnn")
LANL_FOLDER = os.path.join(DATA_ROOT, "0501", "0501") + os.sep

TIMES = {
    "all": 86400000,
}

def empty_lanl():
    return make_data_obj([], None, None)

def load_lanl_dist(workers, start=0, end=635015, delta=8640, is_test=False, ew_fn=std_edge_w):
    if start is None or end is None:
        return empty_lanl()

    num_slices = ((end - start) // delta)
    remainder = (end - start) % delta
    num_slices = num_slices + 1 if remainder else num_slices
    workers = min(num_slices, workers)

    if workers <= 1:
        return load_partial_lanl(start, end, delta, is_test, ew_fn)

    per_worker = [num_slices // workers] * workers
    remainder = num_slices % workers
    if remainder:
        for i in range(workers, workers - remainder, -1):
            per_worker[i - 1] += 1

    kwargs = []
    prev = start
    for i in range(workers):
        end_t = prev + delta * per_worker[i]
        kwargs.append(
            {
                "start": prev,
                "end": min(end_t - 1, end),
                "delta": delta,
                "is_test": is_test,
                "ew_fn": ew_fn,
            }
        )
        prev = end_t

    datas = Parallel(n_jobs=workers, prefer="processes")(
        delayed(load_partial_lanl_job)(i, kwargs[i]) for i in range(workers)
    )

    data_reduce = lambda key: sum([(getattr(datas[i], key, None) or []) for i in range(workers)], [])
    x = datas[0].xs
    eis = data_reduce("eis")
    masks = data_reduce("masks")
    ews = data_reduce("ews")
    node_map = datas[0].node_map
    campaigns = data_reduce("campaigns") if is_test else None

    if is_test:
        ys = data_reduce("ys")
        cnt = data_reduce("cnt")
    else:
        ys = None
        cnt = None

    return TData(
        eis, x, ys, masks, ews=ews, node_map=node_map, cnt=cnt, campaigns=campaigns
    )

def load_partial_lanl_job(pid, args):
    return load_partial_lanl(**args)

def make_data_obj(eis, ys, ew_fn, ews=None, campaigns=None, **kwargs):
    if "node_map" in kwargs:
        nm = kwargs["node_map"]
    else:
        nm = pickle.load(open(LANL_FOLDER + "nmap.pkl", "rb"))

    cl_cnt = len(nm)
    x = torch.eye(cl_cnt + 1)
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
        eis_t, x, ys, masks, ews=ews, cnt=cnt, node_map=nm, campaigns=campaigns
    )

def load_partial_lanl(start=140000, end=156659, delta=8640, is_test=False, ew_fn=standardized):
    cur_slice = int(start - (start % FILE_DELTA))
    start_f = str(cur_slice) + ".txt"
    in_f = open(LANL_FOLDER + start_f, "r")

    edges = []
    ews = []
    edges_t = {}
    ys = []
    campaigns = []

    node_map = pickle.load(open(LANL_FOLDER + "nmap.pkl", "rb"))

    def parse_line(raw):
        if not raw:
            return None
        parts = raw.strip().split(",")
        if len(parts) < 4:
            return None
        if parts[0] == "" or parts[1] == "" or parts[2] == "" or parts[3] == "":
            return None
        try:
            return (
                int(float(parts[0])),
                int(float(parts[1])),
                int(float(parts[2])),
                int(float(parts[3])),
                parts[4] if len(parts) > 4 else "",
            )
        except ValueError:
            return None

    def add_edge(et, is_anom=0, campaign=""):
        current_campaign = campaign if is_anom else ""
        if et in edges_t:
            old_label, old_count, old_campaign = edges_t[et]
            edges_t[et] = (max(is_anom, old_label), old_count + 1, old_campaign or current_campaign)
        else:
            edges_t[et] = (is_anom, 1, current_campaign)

    keep_reading = True
    next_split = start + delta

    line = in_f.readline()
    parsed = parse_line(line)
    while line and parsed is None:
        line = in_f.readline()
        parsed = parse_line(line)
    if not line:
        in_f.close()
        return empty_lanl()

    while keep_reading:
        while line:
            parsed = parse_line(line)
            if parsed is None:
                line = in_f.readline()
                continue
            ts, src, dst, label, campaign = parsed
            if ts < start:
                line = in_f.readline()
                continue

            if ts >= next_split:
                if edges_t:
                    ei = list(zip(*edges_t.keys()))
                    edges.append(ei)
                    y, ew, cvals = list(zip(*edges_t.values()))
                    ews.append(torch.tensor(ew))
                    if is_test:
                        ys.append(torch.tensor(y))
                        campaigns.append(list(cvals))
                    edges_t = {}

                next_split += delta
                if ts >= end:
                    keep_reading = False
                    break

            if src != dst:
                add_edge((src, dst), is_anom=label, campaign=campaign)
            line = in_f.readline()

        in_f.close()
        cur_slice += FILE_DELTA
        next_path = LANL_FOLDER + str(cur_slice) + ".txt"
        if os.path.exists(next_path):
            in_f = open(next_path, "r")
            line = in_f.readline()
        else:
            keep_reading = False

    if edges_t:
        ei = list(zip(*edges_t.keys()))
        edges.append(ei)
        y, ew, cvals = list(zip(*edges_t.values()))
        ews.append(torch.tensor(ew))
        if is_test:
            ys.append(torch.tensor(y))
            campaigns.append(list(cvals))

    ys = ys if is_test else None
    campaigns = campaigns if is_test else None
    return make_data_obj(edges, ys, ew_fn, ews=ews, node_map=node_map, campaigns=campaigns)
