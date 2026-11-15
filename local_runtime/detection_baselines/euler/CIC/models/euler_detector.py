import os

import torch
from torch.distributed import rpc

from .euler_interface import Euler_Encoder, Euler_Recurrent
from .utils import _remote_method, _remote_method_async

try:
    from gids_eval.cdr import build_edge_campaigns
except Exception:
    build_edge_campaigns = None

def _worker_name() -> str:
    raw = os.environ.get("EG1_EULER_LOCAL_MODE", "")
    if raw.strip().lower() not in {"", "0", "false", "no"}:
        return "local"
    return rpc.get_worker_info().name

class DetectorEncoder(Euler_Encoder):

    def decode_all(self, zs, unsqueeze=False):
        assert not zs.size(0) < self.module.data.T,\
            "%s was given fewer embeddings than it has time slices"\
            % _worker_name()

        assert not zs.size(0) > self.module.data.T,\
            "%s was given more embeddings than it has time slices"\
            % _worker_name()

        preds = []
        ys = []
        cnts = []
        campaigns = []
        dataset = os.environ.get("GIDS_JOB_DATASET", "dataset")
        dataset_campaigns = getattr(self.module.data, "campaigns", [])
        for i in range(self.module.data.T):
            preds.append(
                self.decode(self.module.data.eis[i], zs[i])
            )
            ys.append(self.module.data.ys[i])
            cnts.append(self.module.data.cnt[i])
            if dataset_campaigns and i < len(dataset_campaigns):
                campaigns.append(list(dataset_campaigns[i]))
            else:
                campaigns.append(
                    build_edge_campaigns(
                        dataset=dataset,
                        edge_index=self.module.data.eis[i],
                        node_map=getattr(self.module.data, "nmap", None),
                    )
                    if build_edge_campaigns
                    else []
                )

        return preds, ys, cnts, campaigns

    def score_edges(self, z, partition, nratio):
        n = self.module.data.get_negative_edges(partition, nratio)

        p_scores = []
        n_scores = []

        for i in range(len(z)):
            p = self.module.data.ei_masked(partition, i)
            if p.size(1) == 0:
                continue

            p_scores.append(self.decode(p, z[i]))
            n_scores.append(self.decode(n[i], z[i]))

        if not p_scores or not n_scores:
            device = z.device if hasattr(z, "device") else None
            empty = torch.empty((0,), device=device)
            return empty, empty

        p_scores = torch.cat(p_scores, dim=0)
        n_scores = torch.cat(n_scores, dim=0)

        return p_scores, n_scores

    def calc_loss(self, z, partition, nratio):
        device = z[0].device if len(z) else None
        tot_loss = torch.zeros(1, device=device)
        if len(z) == 0:
            return tot_loss
        ns = self.module.data.get_negative_edges(partition, nratio)

        for i in range(len(z)):
            ps = self.module.data.ei_masked(partition, i)

            if ps.size(1) == 0:
                continue

            tot_loss += self.bce(
                self.decode(ps, z[i]),
                self.decode(ns[i], z[i])
            )

        return tot_loss.true_divide(len(z))

class DetectorRecurrent(Euler_Recurrent):
    def score_all(self, zs, unsqueeze=False):
        futs = []
        for i in range(self.num_workers):
            length = self.len_from_each[i] if i < len(self.len_from_each) else 0
            start = getattr(self, "slice_map", [None]*self.num_workers)[i] if i < len(getattr(self, "slice_map", [])) else None
            if length == 0 or start is None:
                continue
            end = start + length
            futs.append(
                _remote_method_async(
                    DetectorEncoder.decode_all,
                    self.gcns[i],
                    zs[start : end],
                    unsqueeze=unsqueeze
                )
            )

        if not futs:
            return [], [], [], []
        obj = [f.wait() for f in futs]
        scores, ys, cnts, campaigns = zip(*obj)

        scores = sum(scores, [])
        ys = sum(ys, [])
        cnts = sum(cnts, [])
        campaigns = sum(campaigns, [])

        return scores, ys, cnts, campaigns

    def loss_fn(self, zs, partition, nratio=1):
        futs = []
        for i in range(self.num_workers):
            length = self.len_from_each[i] if i < len(self.len_from_each) else 0
            start = getattr(self, "slice_map", [None]*self.num_workers)[i] if i < len(getattr(self, "slice_map", [])) else None
            if length == 0 or start is None:
                continue
            end = start + length
            futs.append(
                _remote_method_async(
                    DetectorEncoder.calc_loss,
                    self.gcns[i],
                    zs[start : end],
                    partition, nratio
                )
            )

        if not futs:
            device = zs.device if isinstance(zs, torch.Tensor) else None
            return [torch.zeros(1, device=device)]

        first = futs[0].wait()
        tot_loss = torch.zeros(1, device=first.device)
        tot_loss += first
        for f in futs[1:]:
            tot_loss += f.wait()

        return [tot_loss.true_divide(len(futs))]

    def score_edges(self, zs, partition, nratio=1):
        futs = []
        for i in range(self.num_workers):
            length = self.len_from_each[i] if i < len(self.len_from_each) else 0
            start = getattr(self, "slice_map", [None]*self.num_workers)[i] if i < len(getattr(self, "slice_map", [])) else None
            if length == 0 or start is None:
                continue
            end = start + length
            futs.append(
                _remote_method_async(
                    DetectorEncoder.score_edges,
                    self.gcns[i],
                    zs[start : end],
                    partition, nratio
                )
            )

        if not futs:
            device = zs.device if isinstance(zs, torch.Tensor) else None
            empty = torch.empty((0,), device=device)
            return empty, empty
        pos, neg = zip(*[f.wait() for f in futs])
        return torch.cat(pos, dim=0), torch.cat(neg, dim=0)
