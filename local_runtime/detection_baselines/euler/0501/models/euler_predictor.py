import os

import torch
from torch.distributed import rpc

from .euler_interface import Euler_Embed_Unit, Euler_Encoder, Euler_Recurrent
from .utils import _remote_method, _remote_method_async

def _worker_name() -> str:
    raw = os.environ.get("EG1_EULER_LOCAL_MODE", "")
    if raw.strip().lower() not in {"", "0", "false", "no"}:
        return "local"
    return rpc.get_worker_info().name

class PredictorEncoder(Euler_Encoder):

    def __init__(self, module: Euler_Embed_Unit, head: bool, **kwargs):
        super().__init__(module, **kwargs)
        self.is_head = 1 if head else 0
        if self.is_head:
            print("%s is head" % _worker_name())

    def decode_all(self, zs, unsqueeze=False):
        preds,ys,cnts = [], [], []
        for i in range(self.is_head, self.module.data.T):
            preds.append(
                self.decode(
                    self.module.data.eis[i],
                    zs[i]
                )
            )

            ys.append(self.module.data.ys[i])
            cnts.append(self.module.data.cnt[i])

        return preds, ys, cnts

    def score_edges(self, z, partition, nratio):

        n = self.module.data.get_negative_edges(
            partition, nratio=nratio, start=self.is_head
        )

        p_scores = []
        n_scores = []

        for i in range(self.is_head, len(z)):
            p = self.module.data.ei_masked(partition, i)
            if p.size(1) == 0:
                continue

            p_scores.append(self.decode(p, z[i]))
            n_scores.append(self.decode(n[i-self.is_head], z[i]))

        p_scores = torch.cat(p_scores, dim=0)
        n_scores = torch.cat(n_scores, dim=0)

        return p_scores, n_scores

    def calc_loss(self, z, partition, nratio):
        tot_loss = torch.zeros(1)
        ns = self.module.data.get_negative_edges(
            partition, nratio=nratio, start=self.is_head
        )

        for i in range(self.is_head, len(z)):

            ps = self.module.data.ei_masked(partition, i)
            if ps.size(1) == 0:
                continue

            tot_loss += self.bce(
                self.decode(ps, z[i]),
                self.decode(ns[i-self.is_head], z[i])
            )

        return tot_loss.true_divide(len(z))

class PredictorRecurrent(Euler_Recurrent):

    def score_all(self, zs, unsqueeze=False):
        zs = torch.cat(
            [torch.zeros(zs[0].size()).unsqueeze(0), zs]
        )

        futs = []
        start = 0

        for i in range(self.num_workers):
            end = start + self.len_from_each[i]
            futs.append(
                _remote_method_async(
                    PredictorEncoder.decode_all,
                    self.gcns[i],
                    zs[start : end],
                    unsqueeze=unsqueeze
                )
            )
            start = end

        obj = [f.wait() for f in futs]
        scores, ys, cnts = zip(*obj)

        scores = sum(scores, [])
        ys = sum(ys, [])
        cnts = sum(cnts, [])

        return scores, ys, cnts

    def loss_fn(self, zs, partition, nratio=1):

        zs = torch.cat(
            [torch.zeros(zs[0].size()).unsqueeze(0), zs]
        )

        futs = []
        start = 0

        for i in range(self.num_workers):
            end = start + self.len_from_each[i]
            futs.append(
                _remote_method_async(
                    PredictorEncoder.calc_loss,
                    self.gcns[i],
                    zs[start : end],
                    partition, nratio
                )
            )
            start = end

        tot_loss = torch.zeros(1)
        for f in futs:
            tot_loss += f.wait()

        return [tot_loss.true_divide(sum(self.len_from_each))]

    def score_edges(self, zs, partition, nratio=1):

        zs = torch.cat(
            [torch.zeros(zs[0].size()).unsqueeze(0), zs]
        )

        futs = []
        start = 0

        for i in range(self.num_workers):
            end = start + self.len_from_each[i]
            futs.append(
                _remote_method_async(
                    PredictorEncoder.score_edges,
                    self.gcns[i],
                    zs[start : end],
                    partition, nratio
                )
            )
            start = end

        pos, neg = zip(*[f.wait() for f in futs])
        return torch.cat(pos, dim=0), torch.cat(neg, dim=0)
