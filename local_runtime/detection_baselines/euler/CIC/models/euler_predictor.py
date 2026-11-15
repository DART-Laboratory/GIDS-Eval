import torch
from torch.distributed import rpc

from .euler_interface import Euler_Embed_Unit, Euler_Encoder, Euler_Recurrent
from .utils import _remote_method, _remote_method_async

class PredictorEncoder(Euler_Encoder):

    def __init__(self, module: Euler_Embed_Unit, head: bool, **kwargs):
        super().__init__(module, **kwargs)
        self.is_head = 1 if head else 0
        if self.is_head:
            print("%s is head" % rpc.get_worker_info().name)

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
        for i in range(self.num_workers):
            length = self.len_from_each[i] if i < len(self.len_from_each) else 0
            start_raw = getattr(self, "slice_map", [None]*self.num_workers)[i] if i < len(getattr(self, "slice_map", [])) else None
            if length == 0 or start_raw is None:
                continue
            start = start_raw + 1
            end = start + length
            futs.append(
                _remote_method_async(
                    PredictorEncoder.decode_all,
                    self.gcns[i],
                    zs[start : end],
                    unsqueeze=unsqueeze
                )
            )

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
        for i in range(self.num_workers):
            length = self.len_from_each[i] if i < len(self.len_from_each) else 0
            start_raw = getattr(self, "slice_map", [None]*self.num_workers)[i] if i < len(getattr(self, "slice_map", [])) else None
            if length == 0 or start_raw is None:
                continue
            start = start_raw + 1
            end = start + length
            futs.append(
                _remote_method_async(
                    PredictorEncoder.calc_loss,
                    self.gcns[i],
                    zs[start : end],
                    partition, nratio
                )
            )

        tot_loss = torch.zeros(1)
        for f in futs:
            tot_loss += f.wait()

        return [tot_loss.true_divide(sum(self.len_from_each))]

    def score_edges(self, zs, partition, nratio=1):

        zs = torch.cat(
            [torch.zeros(zs[0].size()).unsqueeze(0), zs]
        )

        futs = []
        for i in range(self.num_workers):
            length = self.len_from_each[i] if i < len(self.len_from_each) else 0
            start_raw = getattr(self, "slice_map", [None]*self.num_workers)[i] if i < len(getattr(self, "slice_map", [])) else None
            if length == 0 or start_raw is None:
                continue
            start = start_raw + 1
            end = start + length
            futs.append(
                _remote_method_async(
                    PredictorEncoder.score_edges,
                    self.gcns[i],
                    zs[start : end],
                    partition, nratio
                )
            )

        pos, neg = zip(*[f.wait() for f in futs])
        return torch.cat(pos, dim=0), torch.cat(neg, dim=0)
