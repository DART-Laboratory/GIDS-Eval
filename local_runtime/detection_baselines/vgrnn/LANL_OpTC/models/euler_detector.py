import torch
from torch.distributed import rpc

from .euler_interface import Euler_Encoder, Euler_Recurrent
from .utils import _remote_method, _remote_method_async

class DetectorEncoder(Euler_Encoder):

    def decode_all(self, zs, unsqueeze=False):
        assert not zs.size(0) < self.module.data.T,\
            "%s was given fewer embeddings than it has time slices"\
            % rpc.get_worker_info().name

        assert not zs.size(0) > self.module.data.T,\
            "%s was given more embeddings than it has time slices"\
            % rpc.get_worker_info().name

        preds = []
        ys = []
        cnts = []
        for i in range(self.module.data.T):
            preds.append(
                self.decode(self.module.data.eis[i], zs[i])
            )
            ys.append(self.module.data.ys[i])
            cnts.append(self.module.data.cnt[i])

        return preds, ys, cnts

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

        p_scores = torch.cat(p_scores, dim=0)
        n_scores = torch.cat(n_scores, dim=0)

        return p_scores, n_scores

    def calc_loss(self, z, partition, nratio):
        tot_loss = torch.zeros(1)
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
        start = 0

        for i in range(self.num_workers):
            end = start + self.len_from_each[i]
            futs.append(
                _remote_method_async(
                    DetectorEncoder.decode_all,
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
        futs = []
        start = 0

        for i in range(self.num_workers):
            end = start + self.len_from_each[i]
            futs.append(
                _remote_method_async(
                    DetectorEncoder.calc_loss,
                    self.gcns[i],
                    zs[start : end],
                    partition, nratio
                )
            )
            start = end

        tot_loss = torch.zeros(1)
        for f in futs:
            tot_loss += f.wait()

        return [tot_loss.true_divide(self.num_workers)]

    def score_edges(self, zs, partition, nratio=1):
        futs = []
        start = 0

        for i in range(self.num_workers):
            end = start + self.len_from_each[i]
            futs.append(
                _remote_method_async(
                    DetectorEncoder.score_edges,
                    self.gcns[i],
                    zs[start : end],
                    partition, nratio
                )
            )
            start = end

        pos, neg = zip(*[f.wait() for f in futs])
        return torch.cat(pos, dim=0), torch.cat(neg, dim=0)
