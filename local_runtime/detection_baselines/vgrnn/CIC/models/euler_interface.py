from copy import deepcopy

import torch
from torch import nn
from torch.distributed import rpc
from torch.nn.parallel import DistributedDataParallel as DDP

from .utils import _remote_method, _remote_method_async, _param_rrefs

class Euler_Embed_Unit(nn.Module):

    def inner_forward(self, mask_enum):
        raise NotImplementedError

    def forward(self, mask_enum, no_grad):
        if no_grad:
            self.eval()
            with torch.no_grad():
                return self.inner_forward(mask_enum)

        return self.inner_forward(mask_enum)

    def decode(self, src, dst, z):
        return torch.sigmoid(
            (z[src] * z[dst]).sum(dim=1)
        )

class Euler_Encoder(DDP):

    def __init__(self, module: Euler_Embed_Unit, **kwargs):
        super().__init__(module, **kwargs)

    def train(self, mode=True):
        self.module.train(mode=mode)

    def load_new_data(self, loader, kwargs):
        print(rpc.get_worker_info().name + ": Reloading %d - %d" % (kwargs['start'], kwargs['end']))

        jobs = kwargs.pop('jobs')
        self.module.data = loader(jobs, **kwargs)
        return True

    def get_data_field(self, field):
        return self.module.data.__getattribute__(field)

    def get_data(self):
        return self.module.data

    def run_arbitrary_fn(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def decode(self, e,z):
        src,dst = e
        return self.module.decode(
            src,dst,z
        )

    def bce(self, t_scores, f_scores):
        EPS = 1e-6
        pos_loss = -torch.log(t_scores+EPS).mean()
        neg_loss = -torch.log(1-f_scores+EPS).mean()

        return (pos_loss + neg_loss) * 0.5

    def calc_loss(self, z, partition, nratio):
        raise NotImplementedError

    def decode_all(self, zs, unsqueeze=False):
        raise NotImplementedError

    def score_edges(self, z, partition, nratio):
        raise NotImplementedError

class Euler_Recurrent(nn.Module):
    def __init__(self, rnn: nn.Module, remote_rrefs: list):
        super(Euler_Recurrent, self).__init__()

        self.gcns = remote_rrefs
        self.rnn = rnn

        self.num_workers = len(self.gcns)
        self.len_from_each = []

        self.cutoff = 0.5

    def forward(self, mask_enum, include_h=False, h0=None, no_grad=False):
        futs = self.encode(mask_enum, no_grad)

        zs = []
        for f in futs:
            z, h0 = self.rnn(
                f.wait(),
                h0, include_h=True
            )
            zs.append(z)

        self.len_from_each = [
            embed.size(0) for embed in zs
        ]
        zs = torch.cat(zs, dim=0)
        self.z_dim = zs.size(-1)

        if include_h:
            return zs, h0
        else:
            return zs

    def encode(self, mask_enum, no_grad):
        embed_futs = []

        for i in range(self.num_workers):
            embed_futs.append(
                _remote_method_async(
                    DDP.forward,
                    self.gcns[i],
                    mask_enum, no_grad
                )
            )

        return embed_futs

    def parameter_rrefs(self):
        params = []
        for rref in self.gcns:
            params.extend(
                _remote_method(
                    _param_rrefs, rref
                )
            )

        params.extend(_param_rrefs(self.rnn))
        return params

    def save_states(self):
        gcn = _remote_method(
            DDP.state_dict, self.gcns[0]
        )

        return gcn, deepcopy(self.state_dict())

    def load_states(self, gcn_state_dict, rnn_state_dict):
        self.load_state_dict(rnn_state_dict)

        jobs = []
        for rref in self.gcns:
            jobs.append(
                _remote_method_async(
                    DDP.load_state_dict, rref,
                    gcn_state_dict
                )
            )

        [j.wait() for j in jobs]

    def train(self, mode=True):
        super(Euler_Recurrent, self).train()
        [_remote_method(
            Euler_Encoder.train,
            self.gcns[i],
            mode=mode
        ) for i in range(self.num_workers)]

    def eval(self):
        super(Euler_Recurrent, self).train(False)
        [_remote_method(
            Euler_Encoder.train,
            self.gcns[i],
            mode=False
        ) for i in range(self.num_workers)]

    def score_all(self, zs, unsqueeze=False):
        raise NotImplementedError

    def loss_fn(self, zs, partition, nratio=1.0):
        raise NotImplementedError

    def score_edges(self, zs, partition, nratio=1):
        raise NotImplementedError
