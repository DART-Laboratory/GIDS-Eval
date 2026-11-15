import torch
from torch import nn
from torch.nn import functional as F
from torch.distributed import rpc
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv.message_passing import MessagePassing

from .euler_interface import Euler_Embed_Unit
from .euler_detector import DetectorEncoder
from .euler_predictor import PredictorEncoder

class DropEdge(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, ei, ew=None):
        if self.training and self.p > 0:
            mask = torch.rand(ei.size(1))
            if ew is None:
                return ei[:, mask > self.p]
            else:
                return ei[:, mask > self.p], ew[mask > self.p]

        if ew is None:
            return ei
        else:
            return ei, ew

class GCN(Euler_Embed_Unit):

    def __init__(self, data_load, data_kws, h_dim, z_dim):
        super(GCN, self).__init__()

        print("%s loading %s-%s" % (
            rpc.get_worker_info().name,
            str(data_kws['start']),
            str(data_kws['end']))
        )

        self.data = data_load(data_kws.pop("jobs"), **data_kws)

        self.c1 = GCNConv(self.data.x_dim, h_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(h_dim, z_dim, add_self_loops=True)
        self.drop = nn.Dropout(0.25)
        self.tanh = nn.Tanh()
        self.de = DropEdge(0.8)

    def inner_forward(self, mask_enum):
        zs = []

        for i in range(self.data.T):

            zs.append(

                self.forward_once(mask_enum, i)
            )

        return torch.stack(zs)

    def forward_once(self, mask_enum, i):
        if self.data.dynamic_feats:
            x = self.data.xs[i]
        else:
            x = self.data.xs

        ei = self.data.ei_masked(mask_enum, i)
        ew = self.data.ew_masked(mask_enum, i)

        ei, ew = self.de(ei, ew=ew)

        x = self.c1(x, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c2(x, ei, edge_weight=ew)

        return self.tanh(x)

def detector_gcn_rref(loader, kwargs, h_dim, z_dim, **kws):
    return DetectorEncoder(
        GCN(loader, kwargs, h_dim, z_dim)
    )

def predictor_gcn_rref(loader, kwargs, h_dim, z_dim, head=False):
    return PredictorEncoder(
        GCN(loader, kwargs, h_dim, z_dim), head
    )

class GAT(GCN):

    def __init__(self, data_load, data_kws, h_dim, z_dim, heads=3):
        super().__init__(data_load, data_kws, h_dim, z_dim)

        self.c1 = GATConv(self.data.x_dim, h_dim, heads=heads, concat=False)
        self.c2 = GATConv(h_dim, z_dim, heads=heads, concat=False)

    def forward_once(self, mask_enum, i):
        if self.data.dynamic_feats:
            x = self.data.xs[i]
        else:
            x = self.data.xs

        ei = self.data.ei_masked(mask_enum, i)
        ei = self.de(ei)

        x = self.c1(x, ei)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c2(x, ei)

        return self.tanh(x)

def detector_gat_rref(loader, kwargs, h_dim, z_dim, **kws):
    return DetectorEncoder(
        GAT(loader, kwargs, h_dim, z_dim)
    )

def predictor_gat_rref(loader, kwargs, h_dim, z_dim, head=False):
    return PredictorEncoder(
        GAT(loader, kwargs, h_dim, z_dim), head
    )

class PoolSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max')

        self.aggr_n = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU()
        )
        self.e_lin = nn.Linear(out_channels, out_channels)
        self.r_lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, ei):
        x_e = self.aggr_n(x)
        x_e = self.propagate(ei, x=x_e, size=None)
        x_e = self.e_lin(x_e)

        x_r = self.r_lin(x)

        x = x_r + x_e
        x = F.normalize(x, p=2., dim=-1)
        return x

class SAGE(GAT):
    def __init__(self, data_load, data_kws, h_dim, z_dim):
        super().__init__(data_load, data_kws, h_dim, z_dim)

        self.c1 = PoolSAGEConv(self.data.x_dim, h_dim)
        self.c2 = PoolSAGEConv(h_dim, z_dim)

def detector_sage_rref(loader, kwargs, h_dim, z_dim, **kws):
    return DetectorEncoder(
        SAGE(loader, kwargs, h_dim, z_dim)
    )

def predictor_sage_rref(loader, kwargs, h_dim, z_dim, head=False):
    return PredictorEncoder(
        SAGE(loader, kwargs, h_dim, z_dim), head
    )
