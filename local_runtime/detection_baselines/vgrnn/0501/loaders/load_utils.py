import torch

def edge_tvt_split(ei):
    ne = ei.size(1)
    val = int(ne*0.85)
    te = int(ne*0.90)

    masks = torch.zeros(3, ne).bool()
    rnd = torch.randperm(ne)
    masks[0, rnd[:val]] = True
    masks[1, rnd[val:te]] = True
    masks[2, rnd[te:]] = True

    return masks[0], masks[1], masks[2]

'''
For the cyber data, all of the test set is the latter time
stamps. So only need train and val partitions
'''
def edge_tv_split(ei, v_size=0.05):
    ne = ei.size(1)
    if ne <= 1:
        masks = torch.zeros(2, ne).bool()
        if ne == 1:
            masks[0, 0] = True
        return masks[0], masks[1]

    val = int(ne*v_size)
    val = max(1, val)
    val = min(val, ne - 1)

    masks = torch.zeros(2, ne).bool()
    rnd = torch.randperm(ne)
    masks[1, rnd[:val]] = True
    masks[0, rnd[val:]] = True

    return masks[0], masks[1]

'''
Various weighting functions for edges
'''
def std_edge_w(ew_ts):
    ews = []
    for ew_t in ew_ts:
        ew_t = ew_t.float()
        ew_t = (ew_t.long() / ew_t.std()).long()
        ew_t = torch.sigmoid(ew_t)
        ews.append(ew_t)

    return ews

def normalized(ew_ts):
    ews = []
    for ew_t in ew_ts:
        ew_t = ew_t.float()
        ew_t = ew_t.true_divide(ew_t.mean())
        ew_t = torch.sigmoid(ew_t)
        ews.append(ew_t)

    return ews

def standardized(ew_ts):
    ews = []
    for ew_t in ew_ts:
        ew_t = ew_t.float()
        std = ew_t.std()

        if std.item() == 0:
            ews.append(torch.full(ew_t.size(), 0.5))
            continue

        ew_t = (ew_t - ew_t.mean()) / std
        ew_t = torch.sigmoid(ew_t)
        ews.append(ew_t)

    return ews

def inv_standardized(ew_ts):
    ews = []
    for ew_t in ew_ts:
        ew_t = ew_t.float()
        ew_t = (ew_t - ew_t.mean()) / ew_t.std()
        ew_t = 1-torch.sigmoid(ew_t)
        ews.append(ew_t)

    return ews
