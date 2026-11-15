from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch_geometric.nn import NNConv
from torch_geometric.nn.dense.linear import Linear

from attack_common import (
    append_target_row,
    build_attack_summary,
    build_budget_rows,
    load_target_rows,
    resolve_budgets,
    target_key,
    unique_target_list,
    write_budget_rows,
    write_progress,
    write_target_rows,
)
from loaders.tdata import TData
from models.recurrent import GRU

class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda payload: torch.load(io.BytesIO(payload), map_location="cpu")
        return super().find_class(module, name)

class DenseGCNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, add_self_loops: bool = True, bias: bool = True):
        super().__init__()
        assert add_self_loops
        self.lin = Linear(in_channels, out_channels, bias=False, weight_initializer="glorot")
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)
        out = adj @ x
        if self.bias is not None:
            out += self.bias
        return out

class DropEdge(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, ei, ewa=None):
        if self.training and self.p > 0:
            mask = torch.rand(ei.size(1))
            if ewa is None:
                return ei[:, mask > self.p]
            return ei[:, mask > self.p], ewa[mask > self.p]
        if ewa is None:
            return ei
        return ei, ewa

class ArgusOptcAttack(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, z_dim: int, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.de = DropEdge(0.5)
        self.c1 = DenseGCNConv(in_dim, hidden_dim, add_self_loops=True)
        self.c2 = DenseGCNConv(hidden_dim, hidden_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.1)
        self.c3 = DenseGCNConv(hidden_dim, z_dim, add_self_loops=True)
        self.c4 = DenseGCNConv(hidden_dim, z_dim, add_self_loops=True)
        self.ac = nn.Tanh()

    def forward(self, data, mask_enum, no_grad, ret_adj=False, extra_edges=None):
        if no_grad:
            self.eval()
            with torch.no_grad():
                return self.inner_forward(data, mask_enum, ret_adj=ret_adj, extra_edges=extra_edges)
        return self.inner_forward(data, mask_enum, ret_adj=ret_adj, extra_edges=extra_edges)

    def inner_forward(self, data, mask_enum, ret_adj=False, extra_edges=None):
        zs = []
        adjs = []
        for i in range(data.T):
            extra = extra_edges[i] if extra_edges is not None else None
            if ret_adj:
                z, adj = self.forward_once(data, mask_enum, i, ret_adj=ret_adj, extra=extra)
                zs.append(z)
                adjs.append(adj)
            else:
                zs.append(self.forward_once(data, mask_enum, i, ret_adj=ret_adj, extra=extra))
        if ret_adj:
            return torch.stack(zs), adjs
        return torch.stack(zs)

    def forward_once(self, data, mask_enum, i, ret_adj=False, extra=None):
        x = data.xs[i].to(self.device) if data.dynamic_feats else data.xs.to(self.device)
        ei = data.ei_masked(mask_enum, i).to(self.device)
        ew = data.ew_masked(mask_enum, i).to(self.device)
        ei, ew = self.de(ei, ewa=ew)
        if extra is not None and len(extra) > 0:
            ei = torch.cat((ei, torch.LongTensor(extra).to(self.device).T), 1)
            ew = torch.cat((ew, torch.ones(len(extra), device=self.device) * ew.mean()), 0)
        raw_adj, normed_adj = self.gcn_norm_adj(ei, ew, len(x), need_grad=ret_adj)
        x = self.c1(x, normed_adj)
        x = self.c2(x, normed_adj)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c3(x, normed_adj)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c4(x, normed_adj)
        if ret_adj:
            return self.ac(x), raw_adj
        return self.ac(x)

    def gcn_norm_adj(self, ei, ew, num_nodes, need_grad):
        raw_adj = torch.eye(num_nodes, device=self.device)
        raw_adj[ei[0], ei[1]] = ew
        if need_grad:
            raw_adj.requires_grad_()
        deg_inv_sqrt = raw_adj.sum(0).pow_(-0.5)
        normed_adj = (raw_adj * deg_inv_sqrt.unsqueeze(0) * deg_inv_sqrt.unsqueeze(1)).T
        return raw_adj, normed_adj

    def decode(self, src, dst, z):
        return torch.sigmoid((z[src] * z[dst]).sum(dim=1))

class ArgusLanlAttack(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, z_dim: int, edge_attr_dim: int, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.c1 = DenseGCNConv(in_dim, hidden_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.c2 = DenseGCNConv(hidden_dim, hidden_dim, add_self_loops=True)
        self.drop = nn.Dropout(0.1)
        self.ac = nn.Tanh()
        self.c3 = DenseGCNConv(hidden_dim, z_dim, add_self_loops=True)
        nn4 = nn.Sequential(nn.Linear(edge_attr_dim, 8), nn.ReLU(), nn.Linear(8, hidden_dim * z_dim))
        self.c4 = NNConv(hidden_dim, z_dim, nn4, aggr="mean")

    def forward(self, data, mask_enum, no_grad, ret_adj=False, extra_edges=None):
        if no_grad:
            self.eval()
            with torch.no_grad():
                return self.inner_forward(data, mask_enum, ret_adj=ret_adj, extra_edges=extra_edges)
        return self.inner_forward(data, mask_enum, ret_adj=ret_adj, extra_edges=extra_edges)

    def inner_forward(self, data, mask_enum, ret_adj=False, extra_edges=None):
        zs = []
        adjs = []
        for i in range(data.T):
            extra = extra_edges[i] if extra_edges is not None else None
            if ret_adj:
                z, adj = self.forward_once(data, mask_enum, i, ret_adj=ret_adj, extra=extra)
                zs.append(z)
                adjs.append(adj)
            else:
                zs.append(self.forward_once(data, mask_enum, i, ret_adj=ret_adj, extra=extra))
        if ret_adj:
            return torch.stack(zs), adjs
        return torch.stack(zs)

    def forward_once(self, data, mask_enum, i, ret_adj=False, extra=None):
        x = data.xs[i].to(self.device) if data.dynamic_feats else data.xs.to(self.device)
        ei = data.ei_masked(mask_enum, i).to(self.device)
        ea = data.ea_masked(mask_enum, i).to(self.device)
        ew = data.ew_masked(mask_enum, i).to(self.device)
        ea = torch.transpose(ea, 0, 1)
        if extra is not None and len(extra) > 0:
            ei = torch.cat((ei, torch.LongTensor(extra).to(self.device).T), 1)
            ew = torch.cat((ew, torch.ones(len(extra), device=self.device) * ew.mean()), 0)
            ea = torch.cat((ea, torch.zeros((len(extra), ea.shape[1]), device=self.device)), 0)
        raw_adj, normed_adj = self.gcn_norm_adj(ei, ew, len(x), need_grad=ret_adj)
        x1 = self.c1(x, normed_adj)
        x = self.c2(x1, normed_adj)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c3(x, normed_adj)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c4(x, ei, edge_attr=ea)
        if ret_adj:
            return self.ac(x), raw_adj
        return self.ac(x)

    def gcn_norm_adj(self, ei, ew, num_nodes, need_grad):
        raw_adj = torch.eye(num_nodes, device=self.device)
        raw_adj[ei[0], ei[1]] = ew
        if need_grad:
            raw_adj.requires_grad_()
        deg_inv_sqrt = raw_adj.sum(0).pow_(-0.5)
        normed_adj = (raw_adj * deg_inv_sqrt.unsqueeze(0) * deg_inv_sqrt.unsqueeze(1)).T
        return raw_adj, normed_adj

    def decode(self, src, dst, z):
        return torch.sigmoid((z[src] * z[dst]).sum(dim=1))

def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload

def slice_data(data, t):
    masks_t = data.masks[t] if len(data.masks) > 0 else data.masks
    eas_list = getattr(data, "eas", None)
    ys_list = getattr(data, "ys", None)
    cnt_list = getattr(data, "cnt", None)
    eas = [eas_list[t]] if eas_list is not None and len(eas_list) > t else None
    return TData(
        getattr(data, "slices", None),
        [data.eis[t]],
        data.xs,
        [ys_list[t]] if ys_list is not None and len(ys_list) > t else None,
        masks_t,
        ews=[data.ews[t]],
        eas=eas,
        cnt=[cnt_list[t]] if cnt_list is not None and len(cnt_list) > t else None,
        node_map=None,
    )

def evaluate_attack_state(enc, rnn, rnn_h0, data, tgt_t, tgt_u, tgt_v, extra_edges):
    rnn.eval()
    with torch.no_grad():
        enc_out = enc.forward(data, mask_enum=2, no_grad=True, extra_edges=extra_edges)
        rnn_out = rnn.forward(enc_out, rnn_h0)
        final_pred = enc.decode([tgt_u], [tgt_v], rnn_out[tgt_t]).item()
        cover_preds = []
        for prev_t, edges in extra_edges.items():
            for prev_u, prev_v in edges:
                cover_preds.append(enc.decode([prev_u], [prev_v], rnn_out[prev_t]).item())
    return final_pred, cover_preds

def propose_next_edge(enc, rnn, rnn_h0, data, tgt_t, tgt_u, tgt_v, cutoff, extra_edges, *, n_choice=10000):
    rnn.train()
    trial_enc_out, raw_adj = enc.forward(data, mask_enum=2, no_grad=False, ret_adj=True, extra_edges=extra_edges)
    trial_rnn_out = rnn.forward(trial_enc_out, rnn_h0)
    pred = enc.decode([tgt_u], [tgt_v], trial_rnn_out[tgt_t])
    pred.backward()
    all_grad = -torch.stack([raw_adj[t].grad.detach() for t in range(len(raw_adj))])
    _, num_nodes, _ = all_grad.shape
    candidate = None
    for index in torch.topk(all_grad.view(-1), n_choice, largest=False).indices:
        new_t = (index // num_nodes // num_nodes).item()
        new_u = ((index // num_nodes) % num_nodes).item()
        new_v = (index % num_nodes).item()
        if raw_adj[new_t][new_u, new_v] != 0 or (new_u, new_v) in extra_edges[new_t]:
            continue
        new_score = enc.decode([new_u], [new_v], trial_rnn_out[new_t]).item()
        if new_score > cutoff:
            candidate = (new_t, new_u, new_v)
            break
    del all_grad
    for raw in raw_adj:
        raw.grad = None
    del raw_adj
    return candidate

def build_model(checkpoint: dict[str, Any], architecture: str, device: str):
    state = checkpoint["states"][1]
    if architecture == "lanl":
        in_dim = int(state["gcns.module.c1.lin.weight"].shape[1])
        hidden_dim = int(state["gcns.module.c1.lin.weight"].shape[0])
        z_dim = int(state["gcns.module.c3.lin.weight"].shape[0])
        edge_attr_dim = int(state["gcns.module.c4.nn.0.weight"].shape[1])
        enc = ArgusLanlAttack(in_dim, hidden_dim, z_dim, edge_attr_dim, device=device).to(device)
    else:
        in_dim = int(state["gcns.module.c1.lin.weight"].shape[1])
        hidden_dim = int(state["gcns.module.c1.lin.weight"].shape[0])
        z_dim = int(state["gcns.module.c3.lin.weight"].shape[0])
        enc = ArgusOptcAttack(in_dim, hidden_dim, z_dim, device=device).to(device)

    enc_state = {}
    rnn_state = {}
    for key, value in state.items():
        if key.startswith("gcns.module."):
            enc_state[key[len("gcns.module."):]] = value
        elif key.startswith("rnn."):
            rnn_state[key[len("rnn."):]] = value
    enc.load_state_dict(enc_state)
    enc.eval()
    for param in enc.parameters():
        param.requires_grad = False

    hidden_dim = int(state["rnn.rnn.weight_hh_l0"].shape[1])
    out_dim = int(state["rnn.lin.weight"].shape[0])
    rnn = GRU(hidden_dim, hidden_dim, out_dim).to(device)
    rnn.load_state_dict(rnn_state)
    rnn.eval()
    for param in rnn.parameters():
        param.requires_grad = False

    rnn_h0 = checkpoint["h0"].to(device)
    return enc, rnn, rnn_h0

def run_target_attack(
    config: dict[str, Any],
    enc,
    rnn,
    rnn_h0,
    data_t,
    *,
    target_index: int,
    tgt_t: int,
    tgt_u: int,
    tgt_v: int,
    cutoff: float,
    max_budget: int,
    n_choice: int,
):
    started = time.time()
    extra_edges = {t: [] for t in range(data_t.T)}
    init_pred, _ = evaluate_attack_state(enc, rnn, rnn_h0, data_t, 0, tgt_u, tgt_v, extra_edges)
    initially_stealthy = init_pred > cutoff
    min_target_stealthy_edges = 0 if initially_stealthy else None
    min_cover_stealthy_edges = 0 if initially_stealthy else None
    minimal_success_edges = 0 if initially_stealthy else None
    generated_edge_count = 0
    final_target_score = init_pred
    final_cover_min_score = 1.0
    status = "initially_stealthy" if initially_stealthy else "failed_within_max_budget"

    if not initially_stealthy:
        while generated_edge_count < max_budget:
            candidate = propose_next_edge(
                enc,
                rnn,
                rnn_h0,
                data_t,
                0,
                tgt_u,
                tgt_v,
                cutoff,
                extra_edges,
                n_choice=n_choice,
            )
            if candidate is None:
                status = "no_candidate" if generated_edge_count == 0 else "failed_within_max_budget"
                break

            one_t, one_u, one_v = candidate
            extra_edges[one_t].append((one_u, one_v))
            generated_edge_count += 1

            final_target_score, cover_preds = evaluate_attack_state(enc, rnn, rnn_h0, data_t, 0, tgt_u, tgt_v, extra_edges)
            final_cover_min_score = min(cover_preds) if cover_preds else 1.0
            target_ok = final_target_score > cutoff
            cover_ok = all(score > cutoff for score in cover_preds)

            if min_target_stealthy_edges is None and target_ok:
                min_target_stealthy_edges = generated_edge_count
            if min_cover_stealthy_edges is None and cover_ok:
                min_cover_stealthy_edges = generated_edge_count
            if target_ok and cover_ok:
                minimal_success_edges = generated_edge_count
                status = "success"
                break

    return {
        "detector": config["detector"],
        "dataset": config["dataset"],
        "repro_level": config["repro_level"],
        "window_seconds": config["window_seconds"],
        "target_index": target_index,
        "tgt_t": tgt_t,
        "tgt_u": tgt_u,
        "tgt_v": tgt_v,
        "status": status,
        "init_score": init_pred,
        "initially_stealthy": initially_stealthy,
        "generated_edge_count": generated_edge_count,
        "min_target_stealthy_edges": min_target_stealthy_edges,
        "min_cover_stealthy_edges": min_cover_stealthy_edges,
        "minimal_success_edges": minimal_success_edges,
        "final_target_score": final_target_score,
        "final_cover_min_score": final_cover_min_score,
        "elapsed_sec": time.time() - started,
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

def main() -> int:
    parser = argparse.ArgumentParser(prog="argus_attack")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attack_cfg = config["attack"]
    checkpoint = CPUUnpickler(open(config["checkpoint"]["path"], "rb")).load()
    device = str(attack_cfg.get("device", "cpu"))
    budgets = resolve_budgets([int(value) for value in attack_cfg["budgets"]])
    max_budget = max([budget for budget in budgets if budget > 0], default=0)
    max_targets = int(os.environ.get("EG6_MAX_TARGETS", "0")) or None
    n_choice = int(os.environ.get("EG6_N_CHOICE", str(attack_cfg["n_choice"])))
    progress_every = max(1, int(os.environ.get("EG6_PROGRESS_EVERY", "10")))
    target_results_path = output_dir / "target_results.csv"
    progress_path = output_dir / "attack_progress.json"
    budget_results_path = output_dir / "budget_results.csv"
    attack_summary_path = output_dir / "attack_summary.json"

    loader_module = __import__(f"loaders.{attack_cfg['loader']['module']}", fromlist=["_"])
    loader_fn = getattr(loader_module, attack_cfg["loader"]["function"])
    data = loader_fn(**attack_cfg["loader"]["kwargs"])

    enc, rnn, rnn_h0 = build_model(checkpoint, str(attack_cfg["architecture"]), device)
    cutoff = float(attack_cfg["threshold"])

    all_mals = []
    for t, (one_es, one_ys) in enumerate(zip(data.eis, data.ys)):
        all_mals.extend([(t, u.item(), v.item()) for u, v, y in zip(one_es[0], one_es[1], one_ys) if y == 1])
    all_mals = unique_target_list(all_mals)
    if max_targets is not None:
        all_mals = all_mals[:max_targets]

    target_rows = load_target_rows(target_results_path)
    if target_rows:
        write_target_rows(target_results_path, target_rows)
    completed = {target_key(row["tgt_t"], row["tgt_u"], row["tgt_v"]): row for row in target_rows}

    print(f"target_count={len(all_mals)}", flush=True)
    if completed:
        print(f"resume_processed={len(completed)}", flush=True)

    def persist(latest_row: dict[str, Any] | None = None) -> None:
        rows = sorted(completed.values(), key=lambda row: row["target_index"])
        write_budget_rows(
            budget_results_path,
            build_budget_rows(config, budgets, rows, total_target_count=len(all_mals)),
        )
        attack_summary_path.write_text(
            json.dumps(build_attack_summary(config, budgets, rows, total_target_count=len(all_mals)), indent=2) + "\n",
            encoding="utf-8",
        )
        write_progress(progress_path, config=config, total_targets=len(all_mals), target_rows=rows, latest_row=latest_row)

    persist()

    for target_index, (tgt_t, tgt_u, tgt_v) in enumerate(all_mals):
        key = target_key(tgt_t, tgt_u, tgt_v)
        if key in completed:
            continue
        row = run_target_attack(
            config,
            enc,
            rnn,
            rnn_h0,
            slice_data(data, tgt_t),
            target_index=target_index,
            tgt_t=tgt_t,
            tgt_u=tgt_u,
            tgt_v=tgt_v,
            cutoff=cutoff,
            max_budget=max_budget,
            n_choice=n_choice,
        )
        append_target_row(target_results_path, row)
        completed[key] = row
        persist(latest_row=row)

        processed = len(completed)
        if processed == 1 or processed % progress_every == 0 or processed == len(all_mals):
            print(
                f"[{processed}/{len(all_mals)}] "
                f"target_index={row['target_index']} status={row['status']} "
                f"minimal_success_edges={row['minimal_success_edges']} "
                f"generated_edge_count={row['generated_edge_count']} "
                f"elapsed_sec={row['elapsed_sec']:.2f}",
                flush=True,
            )

    persist()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
