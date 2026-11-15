from __future__ import annotations

import argparse
import inspect as pyinspect
import io
import json
import os
import pickle
import sys
import time
import types
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
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

try:
    from model_1 import VGRNN as AttackVGRNN
except ModuleNotFoundError:
    AttackVGRNN = None

def install_pyg_inspector_shim() -> None:
    module = types.ModuleType("torch_geometric.inspector")
    module.Signature = pyinspect.Signature
    module.Parameter = pyinspect.Parameter

    class Inspector:
        def __init__(self, base_class):
            self.base_class = base_class
            self.params = {}

        def inspect(self, func, pop_first: bool = False):
            params = OrderedDict(pyinspect.signature(func).parameters)
            if pop_first:
                params.popitem(last=False)
            self.params[func.__name__] = params

        def keys(self, func_names=None):
            keys = []
            for func_name in func_names or list(self.params.keys()):
                keys += list(self.params[func_name].keys())
            return set(keys)

        def __implements__(self, cls, func_name: str) -> bool:
            if cls.__name__ == "MessagePassing":
                return False
            if func_name in cls.__dict__.keys():
                return True
            return any(self.__implements__(base, func_name) for base in cls.__bases__)

        def implements(self, func_name: str) -> bool:
            return self.__implements__(self.base_class.__class__, func_name)

        def types(self, func_names=None):
            return {}

        def distribute(self, func_name: str, kwargs: dict[str, Any]):
            out = {}
            for key, param in self.params[func_name].items():
                data = kwargs.get(key, pyinspect.Parameter.empty)
                if data is pyinspect.Parameter.empty:
                    if param.default is pyinspect.Parameter.empty:
                        raise TypeError(f"Required parameter {key} is empty.")
                    data = param.default
                out[key] = data
            return out

    module.Inspector = Inspector
    sys.modules["torch_geometric.inspector"] = module

class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda payload: torch.load(io.BytesIO(payload), map_location="cpu")
        return super().find_class(module, name)

def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload

def slice_data(data, t):
    masks_t = data.masks[t] if len(data.masks) > 0 else data.masks
    return TData(
        [data.eis[t]],
        data.xs,
        [data.ys[t]] if hasattr(data, "ys") else None,
        masks_t,
        ews=[data.ews[t]],
        node_map=None,
    )

def evaluate_attack_state(model, h0, data, tgt_t, tgt_u, tgt_v, extra_edges):
    model.eval()
    with torch.no_grad():
        enc_out, _, _ = model.forward(data, mask_enum=2, h0=h0, extra_edges=extra_edges)
        final_pred = model.decode([tgt_u], [tgt_v], enc_out[tgt_t]).item()
        cover_preds = []
        for prev_t, edges in extra_edges.items():
            for prev_u, prev_v in edges:
                cover_preds.append(model.decode([prev_u], [prev_v], enc_out[prev_t]).item())
    return final_pred, cover_preds

def propose_next_edge(model, h0, data, tgt_t, tgt_u, tgt_v, cutoff, extra_edges, *, n_choice=10000):
    trial_enc_out, _, raw_adj = model.forward(data, mask_enum=2, h0=h0, extra_edges=extra_edges)
    pred = model.decode([tgt_u], [tgt_v], trial_enc_out[tgt_t])
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
        new_score = model.decode([new_u], [new_v], trial_enc_out[new_t]).item()
        if new_score > cutoff:
            candidate = (new_t, new_u, new_v)
            break
    del all_grad
    for raw in raw_adj:
        raw.grad = None
    del raw_adj
    return candidate

def run_target_attack(
    config: dict[str, Any],
    model,
    h0,
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
    init_pred, _ = evaluate_attack_state(model, h0, data_t, 0, tgt_u, tgt_v, extra_edges)
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
                model,
                h0,
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

            final_target_score, cover_preds = evaluate_attack_state(model, h0, data_t, 0, tgt_u, tgt_v, extra_edges)
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
    parser = argparse.ArgumentParser(prog="vgrnn_attack")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    install_pyg_inspector_shim()
    config = load_yaml(Path(args.config))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attack_cfg = config["attack"]
    checkpoint = CPUUnpickler(open(config["checkpoint"]["path"], "rb")).load()
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

    stored_model = checkpoint["states"]
    in_dim = int(stored_model.phi_x[0].weight.shape[1])
    hidden_dim = int(getattr(stored_model, "h_dim"))
    z_dim = int(stored_model.prior_mean[0].weight.shape[0])
    attack_model_cls = AttackVGRNN or stored_model.__class__
    try:
        model = attack_model_cls(in_dim, hidden_dim, z_dim, pred=False)
    except TypeError:
        model = attack_model_cls(in_dim, hidden_dim, z_dim)
    model.load_state_dict(stored_model.state_dict(), strict=False)
    model.recurrent.weight_xz[0].load_state_dict(stored_model.recurrent.weight_xz[0].state_dict())
    model.recurrent.weight_hz[0].load_state_dict(stored_model.recurrent.weight_hz[0].state_dict())
    model.recurrent.weight_xr[0].load_state_dict(stored_model.recurrent.weight_xr[0].state_dict())
    model.recurrent.weight_hr[0].load_state_dict(stored_model.recurrent.weight_hr[0].state_dict())
    model.recurrent.weight_xh[0].load_state_dict(stored_model.recurrent.weight_xh[0].state_dict())
    model.recurrent.weight_hh[0].load_state_dict(stored_model.recurrent.weight_hh[0].state_dict())
    model.eval()
    h0 = checkpoint["h0"]
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
            model,
            h0,
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
