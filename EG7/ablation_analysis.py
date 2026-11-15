
from __future__ import annotations

import csv
import itertools
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

def _newest_eg7_raw() -> Path:
    root = Path(__file__).resolve().parent / "results"
    snapshots = sorted(
        (p for p in root.glob("*") if p.is_dir() and (p / "raw_results.csv").is_file()),
        key=lambda p: p.name,
    )
    if not snapshots:
        raise FileNotFoundError(f"no results snapshot with raw_results.csv under {root}")
    return snapshots[-1] / "raw_results.csv"


RAW_CSV = _newest_eg7_raw()
OUT_DIR = Path(__file__).resolve().parent / "results" / f"{date.today()}-ablation"

DIMENSIONS = ["featurization", "graph_encoder", "temporal_encoder", "loss"]
METRICS = ["ap", "cdr_auc", "mcc"]
COST_METRIC = "total_wall_time_sec"
RSS_METRIC = "run_peak_rss_kb"

def load_raw(path: Path = RAW_CSV) -> list[dict[str, Any]]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            for m in METRICS + [COST_METRIC, RSS_METRIC]:
                r[m] = float(r[m])
            rows.append(r)
    return rows

def by_dataset(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["dataset"]].append(r)
    return dict(out)

def _ss_total(values: list[float]) -> float:
    mean = np.mean(values)
    return float(np.sum((np.array(values) - mean) ** 2))

def _ss_between(rows: list[dict], dim: str, metric: str) -> float:
    groups: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        groups[r[dim]].append(r[metric])
    grand_mean = np.mean([r[metric] for r in rows])
    ss = 0.0
    for vals in groups.values():
        n_k = len(vals)
        ss += n_k * (np.mean(vals) - grand_mean) ** 2
    return float(ss)

def eta_squared_table(
    ds_rows: dict[str, list[dict]],
) -> list[dict[str, Any]]:
    results = []
    for ds, rows in sorted(ds_rows.items()):
        for metric in METRICS:
            vals = [r[metric] for r in rows]
            ss_tot = _ss_total(vals)
            if ss_tot < 1e-15:
                for dim in DIMENSIONS:
                    results.append(dict(dataset=ds, metric=metric, dimension=dim, eta_sq=0.0))
                continue
            eta_list = []
            for dim in DIMENSIONS:
                ss_b = _ss_between(rows, dim, metric)
                eta = ss_b / ss_tot
                eta_list.append((dim, eta))
            eta_list.sort(key=lambda x: x[1], reverse=True)
            for rank, (dim, eta) in enumerate(eta_list, 1):
                results.append(dict(dataset=ds, metric=metric, dimension=dim, eta_sq=round(eta, 6), rank=rank))
    return results

def marginal_delta_table(ds_rows: dict[str, list[dict]]) -> list[dict]:
    results = []
    for ds, rows in sorted(ds_rows.items()):
        for metric in METRICS:
            grand_mean = np.mean([r[metric] for r in rows])
            for dim in DIMENSIONS:
                groups: dict[str, list[float]] = defaultdict(list)
                for r in rows:
                    groups[r[dim]].append(r[metric])
                for opt, vals in sorted(groups.items()):
                    opt_mean = np.mean(vals)
                    results.append(dict(
                        dataset=ds, metric=metric, dimension=dim, option=opt,
                        option_mean=round(float(opt_mean), 6),
                        grand_mean=round(float(grand_mean), 6),
                        delta=round(float(opt_mean - grand_mean), 6),
                    ))
    return results

def _interaction_strength(rows: list[dict], dim_a: str, dim_b: str, metric: str) -> float:
    vals = [r[metric] for r in rows]
    ss_tot = _ss_total(vals)
    if ss_tot < 1e-15:
        return 0.0
    ss_a = _ss_between(rows, dim_a, metric)
    ss_b = _ss_between(rows, dim_b, metric)

    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        groups[(r[dim_a], r[dim_b])].append(r[metric])
    grand_mean = np.mean(vals)
    ss_cells = sum(len(v) * (np.mean(v) - grand_mean) ** 2 for v in groups.values())
    ss_ab = max(ss_cells - ss_a - ss_b, 0.0)
    return float(ss_ab / ss_tot)

def interaction_matrix(ds_rows: dict[str, list[dict]]) -> list[dict]:
    results = []
    pairs = list(itertools.combinations(DIMENSIONS, 2))
    for ds, rows in sorted(ds_rows.items()):
        for metric in METRICS:
            for dim_a, dim_b in pairs:
                eta_ab = _interaction_strength(rows, dim_a, dim_b, metric)
                results.append(dict(
                    dataset=ds, metric=metric,
                    dim_a=dim_a, dim_b=dim_b,
                    interaction_eta_sq=round(eta_ab, 6),
                ))
    return results

def leave_one_out_table(ds_rows: dict[str, list[dict]]) -> list[dict]:
    loo_dims = ["graph_encoder", "temporal_encoder"]
    results = []
    for ds, rows in sorted(ds_rows.items()):
        for dim in loo_dims:
            for metric in METRICS:
                groups: dict[str, list[float]] = defaultdict(list)
                for r in rows:
                    groups[r[dim]].append(r[metric])
                none_mean = float(np.mean(groups.get("none", [0.0])))
                best_opt, best_mean = None, -1e9
                for opt, vals in groups.items():
                    if opt == "none":
                        continue
                    m = float(np.mean(vals))
                    if m > best_mean:
                        best_mean = m
                        best_opt = opt
                results.append(dict(
                    dataset=ds, dimension=dim, metric=metric,
                    none_mean=round(none_mean, 6),
                    best_option=best_opt,
                    best_mean=round(best_mean, 6),
                    delta=round(best_mean - none_mean, 6),
                    pct_change=round(100 * (best_mean - none_mean) / abs(none_mean), 2) if abs(none_mean) > 1e-9 else float("nan"),
                ))
    return results

def pareto_frontier(ds_rows: dict[str, list[dict]]) -> list[dict]:
    results = []
    for ds, rows in sorted(ds_rows.items()):

        pts = sorted(rows, key=lambda r: r[COST_METRIC])

        pareto_ids = set()
        best_ap = -1.0
        for p in pts:
            if p["ap"] > best_ap:
                pareto_ids.add(p["experiment_id"])
                best_ap = p["ap"]
        for r in rows:
            results.append(dict(
                dataset=ds,
                experiment_id=r["experiment_id"],
                featurization=r["featurization"],
                graph_encoder=r["graph_encoder"],
                temporal_encoder=r["temporal_encoder"],
                loss=r["loss"],
                ap=round(r["ap"], 6),
                cdr_auc=round(r["cdr_auc"], 6),
                mcc=round(r["mcc"], 6),
                wall_time_sec=round(r[COST_METRIC], 3),
                peak_rss_kb=round(r[RSS_METRIC], 0),
                is_pareto=r["experiment_id"] in pareto_ids,
            ))
    return results

def aggregated_ranking(eta_table: list[dict]) -> list[dict]:

    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in eta_table:
        acc[r["dimension"]][r["metric"]].append(r["eta_sq"])

    results = []
    for metric in METRICS:
        ranked = []
        for dim in DIMENSIONS:
            avg_eta = float(np.mean(acc[dim][metric]))
            ranked.append((dim, avg_eta))
        ranked.sort(key=lambda x: x[1], reverse=True)
        for rank, (dim, avg_eta) in enumerate(ranked, 1):
            results.append(dict(metric=metric, dimension=dim, avg_eta_sq=round(avg_eta, 6), rank=rank))
    return results

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")

def _write_conclusion(path: Path, agg: list[dict], eta: list[dict],
                      interactions: list[dict], loo: list[dict],
                      pareto_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# EG7 Ablation Deep Analysis — Conclusion\n"]

    lines.append("## 1. Aggregated Module Contribution Ranking\n")
    lines.append("| Metric | #1 | #2 | #3 | #4 |")
    lines.append("|--------|----|----|----|----|")
    for metric in METRICS:
        ranked = [r for r in agg if r["metric"] == metric]
        ranked.sort(key=lambda x: x["rank"])
        cells = [f"{r['dimension']} ({r['avg_eta_sq']:.4f})" for r in ranked]
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 2. Per-Dataset Module Contribution (η²)\n")
    datasets = sorted(set(r["dataset"] for r in eta))
    for ds in datasets:
        lines.append(f"### {ds}\n")
        lines.append("| Metric | #1 | #2 | #3 | #4 |")
        lines.append("|--------|----|----|----|----|")
        for metric in METRICS:
            ranked = [r for r in eta if r["dataset"] == ds and r["metric"] == metric]
            ranked.sort(key=lambda x: x["rank"])
            cells = [f"{r['dimension']} ({r['eta_sq']:.4f})" for r in ranked]
            lines.append(f"| {metric} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## 3. Top Interaction Effects\n")
    lines.append("| Dataset | Metric | Dim A × Dim B | Interaction η² |")
    lines.append("|---------|--------|---------------|----------------|")
    top_int = sorted(interactions, key=lambda x: x["interaction_eta_sq"], reverse=True)[:20]
    for r in top_int:
        lines.append(f"| {r['dataset']} | {r['metric']} | {r['dim_a']} × {r['dim_b']} | {r['interaction_eta_sq']:.4f} |")
    lines.append("")

    lines.append("## 4. Leave-One-Out: Does Adding a Module Help?\n")
    lines.append("| Dataset | Dimension | Metric | none mean | Best option | Best mean | Δ | % change |")
    lines.append("|---------|-----------|--------|-----------|-------------|-----------|---|----------|")
    for r in loo:
        pct = f"{r['pct_change']:.1f}%" if not math.isnan(r["pct_change"]) else "n/a"
        lines.append(f"| {r['dataset']} | {r['dimension']} | {r['metric']} | {r['none_mean']:.4f} | {r['best_option']} | {r['best_mean']:.4f} | {r['delta']:+.4f} | {pct} |")
    lines.append("")

    lines.append("## 5. Pareto Frontier (AP vs Wall-Time)\n")
    pareto_only = [r for r in pareto_rows if r["is_pareto"]]
    for ds in datasets:
        ds_pareto = [r for r in pareto_only if r["dataset"] == ds]
        ds_pareto.sort(key=lambda x: x["wall_time_sec"])
        lines.append(f"### {ds} ({len(ds_pareto)} Pareto-optimal points)\n")
        lines.append("| Config | AP | CDR-AUC | Wall-time (s) | Peak RSS (KB) |")
        lines.append("|--------|----|---------|---------------|---------------|")
        for r in ds_pareto:
            cfg = f"{r['featurization']}/{r['graph_encoder']}/{r['temporal_encoder']}/{r['loss']}"
            lines.append(f"| {cfg} | {r['ap']:.4f} | {r['cdr_auc']:.4f} | {r['wall_time_sec']:.1f} | {r['peak_rss_kb']:.0f} |")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {path}")

def main() -> None:
    print(f"Loading {RAW_CSV} ...")
    rows = load_raw()
    ds_rows = by_dataset(rows)
    print(f"  {len(rows)} rows across {sorted(ds_rows.keys())}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- 1. Module contribution ranking (ANOVA η²) ---")
    eta = eta_squared_table(ds_rows)
    _write_csv(OUT_DIR / "eta_squared_by_dataset.csv", eta)

    agg = aggregated_ranking(eta)
    _write_csv(OUT_DIR / "contribution_ranking_aggregated.csv", agg)

    print("\n--- 2. Marginal deltas ---")
    deltas = marginal_delta_table(ds_rows)
    _write_csv(OUT_DIR / "marginal_deltas.csv", deltas)

    print("\n--- 3. Interaction matrix ---")
    interactions = interaction_matrix(ds_rows)
    _write_csv(OUT_DIR / "interaction_matrix.csv", interactions)

    print("\n--- 4. Leave-one-out ---")
    loo = leave_one_out_table(ds_rows)
    _write_csv(OUT_DIR / "leave_one_out.csv", loo)

    print("\n--- 5. Pareto frontier ---")
    pareto_rows = pareto_frontier(ds_rows)
    _write_csv(OUT_DIR / "pareto_frontier.csv", pareto_rows)
    pareto_only = [r for r in pareto_rows if r["is_pareto"]]
    print(f"  {len(pareto_only)} Pareto-optimal points across all datasets")

    print("\n--- 6. Conclusion ---")
    _write_conclusion(OUT_DIR / "conclusion.md", agg, eta, interactions, loo, pareto_rows)

    print(f"\nDone. All outputs in {OUT_DIR}")

if __name__ == "__main__":
    main()
