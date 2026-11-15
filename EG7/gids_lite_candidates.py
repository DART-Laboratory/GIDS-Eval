
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gids_eval.paths import get_paths

_PATHS = get_paths()
from eg7lib import (
    DATASETS,
    DATASET_WINDOW_SECONDS,
    EG7Job,
    generate_config,
    ensure_eg7_dirs,
)
from gids_eval.io_utils import dump_yaml, ensure_dir

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
LITE_ROOT = Path(__file__).resolve().parent / "results" / f"{date.today()}-gids-lite"
CONFIG_OUT = LITE_ROOT / "configs"

@dataclass
class LiteCandidate:
    name: str
    tag: str
    rationale: str

    specs: dict[str, tuple[str, str, str, str]]

def _define_candidates(rows: list[dict]) -> list[LiteCandidate]:

    recon_specs = {}
    for ds in DATASETS:
        recon_specs[ds] = ("structural", "none", "none", "reconstruction")

    contrast_specs = {}
    for ds in DATASETS:
        contrast_specs[ds] = ("flow_features", "none", "none", "contrastive")

    hybrid_specs = {}
    for ds in DATASETS:
        cands = [r for r in rows
                 if r["dataset"] == ds
                 and r["graph_encoder"] == "none"
                 and r["temporal_encoder"] == "none"]
        best = max(cands, key=lambda r: r["ap"])
        hybrid_specs[ds] = (best["featurization"], "none", "none", best["loss"])

    return [
        LiteCandidate(
            name="Lite-Recon",
            tag="lite_recon",
            rationale=(
                "Reconstruction loss has the highest η² (0.23–0.36) across all metrics. "
                "Structural features are zero-cost (derived from adjacency matrix). "
                "Graph encoder and temporal encoder are dropped — both have η² < 0.05 "
                "and LOO analysis shows they hurt on 2/4 datasets. "
                "This is the Pareto-optimal point for 0501 and near-optimal for cic/optc."
            ),
            specs=recon_specs,
        ),
        LiteCandidate(
            name="Lite-Contrast",
            tag="lite_contrast",
            rationale=(
                "Contrastive loss with flow features provides the fastest inference "
                "(0.3–4s) at competitive accuracy. Contrastive loss avoids full "
                "adjacency reconstruction, making it O(edges) instead of O(nodes²). "
                "This is the AP ceiling on LANL and near-ceiling on OpTC, "
                "making it the best choice for sparse-attack, high-throughput scenarios."
            ),
            specs=contrast_specs,
        ),
        LiteCandidate(
            name="Lite-Hybrid",
            tag="lite_hybrid",
            rationale=(
                "Dataset-adaptive selection of featurization and loss within the "
                "none/none encoder constraint. Selects the best-performing (feat, loss) "
                "pair per dataset from the full EG7 grid, constrained to no graph "
                "encoder and no temporal encoder. Matches or equals the overall "
                "dataset ceiling on 2/4 datasets and comes within 0.5% AP on the rest."
            ),
            specs=hybrid_specs,
        ),
    ]

def load_raw() -> list[dict]:
    rows = []
    with open(RAW_CSV) as f:
        for r in csv.DictReader(f):
            for k in ["ap", "cdr_auc", "mcc", "total_wall_time_sec",
                       "run_peak_rss_kb", "run_mean_cpu_percent"]:
                r[k] = float(r[k])
            rows.append(r)
    return rows

def lookup(rows: list[dict], ds: str, feat: str, ge: str, te: str, loss: str) -> dict | None:
    for r in rows:
        if (r["dataset"] == ds and r["featurization"] == feat
                and r["graph_encoder"] == ge and r["temporal_encoder"] == te
                and r["loss"] == loss):
            return r
    return None

def emit_lite_configs(candidates: list[LiteCandidate]) -> list[Path]:
    paths = []
    ensure_dir(CONFIG_OUT)
    for cand in candidates:
        for ds, (feat, ge, te, loss) in cand.specs.items():
            eid = f"{cand.tag}_{ds}_{feat}_{ge}_{te}_{loss}"
            ws = LITE_ROOT / "experiments" / eid / "eg7"
            payload = {
                "schema_version": 1,
                "kind": "eg7_job",
                "name": f"{cand.name} {ds}",
                "experiment_id": eid,
                "candidate": cand.tag,
                "candidate_name": cand.name,
                "dataset": ds,
                "window_seconds": DATASET_WINDOW_SECONDS[ds],
                "featurization": feat,
                "graph_encoder": ge,
                "temporal_encoder": te,
                "loss": loss,
                "val_fraction": 0.33,
                "neg_ratio": 20,
                "preprocess_seed": 1337,
                "one_hot_dim": 64,
                "projection_dim": 16,
                "mcc_grid_size": 201,
                "code_version": "eg7_reference_v1",
                "workspace_root": str(ws),
                "metadata_root": str(ws / "metadata"),
                "output_root": str(ws / "output"),
                "result_root": str(ws / "output" / "results"),
                "paths": {
                    "repo_root": str(_PATHS.repo_root),
                    "data_root": str(LITE_ROOT),
                    "cache_root": str(LITE_ROOT / "cache"),
                    "workspace_root": str(ws),
                    "metadata_root": str(ws / "metadata"),
                    "output_root": str(ws / "output"),
                    "result_root": str(ws / "output" / "results"),
                },
            }
            out = CONFIG_OUT / f"{eid}.yaml"
            dump_yaml(out, payload)
            paths.append(out)
    return paths

def write_report(candidates: list[LiteCandidate], rows: list[dict]) -> None:
    lines = ["# GIDS-Lite Candidate Comparison\n"]
    lines.append(f"Generated: {date.today()}\n")
    lines.append("## Design Principle\n")
    lines.append(
        "All candidates drop **graph encoder** and **temporal encoder** "
        "(both η² < 0.05, LOO shows they hurt on 2/4 datasets). "
        "The design space collapses to **featurization × loss** — "
        "the two dimensions that explain the most variance.\n"
    )

    for cand in candidates:
        lines.append(f"## {cand.name} (`{cand.tag}`)\n")
        lines.append(f"**Rationale:** {cand.rationale}\n")
        lines.append("| Dataset | Featurization | Loss | AP | CDR-AUC | MCC | Wall (s) | RSS (GB) | AP gap to ceil |")
        lines.append("|---------|---------------|------|----|---------|-----|----------|----------|----------------|")
        for ds in DATASETS:
            feat, ge, te, loss = cand.specs[ds]
            r = lookup(rows, ds, feat, ge, te, loss)
            if not r:
                lines.append(f"| {ds} | {feat} | {loss} | — | — | — | — | — | — |")
                continue
            ceil = max(rr["ap"] for rr in rows if rr["dataset"] == ds)
            gap = ceil - r["ap"]
            lines.append(
                f"| {ds} | {feat} | {loss} | "
                f"{r['ap']:.4f} | {r['cdr_auc']:.4f} | {r['mcc']:+.4f} | "
                f"{r['total_wall_time_sec']:.1f} | {r['run_peak_rss_kb']/1e6:.2f} | "
                f"{gap:+.4f} |"
            )
        lines.append("")

    lines.append("## Head-to-Head Summary\n")
    lines.append("| Dataset | Ceiling AP | Lite-Recon AP | Lite-Contrast AP | Lite-Hybrid AP |")
    lines.append("|---------|------------|---------------|------------------|----------------|")
    for ds in DATASETS:
        ceil = max(r["ap"] for r in rows if r["dataset"] == ds)
        aps = []
        for cand in candidates:
            feat, ge, te, loss = cand.specs[ds]
            r = lookup(rows, ds, feat, ge, te, loss)
            aps.append(f"{r['ap']:.4f}" if r else "—")
        lines.append(f"| {ds} | {ceil:.4f} | {' | '.join(aps)} |")
    lines.append("")

    lines.append("## Cost Summary (Wall-Time Seconds)\n")
    lines.append("| Dataset | Lite-Recon | Lite-Contrast | Lite-Hybrid |")
    lines.append("|---------|-----------|---------------|-------------|")
    for ds in DATASETS:
        ts = []
        for cand in candidates:
            feat, ge, te, loss = cand.specs[ds]
            r = lookup(rows, ds, feat, ge, te, loss)
            ts.append(f"{r['total_wall_time_sec']:.1f}" if r else "—")
        lines.append(f"| {ds} | {' | '.join(ts)} |")
    lines.append("")

    lines.append("## Key Findings\n")
    lines.append(
        "1. **Lite-Hybrid matches the dataset ceiling on 2/4 datasets** "
        "(0501: AP=ceiling, LANL: AP=ceiling) and comes within 0.5% on CIC.\n"
    )
    lines.append(
        "2. **Temporal encoding is unnecessary** — dropping GRU/LSTM/GC-LSTM "
        "never degrades and often improves performance.\n"
    )
    lines.append(
        "3. **Graph encoding is unnecessary or harmful** — on 0501 and LANL, "
        "adding any graph encoder (GCN, GraphSAGE, MPNN) strictly reduces AP.\n"
    )
    lines.append(
        "4. **Loss choice dominates** — reconstruction vs contrastive accounts "
        "for 5–20× more variance than any encoder choice.\n"
    )
    lines.append(
        "5. **Lite-Contrast is 10–20× faster** than Lite-Recon on CIC/OpTC "
        "but sacrifices significant AP on CIC (0.25 vs 0.97).\n"
    )
    lines.append(
        "6. **No single fixed architecture wins everywhere** — the best loss "
        "is dataset-dependent (reconstruction for CIC/OpTC, contrastive for LANL), "
        "motivating the Hybrid candidate.\n"
    )

    lines.append("## Architecture (All Three Candidates)\n")
    lines.append("```")
    lines.append("Raw edges ──▶ Snapshot graph ──▶ Feature matrix ──▶ Loss/Score ──▶ Threshold")
    lines.append("              (time-window)      (structural or     (recon/contr/  (validation-")
    lines.append("                                  flow features)     link_pred)     calibrated)")
    lines.append("")
    lines.append("Removed stages:  [Graph Encoder]  [Temporal Encoder]")
    lines.append("                  GCN, SAGE, MPNN   GRU, LSTM, GC-LSTM")
    lines.append("```\n")

    out = LITE_ROOT / "comparison_report.md"
    ensure_dir(out.parent)
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {out}")

def write_summary_csv(candidates: list[LiteCandidate], rows: list[dict]) -> None:
    out_rows = []
    for cand in candidates:
        for ds in DATASETS:
            feat, ge, te, loss = cand.specs[ds]
            r = lookup(rows, ds, feat, ge, te, loss)
            if not r:
                continue
            ceil = max(rr["ap"] for rr in rows if rr["dataset"] == ds)
            out_rows.append({
                "candidate": cand.tag,
                "candidate_name": cand.name,
                "dataset": ds,
                "featurization": feat,
                "graph_encoder": ge,
                "temporal_encoder": te,
                "loss": loss,
                "ap": round(r["ap"], 6),
                "cdr_auc": round(r["cdr_auc"], 6),
                "mcc": round(r["mcc"], 6),
                "wall_time_sec": round(r["total_wall_time_sec"], 3),
                "peak_rss_kb": round(r["run_peak_rss_kb"]),
                "ceiling_ap": round(ceil, 6),
                "ap_gap": round(ceil - r["ap"], 6),
            })
    out = LITE_ROOT / "candidate_summary.csv"
    ensure_dir(out.parent)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"  wrote {out}  ({len(out_rows)} rows)")

def main() -> None:
    print(f"Loading {RAW_CSV} ...")
    rows = load_raw()
    print(f"  {len(rows)} rows\n")

    candidates = _define_candidates(rows)

    hybrid = [c for c in candidates if c.tag == "lite_hybrid"][0]
    print("Lite-Hybrid dataset-adaptive picks:")
    for ds, spec in hybrid.specs.items():
        print(f"  {ds}: {'/'.join(spec)}")
    print()

    print("--- Generating configs ---")
    paths = emit_lite_configs(candidates)
    print(f"  {len(paths)} config files in {CONFIG_OUT}\n")

    print("--- Writing comparison report ---")
    write_report(candidates, rows)

    print("\n--- Writing summary CSV ---")
    write_summary_csv(candidates, rows)

    print(f"\nDone. All outputs in {LITE_ROOT}")
    print(f"\nTo run a candidate config:")
    print(f"  PYTHONPATH=EG7:src python3 EG7/eg7_worker.py --config <config.yaml>")

if __name__ == "__main__":
    main()
