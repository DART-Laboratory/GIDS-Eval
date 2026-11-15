# EG5 — Scalability Measurement

Quantify runtime and memory scaling as detector-ready EG1 inputs are expanded
from `25%` to `100%` of one fixed anchor workload per detector and dataset.

EG5 follows the same orchestration pattern as the other shortcomings studies:

- generated configs under `<work_root>/eg5/generated-configs/`
- per-job workspaces under `<work_root>/eg5/experiments/`
- progress and status under `<work_root>/eg5/orchestrator/`
- aggregated reports under `<work_root>/eg5/reports/`

## Default Matrix

- detectors: `argus`, `euler`, `pikachu`, `vgrnn`, `anomal`
- datasets:
  - SC1-derived: `0501`, `cic`, `lanl`, `optc`
  - Anomal-only: `0501`, `cic`
- scale levels: `25`, `50`, `75`, `100`
- seed: `0`

Representative anchor windows:

- `argus`: `0501=150s`, `cic=15m`, `lanl=1h`, `optc=6m`
- `euler`: `0501=150s`, `cic=50m`, `lanl=30m`, `optc=150m`
- `pikachu`: `0501=5m`, `cic=50m`, `lanl=1h`, `optc=1h`
- `vgrnn`: `0501=150s`, `cic=10m`, `lanl=90m`, `optc=30m`
- `anomal`: `0501`, `cic`

Total: **72 points**

The `100%` rows for the SC1-derived detectors are inherited from the matching
successful EG1 anchors. Reduced-scale rows are rebuilt from detector-native
SC1-ready inputs rather than raw-row prefixes.

## Commands

```bash
cd <repo_root>
bash EG5/run_sc5.sh matrix
bash EG5/run_sc5.sh run
bash EG5/run_sc5.sh progress
bash EG5/run_sc5.sh summarize
```

## Output

```
<work_root>/eg5/
├── generated-configs/
├── experiments/
├── orchestrator/
└── reports/
    ├── raw_results.csv
    ├── summary_by_scale.csv
    ├── fit_summary.csv
    └── conclusion.md
```

## Key Metrics

EG5 tracks wall-clock time, recorded peak RSS, and simple scaling-fit quality
for each detector and dataset combination.
