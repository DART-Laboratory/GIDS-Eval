# EG2

Resumable orchestration for EG2 ("Threshold Stability") experiments.

Code lives under `<repo_root>/EG2`.
All generated configs, caches, logs, metrics, and reports go under `<work_root>/eg2`.

## Scope

- Detectors: `argus`, `euler`
- Datasets: `lanl`, `optc`, `cic`, `0501`
- Seeds: 0 through 9 (10 seeds per detector-dataset pair)
- Window sizes: fixed at each detector's default from EG1 (not swept)
- `pikachu` and `vgrnn` are excluded because they do not use random validation sampling for threshold calibration. PIKACHU's threshold issue (transductive evaluation) is covered by EG3.

## Entry points

- One-click shell wrapper: `./EG2/run_sc2.sh`
- Python CLI: `python3 EG2/run_sc2.py`
- Environment probe: `./EG2/run_sc2.sh env-status`
- Progress snapshot: `./EG2/run_sc2.sh progress`

## Environment layout

EG2 reuses the EG1 Conda environments:

- `argus`: `<work_root>/conda/envs/sc1-argus`
- `euler`: `<work_root>/conda/envs/sc1-euler`

## Common commands

Generate configs and inspect the matrix:

```bash
cd <repo_root>
./EG2/run_sc2.sh matrix
```

Run the full EG2 matrix with resume support:

```bash
cd <repo_root>
./EG2/run_sc2.sh run
```

Run only a subset:

```bash
cd <repo_root>
./EG2/run_sc2.sh run --detectors argus --datasets lanl --seeds 0,1,2
```

Refresh the job state and aggregate CSV reports:

```bash
cd <repo_root>
./EG2/run_sc2.sh status
./EG2/run_sc2.sh summarize
```

## Experiment matrix

| Dimension | Values |
|-----------|--------|
| Detectors | argus, euler |
| Datasets  | lanl, optc, cic, 0501 |
| Seeds     | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| Window    | Fixed default per (detector, dataset) |
| **Total** | **2 x 4 x 10 = 80 jobs** |

## Resume model

Same as EG1. Each matrix point is one experiment config. A job is complete when its experiment metadata contains a successful detection stage and a non-empty `baseline_metrics.csv`. Re-running skips completed jobs unless `--force` is given.

Output paths:

- `<work_root>/eg2/generated-configs/`
- `<work_root>/eg2/orchestrator/`
- `<work_root>/eg2/reports/`

## Reports

`summarize` writes:

- `raw_results.csv` — per-job metrics including threshold, TPR, FPR, precision, recall, F1, AP, AUC
- `variance_by_group.csv` — per (detector, dataset) group: mean, std, min, max, range, CV for each metric
- `evidence_table.csv` — per group: which metrics are unstable (CV > 10%), verdict
- `plot_threshold_spread.svg` — box-plot showing threshold distribution across seeds
- `plot_cv_heatmap.svg` — heatmap of CV values across groups and metrics
- `conclusion.md` — interpretation and recommendation
