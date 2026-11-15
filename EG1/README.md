# EG1

Resumable orchestration for EG1 ("Snapshot Sensitivity") experiments.

Code lives under `<repo_root>/EG1`.
All generated configs, caches, logs, metrics, and reports go under `<work_root>/eg1`.

## Scope

- Detectors: `argus`, `euler`, `vgrnn`, `pikachu`
- Datasets: `lanl`, `optc`, `cic`, `0501`
- `0501` is treated as one EG1 dataset and one detector run. EG1 now builds a unified 24-hour timeline by taking a clean pre-attack background day and inserting the `oilrig`, `sandworm`, and `wizard_spider` attack segments into fixed time windows within that day.
- `Anomal-E` is intentionally excluded from EG1 execution because it is the static-graph baseline.

## Entry points

- One-click shell wrapper: `./EG1/run_sc1.sh`
- Python CLI: `python3 EG1/run_sc1.py`
- Environment probe: `./EG1/run_sc1.sh env-status`
- Progress snapshot: `./EG1/run_sc1.sh progress`
- Conda bootstrap: `./EG1/setup_conda_envs.sh`
- Conda verification: `./EG1/verify_conda_envs.sh`

## Environment layout

- Miniforge root: `<work_root>/conda/miniforge3`
- Detector envs:
  - `argus`: `<work_root>/conda/envs/sc1-argus`
  - `euler`: `<work_root>/conda/envs/sc1-euler`
  - `vgrnn`: `<work_root>/conda/envs/sc1-vgrnn`
  - `pikachu`: `<work_root>/conda/envs/sc1-pikachu`

`run_sc1.py` does not require shell-level `conda activate`. For `run`, it resolves the detector-specific interpreter and injects it through `PYTHON`, `PATH`, and the generated config `python_bin`.

EG1 now defaults to one fixed detector seed: `seed=0`. You can still override it with `--seeds`, but the default execution path is a pure time-window sweep rather than a multi-seed study.

For `PIKACHU`, preprocessing is intentionally kept fixed across detector seeds. The EG1 launcher uses a constant preprocessing seed and derives `trainwin` from the first attack snapshot so detector-seed variance does not silently change the dataset timeline.

## Common commands

Generate configs and inspect the matrix:

```bash
cd <repo_root>
./EG1/run_sc1.sh matrix
```

Run the full EG1 matrix with resume support:

```bash
cd <repo_root>
./EG1/run_sc1.sh run
```

That default command now expands to one fixed seed only.

Dry-run the generated configs without executing detector code:

```bash
cd <repo_root>
./EG1/run_sc1.sh run --dry-run
```

Run only a subset:

```bash
cd <repo_root>
./EG1/run_sc1.sh run --detectors argus,euler --datasets lanl,optc
```

Refresh the job state and aggregate CSV reports:

```bash
cd <repo_root>
./EG1/run_sc1.sh status
./EG1/run_sc1.sh summarize
```

Check which detector envs exist and whether they can import their required modules:

```bash
cd <repo_root>
./EG1/run_sc1.sh env-status
```

Recreate the detector-specific Conda environments:

```bash
cd <repo_root>
./EG1/setup_conda_envs.sh
./EG1/verify_conda_envs.sh
```

## Resume model

The launcher writes generated configs, state files, and reports under:

- `<work_root>/eg1/generated-configs/`
- `<work_root>/eg1/orchestrator/`
- `<work_root>/eg1/reports/`

Each matrix point is emitted as one experiment config and one detector run. For `0501`, the featurization stage first builds a unified synthetic day, then the detector trains and evaluates once on that merged dataset. A job is considered complete when its experiment metadata contains a successful detection stage and a non-empty `baseline_metrics.csv`. Re-running the launcher skips completed jobs unless `--force` is provided.

This gives two layers of recovery:

- launcher-level resume via per-job status checks
- stage-level reuse via the existing `gids_eval` cache under `<work_root>/eg1/cache/`

Filtered commands do not overwrite the global 64-job view anymore. The canonical files remain:

- `<work_root>/eg1/orchestrator/job_matrix.csv`
- `<work_root>/eg1/orchestrator/job_status.csv`
- `<work_root>/eg1/orchestrator/progress.json`
- `<work_root>/eg1/orchestrator/progress.md`
- `<work_root>/eg1/reports/*`

If you run a filtered command, EG1 also writes selection-only artifacts under:

- `<work_root>/eg1/orchestrator/job_matrix_selection.csv`
- `<work_root>/eg1/orchestrator/job_status_selection.csv`
- `<work_root>/eg1/orchestrator/progress_selection.json`
- `<work_root>/eg1/orchestrator/progress_selection.md`
- `<work_root>/eg1/reports/selection/*`

## Reports

`summarize` now writes:

- `raw_results.csv`
- `summary_by_window.csv`
- `metric_tables/index.csv`
- `metric_tables/*.csv` (one table per metric, including accuracy, runtime, and memory metrics)
- `evidence_table_pairs.csv`
- `evidence_table_overall.csv`
- `plot_ap_curves.svg`
- `plot_auc_curves.svg`
- `plot_ap_swing_heatmap.svg`
- `conclusion.md`

`raw_results.csv` now also carries stage-cost measurements for window studies, including:

- `featurization_wall_time_sec`
- `detection_wall_time_sec`
- `total_wall_time_sec`
- `featurization_peak_rss_gib`
- `detection_peak_rss_gib`
- `run_peak_rss_gib`
- `featurization_mean_rss_gib`
- `detection_mean_rss_gib`
- `run_mean_rss_gib`
- `featurization_mean_cpu_percent`
- `detection_mean_cpu_percent`
- `run_mean_cpu_percent`

For unified `0501`, `CDR` is computed from the real family labels (`oilrig`, `sandworm`, `wizard_spider`) exported by the single merged run rather than from post-hoc averages over three separate detector jobs.
