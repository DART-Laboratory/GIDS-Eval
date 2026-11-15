# EG3 — Inductive Calibration

Measure how much PIKACHU benefits from using test-set information during
threshold calibration instead of a deployment-realistic inductive setup.

## Background

PIKACHU's original evaluation computes anomaly scores on all post-training
snapshots, then selects the detection threshold by optimising over those
same scores (transductive thresholding).  In deployment, test data is unseen
at calibration time.  EG3 quantifies the performance gap between the original
path and a realistic inductive alternative.

## Calibration modes

| Mode | Threshold source | Evaluation set |
|------|-----------------|----------------|
| **transductive** | Imported from EG1 PIKACHU runs at the matched window | All post-training edges |
| **inductive** | First 50% of post-training snapshots (validation) | Remaining 50% (test) |

Model training, graph embedding, and edge-probability estimation are
identical for both modes — only threshold selection differs.

In the current workflow, `EG3` executes only the **inductive** jobs locally.
The transductive baseline is imported from `<work_root>/eg1/reports/raw_results.csv`
for the matching PIKACHU window:

- `lanl`: `3600`
- `optc`: `3600`
- `cic`: `3000`
- `0501`: `300`

If the matched EG1 baseline is unavailable locally, the summarizer falls back
to a successful local transductive rerun instead of dropping that row from the
EG3 report tables.

## Experiment matrix

- 1 detector (PIKACHU)
- 4 datasets (LANL, OpTC, CIC, 0501)
- 2 calibration-result modes (transductive imported from EG1, inductive run in EG3)
- 1 seed

Total: **8 result slots**, with **4 local EG3 runs** for inductive.

## Usage

```bash
# Generate configs
bash EG3/run_sc3.sh matrix

# Run all inductive jobs and refresh imported transductive status
bash EG3/run_sc3.sh run

# Check progress
bash EG3/run_sc3.sh progress

# Aggregate results
bash EG3/run_sc3.sh summarize

# Run only inductive jobs explicitly
bash EG3/run_sc3.sh run --calibrations inductive
```

## Output (on mounted disk)

```
<work_root>/eg3/
├── generated-configs/       # YAML configs for SC3-run jobs
├── experiments/             # Per-job workspaces for inductive runs
├── orchestrator/            # job_matrix.csv, job_status.csv, progress.md
└── reports/
    ├── raw_results.csv      # Imported transductive + local inductive metrics
    ├── delta_table.csv      # Inductive - transductive per (dataset, seed)
    ├── evidence_table.csv   # Per-dataset mean deltas
    ├── evidence_table_overall.csv
    ├── plot_calibration_comparison.svg
    ├── plot_delta_by_dataset.svg
    └── conclusion.md
```

## Key metrics

AP, AUC, precision, recall, F1 — reported for both modes.
Negative deltas indicate PIKACHU's transductive path inflates performance.
