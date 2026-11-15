# EG4 — Edge Feature Robustness

Measure when ARGUS benefits from edge or flow features, and when those
features hurt generalization.

EG4 reuses the repo's standard orchestration layout:

- generated configs under `<work_root>/eg4/generated-configs/`
- per-job status under `<work_root>/eg4/orchestrator/`
- aggregated reports under `<work_root>/eg4/reports/`

## Experiment Matrix

- detector: `argus`
- datasets: `0501`, `cic`, `lanl`, `optc`
- modes: `flow`, `noflow`
- seed: `0`

Total: **8 jobs**

For `0501`, EG4 uses the unified merged dataset with the three attack
campaigns `oilrig`, `sandworm`, and `wizard_spider`, matching the EG1 data
contract.

## Commands

```bash
cd <repo_root>
bash EG4/run_sc4.sh matrix
bash EG4/run_sc4.sh run
bash EG4/run_sc4.sh progress
bash EG4/run_sc4.sh summarize
```

## Output

```
<work_root>/eg4/
├── generated-configs/
├── experiments/
├── orchestrator/
└── reports/
    ├── raw_results.csv
    ├── summary_by_mode.csv
    ├── paired_deltas.csv
    ├── feature_overlap.csv
    └── conclusion.md
```

## Key Question

Do edge or flow features improve ARGUS consistently, or do they mainly help
when malicious-edge statistics separate cleanly from benign traffic?
