# EG7 — Component Ablations

EG7 fixes graph construction and detection, and ablates four core modules:

- `featurization`
- `graph_encoder`
- `temporal_encoder`
- `loss`

This implementation is a unified reference ablation engine rather than a
detector-fidelity re-run of the five vendored baselines. It reuses the repo's
tracking style from EG1/EG3:

- generated configs under `<work_root>/eg7/generated-configs/`
- per-job workspaces under `<work_root>/eg7/experiments/`
- global progress under `<work_root>/eg7/orchestrator/`
- aggregated reports under `<work_root>/eg7/reports/`

## Default matrix

- datasets: `0501`, `cic`, `optc`, `lanl`
- featurizations: `one_hot`, `flow_features`, `structural`, `one_hot_structural`
- graph encoders: `none`, `linear`, `gcn`, `graphsage`, `mpnn`
- temporal encoders: `none`, `gru`, `lstm`, `gc_lstm`
- losses: `reconstruction`, `contrastive`, `link_prediction`

Total jobs: `4 x 4 x 5 x 4 x 3 = 960`

## Commands

```bash
# Generate configs
bash EG7/run_sc7.sh matrix

# Run all jobs
bash EG7/run_sc7.sh run

# Refresh status / progress
bash EG7/run_sc7.sh status
bash EG7/run_sc7.sh progress

# Aggregate reports
bash EG7/run_sc7.sh summarize
```
