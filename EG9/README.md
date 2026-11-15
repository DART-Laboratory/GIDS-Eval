# EG9

`EG9` studies `Streaming Readiness` for the previously evaluated detectors only. It intentionally excludes `GIDS-Lite` or any new design work and instead replays readiness over archived `EG1` / `EG3` / `EG5` evidence.

## Scope

- detectors: `argus`, `euler`, `vgrnn`, `pikachu`, `anomal`
- datasets:
  - `argus`, `euler`, `vgrnn`, `pikachu`: `lanl`, `optc`, `cic`, `0501`
  - `anomal`: `cic`, `0501`
- replay rates: `1x`, `2x`, `5x`
- matrix size: `54` jobs

## Method

- Reuse archived runtime and quality results instead of retraining or re-running detector training.
- Use `EG5` anchor rows for deployment-scale input statistics and batching structure.
- Use `EG1` runtime/quality for `argus`, `euler`, and `vgrnn`.
- Use `EG3` inductive runtime/quality for `pikachu`.
- Use `EG5` runtime/quality for `anomal`.
- Reconstruct per-job replay traces with queueing simulation so each job emits:
  - overall orchestrator progress
  - per-job progress snapshots
  - per-unit latency traces

## Commands

```bash
./EG9/run_sc9.sh matrix
./EG9/run_sc9.sh env-status
./EG9/run_sc9.sh status
./EG9/run_sc9.sh progress
./EG9/run_sc9.sh summarize
./EG9/run_sc9.sh run
```

Filters match the previous `SC` tasks:

```bash
./EG9/run_sc9.sh run --detectors argus,euler --datasets lanl,optc --rates 1,5
```

## Output Layout

The runner writes under `<work_root>/eg9`:

- `generated-configs/`: one YAML config per `(detector, dataset, replay_rate)`
- `experiments/eg9_<detector>_<dataset>_<rate>/eg9/`: workspace, logs, job metadata
- `orchestrator/`: matrix, status, and progress files
- `reports/`: aggregate CSV/Markdown reports

Main report files:

- `raw_results.csv`
- `readiness_summary.csv`
- `evidence_table.csv`
- `conclusion.md`

Per-job metadata includes:

- `readiness_metrics.csv`: single-row operational summary for that job
- `latency_trace.csv`: simulated queue / lag trace
- `replay_progress.json`: lightweight snapshot for the currently running job
- `logs/replay.stdout.log`: concrete task log for that job

## Notes

- `EG9` is a deployment-readiness evaluation layer over archived evidence, not a new detector-training benchmark.
- `keep_up` is judged from replay backlog and final lag, not from accuracy alone.
- A job can `keep_up` at `5x` and still be classified as not real-time-ready if batching delay dominates alert latency.
