# EG6

`EG6` studies `Vulnerability to Minimal Adversarial Perturbation` with a resumable runner that mirrors the `EG2` layout and reporting style.

## Scope

- detectors: `argus`, `euler`, `vgrnn`
- datasets: `lanl`, `optc`, `0501`, `cic`
- matrix size: `12` jobs (`3 detectors x 4 datasets`)
- per-job attack budgets: `0,2,5,10,20,50`
- per-target attack execution now searches for the first successful perturbation and records `minimal_success_edges`; budget-level reports are derived as "success within budget K"

`LANL` and `OpTC` are treated as `paper_repro`, `0501` as `code_port`, and `CIC` as `extension`.

## Commands

```bash
./EG6/run_sc6.sh matrix
./EG6/run_sc6.sh env-status
./EG6/run_sc6.sh status
./EG6/run_sc6.sh progress
./EG6/run_sc6.sh summarize
./EG6/run_sc6.sh run
```

Filters match `EG2`:

```bash
./EG6/run_sc6.sh run --detectors euler,vgrnn --datasets lanl,optc
```

## Output Layout

The runner writes under `<work_root>/eg6`:

- `generated-configs/`: one YAML config per `(detector, dataset)`
- `experiments/eg6_<detector>_<dataset>/eg6/`: workspace, logs, runtime, per-job outputs
- `orchestrator/`: matrix, status, and progress files
- `reports/`: aggregate CSV/Markdown reports

Main report files:

- `raw_results.csv`
- `minimal_budget_by_group.csv`
- `evidence_table.csv`
- `conclusion.md`

Per-job metadata also includes:

- `target_results.csv`: per-target minimal-success records with incremental resume support
- `attack_progress.json`: lightweight progress snapshot for the currently running attack job

## Notes

- The runner reuses official checkpoints where they are usable and falls back to local bootstrap only when required.
- `VGRNN` runs in the `sc1-torch-template` environment because that is the available environment that matches the current serialization/runtime needs.
- Smoke tests can be shortened with environment overrides:

```bash
EG6_MAX_TARGETS=1 EG6_BUDGETS=2 ./EG6/run_sc6.sh run --detectors euler --datasets 0501 --max-jobs 1
```
