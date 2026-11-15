# EG8 Preprocessing Alignment

EG8 asks one question:

When every detector is forced to consume the same PIKACHU-bounded input budget, how much of the detector gap remains?

This implementation uses one unified protocol:

- Canonical source: reuse the existing EG1 PIKACHU representative-window inputs
- Fixed windows:
  - `0501 = 5m`
  - `cic = 50m`
  - `lanl = 1h`
  - `optc = 1h`
- Aligned matrix:
  - `argus`, `euler`, `pikachu`, `vgrnn` on `0501`, `cic`, `lanl`, `optc`
  - `anomal` on `0501`, `cic`
  - total `18` aligned jobs

The canonical input is "based on PIKACHU" because the retained rows come from the already validated PIKACHU-feasible preprocessing outputs. It is still fair because detector-specific extra filters, extra feature retention, and detector-specific sampling are not reintroduced downstream.

## What Gets Reused

To save time and avoid rebuilding expensive upstream preprocessing, EG8 reuses the following EG1 PIKACHU outputs as the canonical source:

- `eg1_pikachu_0501_5m_s00`
- `eg1_pikachu_cic_50m_s00`
- `eg1_pikachu_lanl_1h_s00`
- `eg1_pikachu_optc_1h_s00`

EG8 then writes thin detector adapters from those canonical CSVs:

- graph-detector roots for `argus`, `euler`, `vgrnn`
- direct canonical metadata for `pikachu`
- compact tabular inputs for `anomal`

## Layout

Runtime artifacts live under `<work_root>/eg8/`.

- `canonical/`: reused PIKACHU-bounded canonical inputs
- `shared_graph/`: shared graph-format adapters
- `adapters/`: detector-facing adapter roots
- `experiments/`: one workspace per EG8 job
- `orchestrator/`: global matrix, status, progress, and launch logs
- `reports/`: aligned/native comparison reports

## Logging

EG8 writes two log layers during runtime:

- Global progress logs:
  - `orchestrator/job_status.csv`
  - `orchestrator/progress.md`
  - `orchestrator/full_run.log`
- Per-job logs:
  - `experiments/<job>/eg8/metadata/logs/prepare.stdout.log`
  - `experiments/<job>/eg8/metadata/logs/prepare.stderr.log`
  - `experiments/<job>/eg8/metadata/logs/detection.stdout.log`
  - `experiments/<job>/eg8/metadata/logs/detection.stderr.log`

## Commands

Run with the repository Python:

```bash
python3 <repo_root>/EG8/run_sc8.py matrix
python3 <repo_root>/EG8/run_sc8.py status
python3 <repo_root>/EG8/run_sc8.py progress
python3 <repo_root>/EG8/run_sc8.py summarize
python3 <repo_root>/EG8/run_sc8.py run
```

Filtered debug examples:

```bash
python3 <repo_root>/EG8/run_sc8.py run --detectors argus --datasets 0501 --max-jobs 1 --stop-on-error
python3 <repo_root>/EG8/run_sc8.py run --detectors anomal --datasets cic --max-jobs 1 --stop-on-error
```

Background launch helper:

```bash
bash <repo_root>/EG8/run_sc8.sh
```

## Outputs

`summarize` writes:

- `reports/raw_results.csv`
- `reports/delta_native_vs_aligned.csv`
- `reports/summary_by_detector.csv`
- `reports/summary_by_dataset.csv`
- `reports/evidence_table.csv`
- `reports/conclusion.md`

