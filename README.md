# GIDS-EVAL Artifact

<p>
  <img src="docs/badges/acm-artifacts-available.png" alt="ACM Artifacts Available" height="100">
  <img src="docs/badges/acm-artifacts-evaluated-functional.png" alt="ACM Artifacts Evaluated: Functional" height="100">
  <img src="docs/badges/acm-results-reproduced.png" alt="ACM Results Reproduced" height="100">
</p>

This repository contains the artifact implementation for **GIDS-EVAL**, the evaluation framework used in the paper:

> **A First-Principles Evaluation of Graph-Based Network Intrusion Detection Systems**
>
> Paper: [https://doi.org/10.1145/3830454.3846576](https://doi.org/10.1145/3830454.3846576)

---

## Citation

```bibtex
@inproceedings{zhao2026gidseval,
  author    = {Zhao, Rui and Hassan, Wajih Ul},
  title     = {A First-Principles Evaluation of Graph-Based Network Intrusion Detection Systems},
  booktitle = {Proceedings of the 2026 ACM SIGSAC Conference on Computer and Communications Security (CCS '26)},
  year      = {2026},
  address   = {The Hague, Netherlands},
  publisher = {ACM},
  doi       = {10.1145/3830454.3846576}
}
```

---

## Repository Layout

```text
GIDS-Eval/
  README.md
  pyproject.toml
  gids_eval.yaml
  paths_env.sh
  src/gids_eval/
  configs/
    data/
    detectors/
    experiments/
  local_runtime/
  EG1/ ... EG9/
  gids-lite/
  scripts/
  tests/
```

Runtime outputs are intentionally written outside the source tree, under the user-supplied `work_root`.

---

## Configuration

All filesystem locations are resolved through a single root config file at the repository root:

```text
gids_eval.yaml
```

After cloning the artifact, edit `gids_eval.yaml` and fill in the required fields:

```yaml
repo_root: ""
raw_data_root: ""
work_root: ""
mirage_official_root: ""
conda_root: ""
eg6_official_rq5_root: ""
```

| Field | Required | Meaning |
| --- | --- | --- |
| `repo_root` | optional | Absolute path to this repo. Defaults to the directory containing `gids_eval.yaml`. |
| `raw_data_root` | required | Directory holding the raw datasets (`LANL/`, `OpTC/`, `CIC-new/`, `0501/`). |
| `work_root` | required | Output base directory. EG tracks write to `<work_root>/eg1/`, `<work_root>/eg2/`, etc. Detector workspaces, caches, conda envs, and reports also live here. |
| `mirage_official_root` | optional | Mirage official drop directory used by EG3. Defaults to `<raw_data_root>/../official`. |
| `conda_root` | optional | Where Miniforge and detector envs live. Defaults to `<work_root>/conda`. |
| `eg6_official_rq5_root` | optional | Robustness-assessment drop used by EG6. Defaults to `<work_root>/eg6_official_rq5`. |

Any field can also be overridden through an environment variable of the form `GIDS_EVAL_<FIELD>` (e.g., `GIDS_EVAL_RAW_DATA_ROOT`). Environment variables win over the YAML.

Python entry points read the config through `gids_eval.paths.get_paths()`. Shell entry points source `paths_env.sh` from the repo root, which exports:

```text
GIDS_EVAL_REPO_ROOT
GIDS_EVAL_RAW_DATA_ROOT
GIDS_EVAL_WORK_ROOT
GIDS_EVAL_MIRAGE_OFFICIAL_ROOT
GIDS_EVAL_CONDA_ROOT
GIDS_EVAL_EG6_OFFICIAL_RQ5_ROOT
GIDS_EVAL_RUNTIME_ROOT
GIDS_EVAL_CONFIG
```

The dataset manifests under `configs/data/` and the modular experiment configs under `configs/experiments/` use `${raw_data_root}`, `${work_root}`, `${repo_root}`, etc., as template placeholders that the loader expands at runtime, so they do not need to be edited per machine.

---

## Requirements

The artifact is designed for Linux execution. The code was organized around the following environment:

- Ubuntu 20.04 or 22.04
- Python 3.10 or newer
- Bash
- Git
- Conda or Mamba for detector-specific environments
- Sufficient disk space for raw datasets, staged detector inputs, cached module outputs, and model artifacts

The core `gids_eval` runner has a small Python dependency surface:

```bash
cd GIDS-Eval
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

After installing, fill in `gids_eval.yaml` (see Configuration above) before running any sanity check or experiment.

The vendored detectors require their original ML stacks. The helper scripts bootstrap the main CPU-oriented environments used by the EG runners:

```bash
cd GIDS-Eval
bash EG1/setup_conda_envs.sh
bash EG1/verify_conda_envs.sh
```

EG2 reuses the EG1 detector environments. Other EG runners use the same staged runtime layout and document any additional assumptions in their local `README.md` files.

---

## Datasets

GIDS-EVAL is configured for the datasets used in the paper evaluation, including:

- `LANL`
- `OpTC`
- `CIC-IDS-2017`
- the `0501` enterprise attack traces, represented by `OilRig`, `Sandworm`, and `WizardSpider` campaign manifests

Dataset manifests live in:

```text
configs/data/
```

Each manifest declares a dataset identifier, family, display name, and the artifacts that should be staged into a detector workspace. Example:

```yaml
schema_version: 1
dataset_id: lanl
family: lanl
display_name: LANL
stage_root: ${stage_root}
artifacts:
  - name: lanl_dir
    kind: directory
    source: <raw_data_root>/LANL
    target: raw_data/LANL
```

The `${raw_data_root}` placeholder is resolved from `gids_eval.yaml`, so the manifests do not need editing per machine. Override the dataset family layout (e.g., a non-standard `LANL/` directory name) by editing the `source` field of the affected manifest.

---

## Quick Sanity Checks

Validate all baseline detector configs without launching detector code:

```bash
cd GIDS-Eval
PYTHONPATH=src python3 -m gids_eval validate configs/detectors/*.yaml
```

Validate modular experiment configs:

```bash
PYTHONPATH=src python3 -m gids_eval validate configs/experiments/*.yaml
```

Resolve an experiment without executing it:

```bash
PYTHONPATH=src python3 -m gids_eval run --dry-run configs/experiments/smoke_pipeline.yaml
```

These commands are the recommended first checks after unpacking the artifact and adjusting dataset paths.

---

## Running Baseline Detectors

Each baseline detector has a YAML config under:

```text
configs/detectors/
```

Run one detector:

```bash
cd GIDS-Eval
PYTHONPATH=src python3 -m gids_eval run --clean configs/detectors/argus.yaml
```

Run all configured detectors:

```bash
PYTHONPATH=src bash scripts/run_all_detectors.sh
```

The runner stages manifest inputs, executes detector-specific preprocessing, launches the baseline jobs, and exports standardized reports.

---

## Modular Experiments

The modular framework represents every experiment as a six-stage pipeline:

1. `graph_construction`
2. `featurization`
3. `graph_encoder`
4. `temporal_encoder`
5. `loss`
6. `detection`

Run a modular detector pipeline:

```bash
cd GIDS-Eval
PYTHONPATH=src python3 -m gids_eval run --clean configs/experiments/argus_pipeline.yaml
```

Run only through a selected stage:

```bash
PYTHONPATH=src python3 -m gids_eval run --clean --to-stage featurization configs/experiments/argus_pipeline.yaml
```

Dry-run all five detector pipelines:

```bash
PYTHONPATH=src python3 -m gids_eval run --dry-run configs/experiments/*_pipeline.yaml
```

The runner computes stage-level cache keys from the stage config, upstream cache key, stable runtime inputs, and manifest fingerprint. If only a later stage changes, earlier stages can be restored from cache instead of recomputed.

---

## Shortcomings Experiments

The paper structures the evaluation around nine shortcomings tracks. Each track has a dedicated directory and resumable runner:

| Track | Topic | Entry point |
| --- | --- | --- |
| `EG1` | Snapshot sensitivity | `bash EG1/run_sc1.sh run` |
| `EG2` | Threshold stability | `bash EG2/run_sc2.sh run` |
| `EG3` | Inductive calibration | `bash EG3/run_sc3.sh run` |
| `EG4` | Edge feature robustness | `bash EG4/run_sc4.sh run` |
| `EG5` | Scalability measurement | `bash EG5/run_sc5.sh run` |
| `EG6` | Minimal adversarial perturbation | `bash EG6/run_sc6.sh run` |
| `EG7` | Component ablations | `bash EG7/run_sc7.sh run` |
| `EG8` | Preprocessing alignment | `bash EG8/run_sc8.sh run` |
| `EG9` | Streaming readiness | `bash EG9/run_sc9.sh run` |

Most EG runners support the same workflow:

```bash
bash EGx/run_scx.sh matrix
bash EGx/run_scx.sh run
bash EGx/run_scx.sh progress
bash EGx/run_scx.sh summarize
```

See each `EG*/README.md` for the exact experiment matrix, filters, output paths, and reuse assumptions.

---

## gids-lite

The `gids-lite/` directory contains the encoder-free control detector.

- one validation-selected configuration, applied unchanged to every dataset
- no graph encoder
- no temporal encoder

Run gids-lite on all supported datasets:

```bash
cd GIDS-Eval
bash gids-lite/run.sh all
```

Run one dataset:

```bash
bash gids-lite/run.sh cic
```

The system configs are stored under `gids-lite/configs/`.

---

## Outputs

For baseline detector configs, the runner writes:

```text
<work_root>/workspaces/<detector>/
  input/                      # staged raw data and detector preprocessing inputs
  output/                     # detector outputs, models, and native reports
  metadata/baseline/          # resolved config, logs, metrics, run summary
```

For modular experiments, the runner writes:

```text
<work_root>/experiments/<experiment_id>/<run_label>/
  input/                      # staged manifests
  output/modules/<stage>/     # per-stage outputs
  output/final                # symlink to the final executed stage
  metadata/                   # resolved config, per-stage logs, run summary
```

The main standardized report files are:

- `baseline_metrics.csv`: AP, AUC, thresholded metrics, and detector-specific quality metrics when available
- `cost_metrics.csv`: wall-clock time, cache status, memory, and model-reported cost metrics when available
- `run_summary.json`: resolved execution metadata and artifact locations

Shortcomings-specific reports are stored under the corresponding `<work_root>/eg*/` output directories.

