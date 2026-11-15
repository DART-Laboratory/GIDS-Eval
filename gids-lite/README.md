# gids-lite

> **Configuration.** GIDS-Lite is a control, not a tuned detector: the same
> configuration (identity+structural featurization, reconstruction head, no
> graph or temporal encoder) runs on all four datasets. Earlier revisions of
> this entry point selected a featurization and head per dataset; that
> dataset-adaptive variant is not the design reported in the paper.


Our GIDS-Lite system.

## Usage

From the repository root:

```bash
bash gids-lite/run.sh all
bash gids-lite/run.sh cic
```

From inside `gids-lite/`:

```bash
bash run.sh all
bash run.sh cic
```

## Analyses

`adaptive_attack.py` computes the minimum white-box perturbation, in
standardized feature units, that pushes each detected attack edge below the
frozen threshold, and reports retained recall as a function of the attacker
budget. The second argument optionally pins the threshold to a target test FPR
instead of the validation-calibrated one.

```bash
python3 gids-lite/adaptive_attack.py gids-lite/configs/cic.yaml
python3 gids-lite/adaptive_attack.py gids-lite/configs/cic.yaml 0.011
```

`recall_fpr_curve.py` sweeps the operating point and writes recall against the
false-positive rate to CSV.

```bash
python3 gids-lite/recall_fpr_curve.py gids-lite/configs/lanl.yaml curve_lanl.csv
```
