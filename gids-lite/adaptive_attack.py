import os
import sys, json
from pathlib import Path

import numpy as np

ROOT = os.environ.get("GIDS_EVAL_ROOT", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, ROOT + "/EG7")
sys.path.insert(0, ROOT + "/src")
import eg7_worker as W
from sklearn.preprocessing import StandardScaler

def report(mode, cost, detected, n_idx, n_attack, recall0):
    cost = np.asarray(cost, float)
    budgets = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 1e12]
    finite = cost[detected & np.isfinite(cost)]
    med = float(np.median(finite)) if finite.size else float("inf")
    frac_unev = float(np.mean(np.isinf(cost[detected]))) if detected.sum() else 0.0
    print(f"[ATTACK {mode}] recall0={recall0:.4f} detected={int(detected.sum())}/{n_attack} "
          f"median_min_cost={med:.4f} frac_unevadable={frac_unev:.3f}")
    for eps in budgets:
        retained = int(np.sum(detected & (cost > eps)))
        tag = "inf" if eps > 1e11 else f"{eps:.2f}"
        print(f"    budget(std-L2)={tag:>5}  retained_recall={retained / n_attack:.4f}")

CFG_PATH = sys.argv[1]
TARGET_FPR = float(sys.argv[2]) if len(sys.argv) > 2 else None

CAP = {}
_orig = W.run_detection
def cap_run(*, config, bundle, **kw):
    CAP["config"] = config
    CAP["bundle"] = bundle
    return _orig(config=config, bundle=bundle, **kw)
W.run_detection = cap_run

sys.argv = ["eg7_worker.py", "--config", CFG_PATH]
try:
    W.main()
except SystemExit:
    pass

b = CAP["bundle"]; cfg = CAP["config"]
rep_dim = int(b.rep_dim)
loss = str(cfg["loss"]); feat = str(cfg["featurization"])

xtr, _ = W.sanitize_feature_matrix(b.x_train)
xva, _ = W.sanitize_feature_matrix(b.x_val)
xte, _ = W.sanitize_feature_matrix(b.x_test)
sc = StandardScaler().fit(xtr)
utr = sc.transform(xtr); uva = sc.transform(xva); ute = sc.transform(xte)
yte = np.asarray(b.y_test).astype(int); yva = np.asarray(b.y_val).astype(int)

if loss == "contrastive":
    model = W.train_contrastive_head(utr, rep_dim)
    vs = W.score_linear_head(model, uva); ts = W.score_linear_head(model, ute)
elif loss == "reconstruction":
    model = W.train_reconstruction_head(utr)
    vs = W.score_reconstruction(model, uva); ts = W.score_reconstruction(model, ute)
else:
    raise SystemExit("unsupported loss " + loss)

T = float(W.select_threshold(vs, yva, int(cfg["mcc_grid_size"])))
if TARGET_FPR is not None:
    target_fpr = TARGET_FPR
    benign = ts[yte == 0]
    if target_fpr <= 0.0:
        T = float(np.max(benign)) + 1e-9
    else:
        T = float(np.quantile(benign, 1.0 - target_fpr))
pred = (ts >= T).astype(int)
n_attack = int((yte == 1).sum())
recall0 = float(((pred == 1) & (yte == 1)).sum()) / max(1, n_attack)
fpr0 = float(((pred == 1) & (yte == 0)).sum()) / max(1, int((yte == 0).sum()))
d = ute.shape[1]
print(f"[REPRO] dataset={cfg['dataset']} feat={feat} loss={loss} d={d} rep_dim={rep_dim} "
      f"n_attack={n_attack} recall0={recall0:.4f} fpr0={fpr0:.4f} thr={T:.4f}")

controllable = np.ones(d, dtype=bool)
if feat == "one_hot_structural":
    one_hot_dim = int(cfg.get("one_hot_dim", 64))
    for blk in range(4):
        s = blk * rep_dim
        controllable[s : s + one_hot_dim] = False

idx = np.where(yte == 1)[0]
ts_a = ts[idx]
detected = ts_a >= T

cost = np.full(len(idx), np.inf)
if loss == "contrastive":
    w = model.coef_[0].astype(np.float64)
    for mode, mask in (("unconstrained", np.ones(d, bool)), ("controllable", controllable)):
        wn = np.linalg.norm(w[mask])
        c = np.where(detected, np.maximum(0.0, (ts_a - T)) / (wn if wn > 0 else np.inf), 0.0)
        report(mode, c, detected, len(idx), n_attack, recall0)
elif loss == "reconstruction":
    pca = model["pca"]
    mu = pca.mean_.astype(np.float64)
    Vt = pca.components_.astype(np.float64)
    U = ute[idx].astype(np.float64)
    centred = U - mu
    proj = centred @ Vt.T @ Vt
    r = centred - proj
    rnorm = np.linalg.norm(r, axis=1)
    c_unc = np.where(detected, np.maximum(0.0, rnorm - np.sqrt(T * d)), 0.0)
    report("unconstrained", c_unc, detected, len(idx), n_attack, recall0)
    A = (np.eye(d) - Vt.T @ Vt)[:, controllable]
    Apinv = np.linalg.pinv(A)
    Td = T * d
    c_con = np.zeros(len(idx))
    for j in range(len(idx)):
        if not detected[j]:
            c_con[j] = 0.0; continue
        rj = r[j]
        dls = -(Apinv @ rj)
        p = A @ dls
        pn2 = float(p @ p)
        Rj = float(rj @ rj) - Td
        if pn2 <= 1e-12 or (Rj / pn2) > 1.0:
            c_con[j] = np.inf
        else:
            alpha = 1.0 - np.sqrt(max(0.0, 1.0 - Rj / pn2))
            c_con[j] = alpha * float(np.linalg.norm(dls))
    report("controllable", c_con, detected, len(idx), n_attack, recall0)
