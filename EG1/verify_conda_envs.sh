#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/../paths_env.sh"

REPO_ROOT="${GIDS_EVAL_REPO_ROOT}"
ENV_ROOT="${GIDS_EVAL_CONDA_ROOT}/envs"
RUNTIME_ROOT="${REPO_ROOT}/local_runtime/detection_baselines"

run_import_check() {
  local env_python="$1"
  shift
  "${env_python}" - <<'PY'
import importlib
mods = [
    "numpy",
    "pandas",
    "sklearn",
    "networkx",
    "tqdm",
    "yaml",
]
for name in mods:
    importlib.import_module(name)
print("common imports OK")
PY
}

run_import_check "${ENV_ROOT}/sc1-argus/bin/python"
"${ENV_ROOT}/sc1-argus/bin/python" - <<'PY'
import torch, torch_geometric
print("argus torch imports OK")
PY

run_import_check "${ENV_ROOT}/sc1-euler/bin/python"
"${ENV_ROOT}/sc1-euler/bin/python" - <<'PY'
import torch, torch_geometric
print("euler torch imports OK")
PY

run_import_check "${ENV_ROOT}/sc1-vgrnn/bin/python"
"${ENV_ROOT}/sc1-vgrnn/bin/python" - <<'PY'
import torch, torch_geometric
print("vgrnn torch imports OK")
PY

run_import_check "${ENV_ROOT}/sc1-pikachu/bin/python"
"${ENV_ROOT}/sc1-pikachu/bin/python" - <<'PY'
import tensorflow, gensim
print("pikachu tensorflow imports OK")
PY

(
  cd "${RUNTIME_ROOT}/argus/LANL_OpTC"
  "${ENV_ROOT}/sc1-argus/bin/python" run.py --help >/dev/null
)
(
  cd "${RUNTIME_ROOT}/euler/LANL_OpTC"
  "${ENV_ROOT}/sc1-euler/bin/python" run.py --help >/dev/null
)
(
  cd "${RUNTIME_ROOT}/vgrnn/LANL_OpTC"
  "${ENV_ROOT}/sc1-vgrnn/bin/python" run.py --help >/dev/null
)
(
  cd "${RUNTIME_ROOT}/pikachu"
  "${ENV_ROOT}/sc1-pikachu/bin/python" main.py --help >/dev/null
)

(
  cd "${REPO_ROOT}"
  ./EG1/run_sc1.sh env-status --detectors argus,euler,vgrnn,pikachu
)

echo "All EG1 detector environments verified"
