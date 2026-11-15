#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/../paths_env.sh"

ROOT_DATA="${GIDS_EVAL_WORK_ROOT}"
CONDA_ROOT="${GIDS_EVAL_CONDA_ROOT}"
MINIFORGE_ROOT="${CONDA_ROOT}/miniforge3"
MINIFORGE_INSTALLER="/tmp/miniforge.sh"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
MAMBA_BIN="${MINIFORGE_ROOT}/bin/mamba"
ENV_ROOT="${CONDA_ROOT}/envs"
TORCH_TEMPLATE="${ENV_ROOT}/sc1-torch-template"

if [[ ! -x "${MAMBA_BIN}" ]]; then
  mkdir -p "${CONDA_ROOT}"
  echo "[EG2] Downloading Miniforge..."
  wget -q -O "${MINIFORGE_INSTALLER}" "${MINIFORGE_URL}"
  echo "[EG2] Installing Miniforge..."
  bash "${MINIFORGE_INSTALLER}" -b -p "${MINIFORGE_ROOT}"
fi

mkdir -p "${ENV_ROOT}"

if [[ ! -d "${TORCH_TEMPLATE}" ]]; then
  echo "[EG2] Creating PyTorch template environment..."
  "${MAMBA_BIN}" create -y --override-channels \
    -p "${TORCH_TEMPLATE}" \
    -c pyg -c pytorch -c defaults -c conda-forge \
    python=3.10 \
    numpy=1.26.4 \
    pandas \
    scikit-learn \
    scipy \
    tqdm \
    networkx \
    pyyaml \
    psutil \
    joblib \
    pillow \
    requests \
    intel-openmp=2023.1.0 \
    mkl=2023.1.0 \
    mkl-include=2023.1.0 \
    cpuonly \
    pytorch=1.13.0 \
    torchvision=0.14.0 \
    pyg=2.2.0
fi

for detector in argus euler; do
  target="${ENV_ROOT}/sc1-${detector}"
  if [[ -d "${target}" ]]; then
    echo "[EG2] Environment ${target} already exists, skipping."
    continue
  fi
  echo "[EG2] Cloning template -> ${target}..."
  "${MAMBA_BIN}" create -y -p "${target}" --clone "${TORCH_TEMPLATE}"
done

echo "[EG2] Conda environments ready under ${ENV_ROOT}"
