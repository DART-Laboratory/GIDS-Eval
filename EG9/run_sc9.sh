#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -eq 0 ]]; then
  set -- run
fi

EG9_PYTHON_BIN="${EG9_PYTHON_BIN:-python3}"

PYTHONPATH="${REPO_ROOT}/src:${SCRIPT_DIR}:${PYTHONPATH:-}" \
"${EG9_PYTHON_BIN}" "${SCRIPT_DIR}/run_sc9.py" "$@"
