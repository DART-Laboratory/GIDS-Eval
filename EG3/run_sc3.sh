#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -eq 0 ]]; then
  set -- run
fi

EG3_PYTHON_BIN="${EG3_PYTHON_BIN:-python3}"

PYTHONPATH="${SCRIPT_DIR}:${REPO_ROOT}/src:${PYTHONPATH:-}" \
"${EG3_PYTHON_BIN}" "${SCRIPT_DIR}/run_sc3.py" "$@"
