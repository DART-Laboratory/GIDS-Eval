#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}" python3 "${REPO_ROOT}/EG5/run_sc5.py" "$@"
