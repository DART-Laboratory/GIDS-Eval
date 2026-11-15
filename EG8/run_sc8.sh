#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/../paths_env.sh"

REPO_ROOT="${GIDS_EVAL_REPO_ROOT}"
DATA_ROOT="${GIDS_EVAL_WORK_ROOT}/eg8"
ORCH_ROOT="${DATA_ROOT}/orchestrator"
LOG_PATH="${ORCH_ROOT}/full_run.log"

mkdir -p "${ORCH_ROOT}"

{
  echo "[$(date --iso-8601=seconds)] EG8 full run starting"
  python3 "${REPO_ROOT}/EG8/run_sc8.py" run
  python3 "${REPO_ROOT}/EG8/run_sc8.py" summarize
  python3 "${REPO_ROOT}/EG8/run_sc8.py" progress
  echo "[$(date --iso-8601=seconds)] EG8 full run finished"
} >> "${LOG_PATH}" 2>&1
