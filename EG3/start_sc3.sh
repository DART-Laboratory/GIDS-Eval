#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
. "${REPO_ROOT}/paths_env.sh"
cd "${REPO_ROOT}"

LOG_DIR="${GIDS_EVAL_WORK_ROOT}/eg3/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/eg3_run_$(date +%Y%m%d_%H%M%S).log"
LATEST_LOG="${LOG_DIR}/latest.log"

export PYTHONPATH="${SCRIPT_DIR}:${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "=== EG3 Run started at $(date) ===" | tee "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
ln -sf "${LOG_FILE}" "${LATEST_LOG}"

python3 "${SCRIPT_DIR}/run_sc3.py" run "$@" >> "${LOG_FILE}" 2>&1
EXIT_CODE=$?

echo "=== EG3 Run finished at $(date) with exit code ${EXIT_CODE} ===" >> "${LOG_FILE}"
exit ${EXIT_CODE}
