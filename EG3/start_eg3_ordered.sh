#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
. "${REPO_ROOT}/paths_env.sh"
cd "${REPO_ROOT}"

LOG_DIR="${GIDS_EVAL_WORK_ROOT}/eg3/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/eg3_run_$(date +%Y%m%d_%H%M%S).log"
LATEST_LOG="${LOG_DIR}/latest.log"
ln -sf "${LOG_FILE}" "${LATEST_LOG}"

export PYTHONPATH="${SCRIPT_DIR}:${REPO_ROOT}/src:${PYTHONPATH:-}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log "=== EG3 Ordered Run Started ==="
log "Log file: ${LOG_FILE}"

for DATASET in cic 0501 optc lanl; do
    log "--- Starting dataset: ${DATASET} ---"
    python3 "${SCRIPT_DIR}/run_sc3.py" run \
        --datasets "${DATASET}" \
        >> "${LOG_FILE}" 2>&1
    EXIT_CODE=$?
    log "--- Dataset ${DATASET} finished with exit code ${EXIT_CODE} ---"
done

log "--- Retry pass: re-running any failed jobs ---"
python3 "${SCRIPT_DIR}/run_sc3.py" run \
    >> "${LOG_FILE}" 2>&1
EXIT_CODE=$?
log "--- Retry pass finished with exit code ${EXIT_CODE} ---"

log "=== EG3 Ordered Run Finished ==="
