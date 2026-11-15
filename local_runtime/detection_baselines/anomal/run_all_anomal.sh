#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFICIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${OFFICIAL_DIR}/config_env.sh"
load_mirage_config "${OFFICIAL_DIR}"

ANOMAL_ROOT="${SCRIPT_DIR}"
ANOMAL_DATA_ROOT="${MIRAGE_INPUT_ROOT}/anomal"
ANOMAL_RESULTS_ROOT="${MIRAGE_OUTPUT_ROOT}/anomal/results"
ANOMAL_MODELS_ROOT="${MIRAGE_OUTPUT_ROOT}/anomal/models"

CIC_DIR="${ANOMAL_ROOT}/CIC"
REAL_0501_DIR="${ANOMAL_ROOT}/0501"

PYTHON_BIN=${PYTHON:-python3}
ANOMAL_METRICS_CSV="${ANOMAL_RESULTS_ROOT}/anomal_metrics.csv"

mkdir -p "${ANOMAL_RESULTS_ROOT}" "${ANOMAL_MODELS_ROOT}"
rm -f "${ANOMAL_METRICS_CSV}"

run_job() {
  local name="$1"
  local workdir="$2"
  shift 2

  echo ">> Running Anomal-E ${name}"
  (
    cd "${workdir}"
    ANOMAL_DATA_ROOT="${ANOMAL_DATA_ROOT}" \
      ANOMAL_RESULTS_ROOT="${ANOMAL_RESULTS_ROOT}" \
      ANOMAL_MODELS_ROOT="${ANOMAL_MODELS_ROOT}" \
      ANOMAL_RUN_NAME="${name}" \
      PYTHONPATH="${workdir}:${PYTHONPATH:-}" \
      ${PYTHON_BIN} run.py "$@"
  )
}

run_job CIC "${CIC_DIR}" --dataset cic

run_job oilrig "${REAL_0501_DIR}" --dataset oilrig
run_job sandworm "${REAL_0501_DIR}" --dataset sandworm
run_job wizard "${REAL_0501_DIR}" --dataset wizard

metrics_files=()
while IFS= read -r -d '' f; do
  metrics_files+=("$f")
done < <(find "${ANOMAL_RESULTS_ROOT}" -name "metrics.csv" -print0 | sort -z)

if [ ${#metrics_files[@]} -gt 0 ]; then
  head -n 1 "${metrics_files[0]}" > "${ANOMAL_METRICS_CSV}"
  for f in "${metrics_files[@]}"; do
    tail -n +2 "$f" >> "${ANOMAL_METRICS_CSV}"
  done
  echo "Wrote aggregated metrics to ${ANOMAL_METRICS_CSV}"
else
  echo "No metrics.csv files found; did you run the jobs?"
fi

echo "Anomal-E runs finished. Metrics: ${ANOMAL_METRICS_CSV}"
