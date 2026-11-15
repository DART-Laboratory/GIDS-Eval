#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFICIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${OFFICIAL_DIR}/config_env.sh"
load_mirage_config "${OFFICIAL_DIR}"

VGRNN_ROOT="${SCRIPT_DIR}"
VGRNN_DATA_ROOT="${MIRAGE_INPUT_ROOT}/vgrnn"
VGRNN_RESULTS_ROOT="${MIRAGE_OUTPUT_ROOT}/vgrnn/results"
VGRNN_MODELS_ROOT="${MIRAGE_OUTPUT_ROOT}/vgrnn/models"

LANL_OPTC_DIR="${VGRNN_ROOT}/LANL_OpTC"
CIC_DIR="${VGRNN_ROOT}/CIC"
REAL_0501_DIR="${VGRNN_ROOT}/0501"

PYTHON_BIN=${PYTHON:-python3}
VGRNN_METRICS_CSV="${VGRNN_RESULTS_ROOT}/vgrnn_metrics.csv"

mkdir -p "${VGRNN_RESULTS_ROOT}" "${VGRNN_MODELS_ROOT}"
rm -f "${VGRNN_METRICS_CSV}"

run_job() {
  local name="$1"
  local workdir="$2"
  shift 2

  echo ">> Running VGRNN ${name}"
  (
    cd "${workdir}"
    VGRNN_DATA_ROOT="${VGRNN_DATA_ROOT}" \
      VGRNN_RESULTS_ROOT="${VGRNN_RESULTS_ROOT}" \
      VGRNN_MODELS_ROOT="${VGRNN_MODELS_ROOT}" \
      VGRNN_RUN_NAME="${name}" \
      PYTHONPATH="${workdir}:${PYTHONPATH:-}" \
      ${PYTHON_BIN} run.py "$@"
  )
}

run_job LANL "${LANL_OPTC_DIR}" --dataset LANL --snapshot 1.5 --dimension 32 --lr 0.01 --patience 10 --tests 1
run_job OPTC "${LANL_OPTC_DIR}" --dataset OPTC --snapshot 0.5 --dimension 32 --lr 0.005 --patience 10 --tests 1

run_job CIC "${CIC_DIR}" --dataset CIC --snapshot 10 --dimension 32 --lr 0.01 --patience 10 --fw 0.43 --tests 1

run_job oilrig "${REAL_0501_DIR}" --dataset oilrig --snapshot 2.5 --dimension 32 --lr 0.01 --patience 10 --fw 0.43 --tests 1
run_job sandworm "${REAL_0501_DIR}" --dataset sandworm --snapshot 2.5 --dimension 32 --lr 0.001 --patience 10 --fw 0.48 --tests 1
run_job wizard "${REAL_0501_DIR}" --dataset wizard --snapshot 2.5 --dimension 32 --lr 0.005 --patience 10 --fw 0.43 --tests 1

metrics_files=()
while IFS= read -r -d '' f; do
  metrics_files+=("$f")
done < <(find "${VGRNN_RESULTS_ROOT}" -name "metrics.csv" -print0 | sort -z)

if [ ${#metrics_files[@]} -gt 0 ]; then
  head -n 1 "${metrics_files[0]}" > "${VGRNN_METRICS_CSV}"
  for f in "${metrics_files[@]}"; do
    tail -n +2 "$f" >> "${VGRNN_METRICS_CSV}"
  done
  echo "Wrote aggregated metrics to ${VGRNN_METRICS_CSV}"
else
  echo "No metrics.csv files found; did you run the jobs?"
fi

echo "VGRNN runs finished. Metrics: ${VGRNN_METRICS_CSV}"
