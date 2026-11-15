#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFICIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${OFFICIAL_DIR}/config_env.sh"
load_mirage_config "${OFFICIAL_DIR}"

ARGUS_ROOT="${SCRIPT_DIR}"
ARGUS_DATA_ROOT="${MIRAGE_INPUT_ROOT}/argus"
ARGUS_RESULTS_ROOT="${MIRAGE_OUTPUT_ROOT}/argus/results"
ARGUS_MODELS_ROOT="${MIRAGE_OUTPUT_ROOT}/argus/models"

LANL_OPTC_DIR="${ARGUS_ROOT}/LANL_OpTC"
CIC_DIR="${ARGUS_ROOT}/CIC"
REAL_0501_DIR="${ARGUS_ROOT}/0501"

PYTHON_BIN=${PYTHON:-python3}
ARGUS_METRICS_CSV="${ARGUS_RESULTS_ROOT}/argus_metrics.csv"

mkdir -p "${ARGUS_RESULTS_ROOT}" "${ARGUS_MODELS_ROOT}"
rm -f "${ARGUS_METRICS_CSV}"

run_job() {
  local name="$1"
  local workdir="$2"
  shift 2

  echo ">> Running ARGUS ${name}"
  (
    cd "${workdir}"
    ARGUS_DATA_ROOT="${ARGUS_DATA_ROOT}" \
      ARGUS_RESULTS_ROOT="${ARGUS_RESULTS_ROOT}" \
      ARGUS_MODELS_ROOT="${ARGUS_MODELS_ROOT}" \
      ARGUS_RUN_NAME="${name}" \
      ${PYTHON_BIN} run.py "$@"
  )
}

run_job LANL "${LANL_OPTC_DIR}" --dataset LANL --hidden 32 -z 16 --rnn GRU --delta 1 --lr 0.01 --fpweight 0.6 --epoch 100 --patience 3
run_job OPTC "${LANL_OPTC_DIR}" --dataset OPTC --hidden 32 -z 16 --rnn GRU --delta 0.1 --lr 0.005 --fpweight 0.6 --epoch 100 --patience 10
run_job CIC "${CIC_DIR}" --dataset O_cic --delta 15 --lr 0.01 --hidden 32 -z 16 --fpweight 0.46 --epoch 100 --patience 3

run_job oilrig "${REAL_0501_DIR}" --dataset O_oilrig --hidden 32 -z 16 --rnn GRU --delta 7.5 --lr 0.01 --fpweight 0.48 --epoch 100 --patience 5
run_job sandworm "${REAL_0501_DIR}" --dataset O_sandworm --hidden 16 -z 8 --rnn GRU --delta 2.5 --lr 0.0001 --fpweight 0.43 --epoch 100 --patience 3
run_job wizard "${REAL_0501_DIR}" --dataset O_wizard --hidden 16 -z 8 --rnn GRU --delta 2.5 --lr 0.001 --fpweight 0.48 --epoch 100 --patience 3

metrics_files=()
while IFS= read -r -d '' f; do
  metrics_files+=("$f")
done < <(find "${ARGUS_RESULTS_ROOT}" -name "metrics.csv" -print0 | sort -z)

if [ ${#metrics_files[@]} -gt 0 ]; then
  head -n 1 "${metrics_files[0]}" > "${ARGUS_METRICS_CSV}"
  for f in "${metrics_files[@]}"; do
    tail -n +2 "$f" >> "${ARGUS_METRICS_CSV}"
  done
  echo "Wrote aggregated metrics to ${ARGUS_METRICS_CSV}"
else
  echo "No metrics.csv files found; did you run the jobs?"
fi

echo "ARGUS runs finished. Metrics: ${ARGUS_METRICS_CSV}"
