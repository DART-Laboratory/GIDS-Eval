#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFICIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${OFFICIAL_DIR}/config_env.sh"
load_mirage_config "${OFFICIAL_DIR}"

DATA_ROOT="${MIRAGE_INPUT_ROOT}/euler"
RESULTS_ROOT="${MIRAGE_OUTPUT_ROOT}/euler/results"
PYTHON_BIN=${PYTHON:-python3}
GPU=""
WORKERS="4"

export CUDA_VISIBLE_DEVICES="${GPU}"
export TP_ENABLE_SHM=0
export TP_DISABLE_SHM=1
export EULER_DATA_ROOT="${DATA_ROOT}"
export EULER_RESULTS_ROOT="${RESULTS_ROOT}"

mkdir -p "${RESULTS_ROOT}"

METRICS_CSV="${RESULTS_ROOT}/euler_metrics.csv"
rm -f "${METRICS_CSV}"

run_job() {
  local subdir="$1"
  local dataset="$2"
  shift 2
  local script="${SCRIPT_DIR}/${subdir}/run.py"

  echo ">> Running ${dataset} (${subdir})"
  "${PYTHON_BIN}" "${script}" --dataset "${dataset}" "$@"
}


run_job CIC CIC --delta 50 --workers ${WORKERS} --threads 1 --lr 0.005 --fpweight 0.7 --patience 5 --tests 1 --tr-start 0 --tr-end 50000 --val-end 55000

run_job 0501 oilrig   --delta 2.5 --workers ${WORKERS} --threads 1 --lr 0.005 --fpweight 0.48 --tests 1
run_job 0501 sandworm --delta 2.5 --workers ${WORKERS} --threads 1 --lr 0.005 --fpweight 0.42 --tests 1
run_job 0501 wizard   --delta 2.5 --workers ${WORKERS} --threads 1 --lr 0.005 --fpweight 0.47 --tests 1

run_job LANL_OpTC LANL --delta 0.5 --workers ${WORKERS} --threads 1 --lr 0.005 --fpweight 0.6 --patience 5 --tests 1
run_job LANL_OpTC OPTC --delta 2.5 --workers ${WORKERS} --threads 1 --rnn LSTM --lr 0.005 --fpweight 0.6 --patience 5 --tests 1

metrics_files=()
while IFS= read -r -d '' f; do
  metrics_files+=("$f")
done < <(find "${RESULTS_ROOT}" -path "*/learning/metrics.csv" -print0 | sort -z)

if [ ${#metrics_files[@]} -gt 0 ]; then
  head -n 1 "${metrics_files[0]}" > "${METRICS_CSV}"
  for f in "${metrics_files[@]}"; do
    tail -n +2 "$f" >> "${METRICS_CSV}"
  done
  echo "Wrote aggregated metrics to ${METRICS_CSV}"
else
  echo "No metrics.csv files found; did you run the jobs?"
fi

echo "All Euler runs finished. Metrics: ${METRICS_CSV}"
