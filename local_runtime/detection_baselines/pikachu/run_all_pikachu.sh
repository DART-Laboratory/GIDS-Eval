#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFICIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${OFFICIAL_DIR}/config_env.sh"
load_mirage_config "${OFFICIAL_DIR}"

DATA_ROOT="${MIRAGE_INPUT_ROOT}/pikachu"
RESULTS_ROOT="${MIRAGE_OUTPUT_ROOT}/pikachu/results"
PYTHON_BIN=${PYTHON:-python3}
GPU="${GPU:-0,1}"
GPU_ARG=()
if [[ -n "${GPU}" ]]; then
  GPU_ARG=(--gpu "${GPU}")
fi

mkdir -p "${RESULTS_ROOT}"

METRICS_CSV="${RESULTS_ROOT}/pikachu_metrics.csv"
rm -f "${METRICS_CSV}"

run_job() {
  local dataset=$1
  local input=$2
  local trainwin=$3
  local dimensions=$4
  local alpha=$5
  local support=$6
  local out_dir="${RESULTS_ROOT}/${dataset}"

  mkdir -p "${out_dir}"
  echo ">> Running ${dataset} | input=${input} | trainwin=${trainwin}"
  ${PYTHON_BIN} "${SCRIPT_DIR}/main.py" \
    --dataset "${dataset}" \
    --input "${input}" \
    --trainwin "${trainwin}" \
    --dimensions "${dimensions}" \
    --alpha "${alpha}" \
    --support "${support}" \
    --output "${out_dir}" \
    --metrics_csv "${METRICS_CSV}" \
    "${GPU_ARG[@]}"
}

run_job oilrig        "${DATA_ROOT}/0501/oilrig_300.csv"        242 100 0.01 10
run_job sandworm      "${DATA_ROOT}/0501/sandworm_300.csv"      242 100 0.01 10
run_job wizard_spider "${DATA_ROOT}/0501/wizard_spider_300.csv" 243 100 0.01 10

run_job cic "${DATA_ROOT}/cic/cic_20.csv" 11 100 0.01 10

run_job lanl "${DATA_ROOT}/lanl/all_20.csv" 42 100 0.01 10

run_job optc "${DATA_ROOT}/optc/optc.csv" 150 100 0.01 10

echo "All Pikachu runs finished. Metrics: ${METRICS_CSV}"
