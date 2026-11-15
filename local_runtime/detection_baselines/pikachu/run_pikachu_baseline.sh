#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFICIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${OFFICIAL_DIR}/config_env.sh"
load_mirage_config "${OFFICIAL_DIR}"

PYTHON_BIN=${PYTHON:-python3}
INPUT_PATH="${MIRAGE_INPUT_ROOT}/pikachu/lanl/all_20.csv"
OUTPUT_PATH="${MIRAGE_OUTPUT_ROOT}/pikachu/results/lanl/baseline"

mkdir -p "${OUTPUT_PATH}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/main.py" \
  --dataset lanl \
  --input "${INPUT_PATH}" \
  --trainwin 42 \
  --dimensions 100 \
  --alpha 0.01 \
  --support 10 \
  --output "${OUTPUT_PATH}"
