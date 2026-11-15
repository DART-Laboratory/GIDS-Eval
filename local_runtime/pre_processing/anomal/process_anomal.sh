#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFICIAL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${OFFICIAL_DIR}/config_env.sh"
load_mirage_config "${OFFICIAL_DIR}"

RAW_DIR="${MIRAGE_INPUT_ROOT}/raw_data"
OUT_BASE="${MIRAGE_INPUT_ROOT}/anomal"
PYTHON_BIN=${PYTHON:-python3}

mkdir -p "${OUT_BASE}"/{0501,cic}

"${PYTHON_BIN}" process_CIC_anomal.py -i "${RAW_DIR}/CIC-new" -o "${OUT_BASE}/cic/"

"${PYTHON_BIN}" process_0501_anomal.py --input "${RAW_DIR}/0501/label_oilrig.log" --name oilrig --o "${OUT_BASE}/0501"
"${PYTHON_BIN}" process_0501_anomal.py --input "${RAW_DIR}/0501/label_sandworm.log" --name sandworm --o "${OUT_BASE}/0501"
"${PYTHON_BIN}" process_0501_anomal.py --input "${RAW_DIR}/0501/label_wizard_spider.log" --name wizard_spider --o "${OUT_BASE}/0501"
