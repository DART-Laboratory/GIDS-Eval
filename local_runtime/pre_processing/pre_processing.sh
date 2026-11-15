#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OFFICIAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${OFFICIAL_DIR}/config_env.sh"
load_mirage_config "${OFFICIAL_DIR}"

run_baseline() {
  local name="$1"
  local script="$2"
  echo "=== Running preprocessing for ${name} ==="
  (cd "${SCRIPT_DIR}/${name}" && bash "${script}")
}

run_baseline "anomal" "process_anomal.sh"
run_baseline "argus" "process_argus.sh"
run_baseline "euler" "process_euler.sh"
run_baseline "pikachu" "process_pikachu.sh"
run_baseline "VGRNN" "process_vgrnn.sh"

echo "All preprocessing scripts finished."
