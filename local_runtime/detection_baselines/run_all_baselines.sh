#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/argus/run_all_argus.sh" "$@"
bash "${SCRIPT_DIR}/euler/run_all_euler.sh" "$@"
bash "${SCRIPT_DIR}/vgrnn/run_all_vgrnn.sh" "$@"
bash "${SCRIPT_DIR}/pikachu/run_all_pikachu.sh" "$@"
bash "${SCRIPT_DIR}/anomal/run_all_anomal.sh" "$@"
