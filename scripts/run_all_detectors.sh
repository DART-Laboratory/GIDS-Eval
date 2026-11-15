set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

python3 -m gids_eval run \
  "${ROOT_DIR}/configs/detectors/argus.yaml" \
  "${ROOT_DIR}/configs/detectors/euler.yaml" \
  "${ROOT_DIR}/configs/detectors/pikachu.yaml" \
  "${ROOT_DIR}/configs/detectors/vgrnn.yaml" \
  "${ROOT_DIR}/configs/detectors/anomal.yaml"
