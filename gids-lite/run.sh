#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CANDIDATE="gids-lite"
CONFIG_DIR="$SCRIPT_DIR/configs"

export PYTHONPATH="$REPO_ROOT/EG7:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

DATASETS=("0501" "cic" "lanl" "optc")
TARGET="${1:-all}"

run_dataset() {
    local ds="$1"
    local config="$CONFIG_DIR/${ds}.yaml"
    if [[ ! -f "$config" ]]; then
        echo "ERROR: config not found: $config" >&2
        return 1
    fi
    echo "=== ${CANDIDATE} / ${ds} ==="
    python3 "$REPO_ROOT/EG7/eg7_worker.py" --config "$config"
    echo ""
}

if [[ "$TARGET" == "all" ]]; then
    for ds in "${DATASETS[@]}"; do
        run_dataset "$ds"
    done
else
    run_dataset "$TARGET"
fi
