#!/usr/bin/env bash
set -euo pipefail

_cfg_get() {
  local key="$1"
  local file="$2"
  if [[ ! -f "${file}" ]]; then
    return 0
  fi
  awk -v k="${key}" '
    {
      line = $0
      sub(/[[:space:]]*#.*$/, "", line)
      if (line ~ ("^[[:space:]]*" k "[[:space:]]*:")) {
        sub("^[[:space:]]*" k "[[:space:]]*:[[:space:]]*", "", line)
        gsub(/^[\"\047]|[\"\047]$/, "", line)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)
        print line
        exit
      }
    }
  ' "${file}"
}

_abs_path() {
  local path="$1"
  if [[ -z "${path}" ]]; then
    printf '%s\n' ""
    return 0
  fi
  if command -v realpath >/dev/null 2>&1; then
    realpath -m "${path}"
    return 0
  fi
  if [[ "${path}" == /* ]]; then
    printf '%s\n' "${path}"
    return 0
  fi
  printf '%s\n' "$(pwd)/${path}"
}

load_mirage_config() {
  local official_dir="${1:-}"
  if [[ -z "${official_dir}" ]]; then
    official_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  fi
  local config_file="${official_dir}/config.yml"
  local default_root
  default_root="$(cd "${official_dir}/.." && pwd)"

  local cfg_input cfg_output
  cfg_input="$(_cfg_get "input_root" "${config_file}")"
  cfg_output="$(_cfg_get "output_root" "${config_file}")"

  if [[ -z "${cfg_input}" ]]; then
    cfg_input="${default_root}"
  fi
  if [[ -z "${cfg_output}" ]]; then
    cfg_output="${cfg_input}"
  fi

  cfg_input="$(_abs_path "${cfg_input}")"
  cfg_output="$(_abs_path "${cfg_output}")"

  export MIRAGE_INPUT_ROOT="${MIRAGE_INPUT_ROOT:-${cfg_input}}"
  export MIRAGE_OUTPUT_ROOT="${MIRAGE_OUTPUT_ROOT:-${cfg_output}}"
}
