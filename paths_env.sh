_GIDS_EVAL_PATHS_ENV_SOURCED="${_GIDS_EVAL_PATHS_ENV_SOURCED:-0}"
if [[ "${_GIDS_EVAL_PATHS_ENV_SOURCED}" == "1" ]]; then
  return 0 2>/dev/null || true
fi

_gids_eval_script_dir() {
  local src="${BASH_SOURCE[0]}"
  while [[ -h "${src}" ]]; do
    local dir
    dir="$(cd -P "$(dirname "${src}")" && pwd)"
    src="$(readlink "${src}")"
    [[ "${src}" != /* ]] && src="${dir}/${src}"
  done
  cd -P "$(dirname "${src}")" && pwd
}

_gids_eval_yaml_get() {
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

_gids_eval_abspath() {
  local path="$1"
  if [[ -z "${path}" ]]; then
    printf '%s\n' ""
    return 0
  fi
  case "${path}" in
    ~*)
      path="${HOME}${path:1}"
      ;;
  esac
  if command -v realpath >/dev/null 2>&1; then
    realpath -m "${path}"
    return 0
  fi
  if [[ "${path}" == /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "$(pwd)/${path}"
  fi
}

_gids_eval_require() {
  local var_name="$1"
  local key="$2"
  local file="$3"
  if [[ -z "${!var_name:-}" ]]; then
    echo "ERROR: ${key} is not set; edit ${file} or export ${var_name}" >&2
    return 1
  fi
}

load_gids_eval_paths() {
  local explicit="${1:-${GIDS_EVAL_CONFIG:-}}"
  local script_dir
  script_dir="$(_gids_eval_script_dir)"

  local config_file="${explicit}"
  if [[ -z "${config_file}" ]]; then
    local search_dir="${PWD}"
    while [[ "${search_dir}" != "/" && "${search_dir}" != "" ]]; do
      if [[ -f "${search_dir}/gids_eval.yaml" ]]; then
        config_file="${search_dir}/gids_eval.yaml"
        break
      fi
      search_dir="$(dirname "${search_dir}")"
    done
    if [[ -z "${config_file}" && -f "${script_dir}/gids_eval.yaml" ]]; then
      config_file="${script_dir}/gids_eval.yaml"
    fi
  fi

  if [[ -z "${config_file}" || ! -f "${config_file}" ]]; then
    echo "ERROR: gids_eval.yaml not found; pass path or set GIDS_EVAL_CONFIG" >&2
    return 1
  fi

  local cfg_repo cfg_raw cfg_work cfg_off cfg_conda cfg_sc6
  cfg_repo="$(_gids_eval_yaml_get repo_root "${config_file}")"
  cfg_raw="$(_gids_eval_yaml_get raw_data_root "${config_file}")"
  cfg_work="$(_gids_eval_yaml_get work_root "${config_file}")"
  cfg_off="$(_gids_eval_yaml_get mirage_official_root "${config_file}")"
  cfg_conda="$(_gids_eval_yaml_get conda_root "${config_file}")"
  cfg_sc6="$(_gids_eval_yaml_get eg6_official_rq5_root "${config_file}")"

  : "${GIDS_EVAL_REPO_ROOT:=${cfg_repo:-$(dirname "${config_file}")}}"
  : "${GIDS_EVAL_RAW_DATA_ROOT:=${cfg_raw}}"
  : "${GIDS_EVAL_WORK_ROOT:=${cfg_work}}"
  : "${GIDS_EVAL_MIRAGE_OFFICIAL_ROOT:=${cfg_off}}"
  : "${GIDS_EVAL_CONDA_ROOT:=${cfg_conda}}"
  : "${GIDS_EVAL_EG6_OFFICIAL_RQ5_ROOT:=${cfg_sc6}}"

  GIDS_EVAL_REPO_ROOT="$(_gids_eval_abspath "${GIDS_EVAL_REPO_ROOT}")"
  GIDS_EVAL_RAW_DATA_ROOT="$(_gids_eval_abspath "${GIDS_EVAL_RAW_DATA_ROOT}")"
  GIDS_EVAL_WORK_ROOT="$(_gids_eval_abspath "${GIDS_EVAL_WORK_ROOT}")"

  _gids_eval_require GIDS_EVAL_REPO_ROOT repo_root "${config_file}" || return 1
  _gids_eval_require GIDS_EVAL_RAW_DATA_ROOT raw_data_root "${config_file}" || return 1
  _gids_eval_require GIDS_EVAL_WORK_ROOT work_root "${config_file}" || return 1

  if [[ -z "${GIDS_EVAL_MIRAGE_OFFICIAL_ROOT}" ]]; then
    GIDS_EVAL_MIRAGE_OFFICIAL_ROOT="$(_gids_eval_abspath "$(dirname "${GIDS_EVAL_RAW_DATA_ROOT}")/official")"
  else
    GIDS_EVAL_MIRAGE_OFFICIAL_ROOT="$(_gids_eval_abspath "${GIDS_EVAL_MIRAGE_OFFICIAL_ROOT}")"
  fi

  if [[ -z "${GIDS_EVAL_CONDA_ROOT}" ]]; then
    GIDS_EVAL_CONDA_ROOT="$(_gids_eval_abspath "${GIDS_EVAL_WORK_ROOT}/conda")"
  else
    GIDS_EVAL_CONDA_ROOT="$(_gids_eval_abspath "${GIDS_EVAL_CONDA_ROOT}")"
  fi

  if [[ -z "${GIDS_EVAL_EG6_OFFICIAL_RQ5_ROOT}" ]]; then
    GIDS_EVAL_EG6_OFFICIAL_RQ5_ROOT="$(_gids_eval_abspath "${GIDS_EVAL_WORK_ROOT}/eg6_official_rq5")"
  else
    GIDS_EVAL_EG6_OFFICIAL_RQ5_ROOT="$(_gids_eval_abspath "${GIDS_EVAL_EG6_OFFICIAL_RQ5_ROOT}")"
  fi

  GIDS_EVAL_RUNTIME_ROOT="${GIDS_EVAL_REPO_ROOT}/local_runtime"
  GIDS_EVAL_CONFIG="${config_file}"

  export GIDS_EVAL_REPO_ROOT GIDS_EVAL_RAW_DATA_ROOT GIDS_EVAL_WORK_ROOT
  export GIDS_EVAL_MIRAGE_OFFICIAL_ROOT GIDS_EVAL_CONDA_ROOT
  export GIDS_EVAL_EG6_OFFICIAL_RQ5_ROOT GIDS_EVAL_RUNTIME_ROOT GIDS_EVAL_CONFIG

  _GIDS_EVAL_PATHS_ENV_SOURCED=1
}

load_gids_eval_paths "$@"
