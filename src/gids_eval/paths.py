from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import yaml

CONFIG_FILENAME = "gids_eval.yaml"

ENV_REPO_ROOT = "GIDS_EVAL_REPO_ROOT"
ENV_RAW_DATA_ROOT = "GIDS_EVAL_RAW_DATA_ROOT"
ENV_WORK_ROOT = "GIDS_EVAL_WORK_ROOT"
ENV_MIRAGE_OFFICIAL_ROOT = "GIDS_EVAL_MIRAGE_OFFICIAL_ROOT"
ENV_CONDA_ROOT = "GIDS_EVAL_CONDA_ROOT"
ENV_EG6_OFFICIAL_RQ5_ROOT = "GIDS_EVAL_EG6_OFFICIAL_RQ5_ROOT"

ALIAS_REPO_ROOT = ("GIDS_EVAL_ROOT_REPO",)
ALIAS_WORK_ROOT = ("GIDS_EVAL_DATA_ROOT", "GIDS_EVAL_ROOT")


class PathsError(RuntimeError):
    pass


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    raw_data_root: Path
    work_root: Path
    mirage_official_root: Path
    conda_root: Path
    eg6_official_rq5_root: Path
    config_path: Path

    def sc_data_root(self, sc: str) -> Path:
        return self.work_root / sc.lower()


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _walk_up_for_config(start: Path) -> Path | None:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    seen: set[Path] = set()
    while current not in seen:
        seen.add(current)
        candidate = current / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


def _resolve_config_path(explicit: Path | None) -> Path:
    if explicit is not None:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise PathsError(f"gids_eval config not found: {path}")
        return path
    env_path = os.environ.get("GIDS_EVAL_CONFIG")
    if env_path:
        path = Path(env_path).resolve()
        if not path.is_file():
            raise PathsError(f"GIDS_EVAL_CONFIG points to missing file: {path}")
        return path
    here = Path(__file__).resolve().parent
    for start in (Path.cwd(), here):
        found = _walk_up_for_config(start)
        if found is not None:
            return found
    raise PathsError(
        f"could not locate {CONFIG_FILENAME}; "
        "place it at the repo root or set GIDS_EVAL_CONFIG"
    )


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise PathsError(f"{path} must contain a mapping")
    return data


def _coalesce(value: str | None, *fallbacks: str | Path | None) -> str | None:
    if value:
        return str(value)
    for item in fallbacks:
        if item:
            return str(item)
    return None


def _abs(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _require(value: str | None, key: str, config_path: Path) -> str:
    if not value:
        raise PathsError(
            f"{key} is not set; edit {config_path} or export the matching env var"
        )
    return value


@lru_cache(maxsize=None)
def _load(config_path_str: str | None) -> Paths:
    config_path = _resolve_config_path(Path(config_path_str) if config_path_str else None)
    raw = _load_yaml(config_path)

    repo_root_value = _coalesce(
        _first_env(ENV_REPO_ROOT, *ALIAS_REPO_ROOT),
        raw.get("repo_root"),
        config_path.parent,
    )
    raw_data_value = _coalesce(
        _first_env(ENV_RAW_DATA_ROOT),
        raw.get("raw_data_root"),
    )
    work_value = _coalesce(
        _first_env(ENV_WORK_ROOT, *ALIAS_WORK_ROOT),
        raw.get("work_root"),
    )
    mirage_official_value = _coalesce(
        _first_env(ENV_MIRAGE_OFFICIAL_ROOT),
        raw.get("mirage_official_root"),
    )
    conda_value = _coalesce(
        _first_env(ENV_CONDA_ROOT),
        raw.get("conda_root"),
    )
    eg6_value = _coalesce(
        _first_env(ENV_EG6_OFFICIAL_RQ5_ROOT),
        raw.get("eg6_official_rq5_root"),
    )

    repo_root = _abs(_require(repo_root_value, "repo_root", config_path))
    raw_data_root = _abs(_require(raw_data_value, "raw_data_root", config_path))
    work_root = _abs(_require(work_value, "work_root", config_path))

    if mirage_official_value:
        mirage_official_root = _abs(mirage_official_value)
    else:
        mirage_official_root = (raw_data_root.parent / "official").resolve()

    if conda_value:
        conda_root = _abs(conda_value)
    else:
        conda_root = (work_root / "conda").resolve()

    if eg6_value:
        eg6_official_rq5_root = _abs(eg6_value)
    else:
        eg6_official_rq5_root = (work_root / "eg6_official_rq5").resolve()

    return Paths(
        repo_root=repo_root,
        raw_data_root=raw_data_root,
        work_root=work_root,
        mirage_official_root=mirage_official_root,
        conda_root=conda_root,
        eg6_official_rq5_root=eg6_official_rq5_root,
        config_path=config_path,
    )


def get_paths(config_path: str | Path | None = None) -> Paths:
    key = str(Path(config_path).resolve()) if config_path else None
    return _load(key)


def reset_cache() -> None:
    _load.cache_clear()


def expand_template_vars(paths: Paths) -> dict[str, str]:
    return {
        "repo_root": str(paths.repo_root),
        "raw_data_root": str(paths.raw_data_root),
        "work_root": str(paths.work_root),
        "data_root": str(paths.work_root),
        "mirage_official_root": str(paths.mirage_official_root),
        "conda_root": str(paths.conda_root),
        "eg6_official_rq5_root": str(paths.eg6_official_rq5_root),
        "runtime_root": str(paths.repo_root / "local_runtime"),
    }
