from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from gids_eval.io_utils import expand_templates, load_yaml
from gids_eval.paths import get_paths
from gids_eval.schema import detect_config_kind, validate_detector_config, validate_experiment_config


def _resolve_runtime_path(value: str | None, fallback: Path) -> Path:
    if value is None:
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    if text.startswith("${") and text.endswith("}"):
        return fallback
    return Path(text).expanduser().resolve()


def _runtime_block(payload: dict[str, Any]) -> dict[str, Any]:
    block = payload.get("runtime")
    if not isinstance(block, dict):
        return {}
    return block


def load_detector_config(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = Path(path).resolve()
    raw = load_yaml(config_path)
    validate_detector_config(raw, config_path)

    paths = get_paths()
    runtime = _runtime_block(raw)
    detector = str(raw["detector"])
    run_label = str(raw.get("run_label", "baseline"))
    repo_root = _resolve_runtime_path(runtime.get("repo_root"), paths.repo_root)
    data_root = _resolve_runtime_path(runtime.get("data_root"), paths.work_root)
    runtime_root = _resolve_runtime_path(runtime.get("runtime_root"), repo_root / "local_runtime")
    workspace_root = data_root / "workspaces" / detector
    input_root = workspace_root / "input"
    output_root = workspace_root / "output"
    metadata_root = workspace_root / "metadata" / run_label

    context: dict[str, Any] = {
        "config_path": str(config_path),
        "config_dir": str(config_path.parent),
        "repo_root": str(repo_root),
        "data_root": str(data_root),
        "work_root": str(data_root),
        "raw_data_root": str(paths.raw_data_root),
        "mirage_official_root": str(paths.mirage_official_root),
        "conda_root": str(paths.conda_root),
        "runtime_root": str(runtime_root),
        "workspace_root": str(workspace_root),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "metadata_root": str(metadata_root),
        "detector": detector,
        "seed": raw.get("seed", 1337),
        "python_bin": raw.get("python_bin", "python3"),
        "run_label": run_label,
    }

    resolved = expand_templates(raw, context)
    resolved["config_path"] = str(config_path)
    resolved["manifests"] = [
        _resolve_manifest_path(item, config_path.parent)
        for item in resolved.get("manifests", [])
    ]
    resolved.setdefault("stage_mode", "symlink")
    resolved.setdefault("python_bin", context["python_bin"])
    resolved.setdefault("seed", context["seed"])
    resolved.setdefault("run_label", run_label)
    resolved.setdefault("environment", {})
    resolved.setdefault("jobs", [])
    resolved.setdefault("preprocess", {"enabled": False})
    resolved["_context"] = context
    return resolved, context


def load_experiment_config(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = Path(path).resolve()
    raw = load_yaml(config_path)
    validate_experiment_config(raw, config_path)

    paths = get_paths()
    runtime = _runtime_block(raw)
    experiment_id = str(raw["experiment_id"])
    run_label = str(raw.get("run_label", "baseline"))
    repo_root = _resolve_runtime_path(runtime.get("repo_root"), paths.repo_root)
    data_root = _resolve_runtime_path(runtime.get("data_root"), paths.work_root)
    runtime_root = _resolve_runtime_path(runtime.get("runtime_root"), repo_root / "local_runtime")
    workspace_root = data_root / "experiments" / experiment_id / run_label
    input_root = workspace_root / "input"
    output_root = workspace_root / "output"
    modules_root = output_root / "modules"
    metadata_root = workspace_root / "metadata"
    cache_root = data_root / "cache" / "experiments"

    context: dict[str, Any] = {
        "config_path": str(config_path),
        "config_dir": str(config_path.parent),
        "repo_root": str(repo_root),
        "data_root": str(data_root),
        "work_root": str(data_root),
        "raw_data_root": str(paths.raw_data_root),
        "mirage_official_root": str(paths.mirage_official_root),
        "conda_root": str(paths.conda_root),
        "runtime_root": str(runtime_root),
        "workspace_root": str(workspace_root),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "modules_root": str(modules_root),
        "metadata_root": str(metadata_root),
        "cache_root": str(cache_root),
        "experiment_id": experiment_id,
        "seed": raw.get("seed", 1337),
        "python_bin": raw.get("python_bin", "python3"),
        "run_label": run_label,
    }

    resolved = expand_templates(raw, context)
    resolved["config_path"] = str(config_path)
    resolved["manifests"] = [
        _resolve_manifest_path(item, config_path.parent)
        for item in resolved.get("manifests", [])
    ]
    resolved.setdefault("run_label", run_label)
    resolved.setdefault("seed", context["seed"])
    resolved.setdefault("python_bin", context["python_bin"])
    resolved.setdefault("stage_mode", "symlink")
    resolved.setdefault("environment", {})
    resolved["_context"] = context
    resolved["_raw_modules"] = raw["modules"]
    resolved["_raw_environment"] = raw.get("environment", {})
    return resolved, context


def sniff_config_kind(path: str | Path) -> str:
    payload = load_yaml(Path(path).resolve())
    return detect_config_kind(payload)


def _resolve_manifest_path(path_value: str, config_dir: Path) -> str:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return str(candidate.resolve())
    return str((config_dir / candidate).resolve())
