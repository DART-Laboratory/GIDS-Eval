from __future__ import annotations

from pathlib import Path
from typing import Any

MODULE_STAGES = (
    "graph_construction",
    "featurization",
    "graph_encoder",
    "temporal_encoder",
    "loss",
    "detection",
)

MODULE_RUNNERS = {
    "noop",
    "shell_command",
    "command_group",
}

class SchemaValidationError(ValueError):
    pass

def validate_detector_config(payload: dict[str, Any], path: Path) -> None:
    _expect_mapping(payload, str(path))
    required = ["schema_version", "name", "detector", "manifests", "jobs"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise SchemaValidationError(f"Config {path} missing required keys: {missing}")

    _expect_string(payload["name"], f"{path}.name")
    _expect_string(payload["detector"], f"{path}.detector")
    if "run_label" in payload:
        _expect_string(payload["run_label"], f"{path}.run_label")
    if "seed" in payload:
        _expect_int(payload["seed"], f"{path}.seed")
    if "python_bin" in payload:
        _expect_string(payload["python_bin"], f"{path}.python_bin")
    if "stage_mode" in payload:
        stage_mode = _expect_string(payload["stage_mode"], f"{path}.stage_mode")
        if stage_mode not in {"symlink", "copy"}:
            raise SchemaValidationError(f"{path}.stage_mode must be 'symlink' or 'copy'")
    if "runtime" in payload:
        runtime = _expect_mapping(payload["runtime"], f"{path}.runtime")
        for key in ("repo_root", "data_root", "runtime_root"):
            if key in runtime:
                _expect_string(runtime[key], f"{path}.runtime.{key}")

    manifests = _expect_list(payload["manifests"], f"{path}.manifests")
    if not manifests:
        raise SchemaValidationError(f"Config {path} must define at least one manifest")
    for index, manifest in enumerate(manifests):
        _expect_string(manifest, f"{path}.manifests[{index}]")

    jobs = _expect_list(payload["jobs"], f"{path}.jobs")
    if not jobs:
        raise SchemaValidationError(f"Config {path} must define at least one job")
    for index, job in enumerate(jobs):
        job_path = f"{path}.jobs[{index}]"
        job_payload = _expect_mapping(job, job_path)
        for key in ("id", "workdir", "command"):
            if key not in job_payload:
                raise SchemaValidationError(f"{job_path} missing required key: {key}")
        _expect_string(job_payload["id"], f"{job_path}.id")
        _expect_string(job_payload["workdir"], f"{job_path}.workdir")
        _expect_command(job_payload["command"], f"{job_path}.command")
        if "env" in job_payload:
            _expect_string_mapping(job_payload["env"], f"{job_path}.env")
        if "metrics_path" in job_payload:
            _expect_string(job_payload["metrics_path"], f"{job_path}.metrics_path")

    preprocess = payload.get("preprocess")
    if preprocess is not None:
        preprocess_payload = _expect_mapping(preprocess, f"{path}.preprocess")
        enabled = preprocess_payload.get("enabled", False)
        _expect_bool(enabled, f"{path}.preprocess.enabled")
        if enabled:
            for key in ("workdir", "command"):
                if key not in preprocess_payload:
                    raise SchemaValidationError(f"{path}.preprocess missing required key: {key}")
            _expect_string(preprocess_payload["workdir"], f"{path}.preprocess.workdir")
            _expect_command(preprocess_payload["command"], f"{path}.preprocess.command")
    if "environment" in payload:
        _expect_string_mapping(payload["environment"], f"{path}.environment")

def validate_experiment_config(payload: dict[str, Any], path: Path) -> None:
    _expect_mapping(payload, str(path))
    required = [
        "schema_version",
        "kind",
        "name",
        "experiment_id",
        "modules",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise SchemaValidationError(f"Experiment {path} missing required keys: {missing}")

    kind = _expect_string(payload["kind"], f"{path}.kind")
    if kind != "experiment":
        raise SchemaValidationError(f"Experiment {path} kind must be 'experiment', got {kind!r}")

    _expect_string(payload["name"], f"{path}.name")
    _expect_string(payload["experiment_id"], f"{path}.experiment_id")
    if "run_label" in payload:
        _expect_string(payload["run_label"], f"{path}.run_label")
    if "seed" in payload:
        _expect_int(payload["seed"], f"{path}.seed")
    if "python_bin" in payload:
        _expect_string(payload["python_bin"], f"{path}.python_bin")
    if "stage_mode" in payload:
        stage_mode = _expect_string(payload["stage_mode"], f"{path}.stage_mode")
        if stage_mode not in {"symlink", "copy"}:
            raise SchemaValidationError(f"{path}.stage_mode must be 'symlink' or 'copy'")

    if "runtime" in payload:
        runtime = _expect_mapping(payload["runtime"], f"{path}.runtime")
        for key in ("repo_root", "data_root", "runtime_root"):
            if key in runtime:
                _expect_string(runtime[key], f"{path}.runtime.{key}")

    manifests = payload.get("manifests", [])
    manifests_list = _expect_list(manifests, f"{path}.manifests")
    for index, manifest in enumerate(manifests_list):
        _expect_string(manifest, f"{path}.manifests[{index}]")

    if "environment" in payload:
        _expect_string_mapping(payload["environment"], f"{path}.environment")

    modules = _expect_mapping(payload["modules"], f"{path}.modules")
    missing_modules = [stage for stage in MODULE_STAGES if stage not in modules]
    extra_modules = [stage for stage in modules if stage not in MODULE_STAGES]
    if missing_modules:
        raise SchemaValidationError(
            f"Experiment {path} missing required module stages: {missing_modules}"
        )
    if extra_modules:
        raise SchemaValidationError(f"Experiment {path} has unsupported module stages: {extra_modules}")

    for stage in MODULE_STAGES:
        _validate_module_config(_expect_mapping(modules[stage], f"{path}.modules.{stage}"), path, stage)

def validate_manifest(payload: dict[str, Any], path: Path) -> None:
    _expect_mapping(payload, str(path))
    required = ["schema_version", "dataset_id", "stage_root", "artifacts"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise SchemaValidationError(f"Manifest {path} missing required keys: {missing}")

    _expect_string(payload["dataset_id"], f"{path}.dataset_id")
    _expect_string(payload["stage_root"], f"{path}.stage_root")

    artifacts = _expect_list(payload["artifacts"], f"{path}.artifacts")
    if not artifacts:
        raise SchemaValidationError(f"Manifest {path} must define a non-empty artifacts list")

    for index, artifact in enumerate(artifacts):
        artifact_path = f"{path}.artifacts[{index}]"
        artifact_payload = _expect_mapping(artifact, artifact_path)
        for key in ("source", "target"):
            if key not in artifact_payload:
                raise SchemaValidationError(f"{artifact_path} missing required key: {key}")
            _expect_string(artifact_payload[key], f"{artifact_path}.{key}")
        kind = artifact_payload.get("kind", "file")
        _expect_string(kind, f"{artifact_path}.kind")
        if kind not in {"file", "directory"}:
            raise SchemaValidationError(
                f"{artifact_path}.kind must be 'file' or 'directory', got {kind!r}"
            )

def detect_config_kind(payload: dict[str, Any]) -> str:
    if payload.get("kind") == "experiment":
        return "experiment"
    if "modules" in payload and "experiment_id" in payload:
        return "experiment"
    return "detector"

def _validate_module_config(module: dict[str, Any], path: Path, stage: str) -> None:
    module_path = f"{path}.modules.{stage}"
    if "runner" not in module:
        raise SchemaValidationError(f"{module_path} missing required key: runner")

    runner = _expect_string(module["runner"], f"{module_path}.runner")
    if runner not in MODULE_RUNNERS:
        raise SchemaValidationError(
            f"{module_path}.runner must be one of {sorted(MODULE_RUNNERS)}, got {runner!r}"
        )

    if "workdir" in module:
        _expect_string(module["workdir"], f"{module_path}.workdir")
    if "command" in module:
        _expect_command(module["command"], f"{module_path}.command")
    if "env" in module:
        _expect_string_mapping(module["env"], f"{module_path}.env")
    if "inputs" in module:
        _expect_relative_string_mapping(module["inputs"], f"{module_path}.inputs")
    if "artifacts" in module:
        _expect_relative_string_mapping(module["artifacts"], f"{module_path}.artifacts")
    if "metrics_relpath" in module:
        _expect_string(module["metrics_relpath"], f"{module_path}.metrics_relpath")
        if Path(str(module["metrics_relpath"])).is_absolute():
            raise SchemaValidationError(f"{module_path}.metrics_relpath must be relative")
    if "jobs" in module:
        jobs = _expect_list(module["jobs"], f"{module_path}.jobs")
        if not jobs:
            raise SchemaValidationError(f"{module_path}.jobs must be a non-empty list")
        for index, job in enumerate(jobs):
            _validate_group_job(_expect_mapping(job, f"{module_path}.jobs[{index}]"), module_path, index)
    if "cache" in module:
        cache = _expect_mapping(module["cache"], f"{module_path}.cache")
        allowed_cache_keys = {"enabled"}
        extra_cache_keys = sorted(set(cache) - allowed_cache_keys)
        if extra_cache_keys:
            raise SchemaValidationError(
                f"{module_path}.cache has unsupported keys: {extra_cache_keys}"
            )
        if "enabled" in cache:
            _expect_bool(cache["enabled"], f"{module_path}.cache.enabled")
    if "params" in module:
        _expect_mapping(module["params"], f"{module_path}.params")

    if runner == "shell_command" and "command" not in module:
        raise SchemaValidationError(f"{module_path} runner=shell_command requires command")
    if runner == "command_group" and "jobs" not in module:
        raise SchemaValidationError(f"{module_path} runner=command_group requires jobs")

def _validate_group_job(job: dict[str, Any], module_path: str, index: int) -> None:
    job_path = f"{module_path}.jobs[{index}]"
    for key in ("id", "command"):
        if key not in job:
            raise SchemaValidationError(f"{job_path} missing required key: {key}")
    _expect_string(job["id"], f"{job_path}.id")
    _expect_command(job["command"], f"{job_path}.command")
    if "workdir" in job:
        _expect_string(job["workdir"], f"{job_path}.workdir")
    if "dataset" in job:
        _expect_string(job["dataset"], f"{job_path}.dataset")
    if "env" in job:
        _expect_string_mapping(job["env"], f"{job_path}.env")
    if "metrics_relpath" in job:
        _expect_string(job["metrics_relpath"], f"{job_path}.metrics_relpath")
        if Path(str(job["metrics_relpath"])).is_absolute():
            raise SchemaValidationError(f"{job_path}.metrics_relpath must be relative")

def _expect_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaValidationError(f"{path} must be a mapping")
    return value

def _expect_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise SchemaValidationError(f"{path} must be a list")
    return value

def _expect_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or value == "":
        raise SchemaValidationError(f"{path} must be a non-empty string")
    return value

def _expect_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise SchemaValidationError(f"{path} must be a boolean")
    return value

def _expect_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaValidationError(f"{path} must be an integer")
    return value

def _expect_command(value: Any, path: str) -> list[str]:
    command = _expect_list(value, path)
    if not command:
        raise SchemaValidationError(f"{path} must be a non-empty list")
    normalized: list[str] = []
    for index, item in enumerate(command):
        if isinstance(item, (str, int, float, bool)):
            normalized.append(str(item))
            continue
        raise SchemaValidationError(f"{path}[{index}] must be a string-compatible scalar")
    return normalized

def _expect_string_mapping(value: Any, path: str) -> dict[str, Any]:
    mapping = _expect_mapping(value, path)
    for key, item in mapping.items():
        _expect_string(key, f"{path} key")
        if isinstance(item, (str, int, float, bool)):
            continue
        raise SchemaValidationError(f"{path}.{key} must be a string-compatible scalar")
    return mapping

def _expect_relative_string_mapping(value: Any, path: str) -> dict[str, str]:
    mapping = _expect_mapping(value, path)
    normalized: dict[str, str] = {}
    for key, item in mapping.items():
        _expect_string(key, f"{path} key")
        relpath = _expect_string(item, f"{path}.{key}")
        candidate = Path(relpath)
        if candidate.is_absolute():
            raise SchemaValidationError(f"{path}.{key} must be relative to the module output dir")
        if ".." in candidate.parts:
            raise SchemaValidationError(f"{path}.{key} must not escape the module output dir")
        normalized[key] = relpath
    return normalized
