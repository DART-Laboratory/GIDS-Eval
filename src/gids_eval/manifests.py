from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from gids_eval.io_utils import copy_path, ensure_dir, expand_templates, load_yaml
from gids_eval.schema import validate_manifest

def load_manifest(path: str | Path, context: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    payload = load_yaml(manifest_path)
    payload = expand_templates(payload, context)
    payload["_path"] = str(manifest_path)
    payload["_dir"] = str(manifest_path.parent)
    validate_manifest(payload, manifest_path)
    _validate_sources(payload, manifest_path)
    return payload

def stage_manifests(
    manifest_paths: list[str | Path],
    context: dict[str, Any],
    stage_mode: str = "symlink",
) -> list[dict[str, Any]]:
    staged: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        manifest = load_manifest(manifest_path, context)
        staged.append(_stage_manifest(manifest, stage_mode=stage_mode))
    return staged

def preview_manifests(
    manifest_paths: list[str | Path],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        manifest = load_manifest(manifest_path, context)
        previews.append(
            {
                "manifest_path": manifest["_path"],
                "dataset_id": manifest["dataset_id"],
                "family": manifest.get("family"),
                "stage_root": str(Path(manifest["stage_root"]).resolve()),
                "targets": [
                    str(Path(manifest["stage_root"]).resolve() / artifact["target"])
                    for artifact in manifest["artifacts"]
                ],
            }
        )
    return previews

def _stage_manifest(manifest: dict[str, Any], stage_mode: str) -> dict[str, Any]:
    stage_root = Path(manifest["stage_root"]).resolve()
    artifacts = manifest["artifacts"]
    staged_targets: list[str] = []

    for artifact in artifacts:
        source = Path(str(artifact["source"])).resolve()
        target = stage_root / artifact["target"]
        kind = artifact.get("kind", "file")
        ensure_dir(target.parent)

        if target.exists() or target.is_symlink():
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)

        if stage_mode == "symlink":
            os.symlink(source, target, target_is_directory=(kind == "directory"))
        elif stage_mode == "copy":
            copy_path(source, target)
        else:
            raise ValueError(f"Unsupported stage_mode: {stage_mode}")

        staged_targets.append(str(target))

    return {
        "manifest_path": manifest["_path"],
        "dataset_id": manifest["dataset_id"],
        "family": manifest.get("family"),
        "stage_root": str(stage_root),
        "targets": staged_targets,
    }

def _validate_sources(payload: dict[str, Any], path: Path) -> None:
    for artifact in payload["artifacts"]:
        source = Path(str(artifact["source"])).resolve()
        if not source.exists():
            raise ValueError(f"Manifest {path} source does not exist: {source}")
