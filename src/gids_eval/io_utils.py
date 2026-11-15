from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from string import Template
from typing import Any

import yaml

def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target

def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return data

def dump_yaml(path: str | Path, payload: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

def dump_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

def read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    with target.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [_coerce_row(row) for row in reader]

def write_csv_rows(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

def expand_templates(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        normalized = {key: str(val) for key, val in context.items()}
        return Template(value).safe_substitute(normalized)
    if isinstance(value, list):
        return [expand_templates(item, context) for item in value]
    if isinstance(value, dict):
        return {key: expand_templates(item, context) for key, item in value.items()}
    return value

def merge_env(*parts: dict[str, Any]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for part in parts:
        for key, value in part.items():
            if value is None:
                continue
            merged[str(key)] = str(value)
    return merged

def safe_rmtree(path: str | Path, root: str | Path) -> None:
    target = Path(path).resolve()
    guard = Path(root).resolve()
    if guard not in target.parents and target != guard:
        raise ValueError(f"Refusing to delete {target}; outside guard root {guard}")
    if not target.exists() and not target.is_symlink():
        return
    if target.is_symlink() or target.is_file():
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        return
    try:
        shutil.rmtree(target)
    except FileNotFoundError:

        pass

def copy_path(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        ensure_dir(target.parent)
        shutil.copy2(source, target)

def _coerce_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _coerce_value(value) for key, value in row.items()}

def _coerce_value(value: Any) -> Any:
    if value is None:
        return value
    if not isinstance(value, str):
        return value
    trimmed = value.strip()
    if trimmed == "":
        return trimmed
    for caster in (int, float):
        try:
            return caster(trimmed)
        except ValueError:
            continue
    lowered = trimmed.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    return value
