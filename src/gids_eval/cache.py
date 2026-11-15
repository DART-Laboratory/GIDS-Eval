from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from gids_eval.io_utils import dump_json, ensure_dir

def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

def build_path_signature(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    stat_result = target.stat()
    kind = "directory" if target.is_dir() else "file"
    return {
        "path": str(target),
        "kind": kind,
        "size": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
    }

def cache_entry_exists(cache_dir: str | Path) -> bool:
    target = Path(cache_dir)
    return (target / "cache_record.json").exists() and (target / "output").exists()

def materialize_cached_output(cache_dir: str | Path, output_dir: str | Path) -> None:
    cache_path = Path(cache_dir)
    cache_output = cache_path / "output"
    if not cache_output.exists():
        raise ValueError(f"Cache output missing: {cache_output}")
    remove_path(output_dir)
    ensure_dir(Path(output_dir).parent)
    os.symlink(cache_output, output_dir, target_is_directory=True)

def store_cache_entry(cache_dir: str | Path, output_dir: str | Path, record: dict[str, Any]) -> None:
    cache_path = Path(cache_dir)
    output_path = Path(output_dir)

    if cache_entry_exists(cache_path):
        materialize_cached_output(cache_path, output_path)
        return

    temp_cache = cache_path.with_name(f"{cache_path.name}.tmp")
    remove_path(temp_cache)
    ensure_dir(temp_cache)

    cached_output = temp_cache / "output"
    if output_path.exists() or output_path.is_symlink():
        output_path.rename(cached_output)
    else:
        ensure_dir(cached_output)

    dump_json(temp_cache / "cache_record.json", record)

    ensure_dir(cache_path.parent)
    temp_cache.rename(cache_path)
    materialize_cached_output(cache_path, output_path)

def remove_path(path: str | Path) -> None:
    target = Path(path)
    if not target.exists() and not target.is_symlink():
        return
    if target.is_symlink() or target.is_file():
        target.unlink()
        return
    if target.is_dir():
        last_error: OSError | None = None
        for _ in range(5):
            try:
                shutil.rmtree(target)
                return
            except FileNotFoundError:
                return
            except OSError as exc:
                last_error = exc
                if exc.errno != 39:
                    raise
                time.sleep(0.2)
        if last_error is not None:
            raise last_error
