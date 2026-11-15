
from __future__ import annotations

import os

try:
    import numpy as np
except Exception:
    np = None

def _alias(name: str, value: object) -> None:
    if np is None:
        return
    if name not in np.__dict__:
        setattr(np, name, value)

if np is not None:
    _alias("bool", np.bool_)
    _alias("int", int)
    _alias("float", float)
    _alias("complex", complex)
    _alias("object", object)
    _alias("str", str)
    _alias("unicode", str)
    _alias("long", np.int64)

def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default

def _patch_multiprocessing_pool() -> None:
    if os.environ.get("EG1_PIKACHU_POOL_BACKEND", "").strip().lower() != "thread":
        return
    try:
        import multiprocessing
        from concurrent.futures import ThreadPoolExecutor
    except Exception:
        return

    max_workers = _int_env("EG1_PIKACHU_MAX_WORKERS", 4)

    class _ThreadExecutorPool:
        def __init__(
            self,
            processes: int | None = None,
            initializer=None,
            initargs=(),
            maxtasksperchild=None,
            context=None,
        ) -> None:
            del maxtasksperchild, context
            requested = processes if isinstance(processes, int) and processes > 0 else max_workers
            self._initializer = initializer
            self._initargs = tuple(initargs)
            self._executor = ThreadPoolExecutor(max_workers=min(requested, max_workers))

        def _invoke(self, func, args):
            if self._initializer is not None:
                self._initializer(*self._initargs)
            return func(*args)

        def starmap(self, func, iterable, chunksize=None):
            del chunksize
            futures = [self._executor.submit(self._invoke, func, tuple(args)) for args in iterable]
            return [future.result() for future in futures]

        def close(self) -> None:
            self._executor.shutdown(wait=True)

        def terminate(self) -> None:
            self.close()

        def join(self) -> None:
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self.close()

    multiprocessing.Pool = _ThreadExecutorPool

_patch_multiprocessing_pool()
