from __future__ import annotations

import os
import random

def _read_seed() -> int | None:
    raw_value = os.environ.get("GIDS_EVAL_SEED")
    if raw_value is None:
        return None
    try:
        return int(raw_value)
    except ValueError:
        return None

def apply_global_determinism() -> None:
    seed = _read_seed()
    if seed is None:
        return

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        enable = getattr(tf.config.experimental, "enable_op_determinism", None)
        if callable(enable):
            enable()
    except Exception:
        pass
