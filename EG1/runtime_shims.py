from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

def load_eg1_sitecustomize() -> None:
    module_name = "_eg1_sitecustomize"
    if module_name in sys.modules:
        return
    module_path = Path(__file__).resolve().parent / "sitecustomize.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load EG1 runtime shim from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
