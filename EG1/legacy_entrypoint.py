
from __future__ import annotations

import runpy
import sys
from pathlib import Path

from runtime_shims import load_eg1_sitecustomize

load_eg1_sitecustomize()

def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: legacy_entrypoint.py <script> [args...]")
    script_path = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(script_path.parent))
    sys.argv = [str(script_path), *sys.argv[2:]]
    runpy.run_path(str(script_path), run_name="__main__")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
