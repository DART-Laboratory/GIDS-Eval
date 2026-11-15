from __future__ import annotations

from dataclasses import dataclass

@dataclass
class RunOptions:
    dry_run: bool = False
    clean: bool = False
    to_stage: str | None = None
