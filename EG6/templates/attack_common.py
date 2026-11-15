from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TARGET_RESULT_FIELDNAMES = [
    "detector",
    "dataset",
    "repro_level",
    "window_seconds",
    "target_index",
    "tgt_t",
    "tgt_u",
    "tgt_v",
    "status",
    "init_score",
    "initially_stealthy",
    "generated_edge_count",
    "min_target_stealthy_edges",
    "min_cover_stealthy_edges",
    "minimal_success_edges",
    "final_target_score",
    "final_cover_min_score",
    "elapsed_sec",
    "processed_at",
]

BUDGET_RESULT_FIELDNAMES = [
    "detector",
    "dataset",
    "repro_level",
    "window_seconds",
    "budget",
    "threshold",
    "target_count",
    "processed_target_count",
    "skipped_target_count",
    "flagged_target_count",
    "init_stealthy_rate",
    "target_stealthy_rate",
    "cover_stealthy_rate",
    "attack_success_rate",
    "checkpoint_source",
    "checkpoint_path",
]

def safe_mean(values, *, default: float = 0.0) -> float:
    values = list(values)
    if not values:
        return default
    return float(sum(values) / len(values))

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def resolve_budgets(config_budgets: list[int]) -> list[int]:
    override = os.environ.get("EG6_BUDGETS")
    if override:
        budgets = [int(part.strip()) for part in override.split(",") if part.strip()]
    else:
        budgets = list(config_budgets)
    if 0 not in budgets:
        budgets.insert(0, 0)
    return sorted(dict.fromkeys(budgets))

def target_key(tgt_t: int, tgt_u: int, tgt_v: int) -> tuple[int, int, int]:
    return (int(tgt_t), int(tgt_u), int(tgt_v))

def unique_target_list(targets: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    unique: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for tgt_t, tgt_u, tgt_v in targets:
        key = target_key(tgt_t, tgt_u, tgt_v)
        if key not in unique:
            unique[key] = key
    return list(unique.values())

def _maybe_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    return int(float(value))

def _maybe_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)

def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}

def _normalize_target_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "detector": str(row["detector"]),
        "dataset": str(row["dataset"]),
        "repro_level": str(row["repro_level"]),
        "window_seconds": int(float(row["window_seconds"])),
        "target_index": int(float(row["target_index"])),
        "tgt_t": int(float(row["tgt_t"])),
        "tgt_u": int(float(row["tgt_u"])),
        "tgt_v": int(float(row["tgt_v"])),
        "status": str(row["status"]),
        "init_score": float(row["init_score"]),
        "initially_stealthy": _to_bool(row["initially_stealthy"]),
        "generated_edge_count": int(float(row["generated_edge_count"])),
        "min_target_stealthy_edges": _maybe_int(row["min_target_stealthy_edges"]),
        "min_cover_stealthy_edges": _maybe_int(row["min_cover_stealthy_edges"]),
        "minimal_success_edges": _maybe_int(row["minimal_success_edges"]),
        "final_target_score": float(row["final_target_score"]),
        "final_cover_min_score": float(row["final_cover_min_score"]),
        "elapsed_sec": float(row["elapsed_sec"]),
        "processed_at": str(row["processed_at"]),
    }

def load_target_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        deduped: dict[tuple[int, int, int], dict[str, Any]] = {}
        for raw_row in reader:
            row = _normalize_target_row(dict(raw_row))
            deduped[target_key(row["tgt_t"], row["tgt_u"], row["tgt_v"])] = row
    rows = list(deduped.values())
    rows.sort(key=lambda row: row["target_index"])
    return rows

def write_target_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: int(row["target_index"]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TARGET_RESULT_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in TARGET_RESULT_FIELDNAMES})

def append_target_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TARGET_RESULT_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in TARGET_RESULT_FIELDNAMES})

def write_budget_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BUDGET_RESULT_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in BUDGET_RESULT_FIELDNAMES})

def _within_budget(min_edges: int | None, budget: int) -> int:
    return int(min_edges is not None and min_edges <= budget)


def _budget_rate(rows: list[dict[str, Any]], key: str, budget: int) -> float | str:
    if not rows:
        return ""
    return safe_mean(_within_budget(row[key], budget) for row in rows)

def build_budget_rows(
    config: dict[str, Any],
    budgets: list[int],
    target_rows: list[dict[str, Any]],
    *,
    total_target_count: int,
) -> list[dict[str, Any]]:
    target_rows = sorted(target_rows, key=lambda row: row["target_index"])
    processed_rows = [row for row in target_rows if row["status"] != "no_candidate"]
    skipped_rows = [row for row in target_rows if row["status"] == "no_candidate"]
    flagged_rows = [row for row in processed_rows if not row["initially_stealthy"]]
    cutoff = float(config["attack"]["threshold"])
    init_rate = safe_mean(int(row["initially_stealthy"]) for row in target_rows)
    rows: list[dict[str, Any]] = [{
        "detector": config["detector"],
        "dataset": config["dataset"],
        "repro_level": config["repro_level"],
        "window_seconds": config["window_seconds"],
        "budget": 0,
        "threshold": cutoff,
        "target_count": int(total_target_count),
        "processed_target_count": len(processed_rows),
        "skipped_target_count": len(skipped_rows),
        "flagged_target_count": len(flagged_rows),
        "init_stealthy_rate": init_rate,
        "target_stealthy_rate": _budget_rate(flagged_rows, "min_target_stealthy_edges", 0),
        "cover_stealthy_rate": 1.0,
        "attack_success_rate": _budget_rate(flagged_rows, "minimal_success_edges", 0),
        "checkpoint_source": config["checkpoint"]["source"],
        "checkpoint_path": config["checkpoint"]["path"],
    }]

    for budget in sorted(budget for budget in budgets if budget > 0):
        rows.append({
            "detector": config["detector"],
            "dataset": config["dataset"],
            "repro_level": config["repro_level"],
            "window_seconds": config["window_seconds"],
            "budget": budget,
            "threshold": cutoff,
            "target_count": int(total_target_count),
            "processed_target_count": len(processed_rows),
            "skipped_target_count": len(skipped_rows),
            "flagged_target_count": len(flagged_rows),
            "init_stealthy_rate": init_rate,
            "target_stealthy_rate": _budget_rate(flagged_rows, "min_target_stealthy_edges", budget),
            "cover_stealthy_rate": _budget_rate(flagged_rows, "min_cover_stealthy_edges", budget),
            "attack_success_rate": _budget_rate(flagged_rows, "minimal_success_edges", budget),
            "checkpoint_source": config["checkpoint"]["source"],
            "checkpoint_path": config["checkpoint"]["path"],
        })
    return rows

def build_attack_summary(
    config: dict[str, Any],
    budgets: list[int],
    target_rows: list[dict[str, Any]],
    *,
    total_target_count: int,
) -> dict[str, Any]:
    target_rows = sorted(target_rows, key=lambda row: row["target_index"])
    skipped_targets = [
        {"target_index": row["target_index"], "tgt_t": row["tgt_t"], "tgt_u": row["tgt_u"], "tgt_v": row["tgt_v"]}
        for row in target_rows
        if row["status"] == "no_candidate"
    ]
    flagged_rows = [
        row for row in target_rows
        if row["status"] != "no_candidate" and not row["initially_stealthy"]
    ]
    successful_rows = [row for row in flagged_rows if row["minimal_success_edges"] is not None]
    return {
        "detector": config["detector"],
        "dataset": config["dataset"],
        "repro_level": config["repro_level"],
        "budgets": list(budgets),
        "target_count": int(total_target_count),
        "processed_target_count": len(target_rows) - len(skipped_targets),
        "skipped_target_count": len(skipped_targets),
        "flagged_target_count": len(flagged_rows),
        "successful_target_count": len(successful_rows),
        "checkpoint_source": config["checkpoint"]["source"],
        "checkpoint_path": config["checkpoint"]["path"],
        "threshold": float(config["attack"]["threshold"]),
        "skipped_targets": skipped_targets,
    }

def write_progress(
    path: Path,
    *,
    config: dict[str, Any],
    total_targets: int,
    target_rows: list[dict[str, Any]],
    latest_row: dict[str, Any] | None = None,
) -> None:
    processed_rows = sorted(target_rows, key=lambda row: row["target_index"])
    payload = {
        "experiment_id": config["experiment_id"],
        "detector": config["detector"],
        "dataset": config["dataset"],
        "repro_level": config["repro_level"],
        "total_targets": int(total_targets),
        "processed_targets": len(processed_rows),
        "remaining_targets": max(int(total_targets) - len(processed_rows), 0),
        "successful_targets": sum(row["minimal_success_edges"] is not None for row in processed_rows),
        "skipped_targets": sum(row["status"] == "no_candidate" for row in processed_rows),
        "updated_at": now_iso(),
    }
    if latest_row is not None:
        payload["latest"] = {
            "target_index": latest_row["target_index"],
            "status": latest_row["status"],
            "minimal_success_edges": latest_row["minimal_success_edges"],
            "generated_edge_count": latest_row["generated_edge_count"],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
