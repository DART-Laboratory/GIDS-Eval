from __future__ import annotations

from pathlib import Path
from typing import Any

from gids_eval.cdr import (
    compute_cdr_summary,
    compute_grouped_campaign_summary,
    derive_predictions_path,
    load_prediction_groups,
)
from gids_eval.io_utils import read_csv_rows, write_csv_rows
from gids_eval.eg1_reference_campaigns import reference_campaigns_for_dataset

BASELINE_FIELDS = [
    "detector",
    "job_id",
    "dataset",
    "seed",
    "auc",
    "ap",
    "threshold",
    "tp",
    "fp",
    "fn",
    "tn",
    "tpr",
    "fpr",
    "fnr",
    "tnr",
    "precision",
    "recall",
    "f1",
    "train_time",
    "test_time",
    "cdr",
    "cdr_auc",
    "campaign_count",
    "detected_campaigns",
    "cdr_meaningful",
    "metrics_path",
    "predictions_path",
]

COST_FIELDS = [
    "detector",
    "phase",
    "job_id",
    "dataset",
    "seed",
    "status",
    "exit_code",
    "started_at",
    "finished_at",
    "wall_time_sec",
    "peak_rss_kb",
    "mean_rss_kb",
    "peak_cpu_percent",
    "mean_cpu_percent",
    "peak_process_count",
    "sample_count",
    "cpu_sample_count",
    "cache_hit",
    "cache_key",
    "train_time",
    "test_time",
    "command",
    "workdir",
    "stdout_log",
    "stderr_log",
    "metrics_path",
]

def collect_metrics_rows(job: dict[str, Any]) -> list[dict[str, Any]]:
    subjobs = job.get("subjobs") or []
    if subjobs:
        aggregate_dataset = _combined_dataset_name(subjobs)
        if aggregate_dataset:
            return _collect_combined_metrics_rows(job, subjobs, aggregate_dataset)
        result: list[dict[str, Any]] = []
        for subjob in subjobs:
            metrics_path = subjob.get("metrics_path")
            predictions_path = subjob.get("predictions_path") or derive_predictions_path(metrics_path)
            rows = read_csv_rows(metrics_path) if metrics_path else []
            prediction_groups = load_prediction_groups(predictions_path)
            for index, row in enumerate(rows):
                eval_index = _resolve_eval_index(row, index)
                dataset_name = row.get("dataset", subjob.get("dataset", job.get("dataset")))
                reference_campaigns = _reference_campaigns_for_job(job, dataset_name)
                result.append(
                    {
                        "detector": job.get("detector"),
                        "job_id": subjob.get("job_id", job["job_id"]),
                        "dataset": dataset_name,
                        "seed": job.get("seed"),
                        "auc": row.get("auc", ""),
                        "ap": row.get("ap", ""),
                        "threshold": row.get("threshold", ""),
                        "tp": row.get("tp", ""),
                        "fp": row.get("fp", ""),
                        "fn": row.get("fn", ""),
                        "tn": row.get("tn", ""),
                        "tpr": row.get("tpr", ""),
                        "fpr": row.get("fpr", ""),
                        "fnr": row.get("fnr", ""),
                        "tnr": row.get("tnr", ""),
                        "precision": row.get("precision", ""),
                        "recall": row.get("recall", ""),
                        "f1": row.get("f1", ""),
                        "train_time": row.get("train_time", ""),
                        "test_time": row.get("test_time", ""),
                        **compute_cdr_summary(
                            prediction_groups.get(eval_index, []),
                            reference_campaigns=reference_campaigns,
                        ),
                        "metrics_path": metrics_path,
                        "predictions_path": predictions_path if prediction_groups else "",
                    }
                )
        return result

    metrics_path = job.get("metrics_path")
    predictions_path = job.get("predictions_path") or derive_predictions_path(metrics_path)
    rows = read_csv_rows(metrics_path) if metrics_path else []
    prediction_groups = load_prediction_groups(predictions_path)
    dataset = job.get("dataset")
    detector = job.get("detector")
    seed = job.get("seed")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        eval_index = _resolve_eval_index(row, index)
        dataset_name = row.get("dataset", dataset)
        reference_campaigns = _reference_campaigns_for_job(job, dataset_name)
        result.append(
            {
                "detector": detector,
                "job_id": job["job_id"],
                "dataset": dataset_name,
                "seed": seed,
                "auc": row.get("auc", ""),
                "ap": row.get("ap", ""),
                "threshold": row.get("threshold", ""),
                "tp": row.get("tp", ""),
                "fp": row.get("fp", ""),
                "fn": row.get("fn", ""),
                "tn": row.get("tn", ""),
                "tpr": row.get("tpr", ""),
                "fpr": row.get("fpr", ""),
                "fnr": row.get("fnr", ""),
                "tnr": row.get("tnr", ""),
                "precision": row.get("precision", ""),
                "recall": row.get("recall", ""),
                "f1": row.get("f1", ""),
                "train_time": row.get("train_time", ""),
                "test_time": row.get("test_time", ""),
                **compute_cdr_summary(
                    prediction_groups.get(eval_index, []),
                    reference_campaigns=reference_campaigns,
                ),
                "metrics_path": metrics_path,
                "predictions_path": predictions_path if prediction_groups else "",
            }
        )
    return result

def build_cost_row(record: dict[str, Any]) -> dict[str, Any]:
    metrics_rows = record.get("metrics_rows") or []
    first_metrics = metrics_rows[0] if metrics_rows else {}
    has_subjobs = bool(record.get("subjobs"))
    return {
        "detector": record.get("detector"),
        "phase": record.get("phase", "baseline"),
        "job_id": record.get("job_id", ""),
        "dataset": record.get("dataset", first_metrics.get("dataset", "")),
        "seed": record.get("seed", ""),
        "status": record.get("status", ""),
        "exit_code": record.get("exit_code", ""),
        "started_at": record.get("started_at", ""),
        "finished_at": record.get("finished_at", ""),
        "wall_time_sec": record.get("wall_time_sec", ""),
        "peak_rss_kb": record.get("peak_rss_kb", ""),
        "mean_rss_kb": record.get("mean_rss_kb", ""),
        "peak_cpu_percent": record.get("peak_cpu_percent", ""),
        "mean_cpu_percent": record.get("mean_cpu_percent", ""),
        "peak_process_count": record.get("peak_process_count", ""),
        "sample_count": record.get("sample_count", ""),
        "cpu_sample_count": record.get("cpu_sample_count", ""),
        "cache_hit": record.get("cache_hit", ""),
        "cache_key": record.get("cache_key", ""),
        "train_time": "" if has_subjobs else first_metrics.get("train_time", ""),
        "test_time": "" if has_subjobs else first_metrics.get("test_time", ""),
        "command": " ".join(record.get("command", [])),
        "workdir": record.get("workdir", ""),
        "stdout_log": record.get("stdout_log", ""),
        "stderr_log": record.get("stderr_log", ""),
        "metrics_path": record.get("metrics_path", ""),
    }

def write_reports(
    baseline_rows: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]],
    baseline_path: str | Path,
    cost_path: str | Path,
) -> None:
    write_csv_rows(baseline_path, baseline_rows, BASELINE_FIELDS)
    write_csv_rows(cost_path, cost_rows, COST_FIELDS)

def _resolve_eval_index(row: dict[str, Any], default: int) -> int:
    value = row.get("eval_index")
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default

def _combined_dataset_name(subjobs: list[dict[str, Any]]) -> str | None:
    dataset_names = {
        str((subjob.get("env") or {}).get("EG1_COMBINED_DATASET", "")).strip().lower()
        for subjob in subjobs
        if (subjob.get("env") or {}).get("EG1_COMBINED_DATASET")
    }
    if len(dataset_names) != 1:
        return None
    return dataset_names.pop()

def _reference_campaigns_for_job(job: dict[str, Any], dataset: Any) -> tuple[str, ...] | None:
    detector = str(job.get("detector", "") or "")
    dataset_name = str(dataset or "").strip().lower()
    if not detector.startswith("eg1_"):
        return None
    if dataset_name not in {"0501", "cic", "lanl", "optc"}:
        return None
    reference = reference_campaigns_for_dataset(dataset_name)
    return reference or None

def _collect_combined_metrics_rows(
    job: dict[str, Any],
    subjobs: list[dict[str, Any]],
    aggregate_dataset: str,
) -> list[dict[str, Any]]:
    grouped_by_eval: dict[int, list[dict[str, Any]]] = {}
    for subjob in subjobs:
        metrics_path = subjob.get("metrics_path")
        predictions_path = subjob.get("predictions_path") or derive_predictions_path(metrics_path)
        rows = read_csv_rows(metrics_path) if metrics_path else []
        prediction_groups = load_prediction_groups(predictions_path)
        group_name = _combined_campaign_name(subjob)
        for index, row in enumerate(rows):
            eval_index = _resolve_eval_index(row, index)
            grouped_by_eval.setdefault(eval_index, []).append(
                {
                    "group_name": group_name,
                    "metrics_row": row,
                    "prediction_rows": prediction_groups.get(eval_index, []),
                    "metrics_path": metrics_path,
                    "predictions_path": predictions_path,
                }
            )

    result: list[dict[str, Any]] = []
    for eval_index in sorted(grouped_by_eval):
        items = grouped_by_eval[eval_index]
        count_totals = _combined_count_totals(items)
        precision = _safe_div(count_totals["tp"], count_totals["tp"] + count_totals["fp"])
        recall = _safe_div(count_totals["tp"], count_totals["tp"] + count_totals["fn"])
        grouped_predictions = {
            item["group_name"]: item["prediction_rows"]
            for item in items
            if item["prediction_rows"]
        }
        cdr_summary = compute_grouped_campaign_summary(grouped_predictions)
        result.append(
            {
                "detector": job.get("detector"),
                "job_id": job.get("job_id", aggregate_dataset),
                "dataset": aggregate_dataset,
                "seed": job.get("seed"),
                "auc": _mean_metric(items, "auc"),
                "ap": _mean_metric(items, "ap"),
                "threshold": "",
                "tp": count_totals["tp"],
                "fp": count_totals["fp"],
                "fn": count_totals["fn"],
                "tn": count_totals["tn"],
                "tpr": _safe_div(count_totals["tp"], count_totals["tp"] + count_totals["fn"]),
                "fpr": _safe_div(count_totals["fp"], count_totals["fp"] + count_totals["tn"]),
                "fnr": _safe_div(count_totals["fn"], count_totals["tp"] + count_totals["fn"]),
                "tnr": _safe_div(count_totals["tn"], count_totals["fp"] + count_totals["tn"]),
                "precision": precision,
                "recall": recall,
                "f1": _safe_div(2.0 * precision * recall, precision + recall),
                "train_time": _sum_metric(items, "train_time"),
                "test_time": _sum_metric(items, "test_time"),
                **cdr_summary,
                "metrics_path": ";".join(str(item["metrics_path"]) for item in items if item["metrics_path"]),
                "predictions_path": ";".join(
                    str(item["predictions_path"]) for item in items if item["predictions_path"]
                ),
            }
        )
    return result

def _combined_campaign_name(subjob: dict[str, Any]) -> str:
    env = subjob.get("env") or {}
    return str(env.get("EG1_CAMPAIGN_LABEL", subjob.get("dataset", "unknown")))

def _combined_count_totals(items: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "tp": sum(_metric_float(item["metrics_row"], "tp") for item in items),
        "fp": sum(_metric_float(item["metrics_row"], "fp") for item in items),
        "fn": sum(_metric_float(item["metrics_row"], "fn") for item in items),
        "tn": sum(_metric_float(item["metrics_row"], "tn") for item in items),
    }

def _mean_metric(items: list[dict[str, Any]], key: str) -> float | str:
    values = [_metric_optional_float(item["metrics_row"], key) for item in items]
    numeric = [value for value in values if value != ""]
    if not numeric:
        return ""
    return sum(float(value) for value in numeric) / len(numeric)

def _sum_metric(items: list[dict[str, Any]], key: str) -> float | str:
    values = [_metric_optional_float(item["metrics_row"], key) for item in items]
    numeric = [value for value in values if value != ""]
    if not numeric:
        return ""
    return sum(float(value) for value in numeric)

def _metric_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

def _metric_optional_float(row: dict[str, Any], key: str) -> float | str:
    value = row.get(key)
    if value in ("", None):
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""

def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0
