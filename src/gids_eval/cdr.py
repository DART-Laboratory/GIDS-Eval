from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable as IterableCollection
from pathlib import Path
from typing import Any, Iterable

from gids_eval.io_utils import ensure_dir, read_csv_rows

PREDICTION_FIELDS = [
    "eval_index",
    "sample_index",
    "dataset",
    "campaign",
    "label",
    "score",
    "weight",
    "predicted",
]

_CAMPAIGN_ALIASES = {
    "wizard": "wizard_spider",
    "wizard_spider": "wizard_spider",
    "wizard-spider": "wizard_spider",
    "o_wizard": "wizard_spider",
    "o_wizard_spider": "wizard_spider",
    "0501-wizard-spider": "wizard_spider",
    "oilrig": "oilrig",
    "o_oilrig": "oilrig",
    "0501-oilrig": "oilrig",
    "sandworm": "sandworm",
    "o_sandworm": "sandworm",
    "0501-sandworm": "sandworm",
    "lanl": "lanl",
    "optc": "optc",
    "cic": "cic",
}

def canonical_campaign_name(value: Any) -> str:
    text = str(value or "").strip().lower().replace(" ", "_")
    if text in _CAMPAIGN_ALIASES:
        return _CAMPAIGN_ALIASES[text]
    if text.startswith("o_"):
        text = text[2:]
    if text.startswith("0501-"):
        text = text[5:]
    return _CAMPAIGN_ALIASES.get(text, text or "unknown")

def derive_predictions_path(metrics_path: str | Path | None) -> str | None:
    if not metrics_path:
        return None
    return str(Path(metrics_path).with_name("predictions.csv"))

def build_prediction_rows(
    *,
    dataset: Any,
    scores: Iterable[Any],
    labels: Iterable[Any],
    weights: Iterable[Any] | None = None,
    predicted: Iterable[Any] | None = None,
    eval_index: int = 0,
    campaign: Any | None = None,
) -> list[dict[str, Any]]:
    dataset_label = canonical_campaign_name(dataset)
    score_list = list(scores)
    label_list = list(labels)
    weight_list = list(weights) if weights is not None else [1.0] * len(score_list)
    predicted_list = list(predicted) if predicted is not None else [False] * len(score_list)
    campaign_list = _normalize_campaign_values(campaign, len(score_list), dataset_label)

    if not (
        len(score_list)
        == len(label_list)
        == len(weight_list)
        == len(predicted_list)
        == len(campaign_list)
    ):
        raise ValueError("Prediction export inputs must have the same length")

    rows: list[dict[str, Any]] = []
    for sample_index, (score, label, weight, is_predicted, campaign_value) in enumerate(
        zip(score_list, label_list, weight_list, predicted_list, campaign_list)
    ):
        rows.append(
            {
                "eval_index": int(eval_index),
                "sample_index": sample_index,
                "dataset": dataset_label,
                "campaign": canonical_campaign_name(campaign_value),
                "label": int(float(label)),
                "score": float(score),
                "weight": float(weight),
                "predicted": bool(is_predicted),
            }
        )
    return rows

def build_edge_campaigns(
    *,
    dataset: Any,
    edge_index: Any,
    node_map: Any | None = None,
) -> list[str]:
    dataset_label = canonical_campaign_name(dataset)
    src_values, dst_values = _edge_index_pairs(edge_index)
    campaigns: list[str] = []
    for src_value, dst_value in zip(src_values, dst_values):
        src_label = _resolve_node_label(src_value, node_map)
        dst_label = _resolve_node_label(dst_value, node_map)
        campaigns.append(canonical_campaign_name(f"{dataset_label}__{src_label}__{dst_label}"))
    return campaigns

def write_prediction_rows(
    path: str | Path | None,
    rows: list[dict[str, Any]],
    *,
    overwrite: bool = False,
) -> None:
    if not path or not rows:
        return
    target = Path(path)
    ensure_dir(target.parent)
    mode = "w" if overwrite or not target.exists() else "a"
    with target.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        if mode == "w":
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in PREDICTION_FIELDS})

def load_prediction_groups(path: str | Path | None) -> dict[int, list[dict[str, Any]]]:
    if not path:
        return {}
    rows = read_csv_rows(path)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_to_int(row.get("eval_index"), 0)].append(
            {
                "campaign": canonical_campaign_name(row.get("campaign") or row.get("dataset")),
                "label": _to_int(row.get("label"), 0),
                "score": _to_float(row.get("score"), 0.0),
                "weight": max(_to_float(row.get("weight"), 1.0), 0.0),
                "predicted": _to_bool(row.get("predicted")),
            }
        )
    return dict(grouped)

def compute_cdr_summary(
    rows: list[dict[str, Any]],
    *,
    reference_campaigns: Iterable[Any] | None = None,
) -> dict[str, Any]:
    if not rows:
        return _empty_cdr_summary()

    if reference_campaigns is not None:
        positive_campaigns = {
            canonical_campaign_name(item)
            for item in reference_campaigns
            if canonical_campaign_name(item) not in {"", "unknown"}
        }
    else:
        positive_campaigns = {
            row["campaign"]
            for row in rows
            if row["label"] == 1 and row["weight"] > 0 and row["campaign"] not in {"", "unknown"}
        }
    if not positive_campaigns:
        return {
            "cdr": 0.0,
            "cdr_auc": 0.0,
            "campaign_count": 0,
            "detected_campaigns": 0,
            "cdr_meaningful": False,
        }

    detected_campaigns = {
        row["campaign"]
        for row in rows
        if row["predicted"] and row["label"] == 1 and row["weight"] > 0 and row["campaign"] in positive_campaigns
    }
    curve_points = _build_cdr_curve_points(rows, positive_campaigns)
    return {
        "cdr": _safe_div(len(detected_campaigns), len(positive_campaigns)),
        "cdr_auc": _compute_cdr_auc(curve_points),
        "campaign_count": len(positive_campaigns),
        "detected_campaigns": len(detected_campaigns),
        "cdr_meaningful": len(positive_campaigns) >= 2,
    }

def compute_grouped_campaign_summary(grouped_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    if not grouped_rows:
        return _empty_cdr_summary()

    normalized_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group_name, rows in grouped_rows.items():
        normalized_groups[canonical_campaign_name(group_name)].extend(rows)

    positive_groups: dict[str, list[dict[str, Any]]] = {}
    for group_name, rows in normalized_groups.items():
        if any(row["label"] == 1 and row["weight"] > 0 for row in rows):
            positive_groups[group_name] = rows

    if not positive_groups:
        return {
            "cdr": 0.0,
            "cdr_auc": 0.0,
            "campaign_count": 0,
            "detected_campaigns": 0,
            "cdr_meaningful": False,
        }

    detected_groups = {
        group_name
        for group_name, rows in positive_groups.items()
        if any(row["predicted"] and row["label"] == 1 and row["weight"] > 0 for row in rows)
    }
    cdr_auc = _compute_grouped_cdr_auc(positive_groups)
    return {
        "cdr": _safe_div(len(detected_groups), len(positive_groups)),
        "cdr_auc": cdr_auc,
        "campaign_count": len(positive_groups),
        "detected_campaigns": len(detected_groups),
        "cdr_meaningful": len(positive_groups) >= 2,
    }

def _build_cdr_curve_points(
    rows: list[dict[str, Any]],
    positive_campaigns: set[str],
) -> list[dict[str, Any]]:
    grouped_by_score: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_by_score[float(row["score"])].append(row)

    detected: set[str] = set()
    tp_weight = 0.0
    fp_weight = 0.0
    curve_points: list[dict[str, Any]] = []
    for score in sorted(grouped_by_score, reverse=True):
        for row in grouped_by_score[score]:
            if row["label"] == 1:
                tp_weight += row["weight"]
                if row["weight"] > 0 and row["campaign"] in positive_campaigns:
                    detected.add(row["campaign"])
            else:
                fp_weight += row["weight"]
        curve_points.append(
            {
                "precision": _safe_div(tp_weight, tp_weight + fp_weight),
                "cdr": _safe_div(len(detected), len(positive_campaigns)),
            }
        )
    return curve_points

def _build_group_detection_curve_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped_by_score: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_by_score[float(row["score"])].append(row)

    detected = False
    tp_weight = 0.0
    fp_weight = 0.0
    curve_points: list[dict[str, Any]] = []
    for score in sorted(grouped_by_score, reverse=True):
        for row in grouped_by_score[score]:
            if row["label"] == 1:
                tp_weight += row["weight"]
                if row["weight"] > 0:
                    detected = True
            else:
                fp_weight += row["weight"]
        curve_points.append(
            {
                "precision": _safe_div(tp_weight, tp_weight + fp_weight),
                "cdr": 1.0 if detected else 0.0,
            }
        )
    return curve_points

def _normalize_campaign_values(
    campaign: Any | None,
    size: int,
    dataset_label: str,
) -> list[Any]:
    if campaign is None:
        return [dataset_label] * size
    if isinstance(campaign, (str, bytes)):
        return [campaign] * size
    if isinstance(campaign, IterableCollection):
        values = list(campaign)
        if len(values) != size:
            flattened: list[Any] = []
            can_flatten = True
            for value in values:
                if isinstance(value, (str, bytes)):
                    flattened.append(value)
                elif isinstance(value, IterableCollection):
                    flattened.extend(list(value))
                else:
                    can_flatten = False
                    break
            if can_flatten and len(flattened) == size:
                return flattened
            raise ValueError("Campaign export inputs must have the same length as prediction arrays")
        return values
    return [campaign] * size

def _edge_index_pairs(edge_index: Any) -> tuple[list[Any], list[Any]]:
    if hasattr(edge_index, "tolist"):
        edge_index = edge_index.tolist()
    if isinstance(edge_index, tuple) and len(edge_index) == 2:
        src_values, dst_values = edge_index
    elif isinstance(edge_index, list) and len(edge_index) == 2:
        src_values, dst_values = edge_index
    else:
        raise ValueError("edge_index must contain source and destination arrays")
    return list(src_values), list(dst_values)

def _resolve_node_label(value: Any, node_map: Any | None) -> str:
    index = _to_int(value, 0)
    if node_map is not None:
        try:
            if isinstance(node_map, dict):
                mapped = node_map.get(index, node_map.get(str(index), value))
            else:
                mapped = node_map[index]
            return str(mapped)
        except Exception:
            pass
    return str(value)

def _compute_cdr_auc(curve_points: list[dict[str, Any]]) -> float:
    if not curve_points:
        return 0.0

    envelope = _build_precision_envelope(curve_points)

    area = 0.0
    previous_precision = 0.0
    for precision, cdr in envelope:
        area += max(precision - previous_precision, 0.0) * cdr
        previous_precision = precision
    return min(max(area, 0.0), 1.0)

def _compute_grouped_cdr_auc(grouped_rows: dict[str, list[dict[str, Any]]]) -> float:
    if not grouped_rows:
        return 0.0

    envelopes = [
        _build_precision_envelope(_build_group_detection_curve_points(rows))
        for rows in grouped_rows.values()
        if rows
    ]
    if not envelopes:
        return 0.0

    breakpoints = sorted({precision for envelope in envelopes for precision, _ in envelope})
    if not breakpoints:
        return 0.0

    area = 0.0
    previous_precision = 0.0
    for precision in breakpoints:
        cdr = sum(_envelope_value_at_precision(envelope, precision) for envelope in envelopes) / len(envelopes)
        area += max(precision - previous_precision, 0.0) * cdr
        previous_precision = precision
    return min(max(area, 0.0), 1.0)

def _build_precision_envelope(curve_points: list[dict[str, Any]]) -> list[tuple[float, float]]:
    best_by_precision: dict[float, float] = {}
    for point in curve_points:
        precision = min(max(float(point["precision"]), 0.0), 1.0)
        cdr = min(max(float(point["cdr"]), 0.0), 1.0)
        best_by_precision[precision] = max(best_by_precision.get(precision, 0.0), cdr)

    envelope: list[tuple[float, float]] = []
    running_cdr = 0.0
    for precision in reversed(sorted(best_by_precision)):
        running_cdr = max(running_cdr, best_by_precision[precision])
        envelope.append((precision, running_cdr))
    envelope.reverse()
    return envelope

def _envelope_value_at_precision(envelope: list[tuple[float, float]], precision: float) -> float:
    for breakpoint, cdr in envelope:
        if precision <= breakpoint:
            return cdr
    return 0.0

def _empty_cdr_summary() -> dict[str, Any]:
    return {
        "cdr": "",
        "cdr_auc": "",
        "campaign_count": "",
        "detected_campaigns": "",
        "cdr_meaningful": "",
    }

def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0

def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y"}

def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _to_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default
