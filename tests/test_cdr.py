from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gids_eval.cdr import (
    build_edge_campaigns,
    build_prediction_rows,
    compute_grouped_campaign_summary,
    compute_cdr_summary,
    write_prediction_rows,
)
from gids_eval.io_utils import write_csv_rows
from gids_eval.metrics import BASELINE_FIELDS, collect_metrics_rows

class CdrTests(unittest.TestCase):
    def test_build_prediction_rows_accepts_campaign_iterable(self) -> None:
        rows = build_prediction_rows(
            dataset="lanl",
            campaign=["edge_a", "edge_b"],
            scores=[0.9, 0.8],
            labels=[1, 1],
            predicted=[1, 0],
            eval_index=0,
        )

        self.assertEqual([row["campaign"] for row in rows], ["edge_a", "edge_b"])
        summary = compute_cdr_summary(rows)
        self.assertEqual(summary["campaign_count"], 2)
        self.assertEqual(summary["detected_campaigns"], 1)
        self.assertTrue(summary["cdr_meaningful"])

    def test_build_prediction_rows_flattens_one_level_of_campaign_lists(self) -> None:
        rows = build_prediction_rows(
            dataset="cic",
            campaign=[["botnet", "ddos"], ["infiltration"]],
            scores=[0.9, 0.8, 0.7],
            labels=[1, 0, 1],
            predicted=[1, 0, 1],
            eval_index=0,
        )

        self.assertEqual([row["campaign"] for row in rows], ["botnet", "ddos", "infiltration"])

    def test_build_edge_campaigns_uses_node_map_labels(self) -> None:
        campaigns = build_edge_campaigns(
            dataset="oilrig",
            edge_index=([0, 1], [2, 3]),
            node_map={0: "HOST_A", 1: "HOST_B", 2: "HOST_C", 3: "HOST_D"},
        )

        self.assertEqual(campaigns, ["oilrig__host_a__host_c", "oilrig__host_b__host_d"])

    def test_compute_cdr_summary_uses_precision_envelope(self) -> None:
        rows = build_prediction_rows(
            dataset="oilrig",
            campaign="oilrig",
            scores=[0.95, 0.8, 0.6, 0.4],
            labels=[1, 0, 1, 0],
            predicted=[1, 1, 0, 0],
            eval_index=0,
        )
        rows[2]["campaign"] = "sandworm"

        summary = compute_cdr_summary(rows)

        self.assertEqual(summary["cdr"], 0.5)
        self.assertEqual(summary["campaign_count"], 2)
        self.assertEqual(summary["detected_campaigns"], 1)
        self.assertTrue(summary["cdr_meaningful"])
        self.assertTrue(math.isclose(summary["cdr_auc"], 5.0 / 6.0, rel_tol=1e-9))

    def test_compute_grouped_campaign_summary_treats_groups_as_campaigns(self) -> None:
        oilrig_rows = build_prediction_rows(
            dataset="oilrig",
            scores=[0.9, 0.2],
            labels=[1, 0],
            predicted=[1, 0],
            eval_index=0,
        )
        sandworm_rows = build_prediction_rows(
            dataset="sandworm",
            scores=[0.8, 0.7],
            labels=[1, 0],
            predicted=[0, 1],
            eval_index=0,
        )

        summary = compute_grouped_campaign_summary(
            {
                "oilrig": oilrig_rows,
                "sandworm": sandworm_rows,
            }
        )

        self.assertEqual(summary["campaign_count"], 2)
        self.assertEqual(summary["detected_campaigns"], 1)
        self.assertEqual(summary["cdr"], 0.5)
        self.assertTrue(summary["cdr_meaningful"])
        self.assertGreater(summary["cdr_auc"], 0.0)

    def test_compute_cdr_summary_uses_reference_campaigns_and_ignores_unknown(self) -> None:
        rows = build_prediction_rows(
            dataset="lanl",
            campaign=["lanl_campaign_001", "unknown", "lanl_campaign_001"],
            scores=[0.9, 0.85, 0.1],
            labels=[1, 1, 0],
            predicted=[1, 1, 0],
            eval_index=0,
        )

        summary = compute_cdr_summary(
            rows,
            reference_campaigns=["lanl_campaign_001", "lanl_campaign_002"],
        )

        self.assertEqual(summary["campaign_count"], 2)
        self.assertEqual(summary["detected_campaigns"], 1)
        self.assertEqual(summary["cdr"], 0.5)

    def test_collect_metrics_rows_matches_predictions_by_eval_index(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            metrics_path = tmp_path / "metrics.csv"
            predictions_path = tmp_path / "predictions.csv"

            write_csv_rows(
                metrics_path,
                [
                    {
                        "eval_index": 0,
                        "dataset": "oilrig",
                        "auc": 0.9,
                        "ap": 0.4,
                        "threshold": 0.7,
                        "tp": 1,
                        "fp": 1,
                        "fn": 1,
                        "tn": 1,
                        "tpr": 0.5,
                        "fpr": 0.5,
                        "fnr": 0.5,
                        "tnr": 0.5,
                        "precision": 0.5,
                        "recall": 0.5,
                        "f1": 0.5,
                        "train_time": 1.0,
                        "test_time": 2.0,
                    },
                    {
                        "eval_index": 1,
                        "dataset": "oilrig",
                        "auc": 0.95,
                        "ap": 0.5,
                        "threshold": 0.6,
                        "tp": 2,
                        "fp": 0,
                        "fn": 0,
                        "tn": 2,
                        "tpr": 1.0,
                        "fpr": 0.0,
                        "fnr": 0.0,
                        "tnr": 1.0,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                        "train_time": 1.0,
                        "test_time": 2.0,
                    },
                ],
                [
                    "eval_index",
                    "dataset",
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
                ],
            )

            eval0 = build_prediction_rows(
                dataset="oilrig",
                scores=[0.9, 0.8, 0.4],
                labels=[1, 0, 1],
                predicted=[1, 1, 0],
                eval_index=0,
            )
            eval0[2]["campaign"] = "sandworm"
            eval1 = build_prediction_rows(
                dataset="oilrig",
                scores=[0.9, 0.8],
                labels=[1, 1],
                predicted=[1, 1],
                eval_index=1,
            )
            eval1[1]["campaign"] = "sandworm"
            write_prediction_rows(predictions_path, eval0, overwrite=True)
            write_prediction_rows(predictions_path, eval1)

            rows = collect_metrics_rows(
                {
                    "detector": "argus",
                    "job_id": "oilrig",
                    "dataset": "oilrig",
                    "seed": 0,
                    "metrics_path": str(metrics_path),
                    "predictions_path": str(predictions_path),
                }
            )

        self.assertEqual(len(rows), 2)
        self.assertTrue(set(BASELINE_FIELDS).issubset(rows[0]))
        self.assertEqual(rows[0]["cdr"], 0.5)
        self.assertEqual(rows[0]["campaign_count"], 2)
        self.assertEqual(rows[0]["detected_campaigns"], 1)
        self.assertEqual(rows[1]["cdr"], 1.0)
        self.assertEqual(rows[1]["campaign_count"], 2)
        self.assertEqual(rows[1]["detected_campaigns"], 2)

    def test_collect_metrics_rows_uses_eg1_reference_campaigns(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            metrics_path = tmp_path / "metrics.csv"
            predictions_path = tmp_path / "predictions.csv"

            write_csv_rows(
                metrics_path,
                [
                    {
                        "dataset": "lanl",
                        "auc": 0.9,
                        "ap": 0.4,
                        "threshold": 0.7,
                        "tp": 1,
                        "fp": 1,
                        "fn": 1,
                        "tn": 1,
                        "tpr": 0.5,
                        "fpr": 0.5,
                        "fnr": 0.5,
                        "tnr": 0.5,
                        "precision": 0.5,
                        "recall": 0.5,
                        "f1": 0.5,
                        "train_time": 1.0,
                        "test_time": 2.0,
                    }
                ],
                [
                    "dataset",
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
                ],
            )

            rows_in = build_prediction_rows(
                dataset="lanl",
                campaign=["lanl_campaign_001", "unknown"],
                scores=[0.9, 0.8],
                labels=[1, 1],
                predicted=[1, 1],
                eval_index=0,
            )
            write_prediction_rows(predictions_path, rows_in, overwrite=True)

            with patch("gids_eval.metrics.reference_campaigns_for_dataset", return_value=("lanl_campaign_001", "lanl_campaign_002")):
                rows = collect_metrics_rows(
                    {
                        "detector": "eg1_argus_lanl_30m_s00",
                        "job_id": "lanl",
                        "dataset": "lanl",
                        "seed": 0,
                        "metrics_path": str(metrics_path),
                        "predictions_path": str(predictions_path),
                    }
                )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["campaign_count"], 2)
        self.assertEqual(rows[0]["detected_campaigns"], 1)
        self.assertEqual(rows[0]["cdr"], 0.5)

    def test_collect_metrics_rows_aggregates_eg1_combined_0501(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            oilrig_metrics = tmp_path / "oilrig_metrics.csv"
            oilrig_predictions = tmp_path / "oilrig_predictions.csv"
            sandworm_metrics = tmp_path / "sandworm_metrics.csv"
            sandworm_predictions = tmp_path / "sandworm_predictions.csv"

            write_csv_rows(
                oilrig_metrics,
                [
                    {
                        "dataset": "oilrig",
                        "auc": 0.9,
                        "ap": 0.8,
                        "threshold": 0.5,
                        "tp": 3,
                        "fp": 1,
                        "fn": 0,
                        "tn": 6,
                        "tpr": 1.0,
                        "fpr": 1 / 7,
                        "fnr": 0.0,
                        "tnr": 6 / 7,
                        "precision": 0.75,
                        "recall": 1.0,
                        "f1": 6 / 7,
                        "train_time": 1.0,
                        "test_time": 2.0,
                    }
                ],
                [
                    "dataset",
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
                ],
            )
            write_csv_rows(
                sandworm_metrics,
                [
                    {
                        "dataset": "sandworm",
                        "auc": 0.7,
                        "ap": 0.6,
                        "threshold": 0.4,
                        "tp": 0,
                        "fp": 1,
                        "fn": 2,
                        "tn": 7,
                        "tpr": 0.0,
                        "fpr": 1 / 8,
                        "fnr": 1.0,
                        "tnr": 7 / 8,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "train_time": 3.0,
                        "test_time": 4.0,
                    }
                ],
                [
                    "dataset",
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
                ],
            )

            write_prediction_rows(
                oilrig_predictions,
                build_prediction_rows(
                    dataset="oilrig",
                    scores=[0.9, 0.2],
                    labels=[1, 0],
                    predicted=[1, 0],
                    eval_index=0,
                ),
                overwrite=True,
            )
            write_prediction_rows(
                sandworm_predictions,
                build_prediction_rows(
                    dataset="sandworm",
                    scores=[0.8, 0.7],
                    labels=[1, 0],
                    predicted=[0, 1],
                    eval_index=0,
                ),
                overwrite=True,
            )

            rows = collect_metrics_rows(
                {
                    "detector": "argus",
                    "job_id": "detection",
                    "seed": 0,
                    "subjobs": [
                        {
                            "job_id": "oilrig",
                            "dataset": "oilrig",
                            "metrics_path": str(oilrig_metrics),
                            "predictions_path": str(oilrig_predictions),
                            "env": {
                                "EG1_COMBINED_DATASET": "0501",
                                "EG1_CAMPAIGN_LABEL": "oilrig",
                            },
                        },
                        {
                            "job_id": "sandworm",
                            "dataset": "sandworm",
                            "metrics_path": str(sandworm_metrics),
                            "predictions_path": str(sandworm_predictions),
                            "env": {
                                "EG1_COMBINED_DATASET": "0501",
                                "EG1_CAMPAIGN_LABEL": "sandworm",
                            },
                        },
                    ],
                }
            )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["dataset"], "0501")
        self.assertAlmostEqual(row["auc"], 0.8)
        self.assertAlmostEqual(row["ap"], 0.7)
        self.assertEqual(row["campaign_count"], 2)
        self.assertEqual(row["detected_campaigns"], 1)
        self.assertAlmostEqual(row["cdr"], 0.5)
        self.assertAlmostEqual(row["precision"], 0.6)
        self.assertAlmostEqual(row["recall"], 0.6)
        self.assertAlmostEqual(row["f1"], 0.6)
        self.assertAlmostEqual(row["train_time"], 4.0)
        self.assertAlmostEqual(row["test_time"], 6.0)

if __name__ == "__main__":
    unittest.main()
