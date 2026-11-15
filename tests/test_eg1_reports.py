from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

EG1_ROOT = Path(__file__).resolve().parents[1] / "EG1"
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(EG1_ROOT) not in sys.path:
    sys.path.insert(0, str(EG1_ROOT))

import eg1lib
from eg1lib import (
    SUMMARY_METRIC_SPECS,
    _ensure_flag,
    _raw_results_fieldnames,
    build_metric_table_rows,
    job_status,
    write_csv,
    write_metric_tables,
    write_progress_report,
)

class Sc1ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summary_rows = [
            {
                "detector": "argus",
                "dataset": "0501",
                "campaign_mode": "family",
                "window_seconds": 75,
                "window_label": "75s",
                "seed_count": 1,
                "cdr_meaningful": True,
                "mean_ap": 0.11,
                "std_ap": 0.0,
                "mean_auc": 0.91,
                "std_auc": 0.0,
                "mean_tpr": 0.71,
                "std_tpr": 0.0,
                "mean_fpr": 0.09,
                "std_fpr": 0.0,
                "mean_precision": 0.21,
                "std_precision": 0.0,
                "mean_recall": 0.31,
                "std_recall": 0.0,
                "mean_f1": 0.26,
                "std_f1": 0.0,
                "mean_cdr": 0.0,
                "std_cdr": 0.0,
                "mean_cdr_auc": 0.0,
                "std_cdr_auc": 0.0,
                "mean_campaign_count": 3.0,
                "std_campaign_count": 0.0,
                "mean_detected_campaigns": 2.0,
                "std_detected_campaigns": 0.0,
                "mean_train_time_sec": 100.0,
                "std_train_time_sec": 0.0,
                "mean_test_time_sec": 5.0,
                "std_test_time_sec": 0.0,
                "mean_featurization_wall_time_sec": 33.0,
                "std_featurization_wall_time_sec": 0.0,
                "mean_detection_wall_time_sec": 66.0,
                "std_detection_wall_time_sec": 0.0,
                "mean_total_wall_time_sec": 99.0,
                "std_total_wall_time_sec": 0.0,
                "mean_featurization_peak_rss_gib": 1.25,
                "std_featurization_peak_rss_gib": 0.0,
                "mean_detection_peak_rss_gib": 2.5,
                "std_detection_peak_rss_gib": 0.0,
                "mean_run_peak_rss_gib": 2.75,
                "std_run_peak_rss_gib": 0.0,
                "mean_featurization_mean_rss_gib": 0.85,
                "std_featurization_mean_rss_gib": 0.0,
                "mean_detection_mean_rss_gib": 1.75,
                "std_detection_mean_rss_gib": 0.0,
                "mean_run_mean_rss_gib": 1.55,
                "std_run_mean_rss_gib": 0.0,
                "mean_featurization_mean_cpu_percent": 12.0,
                "std_featurization_mean_cpu_percent": 0.0,
                "mean_detection_mean_cpu_percent": 58.0,
                "std_detection_mean_cpu_percent": 0.0,
                "mean_run_mean_cpu_percent": 43.0,
                "std_run_mean_cpu_percent": 0.0,
            },
            {
                "detector": "argus",
                "dataset": "0501",
                "campaign_mode": "family",
                "window_seconds": 150,
                "window_label": "150s",
                "seed_count": 1,
                "cdr_meaningful": True,
                "mean_ap": 0.22,
                "std_ap": 0.0,
                "mean_auc": 0.95,
                "std_auc": 0.0,
                "mean_tpr": 0.81,
                "std_tpr": 0.0,
                "mean_fpr": 0.05,
                "std_fpr": 0.0,
                "mean_precision": 0.41,
                "std_precision": 0.0,
                "mean_recall": 0.51,
                "std_recall": 0.0,
                "mean_f1": 0.46,
                "std_f1": 0.0,
                "mean_cdr": 0.0,
                "std_cdr": 0.0,
                "mean_cdr_auc": 0.0,
                "std_cdr_auc": 0.0,
                "mean_campaign_count": 3.0,
                "std_campaign_count": 0.0,
                "mean_detected_campaigns": 3.0,
                "std_detected_campaigns": 0.0,
                "mean_train_time_sec": 150.0,
                "std_train_time_sec": 0.0,
                "mean_test_time_sec": 8.0,
                "std_test_time_sec": 0.0,
                "mean_featurization_wall_time_sec": 44.0,
                "std_featurization_wall_time_sec": 0.0,
                "mean_detection_wall_time_sec": 88.0,
                "std_detection_wall_time_sec": 0.0,
                "mean_total_wall_time_sec": 132.0,
                "std_total_wall_time_sec": 0.0,
                "mean_featurization_peak_rss_gib": 1.5,
                "std_featurization_peak_rss_gib": 0.0,
                "mean_detection_peak_rss_gib": 3.0,
                "std_detection_peak_rss_gib": 0.0,
                "mean_run_peak_rss_gib": 3.5,
                "std_run_peak_rss_gib": 0.0,
                "mean_featurization_mean_rss_gib": 1.1,
                "std_featurization_mean_rss_gib": 0.0,
                "mean_detection_mean_rss_gib": 2.1,
                "std_detection_mean_rss_gib": 0.0,
                "mean_run_mean_rss_gib": 1.85,
                "std_run_mean_rss_gib": 0.0,
                "mean_featurization_mean_cpu_percent": 16.0,
                "std_featurization_mean_cpu_percent": 0.0,
                "mean_detection_mean_cpu_percent": 61.0,
                "std_detection_mean_cpu_percent": 0.0,
                "mean_run_mean_cpu_percent": 46.0,
                "std_run_mean_cpu_percent": 0.0,
            },
        ]

    def test_build_metric_table_rows_pivots_windows_and_swing(self) -> None:
        ap_spec = next(spec for spec in SUMMARY_METRIC_SPECS if spec["slug"] == "ap")
        rows = build_metric_table_rows(self.summary_rows, ap_spec)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["default_window_seconds"], 150)
        self.assertEqual(row["default_window_observed"], "true")
        self.assertAlmostEqual(row["window_75s_mean"], 0.11)
        self.assertAlmostEqual(row["window_150s_mean"], 0.22)
        self.assertAlmostEqual(row["swing_abs"], 0.11)
        self.assertAlmostEqual(row["swing_rel"], 0.5)

    def test_write_metric_tables_writes_index_and_cost_metrics(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "metric_tables"
            write_metric_tables(self.summary_rows, output_root)
            index_path = output_root / "index.csv"
            self.assertTrue(index_path.exists())
            self.assertTrue((output_root / "ap.csv").exists())
            self.assertTrue((output_root / "run_peak_rss_gib.csv").exists())
            self.assertTrue((output_root / "run_mean_rss_gib.csv").exists())
            self.assertTrue((output_root / "run_mean_cpu_percent.csv").exists())
            self.assertTrue((output_root / "campaign_count.csv").exists())
            contents = index_path.read_text(encoding="utf-8")
            self.assertIn("ap", contents)
            self.assertIn("campaign_count", contents)
            self.assertIn("run_peak_rss_gib", contents)
            self.assertIn("run_mean_rss_gib", contents)
            self.assertIn("run_mean_cpu_percent", contents)

    def test_ensure_flag_appends_or_replaces(self) -> None:
        appended = _ensure_flag(["python", "run.py"], "--te_end", "all")
        self.assertEqual(appended[-2:], ["--te_end", "all"])

        replaced = _ensure_flag(["python", "run.py", "--te_end", "test"], "--te_end", "all")
        self.assertEqual(replaced, ["python", "run.py", "--te_end", "all"])

    def test_raw_results_fields_include_campaign_audit_columns(self) -> None:
        fields = _raw_results_fieldnames()
        self.assertIn("campaign_mode", fields)
        self.assertIn("campaign_count", fields)
        self.assertIn("detected_campaigns", fields)
        self.assertIn("cdr_meaningful", fields)

    def test_job_status_marks_detection_unsupported_as_completed(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            baseline_metrics = root / "baseline_metrics.csv"
            cost_metrics = root / "cost_metrics.csv"
            run_summary = root / "run_summary.json"
            metadata_root = root / "metadata"
            metadata_root.mkdir(parents=True, exist_ok=True)

            write_csv(
                cost_metrics,
                [
                    {
                        "phase": "featurization",
                        "status": "success",
                        "exit_code": 0,
                    },
                    {
                        "phase": "detection",
                        "status": "unsupported",
                        "exit_code": 86,
                    },
                ],
                ["phase", "status", "exit_code"],
            )
            baseline_metrics.write_text(
                "detector,job_id,dataset,seed,auc,ap,threshold,tp,fp,fn,tn,tpr,fpr,fnr,tnr,precision,recall,f1,train_time,test_time,cdr,cdr_auc,campaign_count,detected_campaigns,cdr_meaningful,metrics_path,predictions_path\n",
                encoding="utf-8",
            )
            run_summary.write_text(
                (
                    '{"records": ['
                    '{"stage": "featurization", "status": "success"}, '
                    '{"stage": "detection", "status": "unsupported"}'
                    "]}"
                ),
                encoding="utf-8",
            )

            job = SimpleNamespace(
                detector="pikachu",
                dataset="0501",
                window_seconds=75,
                window_label="75s",
                seed=0,
                experiment_id="eg1_pikachu_0501_75s_s00",
                config_path=root / "config.yaml",
                baseline_metrics_path=baseline_metrics,
                cost_metrics_path=cost_metrics,
                run_summary_path=run_summary,
                metadata_root=metadata_root,
            )

            self.assertEqual(job_status(job), "unsupported")

            progress_json, progress_md = write_progress_report(
                [job],
                json_path=root / "progress.json",
                md_path=root / "progress.md",
                scope_name="unit",
            )
            progress_payload = progress_json.read_text(encoding="utf-8")
            self.assertIn('"unsupported_jobs": 1', progress_payload)
            self.assertIn('"completed_jobs": 1', progress_payload)
            progress_md_text = progress_md.read_text(encoding="utf-8")
            self.assertIn("unsupported_jobs: 1", progress_md_text)
            self.assertIn("unsupported=1", progress_md_text)

    def test_manual_override_marks_job_unsupported(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            override_path = root / "manual_status_overrides.json"
            override_path.write_text(
                (
                    '{'
                    '"eg1_pikachu_cic_10m_s00": {'
                    '"status": "unsupported", '
                    '"reason": "user-approved skip for PIKACHU"'
                    "}"
                    "}"
                ),
                encoding="utf-8",
            )
            job = SimpleNamespace(
                detector="pikachu",
                dataset="cic",
                window_seconds=600,
                window_label="10m",
                seed=0,
                experiment_id="eg1_pikachu_cic_10m_s00",
                config_path=root / "config.yaml",
                baseline_metrics_path=root / "baseline_metrics.csv",
                cost_metrics_path=root / "cost_metrics.csv",
                run_summary_path=root / "run_summary.json",
                metadata_root=root / "metadata",
            )
            with patch.object(eg1lib, "MANUAL_STATUS_OVERRIDES_PATH", override_path):
                self.assertEqual(job_status(job), "unsupported")

if __name__ == "__main__":
    unittest.main()
