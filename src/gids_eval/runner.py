from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gids_eval.cdr import derive_predictions_path
from gids_eval.config import load_detector_config, sniff_config_kind
from gids_eval.io_utils import dump_json, dump_yaml, ensure_dir, merge_env, safe_rmtree
from gids_eval.manifests import preview_manifests, stage_manifests
from gids_eval.metrics import build_cost_row, collect_metrics_rows, write_reports
from gids_eval.process_monitor import derive_phase_profile_path, load_phase_profile_usage, run_command_logged
from gids_eval.runtime import RunOptions
from gids_eval.status_codes import UNSUPPORTED_EXIT_CODE

class PipelineRunner:
    def __init__(self, config_path: str | Path, options: RunOptions | None = None) -> None:
        self.config, self.context = load_detector_config(config_path)
        self.options = options or RunOptions()

        self.detector = str(self.config["detector"])
        self.seed = int(self.config["seed"])
        self.workspace_root = Path(self.context["workspace_root"])
        self.input_root = Path(self.context["input_root"])
        self.output_root = Path(self.context["output_root"])
        self.metadata_root = Path(self.context["metadata_root"])
        self.logs_root = self.metadata_root / "logs"
        self.summary_path = self.metadata_root / "run_summary.json"
        self.baseline_csv = self.metadata_root / "baseline_metrics.csv"
        self.cost_csv = self.metadata_root / "cost_metrics.csv"

    def validate(self) -> dict[str, Any]:
        staged = preview_manifests(
            self.config["manifests"],
            {**self.context, "stage_root": str(self.input_root)},
        )
        return {
            "kind": "detector",
            "detector": self.detector,
            "seed": self.seed,
            "workspace_root": str(self.workspace_root),
            "input_root": str(self.input_root),
            "output_root": str(self.output_root),
            "metadata_root": str(self.metadata_root),
            "manifest_count": len(self.config["manifests"]),
            "job_count": len(self.config["jobs"]),
            "preprocess_enabled": bool(self.config.get("preprocess", {}).get("enabled", False)),
            "manifests": staged,
        }

    def run(self) -> dict[str, Any]:
        if self.options.to_stage:
            raise ValueError("--to-stage is only supported for experiment configs")

        if self.options.clean and self.workspace_root.exists():
            safe_rmtree(self.workspace_root, self.workspace_root.parent)

        ensure_dir(self.input_root)
        ensure_dir(self.output_root)
        ensure_dir(self.logs_root)

        dump_yaml(self.metadata_root / "resolved_config.yaml", self._resolved_config_for_dump())

        if self.options.dry_run:
            staged = preview_manifests(
                self.config["manifests"],
                {**self.context, "stage_root": str(self.input_root)},
            )
        else:
            staged = stage_manifests(
                self.config["manifests"],
                {**self.context, "stage_root": str(self.input_root)},
                stage_mode=str(self.config.get("stage_mode", "symlink")),
            )
        dump_json(self.metadata_root / "staged_manifests.json", staged)

        records: list[dict[str, Any]] = []
        preprocess_cfg = self.config.get("preprocess", {})
        if preprocess_cfg.get("enabled", False):
            records.append(self._run_step("preprocess", preprocess_cfg, dataset=None, phase="preprocess"))

        for job in self.config["jobs"]:
            records.append(self._run_step(str(job["id"]), job, dataset=job.get("dataset"), phase="baseline"))

        baseline_rows: list[dict[str, Any]] = []
        cost_rows: list[dict[str, Any]] = []
        for record in records:
            metrics_rows = collect_metrics_rows(record)
            record["metrics_rows"] = metrics_rows
            baseline_rows.extend(metrics_rows)
            cost_rows.append(build_cost_row(record))

        write_reports(baseline_rows, cost_rows, self.baseline_csv, self.cost_csv)

        summary = {
            "kind": "detector",
            "detector": self.detector,
            "seed": self.seed,
            "run_label": self.config["run_label"],
            "dry_run": self.options.dry_run,
            "workspace_root": str(self.workspace_root),
            "input_root": str(self.input_root),
            "output_root": str(self.output_root),
            "metadata_root": str(self.metadata_root),
            "baseline_metrics_csv": str(self.baseline_csv),
            "cost_metrics_csv": str(self.cost_csv),
            "records": records,
        }
        dump_json(self.summary_path, summary)
        return summary

    def _run_step(
        self,
        step_id: str,
        step_cfg: dict[str, Any],
        dataset: str | None,
        phase: str,
    ) -> dict[str, Any]:
        command = [str(part) for part in step_cfg["command"]]
        workdir = str(step_cfg["workdir"])
        metrics_path = step_cfg.get("metrics_path")
        predictions_path = derive_predictions_path(metrics_path)
        profile_path = derive_phase_profile_path(metrics_path)
        env = self._build_env(
            {
                **step_cfg.get("env", {}),
                **_build_metric_env(step_id, dataset, metrics_path, predictions_path),
            }
        )
        stdout_log = self.logs_root / f"{step_id}.stdout.log"
        stderr_log = self.logs_root / f"{step_id}.stderr.log"

        started_at = _utc_now()
        exit_code = 0
        status = "dry_run" if self.options.dry_run else "success"
        peak_rss_kb = 0
        mean_rss_kb = 0.0
        peak_cpu_percent = 0.0
        mean_cpu_percent = 0.0
        peak_process_count = 0
        sample_count = 0
        cpu_sample_count = 0

        if self.options.dry_run:
            ensure_dir(stdout_log.parent)
            stdout_log.write_text("", encoding="utf-8")
            stderr_log.write_text("", encoding="utf-8")
        else:
            usage = run_command_logged(
                command,
                cwd=workdir,
                env=env,
                stdout_path=stdout_log,
                stderr_path=stderr_log,
                monitor_resources=False,
            )
            profile_usage = load_phase_profile_usage(profile_path)
            resource_usage = profile_usage or usage
            exit_code = usage.exit_code
            peak_rss_kb = resource_usage.peak_rss_kb
            mean_rss_kb = resource_usage.mean_rss_kb
            peak_cpu_percent = resource_usage.peak_cpu_percent
            mean_cpu_percent = resource_usage.mean_cpu_percent
            peak_process_count = resource_usage.peak_process_count
            sample_count = resource_usage.sample_count
            cpu_sample_count = resource_usage.cpu_sample_count
            if exit_code == UNSUPPORTED_EXIT_CODE:
                status = "unsupported"
            elif exit_code != 0:
                status = "failed"

        finished_at = _utc_now()
        wall_time = usage.wall_time_sec if not self.options.dry_run else 0.0

        record = {
            "detector": self.detector,
            "phase": phase,
            "job_id": step_id,
            "dataset": dataset,
            "seed": self.seed,
            "status": status,
            "exit_code": exit_code,
            "started_at": started_at,
            "finished_at": finished_at,
            "wall_time_sec": wall_time,
            "peak_rss_kb": peak_rss_kb,
            "mean_rss_kb": mean_rss_kb,
            "peak_cpu_percent": peak_cpu_percent,
            "mean_cpu_percent": mean_cpu_percent,
            "peak_process_count": peak_process_count,
            "sample_count": sample_count,
            "cpu_sample_count": cpu_sample_count,
            "command": command,
            "workdir": workdir,
            "metrics_path": metrics_path,
            "predictions_path": predictions_path,
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
        }
        dump_json(self.metadata_root / f"{step_id}.json", record)

        if status == "failed":
            raise RuntimeError(f"{self.detector}:{step_id} failed with exit code {exit_code}")
        return record

    def _build_env(self, extra_env: dict[str, Any]) -> dict[str, str]:
        repo_src = str((Path(self.context["repo_root"]) / "src").resolve())
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        pythonpath_parts = [repo_src]
        if current_pythonpath:
            pythonpath_parts.append(current_pythonpath)

        base_env = merge_env(
            dict(os.environ),
            self.config.get("environment", {}),
            {
                "PYTHONPATH": os.pathsep.join(pythonpath_parts),
                "PYTHONHASHSEED": self.seed,
                "PYTHON": self.config["python_bin"],
                "GIDS_EVAL_SEED": self.seed,
                "GIDS_EVAL_REPO_ROOT": self.context["repo_root"],
                "GIDS_EVAL_DATA_ROOT": self.context["data_root"],
                "MIRAGE_INPUT_ROOT": self.input_root,
                "MIRAGE_OUTPUT_ROOT": self.output_root,
            },
            extra_env,
        )
        return base_env

    def _resolved_config_for_dump(self) -> dict[str, Any]:
        payload = dict(self.config)
        payload.pop("_context", None)
        return payload

def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _build_metric_env(
    step_id: str,
    dataset: str | None,
    metrics_path: str | None,
    predictions_path: str | None,
) -> dict[str, str]:
    env = {"GIDS_JOB_ID": step_id}
    if dataset is not None:
        env["GIDS_JOB_DATASET"] = str(dataset)
    if metrics_path:
        env["GIDS_METRICS_PATH"] = str(metrics_path)
        profile_path = derive_phase_profile_path(metrics_path)
        if profile_path:
            env["GIDS_PHASE_PROFILE_PATH"] = str(profile_path)
    if predictions_path:
        env["GIDS_PREDICTIONS_PATH"] = str(predictions_path)
    return env

def create_runner(config_path: str | Path, options: RunOptions | None = None) -> Any:
    kind = sniff_config_kind(config_path)
    if kind == "detector":
        return PipelineRunner(config_path, options=options)
    if kind == "experiment":
        from gids_eval.experiment import ExperimentRunner

        return ExperimentRunner(config_path, options=options)
    raise ValueError(f"Unsupported config kind for {config_path}: {kind}")
