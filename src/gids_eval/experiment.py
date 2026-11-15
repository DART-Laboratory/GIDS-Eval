from __future__ import annotations

import os
import shutil
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gids_eval.cache import (
    build_path_signature,
    cache_entry_exists,
    materialize_cached_output,
    remove_path,
    stable_hash,
    store_cache_entry,
)
from gids_eval.cdr import derive_predictions_path
from gids_eval.config import load_experiment_config
from gids_eval.io_utils import dump_json, dump_yaml, ensure_dir, expand_templates, merge_env, safe_rmtree
from gids_eval.manifests import load_manifest, preview_manifests, stage_manifests
from gids_eval.metrics import build_cost_row, collect_metrics_rows, write_reports
from gids_eval.process_monitor import derive_phase_profile_path, load_phase_profile_usage, run_command_logged
from gids_eval.runtime import RunOptions
from gids_eval.schema import MODULE_STAGES
from gids_eval.status_codes import UNSUPPORTED_EXIT_CODE

class ExperimentRunner:
    def __init__(self, config_path: str | Path, options: RunOptions | None = None) -> None:
        self.config, self.context = load_experiment_config(config_path)
        self.options = options or RunOptions()

        self.experiment_id = str(self.config["experiment_id"])
        self.seed = int(self.config["seed"])
        self.workspace_root = Path(self.context["workspace_root"])
        self.input_root = Path(self.context["input_root"])
        self.output_root = Path(self.context["output_root"])
        self.modules_root = Path(self.context["modules_root"])
        self.metadata_root = Path(self.context["metadata_root"])
        self.cache_root = Path(self.context["cache_root"])
        self.logs_root = self.metadata_root / "logs"
        self.summary_path = self.metadata_root / "run_summary.json"
        self.baseline_csv = self.metadata_root / "baseline_metrics.csv"
        self.cost_csv = self.metadata_root / "cost_metrics.csv"
        self.staged_manifests_path = self.metadata_root / "staged_manifests.json"
        self.final_output_link = self.output_root / "final"
        self.manifest_fingerprint = self._build_manifest_fingerprint()

    def validate(self) -> dict[str, Any]:
        staged = preview_manifests(
            self.config.get("manifests", []),
            {**self.context, "stage_root": str(self.input_root)},
        )
        return {
            "kind": "experiment",
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "workspace_root": str(self.workspace_root),
            "input_root": str(self.input_root),
            "output_root": str(self.output_root),
            "metadata_root": str(self.metadata_root),
            "cache_root": str(self.cache_root),
            "manifest_count": len(self.config.get("manifests", [])),
            "module_count": len(MODULE_STAGES),
            "manifests": staged,
            "modules": [
                {
                    "stage": stage,
                    "runner": self.config["modules"][stage]["runner"],
                    "cache_enabled": bool(
                        self.config["modules"][stage].get("cache", {}).get("enabled", True)
                    ),
                    "inputs": self.config["modules"][stage].get("inputs", {}),
                    "job_count": len(self.config["modules"][stage].get("jobs", [])),
                    "artifacts": self.config["modules"][stage].get("artifacts", {}),
                    "metrics_relpath": self.config["modules"][stage].get("metrics_relpath"),
                }
                for stage in MODULE_STAGES
            ],
        }

    def run(self) -> dict[str, Any]:
        if self.options.clean and self.workspace_root.exists():
            safe_rmtree(self.workspace_root, self.workspace_root.parent)

        ensure_dir(self.input_root)
        ensure_dir(self.modules_root)
        ensure_dir(self.metadata_root)
        ensure_dir(self.logs_root)

        dump_yaml(self.metadata_root / "resolved_config.yaml", self._resolved_config_for_dump())

        if self.options.dry_run:
            staged = preview_manifests(
                self.config.get("manifests", []),
                {**self.context, "stage_root": str(self.input_root)},
            )
        else:
            staged = stage_manifests(
                self.config.get("manifests", []),
                {**self.context, "stage_root": str(self.input_root)},
                stage_mode=str(self.config.get("stage_mode", "symlink")),
            )
        dump_json(self.staged_manifests_path, staged)

        records: list[dict[str, Any]] = []
        previous_records: dict[str, dict[str, Any]] = {}
        selected_stages = self._selected_stages()

        for stage in selected_stages:
            record = self._run_module(stage, previous_records)
            records.append(record)
            previous_records[stage] = record

        remove_path(self.final_output_link)
        ensure_dir(self.final_output_link.parent)
        final_stage = selected_stages[-1]
        os.symlink(previous_records[final_stage]["output_dir"], self.final_output_link, target_is_directory=True)

        baseline_rows: list[dict[str, Any]] = []
        cost_rows: list[dict[str, Any]] = []
        for record in records:
            metrics_rows = collect_metrics_rows(record)
            record["metrics_rows"] = metrics_rows
            baseline_rows.extend(metrics_rows)
            cost_rows.append(build_cost_row(record))

        write_reports(baseline_rows, cost_rows, self.baseline_csv, self.cost_csv)

        summary = {
            "kind": "experiment",
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "run_label": self.config["run_label"],
            "dry_run": self.options.dry_run,
            "workspace_root": str(self.workspace_root),
            "input_root": str(self.input_root),
            "output_root": str(self.output_root),
            "metadata_root": str(self.metadata_root),
            "cache_root": str(self.cache_root),
            "baseline_metrics_csv": str(self.baseline_csv),
            "cost_metrics_csv": str(self.cost_csv),
            "selected_stages": selected_stages,
            "final_stage": final_stage,
            "records": records,
        }
        dump_json(self.summary_path, summary)
        return summary

    def _run_module(
        self,
        stage: str,
        previous_records: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        module_workspace_dir = self.modules_root / stage
        module_output_dir = module_workspace_dir / "output"
        cache_key = self._build_module_cache_key(stage, previous_records)
        cache_dir = self.cache_root / stage / cache_key
        stage_context = self._build_stage_context(stage, previous_records, module_workspace_dir, module_output_dir, cache_dir)
        module_cfg = expand_templates(self.config["_raw_modules"][stage], stage_context)
        self._assert_resolved(module_cfg, f"modules.{stage}")

        command = [str(part) for part in module_cfg.get("command", [])]
        workdir = str(module_cfg.get("workdir", self.context["repo_root"]))
        metrics_relpath = module_cfg.get("metrics_relpath")
        metrics_path = str(module_output_dir / metrics_relpath) if metrics_relpath else None
        predictions_path = derive_predictions_path(metrics_path)
        profile_path = derive_phase_profile_path(metrics_path)
        stdout_log = self.logs_root / f"{stage}.stdout.log"
        stderr_log = self.logs_root / f"{stage}.stderr.log"
        artifacts = self._build_artifact_paths(module_cfg, module_output_dir)
        subjobs = self._build_command_group_jobs(stage, module_cfg, stage_context, module_output_dir)
        if module_cfg["runner"] == "command_group":
            command = [f"command_group:{len(subjobs)}"]
            metrics_path = None
            predictions_path = None

        started_at = _utc_now()
        start_t = time.perf_counter()
        exit_code = 0
        status = "dry_run" if self.options.dry_run else "success"
        cache_hit = False
        peak_rss_kb = 0
        mean_rss_kb = 0.0
        peak_cpu_percent = 0.0
        mean_cpu_percent = 0.0
        peak_process_count = 0
        sample_count = 0
        cpu_sample_count = 0

        ensure_dir(module_workspace_dir)

        if self.options.dry_run:
            ensure_dir(stdout_log.parent)
            stdout_log.write_text("", encoding="utf-8")
            stderr_log.write_text("", encoding="utf-8")
            subjobs = self._mark_subjobs(subjobs, status="dry_run")
        elif bool(module_cfg.get("cache", {}).get("enabled", True)) and cache_entry_exists(cache_dir):
            materialize_cached_output(cache_dir, module_output_dir)
            ensure_dir(stdout_log.parent)
            stdout_log.write_text("", encoding="utf-8")
            stderr_log.write_text("", encoding="utf-8")
            status = "cache_hit"
            cache_hit = True
            subjobs = self._mark_subjobs(subjobs, status="cache_hit")
        else:
            pikachu_resume_backups = self._backup_partial_pikachu_weights(subjobs)
            remove_path(module_output_dir)
            ensure_dir(module_output_dir)
            ensure_dir(stdout_log.parent)
            stdout_log.write_text("", encoding="utf-8")
            stderr_log.write_text("", encoding="utf-8")
            self._materialize_inputs(module_cfg, module_output_dir)
            self._restore_partial_pikachu_weights(pikachu_resume_backups)
            if module_cfg["runner"] == "noop":
                dump_json(module_output_dir / "noop.json", {"stage": stage, "status": "noop"})
            elif module_cfg["runner"] == "shell_command":
                env = self._build_env(
                    stage,
                    stage_context,
                    module_cfg,
                    extra_env=_build_metric_env(None, metrics_path, predictions_path),
                )
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
            elif module_cfg["runner"] == "command_group":
                subjobs = self._run_command_group(stage, stage_context, module_cfg, subjobs)
                peak_rss_kb = max((int(job.get("peak_rss_kb", 0)) for job in subjobs), default=0)
                mean_rss_kb = _weighted_mean(subjobs, value_key="mean_rss_kb", weight_key="sample_count")
                peak_cpu_percent = max((float(job.get("peak_cpu_percent", 0.0)) for job in subjobs), default=0.0)
                mean_cpu_percent = _weighted_mean(subjobs, value_key="mean_cpu_percent", weight_key="cpu_sample_count")
                peak_process_count = max((int(job.get("peak_process_count", 0)) for job in subjobs), default=0)
                sample_count = sum(int(job.get("sample_count", 0)) for job in subjobs)
                cpu_sample_count = sum(int(job.get("cpu_sample_count", 0)) for job in subjobs)
                subjob_statuses = {str(job.get("status", "")) for job in subjobs}
                if subjob_statuses and subjob_statuses.issubset({"unsupported"}):
                    status = "unsupported"
            else:
                raise ValueError(f"Unsupported module runner: {module_cfg['runner']}")

            if status != "failed" and bool(module_cfg.get("cache", {}).get("enabled", True)):
                store_cache_entry(
                    cache_dir,
                    module_output_dir,
                    {
                        "experiment_id": self.experiment_id,
                        "stage": stage,
                        "cache_key": cache_key,
                        "runner": module_cfg["runner"],
                        "artifacts": artifacts,
                    },
                )

        finished_at = _utc_now()
        wall_time = round(time.perf_counter() - start_t, 6)

        record = {
            "detector": self.experiment_id,
            "phase": stage,
            "job_id": stage,
            "dataset": None,
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
            "stage": stage,
            "runner": module_cfg["runner"],
            "cache_key": cache_key,
            "cache_dir": str(cache_dir),
            "cache_hit": cache_hit,
            "workspace_dir": str(module_workspace_dir),
            "output_dir": str(module_output_dir),
            "artifacts": artifacts,
            "subjobs": subjobs,
            "metrics_paths": [job["metrics_path"] for job in subjobs if job.get("metrics_path")],
            "predictions_paths": [job["predictions_path"] for job in subjobs if job.get("predictions_path")],
        }
        dump_json(self.metadata_root / f"{stage}.json", record)

        if status == "failed":
            raise RuntimeError(f"{self.experiment_id}:{stage} failed with exit code {exit_code}")
        return record

    def _build_env(
        self,
        stage: str,
        stage_context: dict[str, Any],
        module_cfg: dict[str, Any],
        extra_env: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        repo_src = str((Path(self.context["repo_root"]) / "src").resolve())
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        pythonpath_parts = [repo_src]
        if current_pythonpath:
            pythonpath_parts.append(current_pythonpath)

        global_env = expand_templates(self.config["_raw_environment"], stage_context)
        module_env = module_cfg.get("env", {})
        return merge_env(
            dict(os.environ),
            global_env,
            {
                "PYTHONPATH": os.pathsep.join(pythonpath_parts),
                "PYTHONHASHSEED": self.seed,
                "PYTHON": self.config["python_bin"],
                "GIDS_EVAL_SEED": self.seed,
                "GIDS_EXPERIMENT_ID": self.experiment_id,
                "GIDS_EXPERIMENT_ROOT": self.workspace_root,
                "GIDS_INPUT_ROOT": self.input_root,
                "GIDS_OUTPUT_ROOT": self.output_root,
                "GIDS_METADATA_ROOT": self.metadata_root,
                "GIDS_CACHE_ROOT": self.cache_root,
                "GIDS_MODULE_STAGE": stage,
                "GIDS_MODULE_WORKSPACE_DIR": stage_context["module_workspace_dir"],
                "GIDS_MODULE_OUTPUT_DIR": stage_context["module_output_dir"],
                "GIDS_STAGED_MANIFESTS_PATH": self.staged_manifests_path,
                "MIRAGE_INPUT_ROOT": self.input_root,
                "MIRAGE_OUTPUT_ROOT": self.output_root,
            },
            module_env,
            extra_env or {},
        )

    def _build_stage_context(
        self,
        stage: str,
        previous_records: dict[str, dict[str, Any]],
        module_workspace_dir: Path,
        module_output_dir: Path,
        cache_dir: Path,
    ) -> dict[str, Any]:
        context: dict[str, Any] = {
            **self.context,
            "staged_manifests_path": str(self.staged_manifests_path),
            "manifest_fingerprint": self.manifest_fingerprint,
            "module_stage": stage,
            "module_workspace_dir": str(module_workspace_dir),
            "module_output_dir": str(module_output_dir),
            "module_cache_dir": str(cache_dir),
        }
        for previous_stage, record in previous_records.items():
            context[f"{previous_stage}_status"] = record["status"]
            context[f"{previous_stage}_cache_key"] = record["cache_key"]
            context[f"{previous_stage}_workspace_dir"] = record["workspace_dir"]
            context[f"{previous_stage}_output_dir"] = record["output_dir"]
            for artifact_name, artifact_path in record["artifacts"].items():
                context[f"{previous_stage}_artifact_{artifact_name}"] = artifact_path
        if previous_records:
            upstream_stage = list(previous_records)[-1]
            context["upstream_stage"] = upstream_stage
            context["upstream_output_dir"] = previous_records[upstream_stage]["output_dir"]
        return context

    def _build_command_group_jobs(
        self,
        stage: str,
        module_cfg: dict[str, Any],
        stage_context: dict[str, Any],
        module_output_dir: Path,
    ) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for job in module_cfg.get("jobs", []):
            job_context = {
                **stage_context,
                "job_id": str(job["id"]),
                "job_dataset": str(job.get("dataset", "")),
            }
            resolved_job = expand_templates(job, job_context)
            self._assert_resolved(resolved_job, f"modules.{stage}.jobs.{resolved_job['id']}")
            metrics_relpath = resolved_job.get("metrics_relpath")
            metrics_path = str(module_output_dir / metrics_relpath) if metrics_relpath else None
            jobs.append(
                {
                    "job_id": str(resolved_job["id"]),
                    "dataset": resolved_job.get("dataset"),
                    "command": [str(part) for part in resolved_job["command"]],
                    "workdir": str(resolved_job.get("workdir", module_cfg.get("workdir", self.context["repo_root"]))),
                    "env": resolved_job.get("env", {}),
                    "metrics_path": metrics_path,
                    "predictions_path": derive_predictions_path(metrics_path),
                    "stdout_log": str(self.logs_root / f"{stage}.{resolved_job['id']}.stdout.log"),
                    "stderr_log": str(self.logs_root / f"{stage}.{resolved_job['id']}.stderr.log"),
                }
            )
        return jobs

    def _materialize_inputs(self, module_cfg: dict[str, Any], module_output_dir: Path) -> None:
        for rel_target, source_value in module_cfg.get("inputs", {}).items():
            source = Path(str(source_value))
            source_resolved = source.resolve()
            if not source_resolved.exists():
                raise ValueError(f"Module input source does not exist: {source_value}")
            target = module_output_dir / str(rel_target)
            ensure_dir(target.parent)
            remove_path(target)
            os.symlink(source_resolved, target, target_is_directory=source_resolved.is_dir())

    def _backup_partial_pikachu_weights(self, jobs: list[dict[str, Any]]) -> list[dict[str, str]]:
        backups: list[dict[str, str]] = []
        for job in jobs:
            command = [str(part) for part in job.get("command", [])]
            if not any(part.endswith("pikachu_run.py") for part in command):
                continue
            output_dir = _command_flag_value(command, "--output-dir")
            if not output_dir:
                continue
            metrics_path = job.get("metrics_path")
            if metrics_path and Path(str(metrics_path)).exists():
                continue
            weights_dir = Path(output_dir) / "_pikachu_runtime" / "weights"
            if not weights_dir.exists():
                continue
            backup_dir = self.metadata_root / "resume_backups" / str(job.get("job_id", "unknown")) / "weights"
            remove_path(backup_dir)
            ensure_dir(backup_dir.parent)
            shutil.copytree(weights_dir, backup_dir)
            backups.append({"backup_dir": str(backup_dir), "target_dir": str(weights_dir)})
        return backups

    def _restore_partial_pikachu_weights(self, backups: list[dict[str, str]]) -> None:
        for item in backups:
            backup_dir = Path(item["backup_dir"])
            target_dir = Path(item["target_dir"])
            if not backup_dir.exists():
                continue
            remove_path(target_dir)
            ensure_dir(target_dir.parent)
            shutil.copytree(backup_dir, target_dir)

    def _mark_subjobs(self, jobs: list[dict[str, Any]], status: str) -> list[dict[str, Any]]:
        marked: list[dict[str, Any]] = []
        for job in jobs:
            stdout_log = Path(job["stdout_log"])
            stderr_log = Path(job["stderr_log"])
            ensure_dir(stdout_log.parent)
            stdout_log.write_text("", encoding="utf-8")
            stderr_log.write_text("", encoding="utf-8")
            marked.append(
                {
                    **job,
                    "status": status,
                    "exit_code": 0,
                    "started_at": "",
                    "finished_at": "",
                    "wall_time_sec": 0.0,
                    "peak_rss_kb": 0,
                    "mean_rss_kb": 0.0,
                    "peak_cpu_percent": 0.0,
                    "mean_cpu_percent": 0.0,
                    "peak_process_count": 0,
                    "sample_count": 0,
                    "cpu_sample_count": 0,
                }
            )
        return marked

    def _run_command_group(
        self,
        stage: str,
        stage_context: dict[str, Any],
        module_cfg: dict[str, Any],
        jobs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for job in jobs:
            stdout_log = Path(job["stdout_log"])
            stderr_log = Path(job["stderr_log"])
            ensure_dir(stdout_log.parent)
            started_at = _utc_now()
            exit_code = 0
            status = "success"
            env = self._build_env(
                stage,
                stage_context,
                module_cfg,
                extra_env=_build_metric_env(job, job.get("metrics_path"), job.get("predictions_path")),
            )
            usage = run_command_logged(
                job["command"],
                cwd=job["workdir"],
                env=env,
                stdout_path=stdout_log,
                stderr_path=stderr_log,
                monitor_resources=False,
            )
            profile_usage = load_phase_profile_usage(derive_phase_profile_path(job.get("metrics_path")))
            resource_usage = profile_usage or usage
            exit_code = usage.exit_code
            if exit_code == UNSUPPORTED_EXIT_CODE:
                status = "unsupported"
            elif exit_code != 0:
                status = "failed"
            result = {
                **job,
                "status": status,
                "exit_code": exit_code,
                "started_at": started_at,
                "finished_at": _utc_now(),
                "wall_time_sec": usage.wall_time_sec,
                "peak_rss_kb": resource_usage.peak_rss_kb,
                "mean_rss_kb": resource_usage.mean_rss_kb,
                "peak_cpu_percent": resource_usage.peak_cpu_percent,
                "mean_cpu_percent": resource_usage.mean_cpu_percent,
                "peak_process_count": resource_usage.peak_process_count,
                "sample_count": resource_usage.sample_count,
                "cpu_sample_count": resource_usage.cpu_sample_count,
            }
            results.append(result)
            if status == "failed":
                raise RuntimeError(
                    f"{self.experiment_id}:{stage}:{job['job_id']} failed with exit code {exit_code}"
                )
        return results

    def _build_artifact_paths(self, module_cfg: dict[str, Any], module_output_dir: Path) -> dict[str, str]:
        artifacts: dict[str, str] = {}
        for name, relpath in module_cfg.get("artifacts", {}).items():
            artifacts[str(name)] = str(module_output_dir / str(relpath))
        return artifacts

    def _build_manifest_fingerprint(self) -> str:
        manifest_payloads: list[dict[str, Any]] = []
        context = {**self.context, "stage_root": "__CACHE_STAGE_ROOT__"}
        for manifest_path in self.config.get("manifests", []):
            manifest = load_manifest(manifest_path, context)
            artifacts = []
            for artifact in manifest["artifacts"]:
                artifacts.append(
                    {
                        "name": artifact.get("name"),
                        "kind": artifact.get("kind", "file"),
                        "source": artifact["source"],
                        "source_signature": build_path_signature(artifact["source"]),
                        "target": artifact["target"],
                    }
                )
            manifest_payloads.append(
                {
                    "manifest_path": manifest["_path"],
                    "dataset_id": manifest["dataset_id"],
                    "family": manifest.get("family"),
                    "artifacts": artifacts,
                }
            )
        return stable_hash(manifest_payloads)

    def _build_module_cache_key(
        self,
        stage: str,
        previous_records: dict[str, dict[str, Any]],
    ) -> str:
        previous_stage = self._previous_stage(stage)
        previous_cache_key = previous_records[previous_stage]["cache_key"] if previous_stage else None
        payload = {
            "schema_version": self.config["schema_version"],
            "experiment_id": self.experiment_id,
            "stage": stage,
            "seed": self.seed,
            "python_bin": self.config["python_bin"],
            "runtime": {
                "repo_root": self.context["repo_root"],
                "data_root": self.context["data_root"],
                "runtime_root": self.context["runtime_root"],
                "stage_mode": self.config.get("stage_mode", "symlink"),
            },
            "environment": self.config["_raw_environment"],
            "module": _strip_cache_settings(self.config["_raw_modules"][stage]),
            "previous_cache_key": previous_cache_key,
            "manifest_fingerprint": self.manifest_fingerprint if previous_stage is None else None,
        }
        return stable_hash(payload)

    def _previous_stage(self, stage: str) -> str | None:
        index = MODULE_STAGES.index(stage)
        if index == 0:
            return None
        return MODULE_STAGES[index - 1]

    def _selected_stages(self) -> list[str]:
        if not self.options.to_stage:
            return list(MODULE_STAGES)
        end_index = MODULE_STAGES.index(self.options.to_stage)
        return list(MODULE_STAGES[: end_index + 1])

    def _resolved_config_for_dump(self) -> dict[str, Any]:
        payload = dict(self.config)
        payload.pop("_context", None)
        payload.pop("_raw_modules", None)
        payload.pop("_raw_environment", None)
        return payload

    def _assert_resolved(self, value: Any, path: str) -> None:
        if isinstance(value, str):
            if "${" in value:
                raise ValueError(f"Unresolved template in {path}: {value}")
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                self._assert_resolved(item, f"{path}[{index}]")
            return
        if isinstance(value, dict):
            for key, item in value.items():
                self._assert_resolved(item, f"{path}.{key}")

def _strip_cache_settings(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_cache_settings(item)
            for key, item in value.items()
            if key != "cache"
        }
    if isinstance(value, list):
        return [_strip_cache_settings(item) for item in value]
    return value

def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _command_flag_value(command: list[str], flag: str) -> str | None:
    for index, part in enumerate(command):
        if part == flag and index + 1 < len(command):
            return str(command[index + 1])
    return None

def _build_metric_env(
    job: dict[str, Any] | None,
    metrics_path: str | None,
    predictions_path: str | None,
) -> dict[str, str]:
    env: dict[str, str] = {}
    if job:
        for key, value in job.get("env", {}).items():
            env[str(key)] = str(value)
        if job.get("job_id") is not None:
            env["GIDS_JOB_ID"] = str(job["job_id"])
        if job.get("dataset") is not None:
            env["GIDS_JOB_DATASET"] = str(job["dataset"])
    if metrics_path:
        env["GIDS_METRICS_PATH"] = str(metrics_path)
        profile_path = derive_phase_profile_path(metrics_path)
        if profile_path:
            env["GIDS_PHASE_PROFILE_PATH"] = str(profile_path)
    if predictions_path:
        env["GIDS_PREDICTIONS_PATH"] = str(predictions_path)
    return env

def _weighted_mean(
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    weight_key: str,
) -> float:
    weighted_total = 0.0
    total_weight = 0
    for row in rows:
        weight = int(row.get(weight_key, 0) or 0)
        if weight <= 0:
            continue
        weighted_total += float(row.get(value_key, 0.0) or 0.0) * weight
        total_weight += weight
    if total_weight == 0:
        return 0.0
    return round(weighted_total / total_weight, 3)
