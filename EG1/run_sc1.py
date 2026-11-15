from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gids_eval.paths import get_paths

_PATHS = get_paths()
from eg1lib import (
    detector_env_status,
    detector_list,
    ensure_eg1_dirs,
    iter_jobs,
    parse_csv_list,
    parse_seed_list,
    same_job_set,
    summarize_jobs,
    write_matrix_report,
    write_progress_report,
    write_status_report,
    run_job,
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_sc1")
    subparsers = parser.add_subparsers(dest="command", required=True)

    matrix = subparsers.add_parser("matrix", help="Generate the EG1 matrix and config files")
    _add_filters(matrix)

    run = subparsers.add_parser("run", help="Run the EG1 matrix with resume support")
    _add_filters(run)
    run.add_argument("--dry-run", action="store_true", help="Resolve configs without running detector code")
    run.add_argument("--force", action="store_true", help="Re-run jobs even if they already completed")
    run.add_argument("--max-jobs", type=int, default=None, help="Stop after N jobs")
    run.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Exit immediately on the first failed job",
    )

    status = subparsers.add_parser("status", help="Refresh the EG1 job status CSV")
    _add_filters(status)

    progress = subparsers.add_parser("progress", help="Refresh and print EG1 progress files")
    _add_filters(progress)

    summarize = subparsers.add_parser("summarize", help="Aggregate raw and summary CSV reports")
    _add_filters(summarize)

    env_status = subparsers.add_parser("env-status", help="Check detector environment availability and imports")
    _add_filters(env_status)

    return parser

def _add_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--detectors",
        default=",".join(detector_list()),
        help="Comma-separated detector list",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset list",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seed list",
    )

def _resolve_jobs(args: argparse.Namespace):
    detectors = parse_csv_list(args.detectors)
    datasets = parse_csv_list(args.datasets, canonicalize=True)
    seeds = parse_seed_list(args.seeds)
    return iter_jobs(detectors=detectors, datasets=datasets, seeds=seeds)

def _global_jobs():
    return iter_jobs()

SELECTION_MATRIX_PATH = _PATHS.sc_data_root("eg1") / "orchestrator/job_matrix_selection.csv"
SELECTION_STATUS_PATH = _PATHS.sc_data_root("eg1") / "orchestrator/job_status_selection.csv"
SELECTION_PROGRESS_JSON_PATH = _PATHS.sc_data_root("eg1") / "orchestrator/progress_selection.json"
SELECTION_PROGRESS_MD_PATH = _PATHS.sc_data_root("eg1") / "orchestrator/progress_selection.md"
SELECTION_REPORT_ROOT = _PATHS.sc_data_root("eg1") / "reports/selection"

def main() -> int:
    args = build_parser().parse_args()
    ensure_eg1_dirs()
    jobs = _resolve_jobs(args)
    global_jobs = _global_jobs()
    filtered = not same_job_set(jobs, global_jobs)

    write_matrix_report(global_jobs)
    if filtered:
        write_matrix_report(jobs, path=SELECTION_MATRIX_PATH)

    def sync_state(*, attempted_jobs: int | None = None, failures: int | None = None) -> None:
        write_status_report(global_jobs)
        summarize_jobs(global_jobs)
        write_progress_report(
            global_jobs,
            scope_name="global",
            attempted_jobs=attempted_jobs,
            failed_jobs=failures,
        )
        if filtered:
            write_status_report(jobs, path=SELECTION_STATUS_PATH)
            summarize_jobs(jobs, output_root=SELECTION_REPORT_ROOT)
            write_progress_report(
                jobs,
                json_path=SELECTION_PROGRESS_JSON_PATH,
                md_path=SELECTION_PROGRESS_MD_PATH,
                scope_name="selection",
                attempted_jobs=attempted_jobs,
                failed_jobs=failures,
            )

    if args.command == "matrix":
        for job in jobs:
            from eg1lib import generate_config

            generate_config(job)
        print(f"generated_jobs={len(jobs)}")
        return 0

    if args.command == "status":
        sync_state()
        print(_PATHS.sc_data_root("eg1") / "orchestrator/job_status.csv")
        print(_PATHS.sc_data_root("eg1") / "orchestrator/progress.md")
        if filtered:
            print(SELECTION_STATUS_PATH)
            print(SELECTION_PROGRESS_MD_PATH)
        return 0

    if args.command == "progress":
        sync_state()
        print(_PATHS.sc_data_root("eg1") / "orchestrator/progress.json")
        print(_PATHS.sc_data_root("eg1") / "orchestrator/progress.md")
        if filtered:
            print(SELECTION_PROGRESS_JSON_PATH)
            print(SELECTION_PROGRESS_MD_PATH)
        return 0

    if args.command == "summarize":
        sync_state()
        print(_PATHS.sc_data_root("eg1") / "reports/raw_results.csv")
        print(_PATHS.sc_data_root("eg1") / "reports/summary_by_window.csv")
        print(_PATHS.sc_data_root("eg1") / "reports/metric_tables/index.csv")
        print(_PATHS.sc_data_root("eg1") / "orchestrator/progress.md")
        if filtered:
            print(SELECTION_STATUS_PATH)
            print(SELECTION_REPORT_ROOT / "raw_results.csv")
            print(SELECTION_REPORT_ROOT / "summary_by_window.csv")
            print(SELECTION_REPORT_ROOT / "metric_tables" / "index.csv")
            print(SELECTION_PROGRESS_MD_PATH)
        return 0

    if args.command == "env-status":
        path, rows = detector_env_status(parse_csv_list(args.detectors))
        print(path)
        for row in rows:
            print(
                ",".join(
                    [
                        row["detector"],
                        row["imports_ok"],
                        row["missing_imports"],
                        row["note"],
                    ]
                )
            )
        return 0

    if args.command == "run":
        failures = 0
        attempted = 0
        for job in jobs:
            result = run_job(job, dry_run=bool(args.dry_run), force=bool(args.force))
            if result != "skipped":
                attempted += 1
            if result == "failed":
                failures += 1
            if result != "skipped":
                sync_state(attempted_jobs=attempted, failures=failures)
            if result == "failed":
                if args.stop_on_error:
                    return 1
            if args.max_jobs is not None and attempted >= args.max_jobs:
                break
        sync_state(attempted_jobs=attempted, failures=failures)
        return 1 if failures else 0

    raise SystemExit(f"Unsupported command: {args.command}")

if __name__ == "__main__":
    sys.exit(main())
