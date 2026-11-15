from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gids_eval.paths import get_paths

_PATHS = get_paths()
from eg3lib import (
    CALIBRATION_MODES,
    DATASETS,
    ensure_eg3_dirs,
    generate_config,
    iter_jobs,
    parse_csv_list,
    parse_seed_list,
    run_job,
    same_job_set,
    summarize_jobs,
    write_matrix_report,
    write_progress_report,
    write_status_report,
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_sc3")
    sub = parser.add_subparsers(dest="command", required=True)

    matrix = sub.add_parser("matrix", help="Generate the EG3 job matrix and configs")
    _add_filters(matrix)

    run = sub.add_parser("run", help="Run EG3 jobs with resume support")
    _add_filters(run)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--force", action="store_true")
    run.add_argument("--max-jobs", type=int, default=None)
    run.add_argument("--stop-on-error", action="store_true")

    status = sub.add_parser("status", help="Refresh the EG3 job status CSV")
    _add_filters(status)

    progress = sub.add_parser("progress", help="Print EG3 progress")
    _add_filters(progress)

    summarize = sub.add_parser("summarize", help="Aggregate EG3 results")
    _add_filters(summarize)

    return parser

def _add_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--datasets", default=",".join(DATASETS),
        help="Comma-separated dataset list",
    )
    parser.add_argument(
        "--calibrations", default=",".join(CALIBRATION_MODES),
        help="Comma-separated calibration modes",
    )
    parser.add_argument("--seeds", default=None, help="Comma-separated seed list")

def _resolve_jobs(args: argparse.Namespace):
    datasets = parse_csv_list(args.datasets)
    calibrations = parse_csv_list(args.calibrations)
    seeds = parse_seed_list(args.seeds)
    return iter_jobs(datasets=datasets, calibrations=calibrations, seeds=seeds)

def _global_jobs():
    return iter_jobs()

def main() -> int:
    args = build_parser().parse_args()
    ensure_eg3_dirs()
    jobs = _resolve_jobs(args)
    global_jobs = _global_jobs()

    write_matrix_report(global_jobs)

    def sync_state(*, attempted_jobs: int | None = None, failures: int | None = None) -> None:
        write_status_report(global_jobs)
        summarize_jobs(global_jobs)
        write_progress_report(
            global_jobs,
            scope_name="global",
            attempted_jobs=attempted_jobs,
            failed_jobs=failures,
        )

    if args.command == "matrix":
        for job in jobs:
            generate_config(job)
        print(f"generated_jobs={len(jobs)}")
        return 0

    if args.command == "status":
        sync_state()
        print(_PATHS.sc_data_root("eg3") / "orchestrator/job_status.csv")
        return 0

    if args.command == "progress":
        sync_state()
        print(_PATHS.sc_data_root("eg3") / "orchestrator/progress.md")
        return 0

    if args.command == "summarize":
        sync_state()
        print(_PATHS.sc_data_root("eg3") / "reports/raw_results.csv")
        print(_PATHS.sc_data_root("eg3") / "reports/delta_table.csv")
        print(_PATHS.sc_data_root("eg3") / "reports/evidence_table.csv")
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
            if result == "failed" and args.stop_on_error:
                return 1
            if args.max_jobs is not None and attempted >= args.max_jobs:
                break
        sync_state(attempted_jobs=attempted, failures=failures)
        return 1 if failures else 0

    raise SystemExit(f"Unsupported command: {args.command}")

if __name__ == "__main__":
    sys.exit(main())
