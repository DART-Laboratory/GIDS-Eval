from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gids_eval.paths import get_paths

_PATHS = get_paths()
from eg8lib import (
    ensure_eg8_dirs,
    iter_jobs,
    parse_csv_list,
    run_job,
    summarize_jobs,
    write_matrix_report,
    write_progress_report,
    write_status_report,
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_sc8")
    subparsers = parser.add_subparsers(dest="command", required=True)

    matrix = subparsers.add_parser("matrix", help="Generate the EG8 matrix report")
    _add_filters(matrix)

    status = subparsers.add_parser("status", help="Refresh the EG8 job status CSV")
    _add_filters(status)

    progress = subparsers.add_parser("progress", help="Refresh and print EG8 progress files")
    _add_filters(progress)

    summarize = subparsers.add_parser("summarize", help="Aggregate aligned/native EG8 reports")
    _add_filters(summarize)

    run = subparsers.add_parser("run", help="Run the EG8 aligned matrix with resume support")
    _add_filters(run)
    run.add_argument("--force", action="store_true", help="Re-run jobs even if they already completed")
    run.add_argument("--max-jobs", type=int, default=None, help="Stop after N attempted jobs")
    run.add_argument("--stop-on-error", action="store_true", help="Exit immediately on the first failed job")

    return parser

def _add_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--detectors",
        default=None,
        help="Comma-separated detector list",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset list",
    )

def _resolve_jobs(args: argparse.Namespace):
    detectors = parse_csv_list(args.detectors)
    datasets = parse_csv_list(args.datasets, canonicalize=True)
    return iter_jobs(detectors=detectors, datasets=datasets)

def _global_jobs():
    return iter_jobs()

def main() -> int:
    args = build_parser().parse_args()
    ensure_eg8_dirs()
    jobs = _resolve_jobs(args)
    global_jobs = _global_jobs()

    def sync_state(*, attempted_jobs: int | None = None, failures: int | None = None) -> None:
        del attempted_jobs, failures
        write_matrix_report(global_jobs)
        write_status_report(global_jobs)
        summarize_jobs(global_jobs)
        write_progress_report(global_jobs)

    if args.command == "matrix":
        write_matrix_report(global_jobs)
        print(_PATHS.sc_data_root("eg8") / "orchestrator/job_matrix.csv")
        return 0

    if args.command == "status":
        sync_state()
        print(_PATHS.sc_data_root("eg8") / "orchestrator/job_status.csv")
        print(_PATHS.sc_data_root("eg8") / "orchestrator/progress.md")
        return 0

    if args.command == "progress":
        sync_state()
        print(_PATHS.sc_data_root("eg8") / "orchestrator/progress.json")
        print(_PATHS.sc_data_root("eg8") / "orchestrator/progress.md")
        return 0

    if args.command == "summarize":
        sync_state()
        print(_PATHS.sc_data_root("eg8") / "reports/raw_results.csv")
        print(_PATHS.sc_data_root("eg8") / "reports/delta_native_vs_aligned.csv")
        print(_PATHS.sc_data_root("eg8") / "reports/summary_by_detector.csv")
        print(_PATHS.sc_data_root("eg8") / "reports/summary_by_dataset.csv")
        print(_PATHS.sc_data_root("eg8") / "reports/evidence_table.csv")
        print(_PATHS.sc_data_root("eg8") / "reports/conclusion.md")
        return 0

    if args.command == "run":
        failures = 0
        attempted = 0
        sync_state(attempted_jobs=attempted, failures=failures)
        for job in jobs:
            print(f"START {job.experiment_id}")
            result = run_job(job, force=bool(args.force))
            if result != "skipped":
                attempted += 1
            if result == "failed":
                failures += 1
                print(f"FAIL {job.experiment_id}")
            elif result == "success":
                print(f"OK {job.experiment_id}")
            else:
                print(f"SKIP {job.experiment_id}")
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
