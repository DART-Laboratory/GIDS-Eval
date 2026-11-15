from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gids_eval.paths import get_paths

_PATHS = get_paths()
from eg7lib import (
    DATASETS,
    FEATURIZATION_OPTIONS,
    GRAPH_ENCODERS,
    LOSS_OPTIONS,
    TEMPORAL_ENCODERS,
    ensure_eg7_dirs,
    generate_config,
    iter_jobs,
    parse_csv_list,
    run_job,
    same_job_set,
    summarize_jobs,
    write_matrix_report,
    write_progress_report,
    write_status_report,
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_sc7")
    sub = parser.add_subparsers(dest="command", required=True)

    matrix = sub.add_parser("matrix", help="Generate EG7 job configs")
    _add_filters(matrix)

    run = sub.add_parser("run", help="Run EG7 jobs with resume support")
    _add_filters(run)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--force", action="store_true")
    run.add_argument("--max-jobs", type=int, default=None)
    run.add_argument("--stop-on-error", action="store_true")

    status = sub.add_parser("status", help="Refresh EG7 status")
    _add_filters(status)

    progress = sub.add_parser("progress", help="Refresh EG7 progress")
    _add_filters(progress)

    summarize = sub.add_parser("summarize", help="Aggregate EG7 results")
    _add_filters(summarize)
    return parser

def _add_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--featurizations", default=",".join(FEATURIZATION_OPTIONS))
    parser.add_argument("--graph-encoders", default=",".join(GRAPH_ENCODERS))
    parser.add_argument("--temporal-encoders", default=",".join(TEMPORAL_ENCODERS))
    parser.add_argument("--losses", default=",".join(LOSS_OPTIONS))

def _resolve_jobs(args: argparse.Namespace):
    return iter_jobs(
        datasets=parse_csv_list(args.datasets),
        featurizations=parse_csv_list(args.featurizations),
        graph_encoders=parse_csv_list(args.graph_encoders),
        temporal_encoders=parse_csv_list(args.temporal_encoders),
        losses=parse_csv_list(args.losses),
    )

def _global_jobs():
    return iter_jobs()

def main() -> int:
    args = build_parser().parse_args()
    ensure_eg7_dirs()
    jobs = _resolve_jobs(args)
    global_jobs = _global_jobs()

    write_matrix_report(global_jobs)

    def sync_state(*, attempted_jobs: int | None = None, failures: int | None = None) -> None:
        write_status_report(global_jobs)
        summarize_jobs(global_jobs)
        write_progress_report(global_jobs, attempted_jobs=attempted_jobs, failed_jobs=failures)

    if args.command == "matrix":
        for job in jobs:
            generate_config(job)
        print(f"generated_jobs={len(jobs)}")
        return 0

    if args.command == "status":
        sync_state()
        print(_PATHS.sc_data_root("eg7") / "orchestrator/job_status.csv")
        return 0

    if args.command == "progress":
        sync_state()
        print(_PATHS.sc_data_root("eg7") / "orchestrator/progress.md")
        return 0

    if args.command == "summarize":
        sync_state()
        print(_PATHS.sc_data_root("eg7") / "reports/raw_results.csv")
        print(_PATHS.sc_data_root("eg7") / "reports/summary_by_option.csv")
        print(_PATHS.sc_data_root("eg7") / "reports/leaderboard.csv")
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
