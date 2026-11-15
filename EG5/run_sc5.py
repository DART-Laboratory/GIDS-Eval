from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gids_eval.paths import get_paths

_PATHS = get_paths()
from eg5lib import (
    detector_list,
    ensure_eg5_dirs,
    generate_config,
    iter_jobs,
    parse_csv_list,
    parse_seed_list,
    run_job,
    scale_list,
    summarize_jobs,
    write_matrix_report,
    write_progress_report,
    write_status_report,
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_sc5")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("matrix", "Generate the EG5 matrix and config files"),
        ("status", "Refresh the EG5 job status CSV"),
        ("progress", "Refresh and print EG5 progress files"),
        ("summarize", "Aggregate EG5 CSV reports"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        _add_filters(sub)

    run = subparsers.add_parser("run", help="Run the EG5 matrix with resume support")
    _add_filters(run)
    run.add_argument("--dry-run", action="store_true", help="Resolve configs without running detector code")
    run.add_argument("--force", action="store_true", help="Re-run jobs even if they already completed")
    run.add_argument("--max-jobs", type=int, default=None, help="Stop after N jobs")
    run.add_argument("--stop-on-error", action="store_true", help="Exit immediately on the first failed job")
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
        "--scales",
        default=",".join(str(item) for item in scale_list()),
        help="Comma-separated scale list in percent",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seed list",
    )

def _resolve_jobs(args: argparse.Namespace):
    detectors = parse_csv_list(args.detectors, kind="detector")
    datasets = parse_csv_list(args.datasets, kind="dataset")
    scales = parse_csv_list(args.scales, kind="scale")
    seeds = parse_seed_list(args.seeds)
    return iter_jobs(detectors=detectors, datasets=datasets, scales=scales, seeds=seeds)

def _global_jobs():
    return iter_jobs()

def main() -> int:
    args = build_parser().parse_args()
    ensure_eg5_dirs()
    jobs = _resolve_jobs(args)
    global_jobs = _global_jobs()

    write_matrix_report(global_jobs)

    def sync_state(
        *,
        attempted_jobs: int | None = None,
        failures: int | None = None,
        include_summary: bool = False,
    ) -> None:
        write_status_report(global_jobs)
        if include_summary:
            summarize_jobs(global_jobs)
        write_progress_report(global_jobs, attempted_jobs=attempted_jobs, failed_jobs=failures)

    if args.command == "matrix":
        for job in jobs:
            generate_config(job)
        print(f"generated_jobs={len(jobs)}")
        return 0

    if args.command == "status":
        sync_state()
        print(_PATHS.sc_data_root("eg5") / "orchestrator/job_status.csv")
        print(_PATHS.sc_data_root("eg5") / "orchestrator/progress.md")
        return 0

    if args.command == "progress":
        sync_state()
        print(_PATHS.sc_data_root("eg5") / "orchestrator/progress.json")
        print(_PATHS.sc_data_root("eg5") / "orchestrator/progress.md")
        return 0

    if args.command == "summarize":
        sync_state(include_summary=True)
        print(_PATHS.sc_data_root("eg5") / "reports/raw_results.csv")
        print(_PATHS.sc_data_root("eg5") / "reports/summary_by_scale.csv")
        print(_PATHS.sc_data_root("eg5") / "reports/fit_summary.csv")
        print(_PATHS.sc_data_root("eg5") / "reports/conclusion.md")
        return 0

    if args.command == "run":
        failures = 0
        attempted = 0
        for job in jobs:
            result = run_job(job, dry_run=bool(args.dry_run), force=bool(args.force))
            if result != "skipped":
                attempted += 1
                sync_state(attempted_jobs=attempted, failures=failures)
            if result == "failed":
                failures += 1
                sync_state(attempted_jobs=attempted, failures=failures)
                if args.stop_on_error:
                    return 1
            if args.max_jobs is not None and attempted >= args.max_jobs:
                break
        sync_state(attempted_jobs=attempted, failures=failures)
        return 1 if failures else 0

    raise SystemExit(f"Unsupported command: {args.command}")

if __name__ == "__main__":
    sys.exit(main())
