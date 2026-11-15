from __future__ import annotations

import argparse
import json

from gids_eval.runner import create_runner
from gids_eval.runtime import RunOptions
from gids_eval.schema import MODULE_STAGES

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gids-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one or more detector configs")
    run_parser.add_argument("configs", nargs="+", help="Detector config paths")
    run_parser.add_argument("--dry-run", action="store_true", help="Resolve configs without executing commands")
    run_parser.add_argument("--clean", action="store_true", help="Clear the detector workspace before running")
    run_parser.add_argument(
        "--to-stage",
        choices=MODULE_STAGES,
        help="For experiment configs, stop after completing the selected stage",
    )

    validate_parser = subparsers.add_parser("validate", help="Validate one or more detector configs")
    validate_parser.add_argument("configs", nargs="+", help="Detector config paths")

    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "validate":
        results = []
        for config in args.configs:
            runner = create_runner(config)
            results.append(runner.validate())
        print(json.dumps(results, indent=2, sort_keys=True))
        return

    if args.command == "run":
        options = RunOptions(
            dry_run=bool(args.dry_run),
            clean=bool(args.clean),
            to_stage=args.to_stage,
        )
        summaries = []
        for config in args.configs:
            runner = create_runner(config, options=options)
            summaries.append(runner.run())
        print(json.dumps(summaries, indent=2, sort_keys=True))
        return

    raise SystemExit(f"Unsupported command: {args.command}")
