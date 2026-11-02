from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from app.scenario import load_scenario, load_scenario_catalog
from load_test import render_report, run_load


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run concurrent simulated voice calls"
    )
    parser.add_argument("--concurrency", type=int, default=20)
    scenario_source = parser.add_mutually_exclusive_group()
    scenario_source.add_argument(
        "--scenario-dir",
        type=Path,
        help="directory containing validated interview scenarios",
    )
    scenario_source.add_argument(
        "--scenario-file",
        type=Path,
        help="single validated interview scenario JSON",
    )
    parser.add_argument(
        "--default-scenario-id",
        default="ecommerce",
        help="catalog ID that must exist (matches server catalog validation)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--server-url", default="ws://127.0.0.1:8000")
    args = parser.parse_args(argv)
    try:
        if args.scenario_file is not None:
            scenarios = (load_scenario(args.scenario_file),)
        else:
            scenarios = load_scenario_catalog(
                args.scenario_dir or Path("scenarios"),
                default_id=args.default_scenario_id,
            ).scenarios
        report = asyncio.run(
            run_load(
                concurrency=args.concurrency,
                scenarios=scenarios,
                server_url=args.server_url,
                seed=args.seed,
            )
        )
    except ValueError as exc:
        parser.error(str(exc))
    render_report(report)
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
