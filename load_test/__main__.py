from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

from load_test import render_report, run_load


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run concurrent simulated voice calls"
    )
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--scenario", choices=("default", "barge_in"), default="default")
    parser.add_argument("--server-url", default="ws://127.0.0.1:8000")
    args = parser.parse_args(argv)
    try:
        report = asyncio.run(
            run_load(
                concurrency=args.concurrency,
                scenario=args.scenario,
                server_url=args.server_url,
            )
        )
    except ValueError as exc:
        parser.error(str(exc))
    render_report(report)
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
