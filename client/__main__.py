from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence

from client import CallResult, render_event, render_summary, run_call


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a scripted simulated voice call")
    parser.add_argument("--server-url", default="ws://127.0.0.1:8000")
    parser.add_argument("--scenario", default="default")
    parser.add_argument("--session-id")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.scenario not in {"default", "barge_in"}:
        print(f"error: unknown scenario: {args.scenario}", file=sys.stderr)
        return 2

    sink = lambda event: render_event(event, verbose=args.verbose)
    try:
        result = asyncio.run(
            run_call(
                args.server_url,
                args.scenario,
                args.session_id,
                on_event=sink,
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.verbose:
        print(json.dumps(_result_json(result), sort_keys=True))
    else:
        render_summary(result)
    return 0 if not result.errors else 1


def _result_json(result: CallResult) -> dict[str, object]:
    return {
        "type": "call.summary",
        "session_id": result.session_id,
        "outcome": result.outcome,
        "interrupted_turns": result.interrupted_turns,
        "errors": list(result.errors),
        "turns": [metrics.model_dump(mode="json") for metrics in result.turn_metrics],
    }


if __name__ == "__main__":
    raise SystemExit(main())
