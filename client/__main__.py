from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from app.scenario import load_scenario
from client import CallResult, ReadableTraceRenderer, render_event, run_call


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a scripted simulated voice call")
    parser.add_argument("--server-url", default="ws://127.0.0.1:8000")
    parser.add_argument(
        "--scenario-file",
        type=Path,
        default=Path("scenarios/ecommerce.json"),
    )
    parser.add_argument("--session-id")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    try:
        scenario = load_scenario(args.scenario_file)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    renderer = None if args.verbose else ReadableTraceRenderer(
        scenario_id=scenario.id,
        bot_name=scenario.bot.name,
    )
    sink = (
        (lambda event: render_event(event, verbose=True))
        if args.verbose
        else renderer.render_event
    )
    try:
        result = asyncio.run(
            run_call(
                args.server_url,
                scenario,
                args.session_id,
                on_event=sink,
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        if renderer is not None:
            renderer.close_open_line()
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.verbose:
        print(json.dumps(_result_json(result), sort_keys=True))
    else:
        renderer.render_summary(result)
    return 0 if not result.errors else 1


def _result_json(result: CallResult) -> dict[str, object]:
    return {
        "type": "call.summary",
        "session_id": result.session_id,
        "outcome": result.outcome,
        "interrupted_turns": result.interrupted_turns,
        "errors": list(result.errors),
        "turns": [metrics.model_dump(mode="json") for metrics in result.turn_metrics],
        "conversation": [
            {
                "turn_id": turn.turn_id,
                "caller_transcript": turn.caller_transcript,
                "generated_text": turn.generated_text,
                "spoken_chunks": list(turn.spoken_chunks),
                "outcome": turn.outcome,
                "reason": turn.reason,
                "metrics": turn.metrics,
            }
            for turn in result.conversation
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
