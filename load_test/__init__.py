from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from app.metrics import TurnOutcome, nearest_rank
from client import CallResult, run_call


CallRunner = Callable[[str, str, str], Awaitable[CallResult]]


@dataclass(frozen=True, slots=True)
class PercentileSummary:
    sample_count: int
    p50: float | None
    p95: float | None


@dataclass(frozen=True, slots=True)
class LoadReport:
    requested: int
    succeeded: int
    failed: int
    interrupted: int
    ttft: PercentileSummary
    ttfa: PercentileSummary
    total_latency: PercentileSummary
    errors: tuple[str, ...]


async def run_load(
    concurrency: int = 20,
    scenario: str = "default",
    server_url: str = "ws://127.0.0.1:8000",
    call_runner: CallRunner = run_call,
) -> LoadReport:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    batch_id = uuid.uuid4().hex[:8]
    results = await asyncio.gather(
        *(
            call_runner(server_url, scenario, f"load-{batch_id}-{index}")
            for index in range(concurrency)
        ),
        return_exceptions=True,
    )

    successes: list[CallResult] = []
    errors: list[str] = []
    for result in results:
        if isinstance(result, BaseException):
            errors.append(_safe_error(result))
            continue
        has_failed_turn = any(
            metrics.outcome is TurnOutcome.FAILED
            for metrics in result.turn_metrics
        )
        if result.errors or result.outcome == "failed" or has_failed_turn:
            errors.extend(result.errors or ("call reported failed outcome",))
            continue
        successes.append(result)

    completed_turns = [
        metrics
        for result in successes
        for metrics in result.turn_metrics
        if metrics.outcome is TurnOutcome.COMPLETED
    ]
    return LoadReport(
        requested=concurrency,
        succeeded=len(successes),
        failed=concurrency - len(successes),
        interrupted=sum(result.interrupted_turns for result in successes),
        ttft=_summary(
            [metrics.ttft_ms for metrics in completed_turns if metrics.ttft_ms is not None]
        ),
        ttfa=_summary(
            [metrics.ttfa_ms for metrics in completed_turns if metrics.ttfa_ms is not None]
        ),
        total_latency=_summary(
            [
                metrics.total_latency_ms
                for metrics in completed_turns
                if metrics.total_latency_ms is not None
            ]
        ),
        errors=tuple(errors),
    )


def _summary(values: Sequence[float]) -> PercentileSummary:
    return PercentileSummary(
        sample_count=len(values),
        p50=nearest_rank(values, 0.50),
        p95=nearest_rank(values, 0.95),
    )


def _safe_error(error: BaseException) -> str:
    message = str(error).strip()
    return message or type(error).__name__


def render_report(report: LoadReport) -> None:
    print(
        f"calls: requested={report.requested} succeeded={report.succeeded} "
        f"failed={report.failed} interrupted={report.interrupted}"
    )
    for name, summary in (
        ("TTFT", report.ttft),
        ("TTFA", report.ttfa),
        ("total", report.total_latency),
    ):
        print(
            f"{name}: samples={summary.sample_count} "
            f"p50={_metric(summary.p50)} p95={_metric(summary.p95)}"
        )
    for error in report.errors:
        print(f"error: {error}")


def _metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}ms"
