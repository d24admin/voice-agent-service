from __future__ import annotations

import asyncio

import pytest

from app.metrics import TurnMetrics, TurnOutcome
from client import CallResult
from load_test import PercentileSummary, render_report, run_load


def completed_metrics(turn_id: int, value: float) -> TurnMetrics:
    metrics = TurnMetrics(turn_id=turn_id, baseline_ns=0)
    metrics.ttft_ms = value
    metrics.ttfa_ms = value + 100
    metrics.total_latency_ms = value + 200
    metrics.outcome = TurnOutcome.COMPLETED
    return metrics


def failed_metrics(turn_id: int) -> TurnMetrics:
    metrics = TurnMetrics(turn_id=turn_id, baseline_ns=0)
    metrics.finalize(TurnOutcome.FAILED, "provider_error", 1_000_000)
    return metrics


@pytest.mark.asyncio
async def test_default_twenty_calls_are_concurrent_unique_and_failure_isolated() -> None:
    active = 0
    peak = 0
    session_ids: set[str] = set()
    calls = 0
    lock = asyncio.Lock()

    async def fake_run_call(
        server_url: str,
        scenario: str,
        session_id: str,
    ) -> CallResult:
        nonlocal active, peak, calls
        assert server_url == "ws://127.0.0.1:8000"
        assert scenario == "default"
        async with lock:
            calls += 1
            number = calls
            active += 1
            peak = max(peak, active)
            session_ids.add(session_id)
        await asyncio.sleep(0.01)
        async with lock:
            active -= 1
        if number == 7:
            raise OSError("one simulated call failed")
        return CallResult(
            session_id=session_id,
            outcome="completed",
            turn_metrics=(completed_metrics(1, float(number)),),
            interrupted_turns=number % 2,
            errors=(),
        )

    report = await run_load(call_runner=fake_run_call)

    assert report.requested == 20
    assert report.succeeded == 19
    assert report.failed == 1
    assert len(session_ids) == 20
    assert peak > 1
    assert len(report.errors) == 1
    assert "one simulated call failed" in report.errors[0]
    assert report.ttft == PercentileSummary(sample_count=19, p50=11.0, p95=20.0)
    assert report.ttfa == PercentileSummary(sample_count=19, p50=111.0, p95=120.0)
    assert report.total_latency == PercentileSummary(
        sample_count=19,
        p50=211.0,
        p95=220.0,
    )


@pytest.mark.asyncio
async def test_result_errors_count_as_failure_without_cancelling_siblings() -> None:
    async def fake_run_call(
        server_url: str,
        scenario: str,
        session_id: str,
    ) -> CallResult:
        del server_url, scenario
        failed = session_id.endswith("-1")
        return CallResult(
            session_id=session_id,
            outcome="failed" if failed else "completed",
            turn_metrics=() if failed else (completed_metrics(1, 5),),
            interrupted_turns=0,
            errors=("server rejected call",) if failed else (),
        )

    report = await run_load(concurrency=3, call_runner=fake_run_call)

    assert report.succeeded == 2
    assert report.failed == 1
    assert report.errors == ("server rejected call",)


@pytest.mark.asyncio
async def test_failed_turn_metrics_cannot_be_hidden_by_completed_call_outcome() -> None:
    async def fake_run_call(
        server_url: str,
        scenario: str,
        session_id: str,
    ) -> CallResult:
        del server_url, scenario
        return CallResult(
            session_id=session_id,
            outcome="completed",
            turn_metrics=(failed_metrics(1), completed_metrics(2, 5)),
            interrupted_turns=0,
            errors=(),
        )

    report = await run_load(concurrency=1, call_runner=fake_run_call)

    assert report.succeeded == 0
    assert report.failed == 1
    assert report.errors == ("call reported failed outcome",)


@pytest.mark.asyncio
async def test_empty_latency_samples_render_null_and_na(capsys: object) -> None:
    async def empty_call(
        server_url: str,
        scenario: str,
        session_id: str,
    ) -> CallResult:
        del server_url, scenario
        return CallResult(
            session_id=session_id,
            outcome="completed",
            turn_metrics=(),
            interrupted_turns=0,
            errors=(),
        )

    report = await run_load(concurrency=2, call_runner=empty_call)
    render_report(report)

    assert report.ttft == PercentileSummary(sample_count=0, p50=None, p95=None)
    assert report.ttfa.p50 is None
    assert report.total_latency.p95 is None
    assert "n/a" in capsys.readouterr().out  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_invalid_concurrency_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        await run_load(concurrency=0)
