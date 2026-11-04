from __future__ import annotations

import pytest

from app.conversation.turn_state import TurnOutcome, TurnState
from app.metrics import MetricsRegistry, SessionMetrics, TurnMetrics, nearest_rank


def completed_metrics(turn_id: int, latency_ms: int) -> TurnMetrics:
    metrics = TurnMetrics(turn_id=turn_id, baseline_ns=1_000_000_000)
    metrics.record_token_sent(1_010_000_000)
    metrics.record_audio_sent(1_020_000_000)
    metrics.finalize(
        TurnOutcome.COMPLETED,
        None,
        1_000_000_000 + latency_ms * 1_000_000,
    )
    return metrics


def test_terminal_transition_wins_once_and_invalidates_only_noncompletion() -> None:
    interrupted = TurnState(
        turn_id=1,
        metrics=TurnMetrics(turn_id=1, baseline_ns=0),
    )
    assert interrupted.interrupt("barge_in") is True
    assert interrupted.output_valid is False
    assert interrupted.complete() is False

    completed = TurnState(
        turn_id=2,
        metrics=TurnMetrics(turn_id=2, baseline_ns=0),
    )
    assert completed.complete() is True
    assert completed.outcome is TurnOutcome.COMPLETED
    assert completed.output_valid is True
    assert completed.fail("late") is False


def test_turn_metrics_record_first_successful_transport_times() -> None:
    metrics = TurnMetrics(turn_id=7, baseline_ns=1_000_000_000)
    metrics.record_token_sent(1_025_000_000)
    metrics.record_token_sent(1_040_000_000)
    metrics.record_audio_sent(1_080_000_000)
    metrics.finalize(TurnOutcome.COMPLETED, None, 1_150_000_000)

    assert metrics.ttft_ms == 25.0
    assert metrics.ttfa_ms == 80.0
    assert metrics.total_latency_ms == 150.0
    assert metrics.token_count == 2
    assert metrics.audio_chunk_count == 1


def test_session_summary_aggregates_completed_latency_and_all_outcomes() -> None:
    session = SessionMetrics("session-1")
    session.record_turn(completed_metrics(1, 100))
    interrupted = TurnMetrics(turn_id=2, baseline_ns=1_000_000_000)
    interrupted.finalize(TurnOutcome.INTERRUPTED, "barge_in", 1_050_000_000)
    session.record_turn(interrupted)

    summary = session.summary()

    assert summary["counts"] == {
        "started": 2,
        "completed": 1,
        "interrupted": 1,
        "failed": 0,
    }
    assert summary["ttft"] == {"sample_count": 1, "p50": 10.0, "p95": 10.0}
    assert [record["turn_id"] for record in summary["turns"]] == [1, 2]
    assert "queue_high_water" not in summary
    assert "overloads" not in summary


def test_start_session_replaces_previous_run_for_same_id() -> None:
    registry = MetricsRegistry()
    first = registry.start_session("session-1")
    first.record_turn(completed_metrics(1, 100))

    second = registry.start_session("session-1")

    assert second is not first
    assert registry.summary("session-1")["turns"] == []  # type: ignore[index]


def test_registry_retains_only_100_most_recent_sessions() -> None:
    registry = MetricsRegistry()

    for index in range(101):
        session = registry.start_session(f"session-{index}")
        session.record_turn(completed_metrics(1, index + 1))

    assert registry.get("session-0") is None
    assert all(registry.get(f"session-{index}") is not None for index in range(1, 101))
    assert registry.summary()["counts"]["completed"] == 100  # type: ignore[index]


def test_replacing_session_moves_it_to_newest_retention_position() -> None:
    registry = MetricsRegistry()
    for index in range(100):
        registry.start_session(f"session-{index}")

    replaced = registry.start_session("session-0")
    registry.start_session("session-100")

    assert registry.get("session-0") is replaced
    assert registry.get("session-1") is None
    assert all(registry.get(f"session-{index}") is not None for index in range(2, 101))


def test_nearest_rank_percentiles_are_deterministic() -> None:
    samples = [40.0, 10.0, 30.0, 20.0]
    assert nearest_rank(samples, 0.50) == 20.0
    assert nearest_rank(samples, 0.95) == 40.0
    assert nearest_rank([], 0.50) is None
    with pytest.raises(ValueError):
        nearest_rank(samples, 0.0)
