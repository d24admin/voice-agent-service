from datetime import UTC, datetime

import pytest

from app.metrics import MetricsRegistry, SessionMetrics, TurnMetrics, nearest_rank
from app.conversation.turn_state import TurnOutcome, TurnState


class FakeClock:
    def __init__(self, monotonic_ns: int = 1_000_000_000) -> None:
        self.now_ns = monotonic_ns

    def monotonic_ns(self) -> int:
        return self.now_ns

    def utc_now(self) -> datetime:
        return datetime(2026, 10, 31, tzinfo=UTC)

    def advance_ms(self, milliseconds: int) -> None:
        self.now_ns += milliseconds * 1_000_000


def make_turn(turn_id: int = 1) -> TurnState:
    metrics = TurnMetrics(turn_id=turn_id, baseline_ns=1_000_000_000)
    return TurnState(turn_id=turn_id, metrics=metrics)


def completed_metrics(turn_id: int, latency_ms: int) -> TurnMetrics:
    metrics = TurnMetrics(turn_id=turn_id, baseline_ns=1_000_000_000)
    metrics.record_token_sent(1_010_000_000)
    metrics.record_audio_sent(1_020_000_000)
    metrics.finalize(TurnOutcome.COMPLETED, None, 1_000_000_000 + latency_ms * 1_000_000)
    return metrics


@pytest.mark.parametrize(
    ("transition", "outcome", "reason"),
    [
        ("interrupt", TurnOutcome.INTERRUPTED, "barge_in"),
        ("fail", TurnOutcome.FAILED, "provider_error"),
    ],
)
def test_terminal_transition_wins_once_and_invalidates_output(
    transition: str, outcome: TurnOutcome, reason: str
) -> None:
    turn = make_turn()

    assert getattr(turn, transition)(reason) is True
    assert turn.outcome is outcome
    assert turn.reason == reason
    assert turn.output_valid is False
    assert turn.interrupt("later") is False
    assert turn.fail("later") is False
    assert turn.complete_from_writer() is False


def test_writer_completion_requires_generation_to_finish_and_closes_playback() -> None:
    turn = make_turn()

    assert turn.complete_from_writer() is False
    assert turn.outcome is TurnOutcome.OPEN
    assert turn.playback_open is True

    turn.mark_generation_finished()

    assert turn.complete_from_writer() is True
    assert turn.outcome is TurnOutcome.COMPLETED
    assert turn.playback_open is False
    assert turn.complete_from_writer() is False


def test_turn_metrics_record_first_transport_timestamps_once() -> None:
    metrics = TurnMetrics(turn_id=7, baseline_ns=1_000_000_000)

    metrics.record_token_sent(1_025_000_000)
    metrics.record_token_sent(1_040_000_000)
    metrics.record_audio_sent(1_080_000_000)
    metrics.record_audio_sent(1_100_000_000)
    metrics.finalize(TurnOutcome.COMPLETED, None, 1_150_000_000)

    assert metrics.ttft_ms == 25.0
    assert metrics.ttfa_ms == 80.0
    assert metrics.total_latency_ms == 150.0
    assert metrics.token_count == 2
    assert metrics.audio_chunk_count == 2
    assert metrics.outcome is TurnOutcome.COMPLETED


def test_interrupted_metrics_serialize_unreached_values_as_null() -> None:
    metrics = TurnMetrics(turn_id=2, baseline_ns=1_000_000_000)
    metrics.finalize(TurnOutcome.INTERRUPTED, "barge_in", 1_015_000_000)

    payload = metrics.model_dump(mode="json")

    assert payload["ttft_ms"] is None
    assert payload["ttfa_ms"] is None
    assert payload["total_latency_ms"] == 15.0
    assert payload["outcome"] == "interrupted"
    assert payload["reason"] == "barge_in"


def test_nearest_rank_percentiles_are_deterministic() -> None:
    samples = [40.0, 10.0, 30.0, 20.0]

    assert nearest_rank(samples, 0.50) == 20.0
    assert nearest_rank(samples, 0.95) == 40.0
    assert nearest_rank([], 0.50) is None
    with pytest.raises(ValueError):
        nearest_rank(samples, 0.0)


def test_session_summary_aggregates_only_completed_turns() -> None:
    session = SessionMetrics(session_id="session-1", turn_capacity=4)
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
    assert summary["ttfa"] == {"sample_count": 1, "p50": 20.0, "p95": 20.0}
    assert summary["total_latency"] == {
        "sample_count": 1,
        "p50": 100.0,
        "p95": 100.0,
    }
    assert [record["turn_id"] for record in summary["turns"]] == [1, 2]


def test_registry_evicts_least_recently_used_session() -> None:
    registry = MetricsRegistry(session_capacity=2, turns_per_session=2)
    registry.get_or_create("session-1")
    registry.get_or_create("session-2")
    assert registry.get("session-1") is not None

    registry.get_or_create("session-3")

    assert registry.get("session-2") is None
    assert registry.get("session-1") is not None
    assert registry.get("session-3") is not None


def test_session_turn_records_are_bounded_and_ordered() -> None:
    session = SessionMetrics(session_id="session-1", turn_capacity=2)
    session.record_turn(completed_metrics(1, 10))
    session.record_turn(completed_metrics(2, 20))
    session.record_turn(completed_metrics(3, 30))

    assert [record["turn_id"] for record in session.summary()["turns"]] == [2, 3]
