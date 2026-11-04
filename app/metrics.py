from __future__ import annotations

import math
from collections.abc import Sequence
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TurnOutcome(str, Enum):
    OPEN = "open"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class TurnMetrics(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    turn_id: int = Field(ge=1)
    baseline_ns: int = Field(ge=0, exclude=True)
    outcome: TurnOutcome = TurnOutcome.OPEN
    reason: str | None = None
    ttft_ms: float | None = None
    ttfa_ms: float | None = None
    total_latency_ms: float | None = None
    token_count: int = 0
    audio_chunk_count: int = 0

    def _elapsed_ms(self, now_ns: int) -> float:
        return max(0.0, (now_ns - self.baseline_ns) / 1_000_000)

    def record_token_sent(self, now_ns: int) -> None:
        if self.ttft_ms is None:
            self.ttft_ms = self._elapsed_ms(now_ns)
        self.token_count += 1

    def record_audio_sent(self, now_ns: int) -> None:
        if self.ttfa_ms is None:
            self.ttfa_ms = self._elapsed_ms(now_ns)
        self.audio_chunk_count += 1

    def finalize(
        self,
        outcome: TurnOutcome,
        reason: str | None,
        now_ns: int,
    ) -> None:
        if self.total_latency_ms is not None:
            return
        self.outcome = outcome
        self.reason = reason
        self.total_latency_ms = self._elapsed_ms(now_ns)


def nearest_rank(values: Sequence[float], percentile: float) -> float | None:
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in (0, 1]")
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil(percentile * len(ordered)) - 1
    return float(ordered[index])


def _percentile_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "sample_count": len(values),
        "p50": nearest_rank(values, 0.50),
        "p95": nearest_rank(values, 0.95),
    }


class SessionMetrics:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._turns: list[TurnMetrics] = []
        self._started_ids: set[int] = set()
        self._recorded_ids: set[int] = set()
        self._counts = {
            "started": 0,
            "completed": 0,
            "interrupted": 0,
            "failed": 0,
        }

    def mark_started(self, turn_id: int) -> None:
        if turn_id in self._started_ids:
            return
        self._started_ids.add(turn_id)
        self._counts["started"] += 1

    def record_turn(self, metrics: TurnMetrics) -> None:
        self.mark_started(metrics.turn_id)
        if metrics.turn_id in self._recorded_ids:
            return
        if metrics.outcome is TurnOutcome.OPEN:
            raise ValueError("cannot record an open turn")
        self._recorded_ids.add(metrics.turn_id)
        self._counts[metrics.outcome.value] += 1
        self._turns.append(metrics.model_copy(deep=True))

    def summary(self, *, include_turns: bool = True) -> dict[str, Any]:
        completed = [
            turn for turn in self._turns if turn.outcome is TurnOutcome.COMPLETED
        ]
        result: dict[str, Any] = {
            "session_id": self.session_id,
            "counts": dict(self._counts),
            "ttft": _percentile_summary(
                [turn.ttft_ms for turn in completed if turn.ttft_ms is not None]
            ),
            "ttfa": _percentile_summary(
                [turn.ttfa_ms for turn in completed if turn.ttfa_ms is not None]
            ),
            "total_latency": _percentile_summary(
                [
                    turn.total_latency_ms
                    for turn in completed
                    if turn.total_latency_ms is not None
                ]
            ),
        }
        if include_turns:
            result["turns"] = [
                turn.model_dump(mode="json") for turn in self._turns
            ]
        return result

    @property
    def turns(self) -> tuple[TurnMetrics, ...]:
        return tuple(turn.model_copy(deep=True) for turn in self._turns)


class MetricsRegistry:
    _MAX_SESSIONS = 100

    def __init__(self) -> None:
        self._sessions: dict[str, SessionMetrics] = {}

    def start_session(self, session_id: str) -> SessionMetrics:
        session = SessionMetrics(session_id)
        self._sessions.pop(session_id, None)
        self._sessions[session_id] = session
        while len(self._sessions) > self._MAX_SESSIONS:
            self._sessions.pop(next(iter(self._sessions)))
        return session

    def get_or_create(self, session_id: str) -> SessionMetrics:
        session = self._sessions.get(session_id)
        if session is None:
            session = self.start_session(session_id)
        return session

    def get(self, session_id: str) -> SessionMetrics | None:
        return self._sessions.get(session_id)

    def summary(self, session_id: str | None = None) -> dict[str, Any] | None:
        if session_id is not None:
            session = self.get(session_id)
            return None if session is None else session.summary()

        completed = [
            turn
            for session in self._sessions.values()
            for turn in session.turns
            if turn.outcome is TurnOutcome.COMPLETED
        ]
        counts = {
            key: sum(
                session.summary(include_turns=False)["counts"][key]
                for session in self._sessions.values()
            )
            for key in ("started", "completed", "interrupted", "failed")
        }
        return {
            "counts": counts,
            "ttft": _percentile_summary(
                [turn.ttft_ms for turn in completed if turn.ttft_ms is not None]
            ),
            "ttfa": _percentile_summary(
                [turn.ttfa_ms for turn in completed if turn.ttfa_ms is not None]
            ),
            "total_latency": _percentile_summary(
                [
                    turn.total_latency_ms
                    for turn in completed
                    if turn.total_latency_ms is not None
                ]
            ),
        }
