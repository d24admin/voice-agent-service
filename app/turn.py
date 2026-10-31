from __future__ import annotations

from dataclasses import dataclass

from app.metrics import TurnMetrics, TurnOutcome

__all__ = ["TurnOutcome", "TurnState"]


@dataclass(slots=True)
class TurnState:
    turn_id: int
    metrics: TurnMetrics
    outcome: TurnOutcome = TurnOutcome.OPEN
    reason: str | None = None
    generation_open: bool = True
    playback_open: bool = True
    output_valid: bool = True

    def interrupt(self, reason: str) -> bool:
        return self._claim_terminal(TurnOutcome.INTERRUPTED, reason, invalidate=True)

    def fail(self, reason: str) -> bool:
        return self._claim_terminal(TurnOutcome.FAILED, reason, invalidate=True)

    def mark_generation_finished(self) -> None:
        self.generation_open = False

    def complete_from_writer(self) -> bool:
        if self.outcome is not TurnOutcome.OPEN or self.generation_open:
            return False
        self.playback_open = False
        return self._claim_terminal(TurnOutcome.COMPLETED, None, invalidate=False)

    def _claim_terminal(
        self, outcome: TurnOutcome, reason: str | None, *, invalidate: bool
    ) -> bool:
        if self.outcome is not TurnOutcome.OPEN:
            return False
        self.outcome = outcome
        self.reason = reason
        if invalidate:
            self.output_valid = False
        return True
