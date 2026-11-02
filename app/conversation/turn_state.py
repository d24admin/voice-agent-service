from __future__ import annotations

from dataclasses import dataclass, field

from app.metrics import TurnMetrics, TurnOutcome

__all__ = ["TurnOutcome", "TurnState"]


@dataclass(slots=True)
class TurnState:
    turn_id: int
    metrics: TurnMetrics
    user_text: str = ""
    outcome: TurnOutcome = TurnOutcome.OPEN
    reason: str | None = None
    output_valid: bool = True
    response_tokens: list[str] = field(default_factory=list)
    history_committed: bool = False

    def complete(self) -> bool:
        return self._claim(TurnOutcome.COMPLETED, None, invalidate=False)

    def interrupt(self, reason: str) -> bool:
        return self._claim(TurnOutcome.INTERRUPTED, reason, invalidate=True)

    def fail(self, reason: str) -> bool:
        return self._claim(TurnOutcome.FAILED, reason, invalidate=True)

    def _claim(
        self,
        outcome: TurnOutcome,
        reason: str | None,
        *,
        invalidate: bool,
    ) -> bool:
        if self.outcome is not TurnOutcome.OPEN:
            return False
        self.outcome = outcome
        self.reason = reason
        if invalidate:
            self.output_valid = False
        return True
