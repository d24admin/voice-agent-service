from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PartialResult:
    utterance_id: int
    generation: int
    text: str
    last_frame_ns: int


@dataclass(frozen=True, slots=True)
class FinalResult:
    utterance_id: int
    generation: int
    text: str
    last_frame_ns: int


class SimulatedSTT:
    def __init__(self) -> None:
        self._utterance_id = 1
        self._generation = 0
        self._fragments: list[str] = []
        self._last_frame_ns: int | None = None
        self._finalized = False

    def accept_frame(self, fragment: str, received_ns: int) -> PartialResult:
        normalized = fragment.strip()
        if not normalized:
            raise ValueError("audio fragment must not be empty")
        if self._finalized:
            self._utterance_id += 1
            self._fragments.clear()
            self._last_frame_ns = None
            self._finalized = False

        self._generation += 1
        self._fragments.append(normalized)
        self._last_frame_ns = received_ns
        return PartialResult(
            utterance_id=self._utterance_id,
            generation=self._generation,
            text=" ".join(self._fragments),
            last_frame_ns=received_ns,
        )

    def current_timer_generation(self) -> int:
        return self._generation

    def finalize(self, timer_generation: int | None = None) -> FinalResult | None:
        if not self._fragments or self._finalized:
            return None
        if timer_generation is not None and timer_generation != self._generation:
            return None
        if self._last_frame_ns is None:
            return None

        self._finalized = True
        return FinalResult(
            utterance_id=self._utterance_id,
            generation=self._generation,
            text=" ".join(self._fragments),
            last_frame_ns=self._last_frame_ns,
        )
