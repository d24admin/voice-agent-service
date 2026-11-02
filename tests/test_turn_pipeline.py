from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

import pytest

from app.api.events import AudioChunk, LLMToken, ServerEvent
from app.clock import SystemClock
from app.config import Settings
from app.conversation.turn_pipeline import ActiveTurnRunner
from app.conversation.turn_state import TurnState
from app.metrics import TurnMetrics, TurnOutcome
from app.providers.llm import ChatMessage


def settings() -> Settings:
    return Settings(
        token_queue_size=8,
        first_token_timeout_ms=100,
        generation_timeout_ms=500,
        tts_sentence_delay_ms=1,
    )


def turn() -> TurnState:
    return TurnState(
        turn_id=1,
        metrics=TurnMetrics(turn_id=1, baseline_ns=0),
        user_text="hello",
    )


@pytest.mark.asyncio
async def test_llm_keeps_producing_while_tts_handles_first_sentence() -> None:
    tts_started = asyncio.Event()
    release_tts = asyncio.Event()
    produced: list[str] = []
    emitted: list[ServerEvent] = []
    finished: list[tuple[TurnOutcome, str | None]] = []

    class Provider:
        async def stream(
            self,
            messages: Sequence[ChatMessage],
            turn_id: int,
            session_id: str | None = None,
        ) -> AsyncIterator[str]:
            del messages, turn_id, session_id
            for token in ("First", "sentence.", "Second", "sentence."):
                produced.append(token)
                yield token

    async def blocked_tts(_delay: float) -> None:
        tts_started.set()
        await release_tts.wait()

    async def emit(event: ServerEvent, _turn: TurnState) -> bool:
        emitted.append(event)
        return True

    async def finish(
        _turn: TurnState, outcome: TurnOutcome, reason: str | None
    ) -> None:
        finished.append((outcome, reason))

    active = turn()
    runner = ActiveTurnRunner(
        provider=Provider(),
        emit=emit,
        finish=finish,
        settings=settings(),
        session_id="session-1",
        clock=SystemClock(),
        sleep=blocked_tts,
    )
    task = asyncio.create_task(
        runner.run(active, (ChatMessage(role="user", content="hello"),))
    )

    await asyncio.wait_for(tts_started.wait(), timeout=1)
    await asyncio.sleep(0)
    assert produced == ["First", "sentence.", "Second", "sentence."]

    release_tts.set()
    await asyncio.wait_for(task, timeout=1)

    assert [event.token for event in emitted if isinstance(event, LLMToken)] == produced
    assert [event.text for event in emitted if isinstance(event, AudioChunk)] == [
        "First sentence.",
        "Second sentence.",
    ]
    assert finished == [(TurnOutcome.COMPLETED, None)]


@pytest.mark.asyncio
async def test_provider_failure_finishes_turn_once() -> None:
    finished: list[tuple[TurnOutcome, str | None]] = []

    class Provider:
        async def stream(
            self,
            messages: Sequence[ChatMessage],
            turn_id: int,
            session_id: str | None = None,
        ) -> AsyncIterator[str]:
            del messages, turn_id, session_id
            yield "visible"
            raise RuntimeError("secret provider detail")

    async def emit(_event: ServerEvent, _turn: TurnState) -> bool:
        return True

    async def finish(
        _turn: TurnState, outcome: TurnOutcome, reason: str | None
    ) -> None:
        finished.append((outcome, reason))

    await ActiveTurnRunner(
        provider=Provider(),
        emit=emit,
        finish=finish,
        settings=settings(),
        session_id="session-1",
        clock=SystemClock(),
        sleep=asyncio.sleep,
    ).run(turn(), (ChatMessage(role="user", content="hello"),))

    assert finished == [(TurnOutcome.FAILED, "provider_error")]
