from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

import pytest

from app.config import Settings
from app.api.events import AudioChunk, LLMToken
from app.metrics import TurnMetrics, TurnOutcome
from app.conversation.outbound_buffer import (
    OutboundBuffer,
    OutputCompleteAction,
    SendAction,
    TerminalCommand,
)
from app.providers.llm import ChatMessage
from app.conversation.turn_pipeline import ActiveTurnRunner
from app.conversation.turn_state import TurnState


MESSAGES = (ChatMessage(role="user", content="hello"),)


def make_turn(turn_id: int = 1) -> TurnState:
    return TurnState(
        turn_id=turn_id,
        metrics=TurnMetrics(turn_id=turn_id, baseline_ns=0),
    )


def make_settings(**overrides: int) -> Settings:
    values = {
        "token_queue_size": 4,
        "outbound_data_capacity": 32,
        "outbound_lifecycle_capacity": 4,
        "tts_sentence_delay_ms": 1,
        "first_token_timeout_ms": 50,
        "generation_timeout_ms": 200,
        "backpressure_timeout_ms": 50,
    }
    values.update(overrides)
    return Settings(**values)


class RecordingProvider:
    def __init__(self, tokens: Sequence[str]) -> None:
        self.tokens = tokens
        self.yielded: list[str] = []
        self.finished = asyncio.Event()

    async def stream(
        self, messages: Sequence[ChatMessage], turn_id: int
    ) -> AsyncIterator[str]:
        del messages, turn_id
        for token in self.tokens:
            self.yielded.append(token)
            yield token
        self.finished.set()


async def drain_until_complete(
    outbound: OutboundBuffer,
) -> list[SendAction | OutputCompleteAction]:
    actions: list[SendAction | OutputCompleteAction] = []
    while True:
        action = await outbound.next_dispatch(lambda _turn_id: True)
        assert isinstance(action, (SendAction, OutputCompleteAction))
        actions.append(action)
        if isinstance(action, OutputCompleteAction):
            return actions


@pytest.mark.asyncio
async def test_llm_production_overlaps_blocked_sentence_tts() -> None:
    provider = RecordingProvider(("First.", "Later", "sentence."))
    outbound = OutboundBuffer(data_capacity=32, lifecycle_capacity=4)
    tts_started = asyncio.Event()
    release_tts = asyncio.Event()

    async def gated_tts_sleep(delay_s: float) -> None:
        assert delay_s > 0
        tts_started.set()
        await release_tts.wait()

    runner = ActiveTurnRunner(
        provider=provider,
        outbound=outbound,
        settings=make_settings(),
        session_id="session-1",
        sleep=gated_tts_sleep,
    )
    turn = make_turn()

    task = asyncio.create_task(runner.run(turn, MESSAGES))
    await asyncio.wait_for(tts_started.wait(), timeout=0.2)
    await asyncio.wait_for(provider.finished.wait(), timeout=0.2)

    assert provider.yielded == ["First.", "Later", "sentence."]
    assert turn.generation_open is False

    release_tts.set()
    await asyncio.wait_for(task, timeout=0.2)
    actions = await drain_until_complete(outbound)

    first_audio_index = next(
        index
        for index, action in enumerate(actions)
        if isinstance(action, SendAction) and isinstance(action.item, AudioChunk)
    )
    complete_index = next(
        index
        for index, action in enumerate(actions)
        if isinstance(action, OutputCompleteAction)
    )
    assert first_audio_index < complete_index
    assert [
        action.item.text
        for action in actions
        if isinstance(action, SendAction) and isinstance(action.item, AudioChunk)
    ] == ["First.", "Later sentence."]
    await outbound.close()


@pytest.mark.asyncio
async def test_cancellation_stops_and_awaits_both_pipeline_children() -> None:
    provider_closed = asyncio.Event()
    tts_started = asyncio.Event()
    tts_cancelled = asyncio.Event()

    class BlockingProvider:
        async def stream(
            self, messages: Sequence[ChatMessage], turn_id: int
        ) -> AsyncIterator[str]:
            del messages, turn_id
            try:
                yield "Speaking."
                await asyncio.Event().wait()
            finally:
                provider_closed.set()

    async def blocking_tts_sleep(delay_s: float) -> None:
        del delay_s
        tts_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            tts_cancelled.set()
            raise

    outbound = OutboundBuffer(data_capacity=32, lifecycle_capacity=4)
    turn = make_turn()
    runner = ActiveTurnRunner(
        provider=BlockingProvider(),
        outbound=outbound,
        settings=make_settings(),
        session_id="session-1",
        sleep=blocking_tts_sleep,
    )
    task = asyncio.create_task(runner.run(turn, MESSAGES))
    await asyncio.wait_for(tts_started.wait(), timeout=0.2)

    assert turn.interrupt("barge_in") is True
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert provider_closed.is_set()
    assert tts_cancelled.is_set()
    await outbound.close()


@pytest.mark.asyncio
async def test_token_yielded_after_invalidation_reaches_no_output_lane() -> None:
    turn = make_turn()

    class LateProvider:
        async def stream(
            self, messages: Sequence[ChatMessage], turn_id: int
        ) -> AsyncIterator[str]:
            del messages, turn_id
            turn.interrupt("barge_in")
            yield "late"

    outbound = OutboundBuffer(data_capacity=32, lifecycle_capacity=4)
    runner = ActiveTurnRunner(
        provider=LateProvider(),
        outbound=outbound,
        settings=make_settings(),
        session_id="session-1",
    )

    await asyncio.wait_for(runner.run(turn, MESSAGES), timeout=0.2)

    assert outbound.data_size == 0
    assert outbound.lifecycle_size == 0
    await outbound.close()


@pytest.mark.asyncio
async def test_normal_end_of_stream_flushes_trailing_fragment() -> None:
    provider = RecordingProvider(("unfinished", "thought"))
    outbound = OutboundBuffer(data_capacity=32, lifecycle_capacity=4)
    runner = ActiveTurnRunner(
        provider=provider,
        outbound=outbound,
        settings=make_settings(),
        session_id="session-1",
    )

    await runner.run(make_turn(), MESSAGES)
    actions = await drain_until_complete(outbound)

    assert [
        action.item.token
        for action in actions
        if isinstance(action, SendAction) and isinstance(action.item, LLMToken)
    ] == ["unfinished", "thought"]
    assert [
        action.item.text
        for action in actions
        if isinstance(action, SendAction) and isinstance(action.item, AudioChunk)
    ] == ["unfinished thought"]
    await outbound.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["provider_error", "first_token_timeout"])
async def test_provider_failure_invalidates_output_and_queues_one_terminal(
    failure: str,
) -> None:
    class FailingProvider:
        async def stream(
            self, messages: Sequence[ChatMessage], turn_id: int
        ) -> AsyncIterator[str]:
            del messages, turn_id
            if failure == "first_token_timeout":
                await asyncio.sleep(1)
                yield "too late"
            raise RuntimeError("provider exploded")
            yield  # pragma: no cover

    outbound = OutboundBuffer(data_capacity=32, lifecycle_capacity=4)
    turn = make_turn()
    settings = make_settings(first_token_timeout_ms=5) if failure.endswith("timeout") else make_settings()
    runner = ActiveTurnRunner(
        provider=FailingProvider(),
        outbound=outbound,
        settings=settings,
        session_id="session-1",
    )

    await asyncio.wait_for(runner.run(turn, MESSAGES), timeout=0.2)

    assert turn.outcome is TurnOutcome.FAILED
    assert turn.output_valid is False
    assert outbound.lifecycle_size == 1
    action = await outbound.next_dispatch(lambda _turn_id: False)
    assert isinstance(action, SendAction)
    assert isinstance(action.item, TerminalCommand)
    assert action.item.turn_id == turn.turn_id
    assert action.item.outcome is TurnOutcome.FAILED
    assert action.item.reason is not None
    assert failure in action.item.reason
    assert outbound.lifecycle_size == 0
    await outbound.ack(action)
    await outbound.close()
