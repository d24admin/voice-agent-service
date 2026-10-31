from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

import pytest

from app.events import (
    AudioChunk,
    AudioEnd,
    AudioFrame,
    LLMToken,
    SessionStop,
    TranscriptFinal,
    TurnCompleted,
    TurnInterrupted,
)
from app.metrics import TurnOutcome
from app.providers import ChatMessage
from tests.conftest import FakeTransport, event_fields, live_session_tasks
from tests.test_session import make_session, settings, start, stop


async def wait_until(predicate: object, timeout: float = 1) -> None:
    async def poll() -> None:
        while not predicate():  # type: ignore[operator]
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout)


class GenerationBargeProvider:
    def __init__(self) -> None:
        self.old_cancelled = asyncio.Event()

    async def stream(
        self, messages: Sequence[ChatMessage], turn_id: int
    ) -> AsyncIterator[str]:
        del messages
        if turn_id == 1:
            try:
                yield "old-token"
                await asyncio.Event().wait()
            finally:
                self.old_cancelled.set()
            return
        yield "New"
        yield "answer."


@pytest.mark.asyncio
async def test_generation_barge_in_is_single_fenced_and_keeps_trigger_frame() -> None:
    transport = FakeTransport()
    selected = settings(llm_token_delay_ms=1)
    session, registry = make_session(transport, service_settings=selected)
    provider = GenerationBargeProvider()
    session._provider = provider
    task = asyncio.create_task(session.run())
    await start(transport)

    await transport.inbound.put(
        AudioFrame(text="first", **event_fields("old-frame"))
    )
    await transport.inbound.put(AudioEnd(**event_fields("old-end")))
    await transport.wait_for_sent("llm.token")

    await transport.inbound.put(
        AudioFrame(text="new", **event_fields("barge-frame"))
    )
    await transport.inbound.put(
        AudioFrame(text="request", **event_fields("new-frame-2"))
    )
    await transport.inbound.put(AudioEnd(**event_fields("new-end")))

    await wait_until(
        lambda: any(
            isinstance(event, TurnCompleted) and event.turn_id == 2
            for event in transport.sent
        )
    )
    await stop(transport, task)

    interrupted = [
        event for event in transport.sent if isinstance(event, TurnInterrupted)
    ]
    assert len(interrupted) == 1
    terminal = interrupted[0]
    assert terminal.turn_id == 1
    assert terminal.reason == "barge_in"
    assert terminal.metrics["outcome"] == "interrupted"
    assert terminal.metrics["reason"] == "barge_in"
    assert terminal.metrics["ttft_ms"] is not None
    assert terminal.metrics["ttfa_ms"] is None
    assert terminal.metrics["total_latency_ms"] is not None

    boundary = transport.sent.index(terminal)
    assert not any(
        isinstance(event, (LLMToken, AudioChunk)) and event.turn_id == 1
        for event in transport.sent[boundary + 1 :]
    )
    assert [
        event.text for event in transport.sent if isinstance(event, TranscriptFinal)
    ] == ["first", "new request"]
    assert provider.old_cancelled.is_set()
    summary = registry.summary("session-1")
    assert summary is not None
    assert summary["counts"]["interrupted"] == 1
    assert live_session_tasks("session-1") == []


@pytest.mark.asyncio
async def test_playback_open_barge_in_drops_queued_old_audio_and_wins_terminal_race() -> None:
    class TwoTurnProvider:
        async def stream(
            self, messages: Sequence[ChatMessage], turn_id: int
        ) -> AsyncIterator[str]:
            del messages
            if turn_id == 1:
                for token in ("First", "sentence.", "Second", "sentence."):
                    yield token
            else:
                yield "Replacement."

    transport = FakeTransport()
    session, _ = make_session(transport)
    session._provider = TwoTurnProvider()
    task = asyncio.create_task(session.run())
    await start(transport)
    transport.block_on("audio.chunk")

    await transport.inbound.put(
        AudioFrame(text="old", **event_fields("old-frame"))
    )
    await transport.inbound.put(AudioEnd(**event_fields("old-end")))
    await transport.wait_for_attempt("audio.chunk")
    await wait_until(lambda: session.turns[1].generation_open is False)

    await transport.inbound.put(
        AudioFrame(text="replace", **event_fields("barge-frame"))
    )
    await transport.inbound.put(AudioEnd(**event_fields("new-end")))
    await wait_until(lambda: session.turns[1].outcome is TurnOutcome.INTERRUPTED)
    transport.release_sends()

    await wait_until(
        lambda: any(
            isinstance(event, TurnCompleted) and event.turn_id == 2
            for event in transport.sent
        )
    )
    await stop(transport, task)

    old_terminals = [
        event
        for event in transport.sent
        if isinstance(event, (TurnInterrupted, TurnCompleted)) and event.turn_id == 1
    ]
    assert len(old_terminals) == 1
    assert isinstance(old_terminals[0], TurnInterrupted)
    boundary = transport.sent.index(old_terminals[0])
    assert not any(
        isinstance(event, AudioChunk) and event.turn_id == 1
        for event in transport.sent[boundary + 1 :]
    )
    assert live_session_tasks("session-1") == []


@pytest.mark.asyncio
async def test_delayed_provider_cancellation_does_not_block_new_endpointing() -> None:
    class SlowCancelProvider:
        def __init__(self) -> None:
            self.cancel_seen = asyncio.Event()
            self.cleanup_done = asyncio.Event()

        async def stream(
            self, messages: Sequence[ChatMessage], turn_id: int
        ) -> AsyncIterator[str]:
            del messages
            if turn_id == 1:
                yield "visible"
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancel_seen.set()
                    await asyncio.sleep(0.02)
                    yield "late-after-cancel"
                finally:
                    self.cleanup_done.set()
                return
            yield "Fresh."

    transport = FakeTransport()
    selected = settings(cancellation_timeout_ms=100)
    session, _ = make_session(transport, service_settings=selected)
    provider = SlowCancelProvider()
    session._provider = provider
    task = asyncio.create_task(session.run())
    await start(transport)

    await transport.inbound.put(
        AudioFrame(text="old", **event_fields("old-frame"))
    )
    await transport.inbound.put(AudioEnd(**event_fields("old-end")))
    await transport.wait_for_sent("llm.token")
    await transport.inbound.put(
        AudioFrame(text="fresh", **event_fields("barge-frame"))
    )
    await transport.inbound.put(AudioEnd(**event_fields("fresh-end")))

    await transport.wait_for_sent("turn.interrupted")
    assert provider.cancel_seen.is_set()
    await wait_until(
        lambda: any(
            isinstance(event, TranscriptFinal) and event.text == "fresh"
            for event in transport.sent
        ),
        timeout=0.05,
    )
    await asyncio.wait_for(provider.cleanup_done.wait(), timeout=0.1)
    await wait_until(lambda: session.retiring_turn_count == 0)
    assert not any(
        isinstance(event, LLMToken) and event.token == "late-after-cancel"
        for event in transport.sent
    )

    await wait_until(
        lambda: any(
            isinstance(event, TurnCompleted) and event.turn_id == 2
            for event in transport.sent
        )
    )
    await stop(transport, task)
    assert live_session_tasks("session-1") == []


@pytest.mark.asyncio
async def test_disconnect_during_terminal_send_retains_internal_not_delivered_state() -> None:
    class DisconnectingTransport(FakeTransport):
        async def send(self, event: object) -> None:
            await self.attempted.put(event)
            if getattr(event, "type", None) == "turn.completed":
                raise ConnectionError("peer disconnected")
            self.sent.append(event)

    transport = DisconnectingTransport()
    session, registry = make_session(transport)
    task = asyncio.create_task(session.run())
    await start(transport)

    await transport.inbound.put(
        AudioFrame(text="hello", **event_fields("frame"))
    )
    await transport.inbound.put(AudioEnd(**event_fields("end")))

    await transport.wait_for_attempt("turn.completed")
    with pytest.raises(ConnectionError, match="peer disconnected"):
        await asyncio.wait_for(task, timeout=1)

    turn = session.turns[1]
    assert turn.outcome is TurnOutcome.COMPLETED
    assert turn.metrics.total_latency_ms is not None
    assert not any(isinstance(event, TurnCompleted) for event in transport.sent)
    summary = registry.summary("session-1")
    assert summary is not None
    assert summary["counts"] == {
        "started": 1,
        "completed": 0,
        "interrupted": 0,
        "failed": 0,
    }
    assert live_session_tasks("session-1") == []
