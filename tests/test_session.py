from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

import pytest

from app.clock import SystemClock
from app.config import Settings
from app.api.events import (
    AudioChunk,
    AudioEnd,
    AudioFrame,
    LLMToken,
    SessionStart,
    SessionStop,
    TranscriptFinal,
    TranscriptPartial,
    TurnCompleted,
    TurnFailed,
)
from app.metrics import MetricsRegistry
from app.providers.llm import (
    ChatMessage,
    LLMProviderError,
    LLMProviderUnavailable,
    MockLLMProvider,
)
from app.conversation.call_session import CallSession
from tests.conftest import FakeTransport, event_fields, live_session_tasks


def settings(**overrides: int) -> Settings:
    values = {
        "stt_partial_delay_ms": 1,
        "silence_timeout_ms": 30,
        "llm_token_delay_ms": 1,
        "tts_sentence_delay_ms": 1,
        "first_token_timeout_ms": 100,
        "generation_timeout_ms": 500,
        "shutdown_grace_ms": 500,
        "token_queue_size": 8,
        "output_queue_size": 32,
    }
    values.update(overrides)
    return Settings(**values)


def make_session(
    transport: FakeTransport,
    *,
    service_settings: Settings | None = None,
) -> tuple[CallSession, MetricsRegistry]:
    selected = settings() if service_settings is None else service_settings
    registry = MetricsRegistry()
    provider = MockLLMProvider(
        responses=("Hi there. How are you?",),
        token_delay_s=selected.llm_token_delay_ms / 1_000,
    )
    return (
        CallSession(
            session_id="session-1",
            settings=selected,
            transport=transport,
            provider=provider,
            registry=registry,
            clock=SystemClock(),
        ),
        registry,
    )


async def start(transport: FakeTransport) -> None:
    await transport.inbound.put(SessionStart(**event_fields("start")))
    await transport.wait_for_sent("session.ready")


async def stop(transport: FakeTransport, task: asyncio.Task[None]) -> None:
    await transport.inbound.put(SessionStop(**event_fields("stop")))
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_scripted_call_streams_ordered_stt_llm_tts_and_metrics() -> None:
    transport = FakeTransport()
    session, registry = make_session(transport)
    task = asyncio.create_task(session.run())
    await start(transport)

    for index, text in enumerate(("hello", "there", "friend"), start=1):
        await transport.inbound.put(
            AudioFrame(text=text, **event_fields(f"frame-{index}"))
        )
    await transport.inbound.put(AudioEnd(**event_fields("end")))

    completed = await transport.wait_for_sent("turn.completed")
    await stop(transport, task)

    assert [
        event.text for event in transport.sent if isinstance(event, TranscriptPartial)
    ] == ["hello", "hello there", "hello there friend"]
    finals = [event for event in transport.sent if isinstance(event, TranscriptFinal)]
    assert [event.text for event in finals] == ["hello there friend"]
    assert [
        event.token for event in transport.sent if isinstance(event, LLMToken)
    ] == ["Hi", "there.", "How", "are", "you?"]
    assert [
        (event.sequence, event.text)
        for event in transport.sent
        if isinstance(event, AudioChunk)
    ] == [(0, "Hi there."), (1, "How are you?")]
    assert isinstance(completed, TurnCompleted)
    assert completed.metrics["ttft_ms"] is not None
    assert completed.metrics["ttfa_ms"] is not None
    assert completed.metrics["total_latency_ms"] is not None
    assert completed.metrics["token_count"] == 5
    assert completed.metrics["audio_chunk_count"] == 2
    summary = registry.summary("session-1")
    assert summary is not None
    assert summary["counts"] == {
        "started": 1,
        "completed": 1,
        "interrupted": 0,
        "failed": 0,
    }
    assert transport.close_code == 1000
    assert live_session_tasks("session-1") == []


@pytest.mark.asyncio
async def test_provider_exhaustion_speaks_fallback_fails_and_closes_provider() -> None:
    class UnavailableProvider:
        def __init__(self) -> None:
            self.closed = False

        async def stream(
            self,
            messages: list[ChatMessage],
            turn_id: int,
            session_id: str | None = None,
        ) -> object:
            del messages, turn_id, session_id
            if False:
                yield ""
            raise LLMProviderUnavailable

        async def aclose(self) -> None:
            self.closed = True

    transport = FakeTransport()
    provider = UnavailableProvider()
    registry = MetricsRegistry()
    session = CallSession(
        session_id="session-1",
        settings=settings(),
        transport=transport,
        provider=provider,
        registry=registry,
        clock=SystemClock(),
    )
    task = asyncio.create_task(session.run())
    await start(transport)
    await transport.inbound.put(AudioFrame(text="hello", **event_fields("frame")))
    await transport.inbound.put(AudioEnd(**event_fields("end")))

    failed = await transport.wait_for_sent("turn.failed")
    await stop(transport, task)

    assert isinstance(failed, TurnFailed)
    assert failed.reason == "provider_error"
    assert " ".join(
        event.token for event in transport.sent if isinstance(event, LLMToken)
    ) == settings().llm_failure_message
    assert [
        event.text for event in transport.sent if isinstance(event, AudioChunk)
    ] == [
        "I’m having trouble connecting right now.",
        "Please try again in a moment.",
    ]
    assert session.turns[1].history_committed is False
    assert provider.closed is True
    summary = registry.summary("session-1")
    assert summary is not None
    assert summary["counts"]["failed"] == 1


@pytest.mark.asyncio
async def test_audio_end_without_frames_starts_no_turn() -> None:
    transport = FakeTransport()
    session, registry = make_session(transport)
    task = asyncio.create_task(session.run())
    await start(transport)

    await transport.inbound.put(AudioEnd(**event_fields("empty-end")))
    await asyncio.sleep(0.02)
    await stop(transport, task)

    assert not any(isinstance(event, TranscriptFinal) for event in transport.sent)
    assert not any(isinstance(event, LLMToken) for event in transport.sent)
    summary = registry.summary("session-1")
    assert summary is not None
    assert summary["counts"]["started"] == 0
    assert live_session_tasks("session-1") == []


@pytest.mark.asyncio
async def test_silence_timeout_finalizes_once_and_stale_expiry_is_harmless() -> None:
    transport = FakeTransport()
    session, _ = make_session(
        transport,
        service_settings=settings(
            stt_partial_delay_ms=30,
            silence_timeout_ms=10,
        ),
    )
    task = asyncio.create_task(session.run())
    await start(transport)

    await transport.inbound.put(
        AudioFrame(text="one", **event_fields("frame-1"))
    )
    await asyncio.sleep(0.005)
    await transport.inbound.put(
        AudioFrame(text="two", **event_fields("frame-2"))
    )

    await transport.wait_for_sent("turn.completed")
    await asyncio.sleep(0.03)
    await stop(transport, task)

    partials = [
        event.text for event in transport.sent if isinstance(event, TranscriptPartial)
    ]
    finals = [event.text for event in transport.sent if isinstance(event, TranscriptFinal)]
    assert partials == ["one", "one two"]
    assert finals == ["one two"]
    assert live_session_tasks("session-1") == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("blocked_type", "metric_name"),
    [("llm.token", "ttft_ms"), ("audio.chunk", "ttfa_ms")],
)
async def test_writer_stamps_latency_only_after_transport_send_succeeds(
    blocked_type: str,
    metric_name: str,
) -> None:
    transport = FakeTransport()
    session, _ = make_session(transport)
    task = asyncio.create_task(session.run())
    await start(transport)
    transport.block_on(blocked_type)

    await transport.inbound.put(
        AudioFrame(text="hello", **event_fields("frame"))
    )
    await transport.inbound.put(AudioEnd(**event_fields("end")))

    await transport.wait_for_attempt(blocked_type)
    turn = session.turns[1]
    assert getattr(turn.metrics, metric_name) is None

    transport.release_sends()
    await transport.wait_for_sent("turn.completed")
    assert getattr(turn.metrics, metric_name) is not None
    await stop(transport, task)
    assert live_session_tasks("session-1") == []


@pytest.mark.asyncio
async def test_stalled_writer_is_cancelled_when_shutdown_grace_expires() -> None:
    transport = FakeTransport()
    selected = settings(shutdown_grace_ms=10)
    session, _ = make_session(transport, service_settings=selected)
    task = asyncio.create_task(session.run())
    await start(transport)
    transport.block_on("llm.token")
    await transport.inbound.put(AudioFrame(text="hello", **event_fields("frame")))
    await transport.inbound.put(AudioEnd(**event_fields("end")))
    await transport.wait_for_attempt("llm.token")

    try:
        await session.request_shutdown()
        done, _ = await asyncio.wait({task}, timeout=0.1)
        assert task in done
        assert transport.close_code == 1000
        assert live_session_tasks("session-1") == []
    finally:
        transport.release_sends()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_session_stop_cannot_block_before_timed_writer_cleanup() -> None:
    transport = FakeTransport()
    selected = settings(shutdown_grace_ms=10, output_queue_size=1)
    session, _ = make_session(transport, service_settings=selected)
    task = asyncio.create_task(session.run())
    await start(transport)
    transport.block_on("llm.token")
    await transport.inbound.put(AudioFrame(text="hello", **event_fields("frame")))
    await transport.inbound.put(AudioEnd(**event_fields("end")))
    await transport.wait_for_attempt("llm.token")

    async def wait_for_full_queue() -> None:
        while session.output_queue_size < 1:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_full_queue(), timeout=0.1)
    await transport.inbound.put(SessionStop(**event_fields("stop")))

    try:
        await asyncio.wait_for(task, timeout=0.1)
        assert transport.close_code == 1000
        assert live_session_tasks("session-1") == []
    finally:
        transport.release_sends()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_competing_provider_and_protocol_failures_emit_one_terminal() -> None:
    class PartialFailureProvider:
        async def stream(
            self,
            messages: Sequence[ChatMessage],
            turn_id: int,
            session_id: str | None = None,
        ) -> AsyncIterator[str]:
            del messages, turn_id, session_id
            yield "Partial"
            raise LLMProviderError

    transport = FakeTransport()
    session, _ = make_session(transport)
    session._provider = PartialFailureProvider()
    task = asyncio.create_task(session.run())
    await start(transport)
    transport.block_on("llm.token")
    await transport.inbound.put(AudioFrame(text="hello", **event_fields("frame")))
    await transport.inbound.put(AudioEnd(**event_fields("end")))
    await transport.wait_for_attempt("llm.token")

    async def competing_failures_are_queued() -> None:
        while session.output_queue_size < 3:
            await asyncio.sleep(0)

    await transport.inbound.put(SessionStart(**event_fields("duplicate-start")))
    await asyncio.wait_for(competing_failures_are_queued(), timeout=0.1)
    transport.release_sends()
    await asyncio.wait_for(task, timeout=1)

    failed = [event for event in transport.sent if isinstance(event, TurnFailed)]
    assert len(failed) == 1
    assert failed[0].reason == "session_aborted"


@pytest.mark.asyncio
async def test_blocking_provider_close_is_bounded_by_shutdown_grace() -> None:
    class BlockingCloseProvider:
        def __init__(self) -> None:
            self.close_started = asyncio.Event()
            self.close_cancelled = asyncio.Event()

        async def stream(
            self,
            messages: Sequence[ChatMessage],
            turn_id: int,
            session_id: str | None = None,
        ) -> AsyncIterator[str]:
            del messages, turn_id, session_id
            if False:
                yield ""

        async def aclose(self) -> None:
            self.close_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.close_cancelled.set()

    transport = FakeTransport()
    session, _ = make_session(
        transport,
        service_settings=settings(shutdown_grace_ms=10),
    )
    provider = BlockingCloseProvider()
    session._provider = provider
    task = asyncio.create_task(session.run())
    await start(transport)

    await session.request_shutdown()
    await asyncio.wait_for(task, timeout=0.1)
    await asyncio.wait_for(session.wait_closed(), timeout=0.1)

    assert provider.close_started.is_set()
    assert provider.close_cancelled.is_set()
    assert transport.close_code == 1000


@pytest.mark.asyncio
async def test_cancellation_during_provider_close_still_finalizes_session() -> None:
    class BlockingCloseProvider:
        def __init__(self) -> None:
            self.close_started = asyncio.Event()

        async def stream(
            self,
            messages: Sequence[ChatMessage],
            turn_id: int,
            session_id: str | None = None,
        ) -> AsyncIterator[str]:
            del messages, turn_id, session_id
            if False:
                yield ""

        async def aclose(self) -> None:
            self.close_started.set()
            await asyncio.Event().wait()

    transport = FakeTransport()
    session, _ = make_session(transport)
    provider = BlockingCloseProvider()
    session._provider = provider
    task = asyncio.create_task(session.run())
    await start(transport)

    await session.request_shutdown()
    await asyncio.wait_for(provider.close_started.wait(), timeout=0.1)
    session.force_cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(session.wait_closed(), timeout=0.1)

    assert transport.close_code == 1000
