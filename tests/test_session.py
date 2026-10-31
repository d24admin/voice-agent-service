from __future__ import annotations

import asyncio

import pytest

from app.clock import SystemClock
from app.config import Settings
from app.events import (
    AudioChunk,
    AudioEnd,
    AudioFrame,
    LLMToken,
    SessionStart,
    SessionStop,
    TranscriptFinal,
    TranscriptPartial,
    TurnCompleted,
)
from app.metrics import MetricsRegistry
from app.providers.llm import MockLLMProvider
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
        "cancellation_timeout_ms": 100,
        "backpressure_timeout_ms": 100,
        "shutdown_grace_ms": 500,
        "inbound_queue_size": 16,
        "token_queue_size": 8,
        "outbound_data_capacity": 32,
        "outbound_lifecycle_capacity": 4,
    }
    values.update(overrides)
    return Settings(**values)


def make_session(
    transport: FakeTransport,
    *,
    service_settings: Settings | None = None,
) -> tuple[CallSession, MetricsRegistry]:
    selected = settings() if service_settings is None else service_settings
    registry = MetricsRegistry(session_capacity=8, turns_per_session=8)
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
