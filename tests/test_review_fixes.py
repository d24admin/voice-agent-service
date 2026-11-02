from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

import pytest

from app.api.events import AudioEnd, AudioFrame, SessionStart, SessionStop, TurnFailed
from app.clock import SystemClock
from app.conversation.call_session import CallSession
from app.metrics import MetricsRegistry
from app.providers.llm import ChatMessage
from tests.conftest import FakeTransport, event_fields, live_session_tasks
from tests.test_session import settings, start


class BlockingProvider:
    async def stream(
        self,
        messages: Sequence[ChatMessage],
        turn_id: int,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        del messages, turn_id, session_id
        yield "working"
        await asyncio.Event().wait()


def session_with(
    transport: FakeTransport,
    provider: object,
    registry: MetricsRegistry,
    *,
    session_id: str = "session-1",
) -> CallSession:
    return CallSession(
        session_id=session_id,
        settings=settings(),
        transport=transport,
        provider=provider,  # type: ignore[arg-type]
        registry=registry,
        clock=SystemClock(),
    )


@pytest.mark.asyncio
async def test_fatal_protocol_violation_fails_active_turn_as_session_aborted() -> None:
    transport = FakeTransport()
    registry = MetricsRegistry()
    session = session_with(transport, BlockingProvider(), registry)
    task = asyncio.create_task(session.run())
    await start(transport)
    await transport.inbound.put(AudioFrame(text="hello", **event_fields("frame")))
    await transport.inbound.put(AudioEnd(**event_fields("end")))
    await transport.wait_for_sent("llm.token")

    await transport.inbound.put(SessionStart(**event_fields("duplicate")))
    failed = await transport.wait_for_sent("turn.failed")
    await asyncio.wait_for(task, timeout=0.3)

    assert isinstance(failed, TurnFailed)
    assert failed.reason == "session_aborted"
    assert failed.metrics["outcome"] == "failed"
    assert transport.close_code == 1008


@pytest.mark.asyncio
async def test_disconnect_during_stream_records_internal_interruption() -> None:
    class DisconnectOnToken(FakeTransport):
        async def send(self, event: object) -> None:
            await self.attempted.put(event)
            if getattr(event, "type", None) == "llm.token":
                raise ConnectionError("peer disconnected")
            self.sent.append(event)

    transport = DisconnectOnToken()
    registry = MetricsRegistry()
    session = session_with(transport, BlockingProvider(), registry)
    task = asyncio.create_task(session.run())
    await start(transport)
    await transport.inbound.put(AudioFrame(text="hello", **event_fields("frame")))
    await transport.inbound.put(AudioEnd(**event_fields("end")))
    await transport.wait_for_attempt("llm.token")

    with pytest.raises(ConnectionError):
        await asyncio.wait_for(task, timeout=0.3)

    summary = registry.summary("session-1")
    assert summary is not None
    assert summary["counts"]["interrupted"] == 1
    assert summary["turns"][0]["reason"] == "session_ended"
    assert live_session_tasks("session-1") == []


@pytest.mark.asyncio
async def test_completed_assistant_text_is_in_next_provider_request_history() -> None:
    class HistoryProvider:
        def __init__(self) -> None:
            self.requests: list[tuple[str, int, tuple[ChatMessage, ...]]] = []

        async def stream(
            self,
            messages: Sequence[ChatMessage],
            turn_id: int,
            session_id: str | None = None,
        ) -> AsyncIterator[str]:
            self.requests.append((session_id or "", turn_id, tuple(messages)))
            yield "Answer"
            yield f"{turn_id}."

    transport = FakeTransport()
    registry = MetricsRegistry()
    provider = HistoryProvider()
    session = session_with(transport, provider, registry)
    task = asyncio.create_task(session.run())
    await start(transport)
    for turn, caller_text in ((1, "first question"), (2, "second question")):
        await transport.inbound.put(
            AudioFrame(text=caller_text, **event_fields(f"frame-{turn}"))
        )
        await transport.inbound.put(AudioEnd(**event_fields(f"end-{turn}")))
        while not any(
            event.type == "turn.completed" and event.turn_id == turn
            for event in transport.sent
        ):
            await asyncio.sleep(0)
    await transport.inbound.put(SessionStop(**event_fields("stop")))
    await task

    assert [message.role for message in provider.requests[0][2]] == ["system", "user"]
    assert [message.role for message in provider.requests[1][2]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert provider.requests[1][2][2].content == "Answer 1."


@pytest.mark.asyncio
async def test_reused_session_id_starts_a_fresh_metrics_run() -> None:
    class OneSentenceProvider:
        async def stream(
            self,
            messages: Sequence[ChatMessage],
            turn_id: int,
            session_id: str | None = None,
        ) -> AsyncIterator[str]:
            del messages, turn_id, session_id
            yield "Done."

    registry = MetricsRegistry()
    for call_number in (1, 2):
        transport = FakeTransport()
        session = session_with(transport, OneSentenceProvider(), registry)
        task = asyncio.create_task(session.run())
        await start(transport)
        await transport.inbound.put(
            AudioFrame(text="hello", **event_fields(f"frame-{call_number}"))
        )
        await transport.inbound.put(AudioEnd(**event_fields(f"end-{call_number}")))
        await transport.wait_for_sent("turn.completed")
        await transport.inbound.put(SessionStop(**event_fields(f"stop-{call_number}")))
        await task

    summary = registry.summary("session-1")
    assert summary is not None
    assert summary["counts"]["completed"] == 1
    assert [turn["turn_id"] for turn in summary["turns"]] == [1]
