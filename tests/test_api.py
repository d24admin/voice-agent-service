from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.clock import SystemClock
from app.config import Settings
from app.api.events import AudioEnd, AudioFrame, SessionStart, SessionStop
from app.api.server import create_app
from app.metrics import TurnMetrics, TurnOutcome
from app.providers.llm import ChatMessage
from app.conversation.call_session import CallSession
from app.conversation.session_manager import SessionManager
from tests.conftest import FakeTransport, event_fields, live_session_tasks


def api_settings(**overrides: int) -> Settings:
    values = {
        "stt_partial_delay_ms": 1,
        "silence_timeout_ms": 30,
        "llm_token_delay_ms": 1,
        "tts_sentence_delay_ms": 1,
        "first_token_timeout_ms": 100,
        "generation_timeout_ms": 500,
        "cancellation_timeout_ms": 100,
        "backpressure_timeout_ms": 50,
        "shutdown_grace_ms": 200,
        "inbound_queue_size": 16,
        "token_queue_size": 8,
        "outbound_data_capacity": 32,
        "outbound_lifecycle_capacity": 4,
        "metrics_session_capacity": 8,
        "metrics_turns_per_session": 8,
    }
    values.update(overrides)
    return Settings(**values)


def payload(event_type: str, session_id: str = "ws-1", **extra: Any) -> dict[str, Any]:
    return {
        "type": event_type,
        "event_id": f"event-{event_type}",
        "session_id": session_id,
        "timestamp": datetime.now(UTC).isoformat(),
        **extra,
    }


def receive_until(ws: Any, event_type: str) -> dict[str, Any]:
    while True:
        event = ws.receive_json()
        if event["type"] == event_type:
            return event


def test_health_and_empty_metrics_routes_are_typed() -> None:
    app = create_app(api_settings())
    with TestClient(app) as client:
        health = client.get("/health")
        metrics = client.get("/metrics")

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "service": "voice-agent-service"}
    assert metrics.status_code == 200
    assert metrics.json()["counts"] == {
        "started": 0,
        "completed": 0,
        "interrupted": 0,
        "failed": 0,
    }
    for name in ("ttft", "ttfa", "total_latency"):
        assert metrics.json()[name] == {"sample_count": 0, "p50": None, "p95": None}


def test_session_metrics_are_scoped_ordered_and_query_is_validated() -> None:
    app = create_app(api_settings())
    first = app.state.metrics.get_or_create("first")
    other = app.state.metrics.get_or_create("other")
    for turn_id, latency in ((1, 10), (2, 20)):
        metrics = TurnMetrics(turn_id=turn_id, baseline_ns=0)
        metrics.ttft_ms = latency
        metrics.ttfa_ms = latency + 1
        metrics.finalize(TurnOutcome.COMPLETED, None, latency * 1_000_000)
        first.record_turn(metrics)
    unrelated = TurnMetrics(turn_id=1, baseline_ns=0)
    unrelated.finalize(TurnOutcome.FAILED, "provider_error", 1)
    other.record_turn(unrelated)

    with TestClient(app) as client:
        response = client.get("/metrics", params={"session_id": "first"})
        missing = client.get("/metrics", params={"session_id": "missing"})
        invalid = client.get("/metrics?session_id=")

    assert response.status_code == 200
    assert response.json()["session_id"] == "first"
    assert [turn["turn_id"] for turn in response.json()["turns"]] == [1, 2]
    assert response.json()["counts"]["failed"] == 0
    assert missing.status_code == 404
    assert invalid.status_code == 422


def test_ping_and_recoverable_errors_leave_websocket_usable() -> None:
    app = create_app(api_settings())
    with TestClient(app) as client:
        with client.websocket_connect("/ws/ws-1") as ws:
            ws.send_json(payload("ping"))
            assert receive_until(ws, "pong")["session_id"] == "ws-1"

            ws.send_text("{not-json")
            error = receive_until(ws, "error")
            assert error["fatal"] is False

            ws.send_json(payload("unsupported.event"))
            assert receive_until(ws, "error")["fatal"] is False

            ws.send_json(payload("session.start"))
            assert receive_until(ws, "session.ready")["type"] == "session.ready"
            ws.send_json(payload("ping"))
            assert receive_until(ws, "pong")["type"] == "pong"
            ws.send_json(payload("session.stop"))
            with pytest.raises(WebSocketDisconnect) as closed:
                ws.receive_json()
            assert closed.value.code == 1000


@pytest.mark.parametrize(
    "events",
    [
        [payload("audio.frame", text="before-start")],
        [payload("session.start"), payload("session.start")],
        [payload("session.start", session_id="wrong")],
    ],
)
def test_fatal_protocol_policy_emits_error_and_closes_1008(
    events: list[dict[str, Any]],
) -> None:
    app = create_app(api_settings())
    with TestClient(app) as client:
        with client.websocket_connect("/ws/ws-1") as ws:
            for index, event in enumerate(events):
                ws.send_json(event)
                if (
                    index == 0
                    and event["type"] == "session.start"
                    and event["session_id"] == "ws-1"
                ):
                    receive_until(ws, "session.ready")
            error = receive_until(ws, "error")
            assert error["fatal"] is True
            with pytest.raises(WebSocketDisconnect) as closed:
                ws.receive_json()
            assert closed.value.code == 1008


def test_oversized_message_closes_1009() -> None:
    app = create_app(api_settings(max_message_bytes=128))
    with TestClient(app) as client:
        with client.websocket_connect("/ws/ws-1") as ws:
            ws.send_text("x" * 129)
            with pytest.raises(WebSocketDisconnect) as closed:
                ws.receive_json()
            assert closed.value.code == 1009


def test_inbound_overload_closes_1013() -> None:
    app = create_app(
        api_settings(
            inbound_queue_size=1,
            stt_partial_delay_ms=100,
            backpressure_timeout_ms=5,
        )
    )
    with TestClient(app) as client:
        with client.websocket_connect("/ws/ws-1") as ws:
            ws.send_json(payload("session.start"))
            receive_until(ws, "session.ready")
            for index in range(8):
                ws.send_json(payload("audio.frame", event_id=f"f-{index}", text="word"))
            with pytest.raises(WebSocketDisconnect) as closed:
                while True:
                    ws.receive_json()
            assert closed.value.code == 1013


def test_oversized_session_id_closes_1008() -> None:
    app = create_app(api_settings())
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as closed:
            with client.websocket_connect(f"/ws/{'x' * 129}") as ws:
                ws.receive_json()
        assert closed.value.code == 1008


@pytest.mark.asyncio
async def test_session_manager_shutdown_interrupts_and_awaits_active_session() -> None:
    class BlockingProvider:
        async def stream(
            self, messages: list[ChatMessage], turn_id: int
        ) -> Any:
            del messages, turn_id
            yield "working"
            await asyncio.Event().wait()

    selected = api_settings(shutdown_grace_ms=200)
    transport = FakeTransport()
    app = create_app(selected)
    session = CallSession(
        session_id="shutdown-1",
        settings=selected,
        transport=transport,
        provider=BlockingProvider(),
        registry=app.state.metrics,
        clock=SystemClock(),
    )
    manager = SessionManager()
    manager.register(session)
    task = asyncio.create_task(session.run())
    await transport.inbound.put(
        SessionStart(**event_fields("start", session_id="shutdown-1"))
    )
    await transport.wait_for_sent("session.ready")
    await transport.inbound.put(
        AudioFrame(
            text="hello",
            **event_fields("frame", session_id="shutdown-1"),
        )
    )
    await transport.inbound.put(
        AudioEnd(**event_fields("end", session_id="shutdown-1"))
    )
    await transport.wait_for_sent("llm.token")

    await manager.close_all(timeout_s=0.2)
    await asyncio.wait_for(task, timeout=0.2)

    terminal = next(event for event in transport.sent if event.type == "turn.interrupted")
    assert terminal.reason == "session_ended"
    assert live_session_tasks("shutdown-1") == []


@pytest.mark.asyncio
async def test_lifecycle_logs_exclude_conversation_and_raw_provider_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret_transcript = "private customer sentence"
    secret_url = "https://user:credential@example.invalid/private"

    class FailingProvider:
        async def stream(self, messages: Any, turn_id: int) -> Any:
            del messages, turn_id
            raise RuntimeError(secret_url)
            yield

    selected = api_settings()
    transport = FakeTransport()
    app = create_app(selected)
    session = CallSession(
        session_id="log-1",
        settings=selected,
        transport=transport,
        provider=FailingProvider(),
        registry=app.state.metrics,
        clock=SystemClock(),
    )
    with caplog.at_level(logging.INFO, logger="voice_agent"):
        task = asyncio.create_task(session.run())
        await transport.inbound.put(
            SessionStart(**event_fields("start", session_id="log-1"))
        )
        await transport.wait_for_sent("session.ready")
        await transport.inbound.put(
            AudioFrame(
                text=secret_transcript,
                **event_fields("frame", session_id="log-1"),
            )
        )
        await transport.inbound.put(
            AudioEnd(**event_fields("end", session_id="log-1"))
        )
        await transport.wait_for_sent("turn.failed")
        await transport.inbound.put(
            SessionStop(**event_fields("stop", session_id="log-1"))
        )
        await task

    assert "session_started" in caplog.text
    assert "turn_started" in caplog.text
    assert "turn_terminal" in caplog.text
    assert secret_transcript not in caplog.text
    assert secret_url not in caplog.text
