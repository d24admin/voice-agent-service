from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

import client
from client import (
    BARGE_IN_SCRIPT,
    DEFAULT_SCRIPT,
    CallResult,
    render_event,
    render_summary,
    run_call,
)
from client.__main__ import main
from app.metrics import TurnMetrics, TurnOutcome


def event(event_type: str, **extra: Any) -> dict[str, Any]:
    return {
        "type": event_type,
        "event_id": f"server-{event_type}",
        "session_id": "demo-1",
        "timestamp": "2026-10-31T12:00:00+00:00",
        **extra,
    }


def completed(turn_id: int) -> dict[str, Any]:
    return event(
        "turn.completed",
        turn_id=turn_id,
        metrics={
            "turn_id": turn_id,
            "outcome": "completed",
            "reason": None,
            "ttft_ms": 20.0,
            "ttfa_ms": 40.0,
            "total_latency_ms": 80.0,
            "token_count": 2,
            "audio_chunk_count": 1,
        },
    )


class FakeSocket:
    def __init__(self, incoming: list[dict[str, Any]], *, guard_barge: bool = False) -> None:
        self._incoming = iter(incoming)
        self.sent: list[dict[str, Any]] = []
        self.guard_barge = guard_barge

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    async def recv(self) -> str:
        item = next(self._incoming)
        if (
            self.guard_barge
            and item["type"] == "audio.chunk"
            and item["turn_id"] == 1
            and item["sequence"] == 0
        ):
            assert not any(
                message.get("text") == "I need to interrupt"
                for message in self.sent
            )
        return json.dumps(item)


class FakeConnection:
    def __init__(self, socket: FakeSocket) -> None:
        self.socket = socket

    async def __aenter__(self) -> FakeSocket:
        return self.socket

    async def __aexit__(self, *args: object) -> None:
        return None


def metric_summary(turns: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "session_id": "demo-1",
        "counts": {
            "started": len(turns),
            "completed": sum(turn["outcome"] == "completed" for turn in turns),
            "interrupted": sum(turn["outcome"] == "interrupted" for turn in turns),
            "failed": 0,
        },
        "ttft": {"sample_count": 1, "p50": 20.0, "p95": 20.0},
        "ttfa": {"sample_count": 1, "p50": 40.0, "p95": 40.0},
        "total_latency": {"sample_count": 1, "p50": 80.0, "p95": 80.0},
        "timeouts": 0,
        "overloads": 0,
        "stale_event_drops": 0,
        "turns": turns,
    }


@pytest.mark.asyncio
async def test_default_script_uses_multiple_frames_and_both_endpoint_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert DEFAULT_SCRIPT.explicit_frames == ("Hello", "I need", "some help")
    assert len(DEFAULT_SCRIPT.silence_frames) >= 1
    incoming = [
        event("session.ready"),
        event("transcript.partial", utterance_id=1, text="Hello"),
        event("transcript.final", utterance_id=1, text="Hello I need some help"),
        event("llm.token", turn_id=1, token="Sure."),
        event("audio.chunk", turn_id=1, sequence=0, text="Sure."),
        completed(1),
        event("transcript.final", utterance_id=2, text="Anything else"),
        completed(2),
    ]
    socket = FakeSocket(incoming)
    monkeypatch.setattr(client, "_open_websocket", lambda _url: FakeConnection(socket))
    monkeypatch.setattr(
        client,
        "_fetch_metrics",
        lambda *_args: async_value(
            metric_summary([completed(1)["metrics"], completed(2)["metrics"]])
        ),
    )

    result = await run_call("ws://localhost:8000", "default", "demo-1")

    assert result.outcome == "completed"
    sent_types = [message["type"] for message in socket.sent]
    assert sent_types.count("audio.frame") >= 4
    assert sent_types.count("audio.end") == 1
    assert sent_types[-1] == "session.stop"


async def async_value(value: Any) -> Any:
    return value


@pytest.mark.asyncio
async def test_barge_script_waits_for_audio_and_rejects_stale_old_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert BARGE_IN_SCRIPT.trigger == "first_audio_chunk"
    interruption_metrics = {
        "turn_id": 1,
        "outcome": "interrupted",
        "reason": "barge_in",
        "ttft_ms": 10.0,
        "ttfa_ms": 30.0,
        "total_latency_ms": 35.0,
        "token_count": 2,
        "audio_chunk_count": 1,
    }
    incoming = [
        event("session.ready"),
        event("transcript.final", utterance_id=1, text="old request"),
        event("audio.chunk", turn_id=1, sequence=0, text="Old audio."),
        event(
            "turn.interrupted",
            turn_id=1,
            reason="barge_in",
            metrics=interruption_metrics,
        ),
        event("transcript.final", utterance_id=2, text="I need to interrupt"),
        event("audio.chunk", turn_id=2, sequence=0, text="New audio."),
        completed(2),
    ]
    socket = FakeSocket(incoming, guard_barge=True)
    monkeypatch.setattr(client, "_open_websocket", lambda _url: FakeConnection(socket))
    monkeypatch.setattr(
        client,
        "_fetch_metrics",
        lambda *_args: async_value(
            metric_summary([interruption_metrics, completed(2)["metrics"]])
        ),
    )

    result = await run_call("ws://localhost:8000", "barge_in", "demo-1")

    assert result.interrupted_turns == 1
    assert any(
        message.get("text") == "I need to interrupt" for message in socket.sent
    )

    stale = incoming[:4] + [
        event("audio.chunk", turn_id=1, sequence=1, text="Stale audio."),
        *incoming[4:],
    ]
    stale_socket = FakeSocket(stale, guard_barge=True)
    monkeypatch.setattr(
        client, "_open_websocket", lambda _url: FakeConnection(stale_socket)
    )
    with pytest.raises(RuntimeError, match="stale audio"):
        await run_call("ws://localhost:8000", "barge_in", "demo-1")


def test_human_rendering_includes_stream_and_metric_details(capsys: Any) -> None:
    for item in (
        event("transcript.partial", utterance_id=1, text="partial words"),
        event("transcript.final", utterance_id=1, text="final words"),
        event("llm.token", turn_id=1, token="hello"),
        event("audio.chunk", turn_id=1, sequence=0, text="Hello."),
        completed(1),
    ):
        render_event(item, verbose=False)
    metrics = TurnMetrics(turn_id=1, baseline_ns=0)
    metrics.ttft_ms = 20
    metrics.ttfa_ms = 40
    metrics.finalize(TurnOutcome.COMPLETED, None, 80_000_000)
    render_summary(
        CallResult(
            session_id="demo-1",
            outcome="completed",
            turn_metrics=(metrics,),
            interrupted_turns=0,
            errors=(),
        )
    )

    output = capsys.readouterr().out
    for expected in (
        "partial words",
        "final words",
        "hello",
        "Hello.",
        "completed",
        "TTFT",
        "TTFA",
        "total",
        "server summary",
    ):
        assert expected in output


def test_verbose_rendering_is_json_lines(capsys: Any) -> None:
    item = event("pong")
    render_event(item, verbose=True)
    assert json.loads(capsys.readouterr().out) == item


def test_unknown_scenario_and_connection_failure_are_concise(
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    assert main(["--scenario", "unknown"]) != 0
    assert "unknown scenario" in capsys.readouterr().err

    async def fail(*_args: object, **_kwargs: object) -> CallResult:
        raise OSError("connection refused")

    monkeypatch.setattr("client.__main__.run_call", fail)
    assert main(["--scenario", "default"]) == 1
    error = capsys.readouterr().err
    assert "connection refused" in error
    assert "Traceback" not in error
